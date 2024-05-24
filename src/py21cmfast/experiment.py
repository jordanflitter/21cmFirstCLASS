"""
This module is responsible for running 21cmSense as part of 21cmFirstCLASS,
for computing the noise of an experiment.
"""
import numpy as np
import tqdm
import warnings
from scipy.interpolate import interp1d
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum, hera
from astropy import units as un
from astropy.cosmology import Planck18
from py21cmsense.theory import TheorySpline
from scipy.interpolate import RectBivariateSpline
from typing import Optional

# This is a useful class that gathers all the information from 21cmSense
class NOISE_DATA():

    def __init__(self,
                 power_spectrum,
                 z_values,
                 signal,
                 noise,
                 k_noise,
                 sensitivities
    ):
        self.power_spectrum = power_spectrum
        self.z_values = z_values
        self.signal = signal
        self.noise = noise
        self.k_noise = k_noise
        self.sensitivities = sensitivities


# We need to define this class in order to give 21cmSense the power spectrum from 21cmFirstCLASS as an input
class THEORY_MODEL(TheorySpline):

    use_littleh = False

    def __init__(self,z_values,k_values,ps_values):
        self.k = k_values
        self.z = z_values
        self.coeval_ps = ps_values
        self.spline = RectBivariateSpline(z_values, k_values, ps_values, ky=1)

# Useful function for defining input dictionaries in the format needed for 21cmSense.
def get_input_dicts(kind,**kwargs):

    all_kwargs = {"beam": ["dish_size"],
                  "layout": ["hex_num", "separation", "row_separation", "split_core", "outriggers"],
                  "observatory": ["latitude", "Trcv", "max_antpos", "min_antpos"],
                  "observation": ["time_per_day", "bandwidth", "n_days", "spectral_index",
                                  "tsky_amplitude", "tsky_ref_freq", "n_channels", "coherent",
                                  "track", "lst_bin_size", "integration_time", "redundancy_tol",
                                  "baseline_filters", "use_approximate_cosmo", "cosmo"],
                  "sensitivity": ["horizon_buffer", "foreground_model", "no_ns_baselines", "systematics_mask"]}

    input_kwargs = {}
    for kwarg in all_kwargs[kind]:
        if kwarg in kwargs:
            input_kwargs[kwarg] = kwargs[kwarg]
    return input_kwargs

# Function that allows to run 21cmSense easily
def run_21cmSense(
    kwargs: [dict],
    z_values: Optional[list] = [7.9,],
):
    """
    Run 21cmSense with the keyword arguments specified by the user.

    Parameters
    ----------
    kwargs: dictionary
        All keywords to pass to 21cmSense.
    z_values: list, optional
        List of redshifts for which the noise will be computed by 21cmSense.

    Returns
    -------
    noise_data: :class:`~py21cmfast.experiment.NOISE_DATA`
        The noise data object that stores the data of the noise.

    """


    # Frequency of the 21cm line
    f_21 = 1420.40575177*un.MHz

    # Define input kwargs for the various modules of 21cmSense
    beam_kwargs = get_input_dicts("beam",**kwargs)
    layout_kwargs = get_input_dicts("layout",**kwargs)
    observatory_kwargs = get_input_dicts("observatory",**kwargs)
    observation_kwargs = get_input_dicts("observation",**kwargs)
    sensitivity_kwargs = get_input_dicts("sensitivity",**kwargs)

    # Set a "theory_model" to be used as an input to 21cmSense
    if "power_spectrum" in kwargs:
        power_spectrum = kwargs["power_spectrum"]
        z_values = power_spectrum.z_values
        sensitivity_kwargs["theory_model"] = THEORY_MODEL(power_spectrum.z_values,
                                                          power_spectrum.k_values,
                                                          power_spectrum.ps_values)
    else:
        power_spectrum = None

    # Set the coherent parameter based on the foreground scenario
    if "coherent" not in observation_kwargs:
        if ("foreground_model" not in sensitivity_kwargs
            or
           (sensitivity_kwargs["foreground_model"] in ["optimistic","moderate"])):
            observation_kwargs["coherent"] = True
        else:
            observation_kwargs["coherent"] = False

    # Initialize output
    if "power_spectrum" in kwargs:
        signal = {}
        h = power_spectrum.cosmo_params.hlittle
    elif "cosmo" in kwargs:
        h = kwargs["cosmo"].h
    else:
        h = Planck18.h
    noise = {}
    k_noise = {}
    sensitivities = {}

    # Run 21cmSense for each redshift.
    # Note: 21cmSense default settings are to display progress bars.
    #       Since we are about to run 21cmSense for each chunk in the lightcone box,
    #       it is recommended to set "PROGRESS = False" in config.py prior
    #       the installation of 21cmSense.
    for z_ind,z in tqdm.tqdm(enumerate(z_values),
                             desc="21cmSense",
                             unit="redshift",
                             disable=False,
                             total=len(z_values)):

        # Set parameters for 21cmSense
        beam = GaussianBeam(frequency = f_21/(1.+z),
                            **beam_kwargs)
        hera_layout = hera(**layout_kwargs)
        observatory = Observatory(antpos = hera_layout,
                                  beam = beam,
                                  **observatory_kwargs)
        observation = Observation(observatory = observatory,
                                  **observation_kwargs)
        sensitivity = PowerSpectrum(observation = observation,
                                    **sensitivity_kwargs)

        # Run 21cmSense!
        with np.errstate(divide='ignore'): # Don't show division by 0 warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Don't show extrapolation warnings
                noise_z = sensitivity.calculate_sensitivity_1d().value
        k_sense = sensitivity.k1d.value * h # 1/Mpc

        # Interpolate the 21cmFast data at the 21cmSense k-bins
        if "power_spectrum" in kwargs:
            signal_z = interp1d(power_spectrum.k_values,
                                power_spectrum.ps_values[z_ind,:],
                                kind='cubic',bounds_error=False)(k_sense)

        # Prepare output
        if "power_spectrum" in kwargs:
            signal[z] = signal_z
        noise[z] = noise_z
        k_noise[z] = k_sense
        sensitivities[z] = sensitivity

    # Prepare output
    if "power_spectrum" not in kwargs:
        signal = None

    noise_data = NOISE_DATA(power_spectrum = power_spectrum,
                            z_values = z_values,
                            signal = signal,
                            noise = noise,
                            k_noise = k_noise,
                            sensitivities = sensitivities)

    # Return output
    return noise_data
