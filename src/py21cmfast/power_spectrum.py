"""
This module is responsible for computing the power spectrum given a lightcone box.
"""

# define functions to calculate PS, following py21cmmc
import numpy as np
from powerbox.tools import get_power
from scipy.interpolate import interp1d
from typing import Optional
from .outputs import LightCone
from .inputs import global_params

# Frequency of the 21cm line
f_21 = 1420.40575177 # MHz

# New structure for convenience
class POWER_SPECTRUM():

    def __init__(self,
                 kind,
                 user_params,
                 cosmo_params,
                 astro_params,
                 flag_options,
                 z_values,
                 k_values,
                 ps_values
    ):
        self.user_params = user_params
        self.kind = kind
        self.cosmo_params = cosmo_params
        self.astro_params = astro_params
        self.flag_options = flag_options
        self.z_values = z_values
        self.k_values = k_values
        self.ps_values = ps_values

def compute_power(
   box,
   length,
   k_bins_edges,
   log_bins=True,
   ignore_kperp_zero=True,
   ignore_kpar_zero=False,
   ignore_k_zero=False,
):
    """
    Compute power spectrum of a given box.

    Parameters
    ----------
    box : nd-array
        A 3D array of some quantity.
    length: float
        Size of the box, in Mpc.
    k_bins_edges: nd-array
        The edges of the k-bins, in units of 1/Mpc.
    log_bins: bool, optional
        Whether the k-bins are logarithmically spaced or not. Default is True.
    ignore_kperp_zero: bool, optional
        Whether to ignore zero perpendicular modes. Default is True.
    ignore_kpar_zero: bool, optional
        Whether to ignore zero parallel modes. Default is False.
    ignore_k_zero: bool, optional
        Whether to ignore zero modes (parallel or perpendicular). Default is False.

    Returns
    -------
    power, k_values : nd-array
        Power spectrum of the box at the associated k_values.

    """
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape).astype(int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=k_bins_edges,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k
    return res

def coeval_power_spectrum(
    lightcone: [LightCone],
    redshift: [float],
    kind: Optional[str] = None,
    k_bins: Optional[np.ndarray] = None,
    logk: Optional[bool] = True,
):
    """
    Compute power spectrum of a coeval box from a lightcone object.

    Parameters
    ----------
    lightcone: :class:`~py21cmfast.wrapper.Lightcone`
        The lightcone object that stores the coeval box whose power spectrum
        shall be evaluated.
    redshift: float
        Redshift of the coeval box whose power spectrum
        shall be evaluated. The redshift of the
        manipulated coeval box has the value which is closest to the input redshift.
    kind: str
        The quantity for which the power spectrum is evaluated.
        Must be in the `coeval_boxes` dict in the lightcone.
        By default, will choose the first entry in the dict.
    k_bins: nd-array, optional
        The k-bins in which the power specturm is computed.
    logk: bool, optional
        Whether the k-bins are logarithmically spaced or not. Default is True.

    Returns
    -------
    power_spectrum : :class:`~py21cmfast.power_spectrum.POWER_SPECTRUM`
        A power spectrum object that contains the computed data.

    """

    # Let's find first what is kind
    if kind is None:
        kind = list(lightcone.lightcones.keys())[0]

    # We find the redshift from the coeval boxes that we have that is closest to the user's redshift
    coeval_redshifts = list(lightcone.coeval_boxes.keys())
    coeval_redshift = coeval_redshifts[np.argmin(np.abs(redshift - np.array(coeval_redshifts)))]
    # We extract the coeval box from the lightcone object
    coeval_box = lightcone.coeval_boxes[coeval_redshift][kind]

    # Get the edges of the k-bins (based on the bins of 21cmFAST version 2)
    if k_bins is None:
        Delta_k = 2*np.pi/lightcone.user_params.BOX_LEN # 1/Mpc
        k_max = Delta_k*lightcone.user_params.HII_DIM # 1/Mpc
        k_factor = 1.5
        k_bins_edges = []
        k_ceil = Delta_k
        while (k_ceil < k_max):
            k_bins_edges.append(k_ceil)
            if logk:
                k_ceil *= k_factor
            else:
                k_ceil += Delta_k
        k_bins_edges = np.array(k_bins_edges)
    else:
        if logk:
            k_bins_edges = np.sqrt(k_bins[1:]*k_bins[:-1])
            k_bins_edges = np.concatenate((
                                            [k_bins_edges[0]/(k_bins_edges[1]/k_bins_edges[0])],
                                             k_bins_edges,
                                            [k_bins_edges[-1]*(k_bins_edges[1]/k_bins_edges[0])]
                                          ))
        else:
            k_bins_edges = (k_bins[1:]+k_bins[:-1])/2
            k_bins_edges = np.concatenate((
                                            [k_bins_edges[0]-(k_bins_edges[1]-k_bins_edges[0])],
                                             k_bins_edges,
                                            [k_bins_edges[-1]+(k_bins_edges[1]-k_bins_edges[0])]
                                          ))

    # Compute the power spectrum
    power, k_values = compute_power(coeval_box,
                                    (lightcone.user_params.BOX_LEN,)*3,
                                    k_bins_edges,
                                    ignore_kperp_zero=False, # For the coeval power spectrum, we treat all the axes of the box to be the same
                                    log_bins=logk)
    # We compute the "dimensionless" power spectrum
    power *= k_values ** 3 / (2 * np.pi ** 2)
    ps_values = power

    # Return output
    power_spectrum = POWER_SPECTRUM(kind = kind,
                                    user_params = lightcone.user_params,
                                    cosmo_params = lightcone.cosmo_params,
                                    astro_params = lightcone.astro_params,
                                    flag_options = lightcone.flag_options,
                                    z_values = np.array([int(coeval_redshift*1e3+0.5)/1e3,]), # We round the coeval_redshift (can only be used for display)
                                    k_values = k_values,
                                    ps_values = ps_values)

    return power_spectrum

def lightcone_power_spectrum(
    lightcone: [LightCone],
    nchunks: [int] = 15,
    kind: Optional[str] = None,
    k_bins: Optional[np.ndarray] = None,
    freq_bands_boundaries: Optional[np.ndarray] = None,
    logk: Optional[bool] = True,
):

    """
    Compute power spectrum of a lightcone box from a lightcone object.

    Parameters
    ----------
    lightcone: :class:`~py21cmfast.wrapper.Lightcone`
        The lightcone object that stores the lightcone box whose power spectrum
        shall be evaluated.
    nchunks: int
        Number of chunks along the line-of-sight for which the lightcone box
        will be split. For each chunk, the power spectrum will be evaluated.
        Default is 15.
    kind: str
        The quantity for which the power spectrum is evaluated.
        Must be in the `coeval_boxes` dict in the lightcone.
        By default, will choose the first entry in the dict.
    k_bins: nd-array, optional
        The k-bins in which the power specturm is computed.
    freq_bands_boundaries: nd-array, optional
        If specified, the power spectrum will be interpolated at frequency bands
        whose centers are determined from freq_bands_boundaries (these are the
        edges of the frequency bins).
    logk: bool, optional
        Whether the k-bins are logarithmically spaced or not. Default is True.

    Returns
    -------
    power_spectrum : :class:`~py21cmfast.power_spectrum.POWER_SPECTRUM`
        A power spectrum object that contains the computed data.

    """

    # Let's find first what is kind
    if kind is None:
        kind = list(lightcone.lightcones.keys())[0]

    # Extract the lightcone box from the lightcone structure
    lightcone_box = lightcone.lightcones[kind]
    lightcone_redshifts = lightcone.lightcone_redshifts

    # Get the edges of the k-bins (based on the bins of 21cmFAST version 2)
    if k_bins is None:
        Delta_k = 2*np.pi/lightcone.user_params.BOX_LEN # 1/Mpc
        k_max = Delta_k*lightcone.user_params.HII_DIM # 1/Mpc
        k_factor = 1.5
        k_bins_edges = []
        k_ceil = Delta_k
        while (k_ceil < k_max):
            k_bins_edges.append(k_ceil)
            if logk:
                k_ceil *= k_factor
            else:
                k_ceil += Delta_k
        k_bins_edges = np.array(k_bins_edges)
    else:
        if logk:
            k_bins_edges = np.sqrt(k_bins[1:]*k_bins[:-1])
            k_bins_edges = np.concatenate((
                                            [k_bins_edges[0]/(k_bins_edges[1]/k_bins_edges[0])],
                                             k_bins_edges,
                                            [k_bins_edges[-1]*(k_bins_edges[1]/k_bins_edges[0])]
                                          ))
        else:
            k_bins_edges = (k_bins[1:]+k_bins[:-1])/2
            k_bins_edges = np.concatenate((
                                            [k_bins_edges[0]-(k_bins_edges[1]-k_bins_edges[0])],
                                             k_bins_edges,
                                            [k_bins_edges[-1]+(k_bins_edges[1]-k_bins_edges[0])]
                                          ))

    # We find the index of cosmic dawn
    ind_CD = np.where(lightcone.lightcone_redshifts >= global_params.Z_HEAT_MAX)[0][0]

    # Split the lightcone and its associated to cosmic dawn and dark ages part
    lightcone_box_CD = lightcone_box[:,:,:ind_CD]
    lightcone_redshifts_CD = lightcone_redshifts[:ind_CD]
    if lightcone.user_params.OUTPUT_AT_DARK_AGES:
        lightcone_box_DA = lightcone_box[:,:,ind_CD:]
        lightcone_redshifts_DA = lightcone_redshifts[ind_CD:]

    # COSMIC DAWN CALCULATION
    # Split the lightcone box into even chunks and find the redshift values that correspond to the middle of
    # these chunks
    chunk_indices_boundaries_array = np.round(np.linspace(0,lightcone_box_CD.shape[2]-1,nchunks+1))
    chunk_indices_boundaries = chunk_indices_boundaries_array.astype(int).tolist()
    chunk_indices = ((chunk_indices_boundaries_array[1:]+chunk_indices_boundaries_array[:-1])/2).astype(int).tolist()
    redshift_grid_CD = interp1d(np.arange(lightcone_box_CD.shape[2]),lightcone_redshifts_CD, kind='cubic')
    z_values_CD = redshift_grid_CD(chunk_indices)
    # Compute the power spectrum for each chunk
    ps_values_CD = np.zeros((len(z_values_CD),len(k_bins_edges)-1))
    for i in range(nchunks):
        # Get the chunk's boundaries
        start = chunk_indices_boundaries[i]
        end = chunk_indices_boundaries[i + 1]
        chunklen = (end - start) * lightcone.user_params.BOX_LEN/lightcone.user_params.HII_DIM
        # Compute the power spectrum
        power, k_values = compute_power(lightcone_box_CD[:, :, start:end],
                                        (lightcone.user_params.BOX_LEN, lightcone.user_params.BOX_LEN, chunklen),
                                        k_bins_edges,
                                        log_bins=logk)
        # We compute the "dimensionless" power spectrum
        power *= k_values ** 3 / (2 * np.pi ** 2)
        ps_values_CD[i,:] = power

    # If necssary, we do the above calculations for the dark ages as well
    if lightcone.user_params.OUTPUT_AT_DARK_AGES:
        # DARK AGES CALCULATION
        # Split the lightcone box into even chunks and find the redshift values that correspond to the middle of
        # these chunks
        chunk_indices_boundaries_array = np.round(np.linspace(0,lightcone_box_DA.shape[2]-1,nchunks+1))
        chunk_indices_boundaries = chunk_indices_boundaries_array.astype(int).tolist()
        chunk_indices = ((chunk_indices_boundaries_array[1:]+chunk_indices_boundaries_array[:-1])/2).astype(int).tolist()
        redshift_grid_DA = interp1d(np.arange(lightcone_box_DA.shape[2]),lightcone_redshifts_DA, kind='cubic')
        z_values_DA = redshift_grid_DA(chunk_indices)
        # Compute the power spectrum for each chunk
        ps_values_DA = np.zeros((len(z_values_DA),len(k_bins_edges)-1))
        for i in range(nchunks):
            # Get the chunk's boundaries
            start = chunk_indices_boundaries[i]
            end = chunk_indices_boundaries[i + 1]
            chunklen = (end - start) * lightcone.user_params.BOX_LEN/lightcone.user_params.HII_DIM
            # Compute the power spectrum
            power, k_values = compute_power(lightcone_box_DA[:, :, start:end],
                                            (lightcone.user_params.BOX_LEN, lightcone.user_params.BOX_LEN, chunklen),
                                            k_bins_edges,
                                            log_bins=logk)
            # We compute the "dimensionless" power spectrum
            power *= k_values ** 3 / (2 * np.pi ** 2)
            ps_values_DA[i,:] = power
        # Now, once we have cosmic dawn and dark ages values, we combine them
        z_values = np.insert(z_values_DA,0,z_values_CD)
        ps_values = np.vstack((ps_values_CD,ps_values_DA))
    else:
        z_values = z_values_CD
        ps_values = ps_values_CD


    # If frequencies are given, we would like to interpolate the power spectrum at the experiment's bands
    if freq_bands_boundaries is not None:
        # Find the redshift boundaries that correspond to the experiment's bands
        z_exp_boundaries = (f_21/freq_bands_boundaries - 1.)[::-1]
        z_exp_boundaries = z_exp_boundaries[z_exp_boundaries >= min(lightcone_redshifts)]
        # Split the lightcone according to the bands and define the redshift of each band to correspond to
        # the middle point in the band
        indices_grid = interp1d(lightcone_redshifts, np.arange(lightcone_box.shape[2]), kind='cubic')
        exp_indices_boundaries_array = indices_grid(z_exp_boundaries).astype(int)
        exp_indices = ((exp_indices_boundaries_array[1:]+exp_indices_boundaries_array[:-1])/2).astype(int).tolist()
        redshift_grid = interp1d(np.arange(lightcone_box.shape[2]),lightcone_redshifts, kind='cubic')
        z_exp_values = redshift_grid(exp_indices)
        # Interpolate the power spectrum at z_exp_values
        ps_exp_values = np.zeros((len(z_exp_values),len(k_values)))
        for k_ind in range(len(k_values)):
            ps_exp_values[:,k_ind] = np.exp(interp1d(z_values,np.log(ps_values[:,k_ind]),
                                                 kind='cubic',bounds_error=False)(z_exp_values))
            # This can happen sometimes (if the experiment's redshift values, that are associated with
            # the middle of the experiment's chunk, are outside the redshift grid, defined by z_values)
            if np.isnan(ps_exp_values[0,k_ind]):
                ps_exp_values[0,k_ind] = ps_values[0,k_ind]
            if np.isnan(ps_exp_values[-1,k_ind]):
                ps_exp_values[-1,k_ind] = ps_values[-1,k_ind]
        # Update output to experiment's values
        z_values = z_exp_values
        ps_values = ps_exp_values

    # Return output
    power_spectrum = POWER_SPECTRUM(kind = kind,
                                    user_params = lightcone.user_params,
                                    cosmo_params = lightcone.cosmo_params,
                                    astro_params = lightcone.astro_params,
                                    flag_options = lightcone.flag_options,
                                    z_values = z_values,
                                    k_values = k_values,
                                    ps_values = ps_values)

    return power_spectrum
