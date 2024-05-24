"""
Input parameter classes.

There are four input parameter/option classes, not all of which are required for any
given function. They are :class:`UserParams`, :class:`CosmoParams`, :class:`AstroParams`
and :class:`FlagOptions`. Each of them defines a number of variables, and all of these
have default values, to minimize the burden on the user. These defaults are accessed via
the ``_defaults_`` class attribute of each class. The available parameters for each are
listed in the documentation for each class below.

Along with these, the module exposes ``global_params``, a singleton object of type
:class:`GlobalParams`, which is a simple class providing read/write access to a number of parameters
used throughout the computation which are very rarely varied.
"""
import contextlib
import logging
import warnings
from astropy.cosmology import Planck15
from os import path
from pathlib import Path

from ._cfg import config
from ._data import DATA_PATH
from ._utils import StructInstanceWrapper, StructWithDefaults
from .c_21cmfast import ffi, lib

# JordanFlitter: Need numpy to make arrays
import numpy as np

logger = logging.getLogger("21cmFAST")

# Cosmology is from https://arxiv.org/pdf/1807.06209.pdf
# Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766 ** 2,
    Ob0=0.02242 / 0.6766 ** 2,
    H0=67.66,
)


class GlobalParams(StructInstanceWrapper):
    """
    Global parameters for 21cmFAST.

    This is a thin wrapper over an allocated C struct, containing parameter values
    which are used throughout various computations within 21cmFAST. It is a singleton;
    that is, a single python (and C) object exists, and no others should be created.
    This object is not "passed around", rather its values are accessed throughout the
    code.

    Parameters in this struct are considered to be options that should usually not have
    to be modified, and if so, typically once in any given script or session.

    Values can be set in the normal way, eg.:

    >>> global_params.ALPHA_UVB = 5.5

    The class also provides a context manager for setting parameters for a well-defined
    portion of the code. For example, if you would like to set ``Z_HEAT_MAX`` for a given
    run:

    >>> with global_params.use(Z_HEAT_MAX=25):
    >>>     p21c.run_lightcone(...)  # uses Z_HEAT_MAX=25 for the entire run.
    >>> print(global_params.Z_HEAT_MAX)
    35.0

    Attributes
    ----------
    ALPHA_UVB : float
        Power law index of the UVB during the EoR.  This is only used if `INHOMO_RECO` is
        True (in :class:`FlagOptions`), in order to compute the local mean free path
        inside the cosmic HII regions.
    EVOLVE_DENSITY_LINEARLY : bool
        Whether to evolve the density field with linear theory (instead of 1LPT or Zel'Dovich).
        If choosing this option, make sure that your cell size is
        in the linear regime at the redshift of interest. Otherwise, make sure you resolve
        small enough scales, roughly we find BOX_LEN/DIM should be < 1Mpc
    SMOOTH_EVOLVED_DENSITY_FIELD : bool
        If True, the zeldovich-approximation density field is additionally smoothed
        (aside from the implicit boxcar smoothing performed when re-binning the ICs from
        DIM to HII_DIM) with a Gaussian filter of width ``R_smooth_density*BOX_LEN/HII_DIM``.
        The implicit boxcar smoothing in ``perturb_field()`` bins the density field on
        scale DIM/HII_DIM, similar to what Lagrangian codes do when constructing Eulerian
        grids. In other words, the density field is quantized into ``(DIM/HII_DIM)^3`` values.
        If your usage requires smooth density fields, it is recommended to set this to True.
        This also decreases the shot noise present in all grid based codes, though it
        overcompensates by an effective loss in resolution. **Added in 1.1.0**.
    R_smooth_density : float
        Determines the smoothing length to use if `SMOOTH_EVOLVED_DENSITY_FIELD` is True.
    HII_ROUND_ERR : float
        Rounding error on the ionization fraction. If the mean xHI is greater than
        ``1 - HII_ROUND_ERR``, then finding HII bubbles is skipped, and a homogeneous
        xHI field of ones is returned. Added in  v1.1.0.
    FIND_BUBBLE_ALGORITHM : int, {1,2}
        Choose which algorithm used to find HII bubbles. Options are: (1) Mesinger & Furlanetto 2007
        method of overlapping spheres: paint an ionized sphere with radius R, centered on pixel
        where R is filter radius. This method, while somewhat more accurate, is slower than (2),
        especially in mostly ionized universes, so only use for lower resolution boxes
        (HII_DIM<~400). (2) Center pixel only method (Zahn et al. 2007). This is faster.
    N_POISSON : int
        If not using the halo field to generate HII regions, we provide the option of
        including Poisson scatter in the number of sources obtained through the conditional
        collapse fraction (which only gives the *mean* collapse fraction on a particular
        scale. If the predicted mean collapse fraction is less than  `N_POISSON * M_MIN`,
        then Poisson scatter is added to mimic discrete halos on the subgrid scale (see
        Zahn+2010).Use a negative number to turn it off.

        .. note:: If you are interested in snapshots of the same realization at several
                  redshifts,it is recommended to turn off this feature, as halos can
                  stochastically "pop in and out of" existence from one redshift to the next.
    R_OVERLAP_FACTOR : float
        When using USE_HALO_FIELD, it is used as a factor the halo's radius, R, so that the
        effective radius is R_eff = R_OVERLAP_FACTOR * R.  Halos whose centers are less than
        R_eff away from another halo are not allowed. R_OVERLAP_FACTOR = 1 is fully disjoint
        R_OVERLAP_FACTOR = 0 means that centers are allowed to lay on the edges of
        neighboring halos.
    DELTA_CRIT_MODE : int
        The delta_crit to be used for determining whether a halo exists in a cell
            0: delta_crit is constant (i.e. 1.686)
            1: delta_crit is the sheth tormen ellipsoidal collapse correction to delta_crit
    HALO_FILTER : int
        Filter for the density field used to generate the halo field with EPS
            0: real space top hat filter
            1: sharp k-space filter
            2: gaussian filter
    OPTIMIZE : bool
        Finding halos can be made more efficient if the filter size is sufficiently large that
        we can switch to the collapse fraction at a later stage.
    OPTIMIZE_MIN_MASS : float
        Minimum mass on which the optimization for the halo finder will be used.
    T_USE_VELOCITIES : bool
        Whether to use velocity corrections in 21-cm fields

        .. note:: The approximation used to include peculiar velocity effects works
                  only in the linear regime, so be careful using this (see Mesinger+2010)

    MAX_DVDR : float
        Maximum velocity gradient along the line of sight in units of the hubble parameter at z.
        This is only used in computing the 21cm fields.

        .. note:: Setting this too high can add spurious 21cm power in the early stages,
                  due to the 1-e^-tau ~ tau approximation (see Mesinger's 21cm intro paper and mao+2011).
                  However, this is still a good approximation at the <~10% level.

    VELOCITY_COMPONENT : int
        Component of the velocity to be used in 21-cm temperature maps (1=x, 2=y, 3=z)
    DELTA_R_FACTOR : float
        Factor by which to scroll through filter radius for halos
    DELTA_R_HII_FACTOR : float
        Factor by which to scroll through filter radius for bubbles
    HII_FILTER : int, {0, 1, 2}
        Filter for the Halo or density field used to generate ionization field:
        0. real space top hat filter
        1. k-space top hat filter
        2. gaussian filter
    INITIAL_REDSHIFT : float
        Used to perturb field
    CRIT_DENS_TRANSITION : float
        A transition value for the interpolation tables for calculating the number of ionising
        photons produced given the input parameters. Log sampling is desired, however the numerical
        accuracy near the critical density for collapse (i.e. 1.69) broke down. Therefore, below the
        value for `CRIT_DENS_TRANSITION` log sampling of the density values is used, whereas above
        this value linear sampling is used.
    MIN_DENSITY_LOW_LIMIT : float
        Required for using the interpolation tables for the number of ionising photons. This is a
        lower limit for the density values that is slightly larger than -1. Defined as a density
        contrast.
    RecombPhotonCons : int
        Whether or not to use the recombination term when calculating the filling factor for
        performing the photon non-conservation correction.
    PhotonConsStart : float
        A starting value for the neutral fraction where the photon non-conservation correction is
        performed exactly. Any value larger than this the photon non-conservation correction is not
        performed (i.e. the algorithm is perfectly photon conserving).
    PhotonConsEnd : float
        An end-point for where the photon non-conservation correction is performed exactly. This is
        required to remove undesired numerical artifacts in the resultant neutral fraction histories.
    PhotonConsAsymptoteTo : float
        Beyond `PhotonConsEnd` the photon non-conservation correction is extrapolated to yield
        smooth reionisation histories. This sets the lowest neutral fraction value that the photon
        non-conservation correction will be applied to.
    HEAT_FILTER : int
        Filter used for smoothing the linear density field to obtain the collapsed fraction:
            0: real space top hat filter
            1: sharp k-space filter
            2: gaussian filter
    CLUMPING_FACTOR : float
        Sub grid scale. If you want to run-down from a very high redshift (>50), you should
        set this to one.
    Z_HEAT_MAX : float
        Maximum redshift used in the Tk and x_e evolution equations.
        Temperature and x_e are assumed to be homogeneous at higher redshifts.
        Lower values will increase performance.
    R_XLy_MAX : float
        Maximum radius of influence for computing X-ray and Lya pumping in cMpc. This
        should be larger than the mean free path of the relevant photons.
    NUM_FILTER_STEPS_FOR_Ts : int
        Number of spherical annuli used to compute df_coll/dz' in the simulation box.
        The spherical annuli are evenly spaced in logR, ranging from the cell size to the box
        size. :func:`~wrapper.spin_temp` will create this many boxes of size `HII_DIM`,
        so be wary of memory usage if values are high.
    ZPRIME_STEP_FACTOR : float
        Logarithmic redshift step-size used in the z' integral.  Logarithmic dz.
        Decreasing (closer to unity) increases total simulation time for lightcones,
        and for Ts calculations.
    TK_at_Z_HEAT_MAX : float
        If positive, then overwrite default boundary conditions for the evolution
        equations with this value. The default is to use the value obtained from RECFAST.
        See also `XION_at_Z_HEAT_MAX`.
    XION_at_Z_HEAT_MAX : float
        If positive, then overwrite default boundary conditions for the evolution
        equations with this value. The default is to use the value obtained from RECFAST.
        See also `TK_at_Z_HEAT_MAX`.
    Pop : int
        Stellar Population responsible for early heating (2 or 3)
    Pop2_ion : float
        Number of ionizing photons per baryon for population 2 stellar species.
    Pop3_ion : float
        Number of ionizing photons per baryon for population 3 stellar species.
    NU_X_BAND_MAX : float
        This is the upper limit of the soft X-ray band (0.5 - 2 keV) used for normalising
        the X-ray SED to observational limits set by the X-ray luminosity. Used for performing
        the heating rate integrals.
    NU_X_MAX : float
        An upper limit (must be set beyond `NU_X_BAND_MAX`) for performing the rate integrals.
        Given the X-ray SED is modelled as a power-law, this removes the potential of divergent
        behaviour for the heating rates. Chosen purely for numerical convenience though it is
        motivated by the fact that observed X-ray SEDs apprear to turn-over around 10-100 keV
        (Lehmer et al. 2013, 2015)
    NBINS_LF : int
        Number of bins for the luminosity function calculation.
    P_CUTOFF : bool
        Turn on Warm-Dark-matter power suppression.
    M_WDM : float
        Mass of WDM particle in keV. Ignored if `P_CUTOFF` is False.
    g_x : float
        Degrees of freedom of WDM particles; 1.5 for fermions.
    OMn : float
        Relative density of neutrinos in the universe.
    OMk : float
        Relative density of curvature.
    OMr : float
        Relative density of radiation.
    OMtot : float
        Fractional density of the universe with respect to critical density. Set to
        unity for a flat universe.
    Y_He : float
        Helium fraction.
    wl : float
        Dark energy equation of state parameter (wl = -1 for vacuum )
    SHETH_b : float
        Sheth-Tormen parameter for ellipsoidal collapse (for HMF).

        .. note:: The best fit b and c ST params for these 3D realisations have a redshift,
                  and a ``DELTA_R_FACTOR`` dependence, as shown
                  in Mesinger+. For converged mass functions at z~5-10, set `DELTA_R_FACTOR=1.1`
                  and `SHETH_b=0.15` and `SHETH_c~0.05`.

                  For most purposes, a larger step size is quite sufficient and provides an
                  excellent match to N-body and smoother mass functions, though the b and c
                  parameters should be changed to make up for some "stepping-over" massive
                  collapsed halos (see Mesinger, Perna, Haiman (2005) and Mesinger et al.,
                  in preparation).

                  For example, at z~7-10, one can set `DELTA_R_FACTOR=1.3` and `SHETH_b=0.15`
                   and `SHETH_c=0.25`, to increase the speed of the halo finder.
    SHETH_c : float
        Sheth-Tormen parameter for ellipsoidal collapse (for HMF). See notes for `SHETH_b`.
    Zreion_HeII : float
        Redshift of helium reionization, currently only used for tau_e
    FILTER : int, {0, 1}
        Filter to use for smoothing.
        0. tophat
        1. gaussian
    external_table_path : str
        The system path to find external tables for calculation speedups. DO NOT MODIFY.
    R_BUBBLE_MIN : float
        Minimum radius of bubbles to be searched in cMpc. One can set this to 0, but should
        be careful with shot noise if running on a fine, non-linear density grid. Default
        is set to L_FACTOR which is (4PI/3)^(-1/3) = 0.620350491.
    M_MIN_INTEGRAL:
        Minimum mass when performing integral on halo mass function.
    M_MAX_INTEGRAL:
        Maximum mass when performing integral on halo mass function.
    T_RE:
        The peak gas temperatures behind the supersonic ionization fronts during reionization.
    VAVG:
        Avg value of the DM-b relative velocity [im km/s], ~0.9*SIGMAVCB (=25.86 km/s) normally.
    A_VCB_PM: float
        The Gaussian's amplitude in the vcb correction to the matter power spectrum (see Eq. 14 in arXiv: 2110.13919).
    KP_VCB_PM: float
        The Gaussian's location in the vcb correction to the matter power spectrum (see Eq. 14 in arXiv: 2110.13919).
    SIGMAK_VCB_PM: float
        The Gaussian's width in the vcb correction to the matter power spectrum (see Eq. 14 in arXiv: 2110.13919).
    Z_REC: float
        Redshift of recombination, where x_e = n_e/(n_H + n_He) = 0.1.
    DELTA_Z: float
        Redshift step size between Z_HEAT_MAX and Z1_VALUE.
    DELTA_Z1: float
        Redshift step size between Z1_VALUE and Z2_VALUE.
    DELTA_Z2: float
        Redshift step size between Z2_VALUE and Z_HIGH_MAX (unless USE_TCA_COMPTON = True, in which case it is DELTA_Z1).
    Z1_VALUE: float
        The redshift step size is DELTA_Z1 between Z1_VALUE and Z2_VALUE.
    Z2_VALUE: float
        The redshift step size is DELTA_Z2 between Z2_VALUE and Z_HIGH_MAX (unless USE_TCA_COMPTON = True, in which case it is DELTA_Z1).
    EPSILON_THRESH_HIGH_Z: float
        The thershold for tight coupling approximation at high redshift (above z=100).
    EPSILON_THRESH_LOW_Z: float
        The thershold for tight coupling approximation at low redshift (below z=100).
    REDSHIFT_2LPT: float
        The transition redshift from linear perturbation theory to 2LPT during the dark ages.
    Z_DARK_AGES_STEP_FACTOR: float
        Logarithmic redshift step size parameter for the output redshifts during the dark ages.
    LOG_Z_ARR: float array of size 70
        Logarithm of the redshift grid for interpolation tables.
    LOG_T_k: float array of size 70
        Logarithm of the gas kinetic temperature for interpolation tables.
    LOG_x_e: float array of size 70
        Logarithm of the free electron fraction for interpolation tables.
    LOG_SIGF: float array of size 70
        Logarithm of the scale-independent growth factor (SIGF) for interpolation tables.
    LOG_T_chi: float array of size 70
        Logarithm of the SDM temperature for interpolation tables.
    LOG_V_chi_b: float array of size 70
        Logarithm of the relative velocity between baryons and SDM for interpolation tables.
    LOG_K_ARR_FOR_TRANSFERS: float array of size 149
        Logarithm of wavenumber grid for transfer functions interpolation tables.
    T_M0_TRANSFER: float array of size 149
        Transfer function of the matter density field at z=0 for interpolation tables.
    T_VCB_KIN_TRANSFER: float array of size 149
        Transfer function of the relative velocity between baryons and CDM at kinematic decoupling (recombination) for interpolation tables.
    T_V_CHI_B_ZHIGH_TRANSFER: float array of size 149
        Transfer function of the relative velocity between baryons and SDM at Z_HIGH_MAX for interpolation tables.
    LOG_K_ARR_FOR_SDGF: float array of size 300
        Logarithm of wavenumber grid for scale-dependent growth factor (SDGF) interpolation tables.
    LOG_SDGF: float array of size 70*300
        Logarithm of the baryons SDGF for interpolation tables.
    LOG_SDGF_SDM: float array of size 70*300
        Logarithm of the scattering dark matter (SDM) SDGF for interpolation tables.

    """

    def __init__(self, wrapped, ffi):
        super().__init__(wrapped, ffi)

        self.external_table_path = ffi.new("char[]", str(DATA_PATH).encode())
        self._wisdoms_path = Path(config["direc"]) / "wisdoms"
        self.wisdoms_path = ffi.new("char[]", str(self._wisdoms_path).encode())

    @property
    def external_table_path(self):
        """An ffi char pointer to the path to which external tables are kept."""
        return self._external_table_path

    @external_table_path.setter
    def external_table_path(self, val):
        self._external_table_path = val

    @property
    def wisdoms_path(self):
        """An ffi char pointer to the path to which external tables are kept."""
        if not self._wisdoms_path.exists():
            self._wisdoms_path.mkdir(parents=True)

        return self._wisdom_path

    @wisdoms_path.setter
    def wisdoms_path(self, val):
        self._wisdom_path = val

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Set given parameters for a certain context.

        .. note:: Keywords are *not* case-sensitive.

        Examples
        --------
        >>> from py21cmfast import global_params, run_lightcone
        >>> with global_params.use(zprime_step_factor=1.1, Sheth_c=0.06):
        >>>     run_lightcone(redshift=7)
        """
        prev = {}
        this_attr_upper = {k.upper(): k for k in self.keys()}

        for k, val in kwargs.items():
            if k.upper() not in this_attr_upper:
                raise ValueError(f"{k} is not a valid parameter of global_params")
            key = this_attr_upper[k.upper()]
            prev[key] = getattr(self, key)
            setattr(self, key, val)

        yield

        # Restore everything back to the way it was.
        for k, v in prev.items():
            setattr(self, k, v)


global_params = GlobalParams(lib.global_params, ffi)
# JordanFlitter: added these lists to global_params. They serve as interpolation tables in the C code.
#                The numerical values were achieved from running CLASS with Planck18 parameters
#                If RUN_CLASS is True, these values will be overwritten.
# JordanFlitterTODO: remove all the lists to a new structure (they should not be part of global_params)
global_params.LOG_Z_ARR = [0.69897, 0.73291816, 0.76686631, 0.80081447, 0.83476262,
                           0.86871078, 0.90265893, 0.93660709, 0.97055524, 1.0045034,
                           1.03845155, 1.07239971, 1.10634786, 1.14029602, 1.17424417,
                           1.20819233, 1.24214048, 1.27608864, 1.31003679, 1.34398495,
                           1.3779331 , 1.41188126, 1.44582941, 1.47977756, 1.51372572,
                           1.54767387, 1.58162203, 1.61557018, 1.64951834, 1.68346649,
                           1.71741465, 1.7513628 , 1.78531096, 1.81925911, 1.85320727,
                           1.88715542, 1.92110358, 1.95505173, 1.98899989, 2.02294804,
                           2.0568962 , 2.09084435, 2.12479251, 2.15874066, 2.19268882,
                           2.22663697, 2.26058512, 2.29453328, 2.32848143, 2.36242959,
                           2.39637774, 2.4303259 , 2.46427405, 2.49822221, 2.53217036,
                           2.56611852, 2.60006667, 2.63401483, 2.66796298, 2.70191114,
                           2.73585929, 2.76980745, 2.8037556 , 2.83770376, 2.87165191,
                           2.90560007, 2.93954822, 2.97349638, 3.00744453, 3.04139269]
global_params.LOG_T_k = [0.84167437, 0.85528507, 0.85644909, 0.83640486, 0.77387952,
                         0.63492755, 0.40380558, 0.31862331, 0.36343072, 0.42369354,
                         0.48551599, 0.54785169, 0.61002035, 0.67236437, 0.73493888,
                         0.79771627, 0.86066324, 0.92374593, 0.98693, 1.05018054,
                         1.11346179, 1.17673702, 1.23996823, 1.30311604, 1.3661394 ,
                         1.42899547, 1.49163939, 1.55402416, 1.61610048, 1.67781671,
                         1.73911879, 1.79995033, 1.86025269, 1.91996521, 1.97902554,
                         2.03737011, 2.09493473, 2.15165536, 2.20746907, 2.26231514,
                         2.31613633, 2.3688803 , 2.42050114, 2.47096095, 2.52023137,
                         2.56829502, 2.61514674, 2.66079446, 2.70525966, 2.74857734,
                         2.79079529, 2.83197294, 2.87217951, 2.91149182, 2.94999175,
                         2.98776362, 3.02489163, 3.06145759, 3.09753898, 3.13320756,
                         3.1685284 , 3.20355935, 3.23835085, 3.27294525, 3.30737377,
                         3.34164854, 3.37576683, 3.40976314, 3.44370331, 3.47762731]
global_params.LOG_x_e = [ 6.00716195e-05, -6.75664429e-05, -3.84636959e-04, -2.14770651e-03,
                          -1.44107014e-02, -1.10169225e-01, -6.21913683e-01, -1.67688373e+00,
                          -2.90846656e+00, -3.65169346e+00, -3.72613619e+00, -3.72808297e+00,
                          -3.72462805e+00, -3.72172436e+00, -3.71860351e+00, -3.71537279e+00,
                          -3.71203631e+00, -3.70859055e+00, -3.70503147e+00, -3.70135472e+00,
                          -3.69755559e+00, -3.69362898e+00, -3.68956938e+00, -3.68537075e+00,
                          -3.68102655e+00, -3.67652960e+00, -3.67187204e+00, -3.66704521e+00,
                          -3.66203958e+00, -3.65684461e+00, -3.65144861e+00, -3.64583856e+00,
                          -3.63999999e+00, -3.63391665e+00, -3.62757037e+00, -3.62094069e+00,
                          -3.61400454e+00, -3.60673584e+00, -3.59910502e+00, -3.59107848e+00,
                          -3.58261794e+00, -3.57367965e+00, -3.56421352e+00, -3.55416200e+00,
                          -3.54345877e+00, -3.53202719e+00, -3.51977835e+00, -3.50660863e+00,
                          -3.49239674e+00, -3.47699982e+00, -3.46024847e+00, -3.44194022e+00,
                          -3.42183063e+00, -3.39962120e+00, -3.37494237e+00, -3.34732895e+00,
                          -3.31618381e+00, -3.28072208e+00, -3.23988206e+00, -3.19217575e+00,
                          -3.13542250e+00, -3.06623897e+00, -2.97898363e+00, -2.86345206e+00,
                          -2.70016313e+00, -2.45567673e+00, -2.10280898e+00, -1.67635415e+00,
                          -1.24912039e+00, -8.67959877e-01]
global_params.LOG_SIGF = [-0.67377048, -0.7019959 , -0.73059096, -0.75953336, -0.78880172,
                          -0.81837566, -0.84823575, -0.87836356, -0.90874152, -0.93935304,
                          -0.97018232, -1.00121446, -1.03243534, -1.06383161, -1.09539064,
                          -1.12710051, -1.15894991, -1.19092821, -1.2230253 , -1.25523164,
                          -1.28753819, -1.31993637, -1.35241804, -1.38497546, -1.41760125,
                          -1.45028835, -1.48303003, -1.51581978, -1.54865136, -1.58151871,
                          -1.61441597, -1.64733738, -1.68027733, -1.71323026, -1.7461907 ,
                          -1.77915316, -1.8121122 , -1.84506229, -1.87799788, -1.9109133 ,
                          -1.9438028 , -1.97666044, -2.00948014, -2.04225558, -2.07498026,
                          -2.10764737, -2.14024983, -2.17278026, -2.20523088, -2.2375936 ,
                          -2.26985986, -2.30202072, -2.33406674, -2.36598802, -2.39777413,
                          -2.42941411, -2.46089643, -2.49220901, -2.52333914, -2.55427354,
                          -2.58499825, -2.61549877, -2.64575993, -2.67576595, -2.70550048,
                          -2.73494659, -2.7640868 , -2.79290315, -2.82137723, -2.84949027]
global_params.LOG_T_chi = list(np.zeros(70))
global_params.LOG_V_chi_b = list(np.zeros(70))
global_params.LOG_K_ARR_FOR_TRANSFERS = [-5.15144838, -5.0514484 , -4.95144828, -4.85144824, -4.75144831,
                                         -4.65144846, -4.55144842, -4.45144844, -4.35144836, -4.2514484 ,
                                         -4.15144838, -4.0514484 , -3.95144828, -3.85144824, -3.75144831,
                                         -3.65144846, -3.55144842, -3.45144844, -3.35144836, -3.2514484 ,
                                         -3.15144838, -3.0514484 , -2.95144828, -2.85144824, -2.75144831,
                                         -2.65144846, -2.55144842, -2.45144844, -2.35144836, -2.2514484 ,
                                         -2.15144838, -2.0514484 , -1.95144828, -1.85144855, -1.75145051,
                                         -1.65167858, -1.55757323, -1.49148227, -1.44915893, -1.41712202,
                                         -1.39038991, -1.36686022, -1.34546231, -1.3255776 , -1.30681547,
                                         -1.28891354, -1.27168632, -1.25499794, -1.23874561, -1.22284979,
                                         -1.20724751, -1.19188819, -1.17673058, -1.1617405 , -1.14688964,
                                         -1.13215425, -1.11751423, -1.1029525 , -1.08845468, -1.07400858,
                                         -1.05960376, -1.04523159, -1.0308847 , -1.01655703, -1.00224362,
                                         -0.98794056, -0.97364449, -0.95935297, -0.9450646 , -0.93077752,
                                         -0.91649165, -0.90220562, -0.88792009, -0.87363446, -0.85934868,
                                         -0.84506291, -0.83077684, -0.81649043, -0.80220246, -0.78791232,
                                         -0.7736182 , -0.75931799, -0.74500867, -0.73068674, -0.71634763,
                                         -0.70198555, -0.6875937 , -0.67316396, -0.65868698, -0.64415096,
                                         -0.6295428 , -0.61484684, -0.60004451, -0.58511402, -0.57002998,
                                         -0.5547616 , -0.53927245, -0.5235185 , -0.50744591, -0.49098831,
                                         -0.47406185, -0.45656015, -0.43834405, -0.41922757, -0.39895473,
                                         -0.3771609 , -0.35330423, -0.32653792, -0.29545819, -0.25758495,
                                         -0.20836373, -0.14039666, -0.05026523,  0.04900858,  0.1489976 ,
                                         0.24899768,  0.34899762,  0.44899771,  0.54899762,  0.64899771,
                                         0.7489977 ,  0.84899769,  0.94899765,  1.04899772,  1.1489976 ,
                                         1.24899768,  1.34899762,  1.44899771,  1.54899762,  1.64899771,
                                         1.7489977 ,  1.84899769,  1.94899765,  2.04899772,  2.1489976 ,
                                         2.24899768,  2.34899762,  2.44899771,  2.54899762,  2.64899771,
                                         2.7489977 ,  2.84899769,  2.94899765,  3.04899772]
                                         #,  3.1489976 ,3.24899768,  3.34899762,  3.44899771,  3.54899762] Don't need such large k's, gsl integration goes up to k=1000/Mpc
global_params.T_M0_TRANSFER = [0.00000000e+00, 1.55624921e-03, 2.46648775e-03, 3.90911461e-03,
                               6.19550822e-03, 9.81915873e-03, 1.55621876e-02, 2.46640708e-02,
                               3.90890991e-02, 6.19499473e-02, 9.81788313e-02, 1.55589884e-01,
                               2.46560940e-01, 3.90692104e-01, 6.19006205e-01, 9.80573033e-01,
                               1.55293700e+00, 2.45848028e+00, 3.89002726e+00, 6.15063815e+00,
                               9.71494169e+00, 1.53227820e+01, 2.41206291e+01, 3.78714209e+01,
                               5.92593234e+01, 9.23226225e+01, 1.43048493e+02, 2.20146952e+02,
                               3.35976481e+02, 5.07456077e+02, 7.56574097e+02, 1.10954469e+03,
                               1.59306521e+03, 2.22547553e+03, 3.00486106e+03, 3.90909549e+03,
                               4.89667106e+03, 5.74593173e+03, 6.40625559e+03, 6.98085295e+03,
                               7.50836901e+03, 8.00143303e+03, 8.46315654e+03, 8.89279281e+03,
                               9.28837818e+03, 9.64818906e+03, 9.97183608e+03, 1.02610350e+04,
                               1.05202049e+04, 1.07567159e+04, 1.09808758e+04, 1.12054189e+04,
                               1.14445341e+04, 1.17122529e+04, 1.20207647e+04, 1.23781379e+04,
                               1.27863056e+04, 1.32396635e+04, 1.37243642e+04, 1.42191799e+04,
                               1.46982102e+04, 1.51354022e+04, 1.55103993e+04, 1.58145143e+04,
                               1.60551952e+04, 1.62574715e+04, 1.64608133e+04, 1.67108098e+04,
                               1.70465264e+04, 1.74866402e+04, 1.80193033e+04, 1.86036094e+04,
                               1.91756683e+04, 1.96667147e+04, 2.00401908e+04, 2.03099111e+04,
                               2.05368820e+04, 2.08033909e+04, 2.11791775e+04, 2.16861054e+04,
                               2.22761025e+04, 2.28515629e+04, 2.33252040e+04, 2.36749609e+04,
                               2.39654506e+04, 2.43105412e+04, 2.47790009e+04, 2.53401483e+04,
                               2.58923525e+04, 2.63480198e+04, 2.67251455e+04, 2.71227361e+04,
                               2.76175362e+04, 2.81706732e+04, 2.86868982e+04, 2.91429787e+04,
                               2.96136459e+04, 3.01517421e+04, 3.07179638e+04, 3.12523946e+04,
                               3.17964776e+04, 3.24035991e+04, 3.30227884e+04, 3.36610439e+04,
                               3.43619717e+04, 3.51062271e+04, 3.59324599e+04, 3.68602489e+04,
                               3.79480629e+04, 3.92822128e+04, 4.10308802e+04, 4.34699200e+04,
                               4.67454762e+04, 5.03989118e+04, 5.41208100e+04, 5.78782997e+04,
                               6.16658634e+04, 6.54781245e+04, 6.93116700e+04, 7.31624393e+04,
                               7.70269411e+04, 8.09028912e+04, 8.47865593e+04, 8.86747644e+04,
                               9.25624204e+04, 9.64445716e+04, 1.00312272e+05, 1.04152254e+05,
                               1.07943459e+05, 1.11652338e+05, 1.15226451e+05, 1.18584080e+05,
                               1.21594008e+05, 1.24067049e+05, 1.25727566e+05, 1.26216346e+05,
                               1.25141562e+05, 1.22221878e+05, 1.17543752e+05, 1.11829082e+05,
                               1.06439206e+05, 1.02869905e+05, 1.01731069e+05, 1.01643363e+05,
                               1.00434901e+05, 1.00051594e+05, 1.01830295e+05, 1.02021246e+05,
                               1.05040039e+05]
global_params.T_VCB_KIN_TRANSFER = [0.00000000e+00, 3.11899486e-10, 5.02290006e-10, 8.40004934e-10,
                                    1.59596557e-09, 3.21130599e-09, 6.42766947e-09, 1.28234813e-08,
                                    2.55891459e-08, 5.10576226e-08, 1.01876989e-07, 2.03270550e-07,
                                    4.05574814e-07, 8.09223432e-07, 1.61458611e-06, 3.22143972e-06,
                                    6.42736114e-06, 1.28234809e-05, 2.55836656e-05, 5.10381805e-05,
                                    1.01809300e-04, 2.03056481e-04, 4.05619784e-04, 8.09002825e-04,
                                    1.61185173e-03, 3.20794439e-03, 6.37463596e-03, 1.26367452e-02,
                                    2.49569535e-02, 4.89887949e-02, 9.52369464e-02, 1.82316190e-01,
                                    3.40497816e-01, 6.11055671e-01, 1.02734862e+00, 1.54774834e+00,
                                    1.92176895e+00, 1.89537323e+00, 1.67868217e+00, 1.40993886e+00,
                                    1.13622904e+00, 8.82538034e-01, 6.66333049e-01, 5.00939656e-01,
                                    3.96273524e-01, 3.58652267e-01, 3.90269279e-01, 4.88598688e-01,
                                    6.45959531e-01, 8.49332479e-01, 1.08058286e+00, 1.31731112e+00,
                                    1.53430184e+00, 1.70582070e+00, 1.80837459e+00, 1.82415783e+00,
                                    1.74455066e+00, 1.57297348e+00, 1.32687207e+00, 1.03764935e+00,
                                    7.47991677e-01, 5.06233837e-01, 3.57998920e-01, 3.36038782e-01,
                                    4.50468407e-01, 6.81584464e-01, 9.79187692e-01, 1.26971736e+00,
                                    1.47282081e+00, 1.52410667e+00, 1.39878061e+00, 1.12709443e+00,
                                    7.93184801e-01, 5.12125982e-01, 3.88572832e-01, 4.70326640e-01,
                                    7.18267167e-01, 1.01282414e+00, 1.20397279e+00, 1.18810132e+00,
                                    9.72408535e-01, 6.82465218e-01, 4.93775547e-01, 5.17691053e-01,
                                    7.16770478e-01, 9.22958312e-01, 9.64874775e-01, 8.13070514e-01,
                                    6.11626337e-01, 5.44419458e-01, 6.49263908e-01, 7.77717222e-01,
                                    7.68271722e-01, 6.43007779e-01, 5.69827564e-01, 6.20858265e-01,
                                    6.75000225e-01, 6.29508338e-01, 5.69284208e-01, 5.83630336e-01,
                                    5.95327223e-01, 5.57702659e-01, 5.45408677e-01, 5.43046439e-01,
                                    5.19607112e-01, 5.09441751e-01, 4.91100044e-01, 4.74898658e-01,
                                    4.54384061e-01, 4.30897828e-01, 4.01658379e-01, 3.63622657e-01,
                                    3.17520621e-01, 2.72192855e-01, 2.32051088e-01, 1.97025952e-01,
                                    1.66675319e-01, 1.40529064e-01, 1.18124724e-01, 9.90160659e-02,
                                    8.27868436e-02, 6.90561834e-02, 5.74786884e-02, 4.77472393e-02,
                                    3.95902723e-02, 3.27710636e-02, 2.70836028e-02, 2.23503321e-02,
                                    1.84190175e-02, 1.51597377e-02, 1.24622710e-02, 1.02334090e-02,
                                    8.39422619e-03, 6.87888650e-03, 5.63189512e-03, 4.60689880e-03,
                                    3.76537718e-03, 3.07527713e-03, 2.50991054e-03, 2.04718427e-03,
                                    1.66884536e-03, 1.35979390e-03, 1.10760161e-03, 9.02059531e-04,
                                    7.34706853e-04, 5.98633907e-04, 4.88111761e-04, 3.98385369e-04,
                                    3.25460580e-04]
global_params.T_V_CHI_B_ZHIGH_TRANSFER = list(np.zeros(149))
global_params.LOG_K_ARR_FOR_SDGF = list(np.zeros(300))
global_params.LOG_SDGF = list(np.zeros(70*300))
global_params.LOG_SDGF_SDM = list(np.zeros(70*300))

class CosmoParams(StructWithDefaults):
    """
    Cosmological parameters (with defaults) which translates to a C struct.

    To see default values for each parameter, use ``CosmoParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Default parameters are based on Plank18, https://arxiv.org/pdf/1807.06209.pdf,
    Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]

    Parameters
    ----------
    SIGMA_8 : float, optional
        RMS mass variance (power spectrum normalisation).
    hlittle : float, optional
        The hubble parameter, H_0/100.
    OMm : float, optional
        Omega matter.
    OMb : float, optional
        Omega baryon, the baryon component.
    POWER_INDEX : float, optional
        Spectral index of the power spectrum.
    A_s : float, optional
        Amplitude of primordial curvature fluctuations.
    tau_reio : float, optional
        Optical depth to reionization.
    f_FDM : float, optional
        Fraction of fuzzy dark matter (this is actually -log10(f_FDM)).
    m_FDM : float, optional
        Mass of the FDM particle (this is actually -log10(m_FDM/eV)).
    f_chi : float, optional
        Fraction of scattering dark matter (this is actually -log10(f_chi)).
    m_chi : float, optional
        Mass of the SDM particle (this is actually log10(m_chi/eV)).
    sigma_SDM : float, optional
        The amplitude of the cross-section between the SDM particle and its target particles,
        see SDM_TARGET_TYPE in user_params (this is actually -log10(sigma/cm^2)).
    SDM_INDEX : float, optional
        The power-law index of the cross-section between the SDM particle and its target particles,
        see SDM_TARGET_TYPE in user_params.
    """

    _ffi = ffi

    _defaults_ = {
        "SIGMA_8": 0.8102,
        "hlittle": Planck18.h,
        "OMm": Planck18.Om0,
        "OMb": Planck18.Ob0,
        "POWER_INDEX": 0.9665,
        "A_s": 2.1e-9, # JordanFlitter: added amplitude of primordial curvature fluctuations
        "tau_reio": 0.0544, # JordanFlitter: added optical depth to reionization
        "m_FDM": 21., # JordanFlitter: added FDM mass (this is actually -log10(m_FDM/eV))
        "f_FDM": 1., # JordanFlitter: added FDM fraction (this is actually -log10(f_FDM))
        "m_chi": 9., # JordanFlitter: added SDM mass (this is actually log10(m_chi/eV))
        "f_chi": 0., # JordanFlitter: added SDM fraction (this is actually -log10(f_chi))
        "sigma_SDM": 41., # JordanFlitter: added SDM cross section prefactor (this is actually -log10(sigma/cm^2))
        "SDM_INDEX": -4., # JordanFlitter: added SDM cross section index
    }

    @property
    def OMl(self):
        """Omega lambda, dark energy density."""
        return 1 - self.OMm

    @property
    def cosmo(self):
        """Return an astropy cosmology object for this cosmology."""
        return Planck15.clone(H0=self.hlittle * 100, Om0=self.OMm, Ob0=self.OMb)


class UserParams(StructWithDefaults):
    """
    Structure containing user parameters (with defaults).

    To see default values for each parameter, use ``UserParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    HII_DIM : int, optional
        Number of cells for the low-res box. Default 200.
    DIM : int,optional
        Number of cells for the high-res box (sampling ICs) along a principal axis. To avoid
        sampling issues, DIM should be at least 3 or 4 times HII_DIM, and an integer multiple.
        By default, it is set to 3*HII_DIM.
    BOX_LEN : float, optional
        Length of the box, in Mpc. Default 300 Mpc.
    HMF: int or str, optional
        Determines which halo mass function to be used for the normalisation of the
        collapsed fraction (default Sheth-Tormen). If string should be one of the
        following codes:
        0: PS (Press-Schechter)
        1: ST (Sheth-Tormen)
        2: Watson (Watson FOF)
        3: Watson-z (Watson FOF-z)
    USE_RELATIVE_VELOCITIES: int, optional
        Flag to decide whether to use relative velocities.
        If True, POWER_SPECTRUM is automatically set to 5. Default True.
    POWER_SPECTRUM: int or str, optional
        Determines which power spectrum to use, default CLASS.
        If string, use the following codes:
        0: EH
        1: BBKS
        2: EFSTATHIOU
        3: PEEBLES
        4: WHITE
        5: CLASS
    N_THREADS : int, optional
        Sets the number of processors (threads) to be used for performing 21cmFAST.
        Default 1.
    PERTURB_ON_HIGH_RES : bool, optional
        Whether to perform the Zel'Dovich or 2LPT perturbation on the low or high
        resolution grid.
    NO_RNG : bool, optional
        Ability to turn off random number generation for initial conditions. Can be
        useful for debugging and adding in new features
    USE_FFTW_WISDOM : bool, optional
        Whether or not to use stored FFTW_WISDOMs for improving performance of FFTs
    USE_INTERPOLATION_TABLES: bool, optional
        If True, calculates and evaluates quantites using interpolation tables, which
        is considerably faster than when performing integrals explicitly.
    FAST_FCOLL_TABLES: bool, optional
        Whether to use fast Fcoll tables, as described in Appendix of Muñoz+21 (2110.13919). Significant speedup for minihaloes.
    USE_2LPT: bool, optional
        Whether to use second-order Lagrangian perturbation theory (2LPT).
        Set this to True if the density field or the halo positions are extrapolated to
        low redshifts. The current implementation is very naive and adds a factor ~6 to
        the memory requirements. Reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118
        Appendix D.
    MINIMIZE_MEMORY: bool, optional
        If set, the code will run in a mode that minimizes memory usage, at the expense
        of some CPU/disk-IO. Good for large boxes / small computers.
    RUN_CLASS: bool, optional
        Whether to run CLASS to get initial conditions that are consistent with the
        chosen cosmological parameters. Note that if FUZZY_DM is True, then AxionCAMB
        is also called. Default is True.
    EVOLVE_BARYONS: bool, optional
        Whether to track the baryons density field. It is well known that the 21cm signal is sourced from neutral hydrogen, i.e. baryons.
        Therefore the baryons density field is the more appropriate quantitiy to consider, especially during the dark ages. If set to True,
        the baryons field is evolved with the scale-dependent growth factor (SDGF) that is evaluated from CLASS (see arXiv: 2309.03948).
        Otherwise, the scale-independent growth factor (SIGF) will be considered, thereby treating the hydrogen atoms as CDM (this is of
        course less accurate but is more efficient computationally). Default is True.
    START_AT_RECOMBINATION: bool, optional
        Whether to begin the simulation at the dark ages, above Z_HEAT_MAX. The initial redshift
        is determined from the value of Z_HIGH_MAX. If Z_HIGH_MAX is negative (default), then
        the initial redshift is at recobmination, when x_e = n_e/(n_H+n_He) = 0.1. Otherwise,
        the simulation begins at the value of Z_HIGH_MAX. If set to True, the simulation begins
        at the dark ages. Otherwise, it begins from the onset of cosmic dawn, at Z_HEAT_MAX.
        Default is True.
    OUTPUT_AT_DARK_AGES: bool, optional
        Whether to have any output during the dark ages. If set to True, output is returned
        also during the dark ages. Otherwise, there is no output during the dark ages and the
        codes runs faster. Note that begining the simulation from the dark ages (via
        START_AT_RECOMBINATION) alters the 21cm power spectrum due to early temperature
        fluctuations, so it reasonable for users to set this flag on False while begninning
        the simulation from the dark ages. Default is True.
    Z_HIGH_MAX: float, optional
        The initial redshift for the simulation, when START_AT_RECOMBINATION is True. If
        this parameter is negative, then the initial redshift is at recobmination, when
        x_e = n_e/(n_H+n_He) = 0.1. Otherwise, the simulation begins at the value of
        Z_HIGH_MAX. Default is -1.
    USE_HYREC: bool, optional
        Whether to use HYREC for sloving for free electron fraction prior to reionization.
        If set to True, HYREC is used. Otherwise, simpler models for the recombination rate
        are considered. Default is True.
    EVALUATE_TAU_REIO: bool, optional
        Whether to compute the optical depth to reionization, tau_reio, from the 21cmFAST simulation. If set to True, tau_reio is evaluated
        from the output of 21cmFAST. Then, if RUN_CLASS is also set to True, CLASS runs the second time, after the 21cmFAST simulation is over
        to calculate the CMB power spectrum, C_ell, with the updated value of tau_reio. Default is True.
    FUZZY_DM: bool, optional
        Whether to consider fuzzy dark matter (FDM) in the simulation. If set to True,
        AxionCAMB is called in order to generate initial conditions that are consistent
        with the chosen cosmological parameters. Default is False.
    SCATTERING_DM: bool, optional
        Whether to consider scattering dark matter (SDM) in the simulation. If set to True,
        CLASS is called in order to generate initial conditions that are consistent
        with the chosen cosmological parameters. Default is False.
    SDM_TARGET_TYPE: int or str, optional
        Determines the type of target particles that interact with SDM, default is BARYONS.
        If string, use the following codes:
        1: BARYONS (all the baryons)
        2: IONIZED (free protons and electrons)
        3: HYDROGEN (hydrogen nuclei, neutralized or not)
        4: PROTONS (free protons)
        5: ELECTRONS (free electrons)
    USE_SDM_FLUCTS: bool, optional
        Whether to begin the simulation with an inhomogeneous box for the relative velocity
        between baryons and SDM. If set to True, the initial box is inhomogeneous. Otherwise,
        it is homogeneous. Default is True.
    NO_INI_MATTER_FLUCTS: bool, optional
        Whether to zero the perturbations in the density field during the dark ages (this is
        useful for comparing with CLASS). If set to True, the perturbations are set to zero.
        Otherwise, non-zero density perturbations are considered. Default is False.
    DO_VCB_FIT: bool, optional
        Whether to find the best fit values for the vcb (relative velocity between baryons
        and CDM) correction to the matter power spectrum (see Eq. 14 in arXiv: 2110.13919).
        If set to True, along with RUN_CLASS, the code finds the best fit values for the
        chosen cosmological parameters. Otherwise, default values are used. This feature
        currently cannot be active while either FUZZY_DM or SCATTERING_DM is True. Default
        is False.
    USE_TCA_COMPTON: bool, optional
        Whether to use the Compton-TCA (tight coupling algorithm) close to the time of recombination.
        See Appendix B in arXiv: 2309.03942. If set to True, the Compton-TCA will be applied when needed,
        allowing us to work with a larger step size and much less redshift iterations during the dark ages.
        Otherwise, the regular ODEs are always solved, forcing us to work with a small step size close
        to the time of recombination and much more redshift iterations. Default is True.
    USE_CS_S_ALPHA: bool, optional
        Whether to compute the S_alpha correction according to Eq. A4 in Mittal & Kulkarni (arXiv: 2009.10746),
        a result that was derived from the work of Chuzhouy & Shapiro (arXiv: astro-ph/0512206). This formula is
        more suited to low temperatures (below 2K) that can be reached if SCATTERING_DM is True. At higher
        temperatures the formula agrees well with Hirata's fit (arXiv: astro-ph/0507102). If set to True,
        the formula from Chuzhouy & Shapiro paper is used. Otherwise, Hirata's fit is used. Default is False.
    MANY_Z_SAMPLES_AT_COSMIC_DAWN: bool, optional
        Whether to have a lot of redhift iterations during cosmic dawn. This might be required to evolve the temperature
        field correctly at low temperatures that can be reached if SCATTERING_DM is True. If set to True,
        there will be more redshift iterations during cosmic dawn (note that this feature can only be applied if
        START_AT_RECOMBINATION is also True). Default is False.
    USE_ALPHA_B: bool, optional
        Whether to use the case-B recombination rate, alpha_B, during cosmic dawn. When approaching very low temperatures
        that can be reached if SCATTERING_DM is True, the fit for the case-A recombination rate, alpha_A, is inappropriate.
        If set to True, alpha_B is used during cosmic dawn, otherwise alpha_A is used. Note that when USE_HYREC is set to True
        the code uses HyRec's recombination rate and is insensitive to the value of this parameter. Default is False.
    DO_PERTURBS_WITH_TS: bool, optional
        Whether to compute the density field alongside the spin temperature. If set to True, these two fields are computed
        together, at the same redshift iteration. Otherwise, the density field at all redshifts is precomputed and stored in
        a list, which can be expensive memorywise, especially when the code begins from the dark ages. Default is True.
    USE_ADIABATIC_FLUCTUATIONS: bool, optional
        Whether to use JBM's fit (arXiv: 2302.08506) to the scale-independent c_T = delta_T/delta_c to generate temperature
        fluctuations at Z_HEAT_MAX. If set to True, initial temperature fluctuations are generated with JBM's fit. Otherwise,
        the initial temperature box is homogeneous. Note that this feature cannot be applied if we either begin the simulation
        from the dark ages (when setting START_AT_RECOMBINATION to True) or when tracking the baryons density field (when setting
        EVOLVE_BARYONS to True). Default is False.
    USE_DICKE_GROWTH_FACTOR: bool, optional
        Whether to use the Dicke's scale-independent growth factor. This fit is particually good during cosmic dawn but is
        inadequate during the dark ages. If set to True, Dicke's growth factor is considered, even if RUN_CLASS is True. Otherwise,
        the scale-independt growth factor from CLASS is considered, even if RUN_CLASS is False. Default is False.
    CLOUD_IN_CELL: bool, optional
        Whether to use Bradley Greig's "cloud in cell" algorithm in 2LPT calculations. If set to True, mass will be redistributed to
        its 8 nearest neighbors during the 2LPT calculations. Otherwise, there will be no such redistribution. Default is True.

    """

    _ffi = ffi

    _defaults_ = {
        "BOX_LEN": 300.0,
        "DIM": None,
        "HII_DIM": 200,
        "USE_FFTW_WISDOM": False,
        "HMF": 1,
        "USE_RELATIVE_VELOCITIES": True, # JordanFlitter: changed default to True
        "POWER_SPECTRUM": 5, # JordanFlitter: changed default to 5
        "N_THREADS": 1,
        "PERTURB_ON_HIGH_RES": False,
        "NO_RNG": False,
        "USE_INTERPOLATION_TABLES": None,
        "FAST_FCOLL_TABLES": False,
        "USE_2LPT": True,
        "MINIMIZE_MEMORY": False,
        "RUN_CLASS": True, # JordanFlitter: added flag for running CLASS
        "DO_VCB_FIT": False, # JordanFlitter: added flag for performing v_cb fit to the matter power spectrum
        "FUZZY_DM": False, # JordanFlitter: added flag for fuzzy DM
        "SCATTERING_DM": False, # JordanFlitter: added flag for scattering DM
        "SDM_TARGET_TYPE": 1, # JordanFlitter: added type of SDM target particles (1=baryons, 2=protons and electrons, 3=hydrogen nuclei, 4=protons, 5=electrons)
        "USE_SDM_FLUCTS": True, # JordanFlitter: added flag to use initial SDM fluctuations
        "START_AT_RECOMBINATION": True, # JordanFlitter: added flag to run the simulation through the dark ages
        "USE_HYREC": True, # JordanFlitter: added flag to use HyRec
        "NO_INI_MATTER_FLUCTS": False, # JordanFlitter: added flag to set on zero the matter fluctuations during the dark ages (useful for comparing with CLASS)
        "Z_HIGH_MAX": -1., # JordanFlitter: added the highest redshift for the evolution (default -1 means recombination)
        "OUTPUT_AT_DARK_AGES": True, # JordanFlitter: added flag to compute output during the dark ages
        "USE_TCA_COMPTON": True, # JordanFlitter: added flag to use tight coupling approximation (for Compton scattering)
        "USE_CS_S_ALPHA": False, # JordanFlitter: added flag to use the S_alpha correction from Chuzhouy & Shapiro (arXiv: astro-ph/0512206)
        "MANY_Z_SAMPLES_AT_COSMIC_DAWN": False, # JordanFlitter: added flag to include many more redshift samples during cosmic dawn (works only if START_AT_RECOMBINATION = True)
        "USE_ALPHA_B": False, # JordanFlitter: added flag to use alpha_B (with the Peebles coefficient) as the recombination rate at cosmic dawn (below Z_HEAT_MAX)
        "DO_PERTURBS_WITH_TS": True, # JordanFlitter: added flag to perturb the density field while evaluating the spin temperature (to reduce required memory)
        "USE_ADIABATIC_FLUCTUATIONS": False, # JordanFlitter: added flag to generate inhomogeneous temperature box at Z_HEAT_MAX, based on arXiv: 2302.08506
        "USE_DICKE_GROWTH_FACTOR": False, # JordanFlitter: added flag to use 21cmFAST default Dicke growth factor even if RUN_CLASS = True
        "CLOUD_IN_CELL": True, # JordanFlitter: added flag to use Bradley Greig's algorithm for cloud in cell in 2LPT calculations
        "EVOLVE_BARYONS": True, # JordanFlitter: added flag to use the scale-dependent growth factor to evolve the baryons density field (see arXiv: 2309.03948)
        "EVALUATE_TAU_REIO": True, # JordanFlitter: added flag to evaluate tau_reio from the simulation
    }

    _hmf_models = ["PS", "ST", "WATSON", "WATSON-Z"]
    _power_models = ["EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"]
    # JordanFlitter: added _SDM_target_type_models
    _SDM_target_type_models = ["BARYONS", "IONIZED","HYDROGEN","PROTONS","ELECTRONS"]

    @property
    def USE_INTERPOLATION_TABLES(self):
        """Whether to use interpolation tables for integrals, speeding things up."""
        if self._USE_INTERPOLATION_TABLES is None:
            # JordanFlitter: I commented out that warning...
            # warnings.warn(
            #    "The USE_INTERPOLATION_TABLES setting has changed in v3.1.2 to be "
            #    "default True. You can likely ignore this warning, but if you relied on"
            #    "having USE_INTERPOLATION_TABLES=False by *default*, please set it "
            #    "explicitly. To silence this warning, set it explicitly to True. This"
            #    "warning will be removed in v4."
            #)
            self._USE_INTERPOLATION_TABLES = True

        return self._USE_INTERPOLATION_TABLES

    @property
    def DIM(self):
        """Number of cells for the high-res box (sampling ICs) along a principal axis."""
        return self._DIM or 3 * self.HII_DIM

    @property
    def tot_fft_num_pixels(self):
        """Total number of pixels in the high-res box."""
        return self.DIM ** 3

    @property
    def HII_tot_num_pixels(self):
        """Total number of pixels in the low-res box."""
        return self.HII_DIM ** 3

    @property
    def POWER_SPECTRUM(self):
        """
        The power spectrum generator to use, as an integer.

        See :func:`power_spectrum_model` for a string representation.
        """
        # JordanFlitter: added RUN_CLASS to the following logic
        if self.USE_RELATIVE_VELOCITIES or self.RUN_CLASS:
            if (self._POWER_SPECTRUM != 5) or (
                isinstance(self._POWER_SPECTRUM, str)
                and self._POWER_SPECTRUM.upper() != "CLASS"
            ):
                if self.USE_RELATIVE_VELOCITIES:
                    logger.warning(
                        "Automatically setting POWER_SPECTRUM to 5 (CLASS) as you are using "
                        "relative velocities"
                    )
                else:
                    logger.warning(
                        "Automatically setting POWER_SPECTRUM to 5 (CLASS) as you RUN_CLASS"
                    )
                self._POWER_SPECTRUM = 5
            return 5
        else:
            if isinstance(self._POWER_SPECTRUM, str):
                val = self._power_models.index(self._POWER_SPECTRUM.upper())
            else:
                val = self._POWER_SPECTRUM

            if not 0 <= val < len(self._power_models):
                raise ValueError(
                    "Power spectrum must be between 0 and {}".format(
                        len(self._power_models) - 1
                    )
                )

            return val

    @property
    def HMF(self):
        """The HMF to use (an int, mapping to a given form).

        See hmf_model for a string representation.
        """
        if isinstance(self._HMF, str):
            val = self._hmf_models.index(self._HMF.upper())
        else:
            val = self._HMF

        try:
            val = int(val)
        except (ValueError, TypeError):
            raise ValueError("Invalid value for HMF")

        if not 0 <= val < len(self._hmf_models):
            raise ValueError(
                f"HMF must be an int between 0 and {len(self._hmf_models) - 1}"
            )

        return val

    # JordanFlitter: added the following property for SDM_TARGET_TYPE
    @property
    def SDM_TARGET_TYPE(self):
        """The SDM_TARGET_TYPE to use (an int, mapping to a given form)."""

        if isinstance(self._SDM_TARGET_TYPE, str):
            try:
                val = self._SDM_target_type_models.index(self._SDM_TARGET_TYPE.upper()) + 1
            except ValueError:
                raise ValueError("Invalid string for SDM_TARGET_TYPE. It can be either 'BARYONS', 'IONIZED', 'HYDROGEN', 'PROTONS' or 'ELECTRONS'.")
        else:
            val = self._SDM_TARGET_TYPE

        try:
            val = int(val)
        except (ValueError, TypeError):
            raise ValueError("Invalid value for SDM_TARGET_TYPE")

        if not 1 <= val <= len(self._SDM_target_type_models):
            raise ValueError(
                f"SDM_TARGET_TYPE must be an int between 1 and {len(self._SDM_target_type_models)}"
            )

        return val

    @property
    def hmf_model(self):
        """String representation of the HMF model used."""
        return self._hmf_models[self.HMF]

    @property
    def power_spectrum_model(self):
        """String representation of the power spectrum model used."""
        return self._power_models[self.POWER_SPECTRUM]

    @property
    def FAST_FCOLL_TABLES(self):
        """Check that USE_INTERPOLATION_TABLES is True."""
        if self._FAST_FCOLL_TABLES and not self.USE_INTERPOLATION_TABLES:
            logger.warning(
                "You cannot turn on FAST_FCOLL_TABLES without USE_INTERPOLATION_TABLES."
            )
            return False
        else:
            return self._FAST_FCOLL_TABLES


class FlagOptions(StructWithDefaults):
    """
    Flag-style options for the ionization routines.

    To see default values for each parameter, use ``FlagOptions._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes
    which should be considered read-only. This is true of all input-parameter classes.

    Note that all flags are set to False by default, giving the simplest "vanilla"
    version of 21cmFAST.

    Parameters
    ----------
    USE_HALO_FIELD : bool, optional
        Set to True if intending to find and use the halo field. If False, uses
        the mean collapse fraction (which is considerably faster).
    USE_MINI_HALOS : bool, optional
        Set to True if using mini-halos parameterization.
        If True, USE_MASS_DEPENDENT_ZETA and INHOMO_RECO must be True.
    USE_MASS_DEPENDENT_ZETA : bool, optional
        Set to True if using new parameterization. Setting to True will automatically
        set `M_MIN_in_Mass` to True.
    SUBCELL_RSDS : bool, optional
        Add sub-cell redshift-space-distortions (cf Sec 2.2 of Greig+2018).
        Will only be effective if `USE_TS_FLUCT` is True.
    INHOMO_RECO : bool, optional
        Whether to perform inhomogeneous recombinations. Increases the computation
        time.
    USE_TS_FLUCT : bool, optional
        Whether to perform IGM spin temperature fluctuations (i.e. X-ray heating).
        Dramatically increases the computation time.
    M_MIN_in_Mass : bool, optional
        Whether the minimum halo mass (for ionization) is defined by
        mass or virial temperature. Automatically True if `USE_MASS_DEPENDENT_ZETA`
        is True.
    PHOTON_CONS : bool, optional
        Whether to perform a small correction to account for the inherent
        photon non-conservation.
    FIX_VCB_AVG: bool, optional
        Determines whether to use a fixed vcb=VAVG (*regardless* of USE_RELATIVE_VELOCITIES). It includes the average effect of velocities but not its fluctuations. See Muñoz+21 (2110.13919).
    USE_VELS_AUX: bool, optional
        Auxiliary variable (not input) to check if minihaloes are being used without relative velocities and complain
    """

    _ffi = ffi

    _defaults_ = {
        "USE_HALO_FIELD": False,
        "USE_MINI_HALOS": True, # JordanFlitter: changed default to True
        "USE_CMB_HEATING": False, # JordanFlitter: added CMB heating
        "USE_Lya_HEATING": False, # JordanFlitter: added Lya heating
        "USE_MASS_DEPENDENT_ZETA": True, # JordanFlitter: changed default to True
        "SUBCELL_RSD": True, # JordanFlitter: changed default to True
        "INHOMO_RECO": True, # JordanFlitter: changed default to True
        "USE_TS_FLUCT": True, # JordanFlitter: changed default to True
        "M_MIN_in_Mass": False,
        "PHOTON_CONS": False,
        "FIX_VCB_AVG": False,
    }

    # This checks if relative velocities are off to complain if minihaloes are on
    def __init__(
        self,
        *args,
        USE_VELS_AUX=UserParams._defaults_["USE_RELATIVE_VELOCITIES"],
        **kwargs,
    ):
        # TODO: same as with inhomo_reco. USE_VELS_AUX used to check that relvels are on if MCGs are too
        self.USE_VELS_AUX = USE_VELS_AUX
        super().__init__(*args, **kwargs)
        if self.USE_MINI_HALOS and not self.USE_VELS_AUX and not self.FIX_VCB_AVG:
            logger.warning(
                "USE_MINI_HALOS needs USE_RELATIVE_VELOCITIES to get the right evolution!"
            )

    @property
    def USE_HALO_FIELD(self):
        """Automatically setting USE_MASS_DEPENDENT_ZETA to False if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and self._USE_HALO_FIELD:
            logger.warning(
                "You have set USE_MINI_HALOS to True but USE_HALO_FIELD is also True! "
                "Automatically setting USE_HALO_FIELD to False."
            )
            return False
        else:
            return self._USE_HALO_FIELD

    @property
    def M_MIN_in_Mass(self):
        """Whether minimum halo mass is defined in mass or virial temperature."""
        if self.USE_MASS_DEPENDENT_ZETA:
            return True

        else:
            return self._M_MIN_in_Mass

    @property
    def USE_MASS_DEPENDENT_ZETA(self):
        """Automatically setting USE_MASS_DEPENDENT_ZETA to True if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and not self._USE_MASS_DEPENDENT_ZETA:
            logger.warning(
                "You have set USE_MINI_HALOS to True but USE_MASS_DEPENDENT_ZETA to False! "
                "Automatically setting USE_MASS_DEPENDENT_ZETA to True."
                )
            return True
        else:
            return self._USE_MASS_DEPENDENT_ZETA

    @property
    def INHOMO_RECO(self):
        """Automatically setting INHOMO_RECO to True if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and not self._INHOMO_RECO:
            logger.warning(
                "You have set USE_MINI_HALOS to True but INHOMO_RECO to False! "
                "Automatically setting INHOMO_RECO to True."
            )
            return True
        else:
            return self._INHOMO_RECO

    @property
    def USE_TS_FLUCT(self):
        """Automatically setting USE_TS_FLUCT to True if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and not self._USE_TS_FLUCT:
            logger.warning(
                "You have set USE_MINI_HALOS to True but USE_TS_FLUCT to False! "
                "Automatically setting USE_TS_FLUCT to True."
            )
            return True
        else:
            return self._USE_TS_FLUCT

    @property
    def PHOTON_CONS(self):
        """Automatically setting PHOTON_CONS to False if USE_MINI_HALOS."""
        if self.USE_MINI_HALOS and self._PHOTON_CONS:
            logger.warning(
                "USE_MINI_HALOS is not compatible with PHOTON_CONS! "
                "Automatically setting PHOTON_CONS to False."
            )
            return False
        else:
            return self._PHOTON_CONS


class AstroParams(StructWithDefaults):
    """
    Astrophysical parameters.

    To see default values for each parameter, use ``AstroParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    INHOMO_RECO : bool, optional
        Whether inhomogeneous recombinations are being calculated. This is not a part of the
        astro parameters structure, but is required by this class to set some default behaviour.
    HII_EFF_FACTOR : float, optional
        The ionizing efficiency of high-z galaxies (zeta, from Eq. 2 of Greig+2015).
        Higher values tend to speed up reionization.
    F_STAR10 : float, optional
        The fraction of galactic gas in stars for 10^10 solar mass haloes.
        Only used in the "new" parameterization,
        i.e. when `USE_MASS_DEPENDENT_ZETA` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
        Given in log10 units.
    F_STAR7_MINI : float, optional
        The fraction of galactic gas in stars for 10^7 solar mass minihaloes.
        Only used in the "minihalo" parameterization,
        i.e. when `USE_MINI_HALOS` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC7_MINI` to determine `HII_EFF_FACTOR_MINI` (which
        is then unused). See Eq. 8 of Qin+2020.
        Given in log10 units.
    ALPHA_STAR : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass.
        See Sec 2.1 of Park+2018.
    ALPHA_STAR_MINI : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass, for MCGs.
        See Sec 2 of Muñoz+21 (2110.13919).
    F_ESC10 : float, optional
        The "escape fraction", i.e. the fraction of ionizing photons escaping into the
        IGM, for 10^10 solar mass haloes. Only used in the "new" parameterization,
        i.e. when `USE_MASS_DEPENDENT_ZETA` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_STAR10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
    F_ESC7_MINI: float, optional
        The "escape fraction for minihalos", i.e. the fraction of ionizing photons escaping
        into the IGM, for 10^7 solar mass minihaloes. Only used in the "minihalo" parameterization,
        i.e. when `USE_MINI_HALOS` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC7_MINI` to determine `HII_EFF_FACTOR_MINI` (which
        is then unused). See Eq. 17 of Qin+2020.
        Given in log10 units.
    ALPHA_ESC : float, optional
        Power-law index of escape fraction as a function of halo mass. See Sec 2.1 of
        Park+2018.
    M_TURN : float, optional
        Turnover mass (in log10 solar mass units) for quenching of star formation in
        halos, due to SNe or photo-heating feedback, or inefficient gas accretion. Only
        used if `USE_MASS_DEPENDENT_ZETA` is set to True in :class:`FlagOptions`.
        See Sec 2.1 of Park+2018.
    R_BUBBLE_MAX : float, optional
        Mean free path in Mpc of ionizing photons within ionizing regions (Sec. 2.1.2 of
        Greig+2015). Default is 50 if `INHOMO_RECO` is True, or 15.0 if not.
    ION_Tvir_MIN : float, optional
        Minimum virial temperature of star-forming haloes (Sec 2.1.3 of Greig+2015).
        Given in log10 units.
    L_X : float, optional
        The specific X-ray luminosity per unit star formation escaping host galaxies.
        Cf. Eq. 6 of Greig+2018. Given in log10 units.
    L_X_MINI: float, optional
        The specific X-ray luminosity per unit star formation escaping host galaxies for
        minihalos. Cf. Eq. 23 of Qin+2020. Given in log10 units.
    NU_X_THRESH : float, optional
        X-ray energy threshold for self-absorption by host galaxies (in eV). Also called
        E_0 (cf. Sec 4.1 of Greig+2018). Typical range is (100, 1500).
    X_RAY_SPEC_INDEX : float, optional
        X-ray spectral energy index (cf. Sec 4.1 of Greig+2018). Typical range is
        (-1, 3).
    X_RAY_Tvir_MIN : float, optional
        Minimum halo virial temperature in which X-rays are produced. Given in log10
        units. Default is `ION_Tvir_MIN`.
    F_H2_SHIELD: float, optional
        Self-shielding factor of molecular hydrogen when experiencing LW suppression.
        Cf. Eq. 12 of Qin+2020. Consistently included in A_LW fit from sims.
        If used we recommend going back to Macachek+01 A_LW=22.86.
    t_STAR : float, optional
        Fractional characteristic time-scale (fraction of hubble time) defining the
        star-formation rate of galaxies. Only used if `USE_MASS_DEPENDENT_ZETA` is set
        to True in :class:`FlagOptions`. See Sec 2.1, Eq. 3 of Park+2018.
    N_RSD_STEPS : int, optional
        Number of steps used in redshift-space-distortion algorithm. NOT A PHYSICAL
        PARAMETER.
    A_LW, BETA_LW: float, optional
        Impact of the LW feedback on Mturn for minihaloes. Default is 22.8685 and 0.47 following Machacek+01, respectively. Latest simulations suggest 2.0 and 0.6. See Sec 2 of Muñoz+21 (2110.13919).
    A_VCB, BETA_VCB: float, optional
        Impact of the DM-baryon relative velocities on Mturn for minihaloes. Default is 1.0 and 1.8, and agrees between different sims. See Sec 2 of Muñoz+21 (2110.13919).
    """

    _ffi = ffi

    _defaults_ = {
        "HII_EFF_FACTOR": 30.0,
        "F_STAR10": -1.3,
        "F_STAR7_MINI": -2.0,
        "ALPHA_STAR": 0.5,
        "ALPHA_STAR_MINI": 0.5,
        "F_ESC10": -1.0,
        "F_ESC7_MINI": -2.0,
        "ALPHA_ESC": -0.5,
        "M_TURN": 8.7,
        "R_BUBBLE_MAX": None,
        "ION_Tvir_MIN": 4.69897,
        "L_X": 40.0,
        "L_X_MINI": 40.0,
        "NU_X_THRESH": 500.0,
        "X_RAY_SPEC_INDEX": 1.0,
        "X_RAY_Tvir_MIN": None,
        "F_H2_SHIELD": 0.0,
        "t_STAR": 0.5,
        "N_RSD_STEPS": 20,
        "A_LW": 2.00,
        "BETA_LW": 0.6,
        "A_VCB": 1.0,
        "BETA_VCB": 1.8,
    }

    def __init__(
        self, *args, INHOMO_RECO=FlagOptions._defaults_["INHOMO_RECO"], **kwargs
    ):
        # TODO: should try to get inhomo_reco out of here... just needed for default of
        #  R_BUBBLE_MAX.
        self.INHOMO_RECO = INHOMO_RECO
        super().__init__(*args, **kwargs)

    def convert(self, key, val):
        """Convert a given attribute before saving it the instance."""
        if key in [
            "F_STAR10",
            "F_ESC10",
            "F_STAR7_MINI",
            "F_ESC7_MINI",
            "M_TURN",
            "ION_Tvir_MIN",
            "L_X",
            "L_X_MINI",
            "X_RAY_Tvir_MIN",
        ]:
            return 10 ** val
        else:
            return val

    @property
    def R_BUBBLE_MAX(self):
        """Maximum radius of bubbles to be searched. Set dynamically."""
        if not self._R_BUBBLE_MAX:
            return 50.0 if self.INHOMO_RECO else 15.0
        else:
            if self.INHOMO_RECO and self._R_BUBBLE_MAX != 50:
                logger.warning(
                    "You are setting R_BUBBLE_MAX != 50 when INHOMO_RECO=True. "
                    "This is non-standard (but allowed), and usually occurs upon manual "
                    "update of INHOMO_RECO"
                )
            return self._R_BUBBLE_MAX

    @property
    def X_RAY_Tvir_MIN(self):
        """Minimum virial temperature of X-ray emitting sources (unlogged and set dynamically)."""
        return self._X_RAY_Tvir_MIN if self._X_RAY_Tvir_MIN else self.ION_Tvir_MIN

    @property
    def NU_X_THRESH(self):
        """Check if the choice of NU_X_THRESH is sensible."""
        if self._NU_X_THRESH < 100.0:
            raise ValueError(
                "Chosen NU_X_THRESH is < 100 eV. NU_X_THRESH must be above 100 eV as it describes X-ray photons"
            )
        elif self._NU_X_THRESH >= global_params.NU_X_BAND_MAX:
            raise ValueError(
                """
                Chosen NU_X_THRESH > {}, which is the upper limit of the adopted X-ray band
                (fiducially the soft band 0.5 - 2.0 keV). If you know what you are doing with this
                choice, please modify the global parameter: NU_X_BAND_MAX""".format(
                    global_params.NU_X_BAND_MAX
                )
            )
        else:
            if global_params.NU_X_BAND_MAX > global_params.NU_X_MAX:
                raise ValueError(
                    """
                    Chosen NU_X_BAND_MAX > {}, which is the upper limit of X-ray integrals (fiducially 10 keV)
                    If you know what you are doing, please modify the global parameter:
                    NU_X_MAX""".format(
                        global_params.NU_X_MAX
                    )
                )
            else:
                return self._NU_X_THRESH

    @property
    def t_STAR(self):
        """Check if the choice of NU_X_THRESH is sensible."""
        if self._t_STAR <= 0.0 or self._t_STAR > 1.0:
            raise ValueError("t_STAR must be above zero and less than or equal to one")
        else:
            return self._t_STAR
