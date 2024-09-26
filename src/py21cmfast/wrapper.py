"""
The main wrapper for the underlying 21cmFAST C-code.

The module provides both low- and high-level wrappers, using the very low-level machinery
in :mod:`~py21cmfast._utils`, and the convenient input and output structures from
:mod:`~py21cmfast.inputs` and :mod:`~py21cmfast.outputs`.

This module provides a number of:

* Low-level functions which simplify calling the background C functions which populate
  these output objects given the input classes.
* High-level functions which provide the most efficient and simplest way to generate the
  most commonly desired outputs.

**Low-level functions**

The low-level functions provided here ease the production of the aforementioned output
objects. Functions exist for each low-level C routine, which have been decoupled as far
as possible. So, functions exist to create :func:`initial_conditions`,
:func:`perturb_field`, :class:`ionize_box` and so on. Creating a brightness temperature
box (often the desired final output) would generally require calling each of these in
turn, as each depends on the result of a previous function. Nevertheless, each function
has the capability of generating the required previous outputs on-the-fly, so one can
instantly call :func:`ionize_box` and get a self-consistent result. Doing so, while
convenient, is sometimes not *efficient*, especially when using inhomogeneous
recombinations or the spin temperature field, which intrinsically require consistent
evolution of the ionization field through redshift. In these cases, for best efficiency
it is recommended to either use a customised manual approach to calling these low-level
functions, or to call a higher-level function which optimizes this process.

Finally, note that :mod:`py21cmfast` attempts to optimize the production of the large amount of
data via on-disk caching. By default, if a previous set of data has been computed using
the current input parameters, it will be read-in from a caching repository and returned
directly. This behaviour can be tuned in any of the low-level (or high-level) functions
by setting the `write`, `direc`, `regenerate` and `match_seed` parameters (see docs for
:func:`initial_conditions` for details). The function :func:`~query_cache` can be used
to search the cache, and return empty datasets corresponding to each (and these can then be
filled with the data merely by calling ``.read()`` on any data set). Conversely, a
specific data set can be read and returned as a proper output object by calling the
:func:`~py21cmfast.cache_tools.readbox` function.


**High-level functions**

As previously mentioned, calling the low-level functions in some cases is non-optimal,
especially when full evolution of the field is required, and thus iteration through a
series of redshift. In addition, while :class:`InitialConditions` and
:class:`PerturbedField` are necessary intermediate data, it is *usually* the resulting
brightness temperature which is of most interest, and it is easier to not have to worry
about the intermediate steps explicitly. For these typical use-cases, two high-level
functions are available: :func:`run_coeval` and :func:`run_lightcone`, whose purpose
should be self-explanatory. These will optimally run all necessary intermediate
steps (using cached results by default if possible) and return all datasets of interest.


Examples
--------
A typical example of using this module would be the following.

>>> import py21cmfast as p21

Get coeval cubes at redshift 7,8 and 9, without spin temperature or inhomogeneous
recombinations:

>>> coeval = p21.run_coeval(
>>>     redshift=[7,8,9],
>>>     cosmo_params=p21.CosmoParams(hlittle=0.7),
>>>     user_params=p21.UserParams(HII_DIM=100)
>>> )

Get coeval cubes at the same redshift, with both spin temperature and inhomogeneous
recombinations, pulled from the natural evolution of the fields:

>>> all_boxes = p21.run_coeval(
>>>                 redshift=[7,8,9],
>>>                 user_params=p21.UserParams(HII_DIM=100),
>>>                 flag_options=p21.FlagOptions(INHOMO_RECO=True),
>>>                 do_spin_temp=True
>>>             )

Get a self-consistent lightcone defined between z1 and z2 (`z_step_factor` changes the
logarithmic steps between redshift that are actually evaluated, which are then
interpolated onto the lightcone cells):

>>> lightcone = p21.run_lightcone(redshift=z2, max_redshift=z2, z_step_factor=1.03)
"""
import logging
import numpy as np
import os
import warnings
from astropy import units
from astropy.cosmology import z_at_value
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.special import erf
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
# JordanFlitter: I import tqdm to visualize the progress of the lightcone calculation
import tqdm
# JordanFlitter: we need to do FFT in case we want to compute the scale-dependent c_T
from scipy.fft import fftn
# JordanFlitter: I import time to pause the program for a short time, just to print messages before getting into the C-codes
import time

from ._cfg import config
from ._utils import (
    OutputStruct,
    StructWrapper,
    _check_compatible_inputs,
    _process_exitcode,
)
from .c_21cmfast import ffi, lib
from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams, global_params
from .outputs import (
    BrightnessTemp,
    Coeval,
    HaloField,
    InitialConditions,
    IonizedBox,
    LightCone,
    PerturbedField,
    PerturbHaloField,
    TsBox,
    _OutputStructZ,
)
# JordanFlitter: import the module to genereate ICs from CLASS
from .generate_ICs import run_ICs 
logger = logging.getLogger(__name__)

# JordanFlitter: I added some logics to prevent conflict between inputs
# I'm not sure if that's the best place for these logics, but it works...
def _configure_user_params(user_params,user_params_dic):
    # First, let's have some SDM logics
    if user_params.SCATTERING_DM:
        if not user_params.RUN_CLASS:
            try:
                if user_params_dic["SCATTERING_DM"] and not user_params_dic["RUN_CLASS"]:
                    logger.warning("You have set SCATTERING_DM to True but RUN_CLASS to False!")
                    logger.warning("Without RUN_CLASS the initial conditions won't be consistent with your SDM model.")
                    logger.warning("Automatically setting RUN_CLASS to True.\n")
            except KeyError:
                pass
            user_params.RUN_CLASS = True
        if user_params.FUZZY_DM:
            try:
                if user_params_dic["SCATTERING_DM"] and user_params_dic["FUZZY_DM"]:
                    logger.warning("You have set both SCATTERING_DM and FUZZY_DM to True!")
                    logger.warning("Current version does not support mixed dark matter models with FDM and SDM.")
                    logger.warning("Automatically setting FUZZY_DM to False.\n")
            except KeyError:
                pass
            user_params.FUZZY_DM = False
        if user_params.DO_VCB_FIT:
            try:
                if user_params_dic["SCATTERING_DM"] and user_params_dic["DO_VCB_FIT"]:
                    logger.warning("You have set both SCATTERING_DM and DO_VCB_FIT to True!")
                    logger.warning("Current version does not support correcting the matter power spectrum "
                                   "with relative velocity effects for models other than Lambda CDM.")
                    logger.warning("Automatically setting DO_VCB_FIT to False.\n")
            except KeyError:
                pass
            user_params.DO_VCB_FIT = False
        if not user_params.START_AT_RECOMBINATION:
            try:
                if user_params_dic["SCATTERING_DM"] and not user_params_dic["START_AT_RECOMBINATION"]:
                    logger.warning("You have set SCATTERING_DM to True but START_AT_RECOMBINATION to False!")
                    logger.warning("For SDM we need to start before cosmic dawn because of the non-linear fluctuations caused by the relative velocity.")
                    logger.warning("Automatically setting START_AT_RECOMBINATION to True.\n")
            except KeyError:
                pass
            user_params.START_AT_RECOMBINATION = True
        if not user_params.USE_SDM_FLUCTS:
            try:
                if user_params_dic["SCATTERING_DM"] and not user_params_dic["USE_SDM_FLUCTS"]:
                    logger.warning("You have set SCATTERING_DM to True but USE_SDM_FLUCTS to False!")
                    logger.warning("This means that the initial V_chi_b box is homogeneous.")
                    logger.warning("Set USE_SDM_FLUCTS to True to have initial inhomogeneous V_chi_b box.\n")
            except KeyError:
                pass
        if not user_params.MANY_Z_SAMPLES_AT_COSMIC_DAWN:
            logger.warning("You have set SCATTERING_DM to True but MANY_Z_SAMPLES_AT_COSMIC_DAWN is False!")
            logger.warning("For large SDM cross sections we need more redshift iterations during cosmic dawn "
                           "in order to evolve the temperature correctly.")
            logger.warning("Consider setting MANY_Z_SAMPLES_AT_COSMIC_DAWN to True to avoid numerical errors.\n")
        if not user_params.USE_CS_S_ALPHA:
            try:
                if user_params_dic["SCATTERING_DM"] and not user_params_dic["USE_CS_S_ALPHA"]:
                    logger.warning("You have set SCATTERING_DM to True but USE_CS_S_ALPHA to False!")
                    logger.warning("For large SDM cross sections we need to use Chuzhouy and Shapiro’s S_alpha "
                                   "because Hirata’s fit gives junk at low temperature.")
                    logger.warning("Automatically setting USE_CS_S_ALPHA to True.\n")
            except KeyError:
                pass
            user_params.USE_CS_S_ALPHA = True
        if not user_params.USE_HYREC and not user_params.USE_ALPHA_B:
            try:
                if user_params_dic["SCATTERING_DM"] and not user_params_dic["USE_HYREC"] and not user_params_dic["USE_ALPHA_B"]:
                    logger.warning("You have set SCATTERING_DM to True but USE_HYREC and USE_ALPHA_B are False!")
                    logger.warning("For large SDM cross sections we need to use to use alpha_B "
                                   "because the fit for alpha_A gives junk at low temperature.")
                    logger.warning("Automatically setting USE_ALPHA_B to True.")
                    logger.warning("Alternatively, consider setting USE_HYREC to True.\n")
            except KeyError:
                pass
            user_params.USE_ALPHA_B = True
    try:
        if not user_params_dic["SCATTERING_DM"] and user_params_dic["USE_SDM_FLUCTS"]:
            logger.warning("You have set USE_SDM_FLUCTS to True but SCATTERING_DM to False!")
            logger.warning("We need SDM in order to compute its initial relative velocity fluctuations.")
            logger.warning("I am not sure what you were trying to do, continuing with SCATTERING_DM=False.")
            logger.warning("If you wish to simulate SDM set SCATTERING_DM to True.\n")
    except KeyError:
        pass
    # Next, FDM logics
    if user_params.FUZZY_DM:
        if not user_params.RUN_CLASS:
            try:
                if user_params_dic["FUZZY_DM"] and not user_params_dic["RUN_CLASS"]:
                    logger.warning("You have set FUZZY_DM to True but RUN_CLASS to False!")
                    logger.warning("Without RUN_CLASS the initial conditions won't be consistent with your FDM model.")
                    logger.warning("Automatically setting RUN_CLASS to True.\n")
            except KeyError:
                pass
            user_params.RUN_CLASS = True
        if user_params.DO_VCB_FIT:
            try:
                if user_params_dic["FUZZY_DM"] and user_params_dic["DO_VCB_FIT"]:
                    logger.warning("You have set both FUZZY_DM and DO_VCB_FIT to True!")
                    logger.warning("Current version does not support correcting the matter power spectrum "
                                   "with relative velocity effects for models other than Lambda CDM.")
                    logger.warning("Automatically setting DO_VCB_FIT to False.\n")
            except KeyError:
                pass
            user_params.DO_VCB_FIT = False
    # Next, dark ages logics
    if user_params.Z_HIGH_MAX > global_params.Z_HEAT_MAX:
        if not user_params.START_AT_RECOMBINATION:
            try:
                if ((user_params_dic["Z_HIGH_MAX"] > global_params.Z_HEAT_MAX) and not user_params_dic["START_AT_RECOMBINATION"]):
                    logger.warning("You have set Z_HIGH_MAX above Z_HEAT_MAX but START_AT_RECOMBINATION to False!")
                    logger.warning("In order to begin above Z_HEAT_MAX we need START_AT_RECOMBINATION to be True.")
                    logger.warning("Automatically setting START_AT_RECOMBINATION to True.\n")
            except KeyError:
                pass
            user_params.START_AT_RECOMBINATION = True
    if user_params.OUTPUT_AT_DARK_AGES:
        if not user_params.START_AT_RECOMBINATION:
            try:
                if not user_params_dic["START_AT_RECOMBINATION"]:
                    logger.warning("You have set START_AT_RECOMBINATION to False but OUTPUT_AT_DARK_AGES is True!")
                    logger.warning("In order to have output at the dark ages we need START_AT_RECOMBINATION to be True.")
                    logger.warning("Automatically setting START_AT_RECOMBINATION to True.\n")
            except KeyError:
                pass
            user_params.START_AT_RECOMBINATION = True
        if not user_params.EVOLVE_BARYONS:
            try:
                if not user_params_dic["EVOLVE_BARYONS"]:
                    logger.warning("You have set EVOLVE_BARYONS to False but OUTPUT_AT_DARK_AGES is True!")
                    logger.warning("Note that the 21cm power spectrum during the dark ages will be inconsistent.")
                    logger.warning("Consider setting EVOLVE_BARYONS to True.\n")
            except KeyError:
                pass
    if user_params.START_AT_RECOMBINATION:
        if ((user_params.Z_HIGH_MAX < global_params.Z_HEAT_MAX) and (user_params.Z_HIGH_MAX > 0.)):
            try:
                if ((user_params_dic["Z_HIGH_MAX"] < global_params.Z_HEAT_MAX) and (user_params_dic["Z_HIGH_MAX"] > 0.)):
                    logger.warning("You have set Z_HIGH_MAX below Z_HEAT_MAX with a positive value but START_AT_RECOMBINATION is True!")
                    logger.warning("In order to do calculations during the dark ages we need either that Z_HIGH_MAX to be above Z_HEAT_MAX "
                                   "or Z_HIGH_MAX to be negative (the latter option would cause the code to start from recombination).")
                    logger.warning("Automatically setting Z_HIGH_MAX to -1.\n")
            except KeyError:
                pass
            user_params.Z_HIGH_MAX = -1.
        if not user_params.USE_TCA_COMPTON:
            try:
                if not user_params_dic["USE_TCA_COMPTON"]:
                    logger.warning("You have set USE_TCA_COMPTON to False but START_AT_RECOMBINATION is True!")
                    logger.warning("This is possible, but it only increases runtime for no good reason.")
                    logger.warning("Consider setting USE_TCA_COMPTON to True.\n")
            except KeyError:
                pass
        if user_params.USE_DICKE_GROWTH_FACTOR:
            try:
                if user_params_dic["USE_DICKE_GROWTH_FACTOR"]:
                    logger.warning("You have set USE_DICKE_GROWTH_FACTOR to True but START_AT_RECOMBINATION is also True!")
                    logger.warning("Note that the Dicke growth factor is inconsistent during the dark ages.")
                    logger.warning("Automatically setting USE_DICKE_GROWTH_FACTOR to False.\n")
            except KeyError:
                pass
            user_params.USE_DICKE_GROWTH_FACTOR = False
        if not user_params.DO_PERTURBS_WITH_TS:
            try:
                if not user_params_dic["DO_PERTURBS_WITH_TS"]:
                    logger.warning("You have set DO_PERTURBS_WITH_TS to False but START_AT_RECOMBINATION is True!")
                    logger.warning("This might cause the code to crash.")
                    logger.warning("Automatically setting DO_PERTURBS_WITH_TS to True.\n")
            except KeyError:
                pass
            user_params.DO_PERTURBS_WITH_TS = True
        if user_params.USE_ADIABATIC_FLUCTUATIONS:
            try:
                if user_params_dic["USE_ADIABATIC_FLUCTUATIONS"]:
                    logger.warning("You have set USE_ADIABATIC_FLUCTUATIONS to True but START_AT_RECOMBINATION is also True!")
                    logger.warning("It only makes sense to USE_ADIABATIC_FLUCTUATIONS when not doing any calculations during the dark ages, "
                                   "as it generates INITIAL temperature fluctuations at the begining of cosmic dawn.")
                    logger.warning("Automatically setting USE_ADIABATIC_FLUCTUATIONS to False.\n")
            except KeyError:
                pass
            user_params.USE_ADIABATIC_FLUCTUATIONS = False
    else:
        if user_params.MANY_Z_SAMPLES_AT_COSMIC_DAWN:
            try:
                if not user_params_dic["START_AT_RECOMBINATION"] and user_params_dic["MANY_Z_SAMPLES_AT_COSMIC_DAWN"]:
                    logger.warning("You have set START_AT_RECOMBINATION to False but MANY_Z_SAMPLES_AT_COSMIC_DAWN to True!")
                    logger.warning("Current version does not support having many iterations during cosmic dawn without doing "
                                   "any calculations during the dark ages.")
                    logger.warning("Automatically setting MANY_Z_SAMPLES_AT_COSMIC_DAWN to False.\n")
            except KeyError:
                pass
            user_params.MANY_Z_SAMPLES_AT_COSMIC_DAWN = False
        if user_params.NO_INI_MATTER_FLUCTS:
            try:
                if not user_params_dic["START_AT_RECOMBINATION"] and user_params_dic["NO_INI_MATTER_FLUCTS"]:
                    logger.warning("You have set START_AT_RECOMBINATION to False but NO_INI_MATTER_FLUCTS to True!")
                    logger.warning("If you do not want initial matter fluctuations then set START_AT_RECOMBINATION to True.")
                    logger.warning("Automatically setting NO_INI_MATTER_FLUCTS to False.\n")
            except KeyError:
                pass
            user_params.NO_INI_MATTER_FLUCTS = False
    # Finally, let's do some FirstCLASS logics
    if user_params.DO_VCB_FIT:
        if not user_params.RUN_CLASS:
            try:
                if user_params_dic["DO_VCB_FIT"] and not user_params_dic["RUN_CLASS"]:
                    logger.warning("You have set DO_VCB_FIT to True but RUN_CLASS to False!")
                    logger.warning("We need to run CLASS in order to find the correction to the matter power spectrum "
                                   "due to relative velocity.")
                    logger.warning("Automatically setting RUN_CLASS to True.\n")
            except KeyError:
                pass
            user_params.RUN_CLASS = True
    if user_params.EVOLVE_BARYONS:
        if not user_params.RUN_CLASS:
            try:
                if not user_params_dic["RUN_CLASS"]:
                    logger.warning("You have set RUN_CLASS to False but EVOLVE_BARYONS is True!")
                    logger.warning("We need to run CLASS in order to find the scale-dependent baryons growth factor.")
                    logger.warning("If you wish to not run CLASS set also EVOLVE_BARYONS to False.")
                    logger.warning("Automatically setting RUN_CLASS to True.\n")
            except KeyError:
                pass
            user_params.RUN_CLASS = True
        if user_params.NO_INI_MATTER_FLUCTS:
            try:
                if user_params_dic["NO_INI_MATTER_FLUCTS"]:
                    logger.warning("You have set NO_INI_MATTER_FLUCTS to True but EVOLVE_BARYONS is also True!")
                    logger.warning("If you do not want initial matter fluctuations then set EVOLVE_BARYONS to False.")
                    logger.warning("Automatically setting NO_INI_MATTER_FLUCTS to False.\n")
            except KeyError:
                pass
            user_params.NO_INI_MATTER_FLUCTS = False
    if user_params.RUN_CLASS:
        if not user_params.POWER_SPECTRUM == 5:
            try:
                if not user_params_dic["POWER_SPECTRUM"]:
                    logger.warning("You have set POWER_SPECTRUM to be not 5 but RUN_CLASS is True!")
                    logger.warning("This would cause the code to ignore CLASS transfer function.")
                    logger.warning("Automatically setting POWER_SPECTRUM to 5.\n")
            except KeyError:
                pass
            user_params.POWER_SPECTRUM = 5
        if user_params.EVOLVE_BARYONS and user_params.USE_ADIABATIC_FLUCTUATIONS:
            try:
                if user_params_dic["USE_ADIABATIC_FLUCTUATIONS"]:
                    logger.warning("You have set USE_ADIABATIC_FLUCTUATIONS to True but EVOLVE_BARYONS is also True!")
                    logger.warning("This means that for consistent initial conditions we need to have a two parameter fit "
                                   "for the temperature fluctuations. We currently do not have such a fit.")
                    logger.warning("Automatically setting USE_ADIABATIC_FLUCTUATIONS to False.\n")
            except KeyError:
                pass
            user_params.USE_ADIABATIC_FLUCTUATIONS = False
    else:
        if not user_params.START_AT_RECOMBINATION and not user_params.USE_ADIABATIC_FLUCTUATIONS:
            try:
                if not user_params_dic["RUN_CLASS"] and not user_params_dic["START_AT_RECOMBINATION"] and not user_params_dic["USE_ADIABATIC_FLUCTUATIONS"]:
                    logger.warning("You have set RUN_CLASS to False but START_AT_RECOMBINATION and USE_ADIABATIC_FLUCTUATIONS are also False!")
                    logger.warning("This means that the code starts with homogeneous temperature box, which would lead to inconsistencies in the 21cm power spectrum.")
                    logger.warning("Automatically setting USE_ADIABATIC_FLUCTUATIONS to True.\n")
            except KeyError:
                pass
            user_params.USE_ADIABATIC_FLUCTUATIONS = True

# JordanFlitterTODO: Remove all the following variables to a new structure and get rid of this function.
def _set_default_globals():
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
                                             2.7489977 ,  2.84899769,  2.94899765,  3.04899772,  3.1489976 ,
                                             3.24899768,  3.34899762,  3.44899771,  3.54899762]
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
    global_params.Y_He = 0.245
    global_params.VAVG = 25.86
    global_params.Z_REC = 1069.
    global_params.A_VCB_PM = 0.24
    global_params.KP_VCB_PM = 300.
    global_params.SIGMAK_VCB_PM = 0.9

def _configure_inputs(
    defaults: list,
    *datasets,
    ignore: list = ["redshift"],
    flag_none: [list, None] = None,
):
    """Configure a set of input parameter structs.

    This is useful for basing parameters on a previous output.
    The logic is this: the struct _cannot_ be present and different in both defaults and
    a dataset. If it is present in _either_ of them, that will be returned. If it is
    present in _neither_, either an error will be raised (if that item is in `flag_none`)
    or it will pass.

    Parameters
    ----------
    defaults : list of 2-tuples
        Each tuple is (key, val). Keys are input struct names, and values are a default
        structure for that input.
    datasets : list of :class:`~_utils.OutputStruct`
        A number of output datasets to cross-check, and draw parameter values from.
    ignore : list of str
        Attributes to ignore when ensuring that parameter inputs are the same.
    flag_none : list
        A list of parameter names for which ``None`` is not an acceptable value.

    Raises
    ------
    ValueError :
        If an input parameter is present in both defaults and the dataset, and is different.
        OR if the parameter is present in neither defaults not the datasets, and it is
        included in `flag_none`.
    """
    # First ensure all inputs are compaible in their parameters
    _check_compatible_inputs(*datasets, ignore=ignore)

    if flag_none is None:
        flag_none = []

    output = [0] * len(defaults)
    for i, (key, val) in enumerate(defaults):
        # Get the value of this input from the datasets
        data_val = None
        for dataset in datasets:
            if dataset is not None and hasattr(dataset, key):
                data_val = getattr(dataset, key)
                break

        # If both data and default have values
        if not (val is None or data_val is None or data_val == val):
            raise ValueError(
                "%s has an inconsistent value with %s"
                % (key, dataset.__class__.__name__)
            )
        else:
            if val is not None:
                output[i] = val
            elif data_val is not None:
                output[i] = data_val
            elif key in flag_none:
                raise ValueError(
                    "For %s, a value must be provided in some manner" % key
                )
            else:
                output[i] = None

    return output


def configure_redshift(redshift, *structs):
    """
    Check and obtain a redshift from given default and structs.

    Parameters
    ----------
    redshift : float
        The default redshift to use
    structs : list of :class:`~_utils.OutputStruct`
        A number of output datasets from which to find the redshift.

    Raises
    ------
    ValueError :
        If both `redshift` and *all* structs have a value of `None`, **or** if any of them
        are different from each other (and not `None`).
    """
    zs = {s.redshift for s in structs if s is not None and hasattr(s, "redshift")}
    zs = list(zs)

    if len(zs) > 1 or (len(zs) == 1 and redshift is not None and zs[0] != redshift):
        raise ValueError("Incompatible redshifts in inputs")
    elif len(zs) == 1:
        return zs[0]
    elif redshift is None:
        raise ValueError(
            "Either redshift must be provided, or a data set containing it."
        )
    else:
        return redshift


def _verify_types(**kwargs):
    """Ensure each argument has a type of None or that matching its name."""
    for k, v in kwargs.items():
        for j, kk in enumerate(
            ["init", "perturb", "ionize", "spin_temp", "halo_field", "pt_halos"]
        ):
            if kk in k:
                break
        cls = [
            InitialConditions,
            PerturbedField,
            IonizedBox,
            TsBox,
            HaloField,
            PerturbHaloField,
        ][j]

        if v is not None and not isinstance(v, cls):
            raise ValueError(f"{k} must be an instance of {cls.__name__}")


def _call_c_simple(fnc, *args):
    """Call a simple C function that just returns an object.

    Any such function should be defined such that the last argument is an int pointer generating
    the status.
    """
    # Parse the function to get the type of the last argument
    cdata = str(ffi.addressof(lib, fnc.__name__))
    kind = cdata.split("(")[-1].split(")")[0].split(",")[-1]
    result = ffi.new(kind)
    status = fnc(*args, result)
    _process_exitcode(status, fnc, args)
    return result[0]


def _get_config_options(
    direc, regenerate, write, hooks
) -> Tuple[str, bool, Dict[Callable, Dict[str, Any]]]:

    direc = str(os.path.expanduser(config["direc"] if direc is None else direc))
    hooks = hooks or {}

    if callable(write) and write not in hooks:
        hooks[write] = {"direc": direc}

    if not hooks:
        if write is None:
            write = config["write"]

        if not callable(write) and write:
            hooks["write"] = {"direc": direc}

    return (
        direc,
        bool(config["regenerate"] if regenerate is None else regenerate),
        hooks,
    )


def get_all_fieldnames(
    arrays_only=True, lightcone_only=False, as_dict=False
) -> Union[Dict[str, str], Set[str]]:
    """Return all possible fieldnames in output structs.

    Parameters
    ----------
    arrays_only : bool, optional
        Whether to only return fields that are arrays.
    lightcone_only : bool, optional
        Whether to only return fields from classes that evolve with redshift.
    as_dict : bool, optional
        Whether to return results as a dictionary of ``quantity: class_name``.
        Otherwise returns a set of quantities.
    """
    classes = [cls(redshift=0) for cls in _OutputStructZ._implementations()]

    if not lightcone_only:
        classes.append(InitialConditions())

    attr = "pointer_fields" if arrays_only else "fieldnames"

    if as_dict:
        return {
            name: cls.__class__.__name__
            for cls in classes
            for name in getattr(cls, attr)
        }
    else:
        return {name for cls in classes for name in getattr(cls, attr)}


# ======================================================================================
# WRAPPING FUNCTIONS
# ======================================================================================
def construct_fftw_wisdoms(*, user_params=None, cosmo_params=None):
    """Construct all necessary FFTW wisdoms.

    Parameters
    ----------
    user_params : :class:`~inputs.UserParams`
        Parameters defining the simulation run.

    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    # Run the C code
    if user_params.USE_FFTW_WISDOM:
        return lib.CreateFFTWWisdoms(user_params(), cosmo_params())
    else:
        return 0


def compute_tau(*, redshifts, global_xHI, user_params=None, cosmo_params=None):
    """Compute the optical depth to reionization under the given model.

    Parameters
    ----------
    redshifts : array-like
        Redshifts defining an evolution of the neutral fraction.
    global_xHI : array-like
        The mean neutral fraction at `redshifts`.
    user_params : :class:`~inputs.UserParams`
        Parameters defining the simulation run.
    cosmo_params : :class:`~inputs.CosmoParams`
        Cosmological parameters.

    Returns
    -------
    tau : float
        The optional depth to reionization

    Raises
    ------
    ValueError :
        If `redshifts` and `global_xHI` have inconsistent length or if redshifts are not
        in ascending order.
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    if len(redshifts) != len(global_xHI):
        raise ValueError("redshifts and global_xHI must have same length")

    if not np.all(np.diff(redshifts) > 0):
        raise ValueError("redshifts and global_xHI must be in ascending order")

    # Convert the data to the right type
    redshifts = np.array(redshifts, dtype="float32")
    global_xHI = np.array(global_xHI, dtype="float32")

    z = ffi.cast("float *", ffi.from_buffer(redshifts))
    xHI = ffi.cast("float *", ffi.from_buffer(global_xHI))

    # Run the C code
    return lib.ComputeTau(user_params(), cosmo_params(), len(redshifts), z, xHI)

# !!! SLTK: get SFR from the C code
def compute_wSFR(*,
              Mh: np.ndarray,
              redshifts: np.array,
              user_params=None,
              cosmo_params=None,
              astro_params=None,
              flag_options=None,
              ):
    """Compute the UV magnitude of a halo of a given halo mass array at a given redshift.
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(flag_options)


    redshifts = np.array(redshifts, dtype="float32")

    SFR = np.zeros(len(redshifts) * np.shape(Mh)[1])

    SFR.shape = (len(redshifts), np.shape(Mh)[1])

    c_Mh_SFR = ffi.cast("double *", ffi.from_buffer(Mh))
    c_SFR = ffi.cast("double *", ffi.from_buffer(SFR))
    errcode = lib.output_wSFR(
        user_params(),
        cosmo_params(),
        astro_params(),
        flag_options(),
        len(redshifts),
        ffi.cast("float *", ffi.from_buffer(redshifts)),
        np.shape(Mh)[1],
        c_Mh_SFR,
        c_SFR,
    )

    _process_exitcode(
        errcode,
        lib.output_wSFR,
        (
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
            len(redshifts),
        ),
    )

    return SFR

# !!! SLTK: conversion from halo mass to UV magnitude 
def Muv_of_Mh(*,
              Mh: np.ndarray,
              redshifts: np.array,
              user_params=None,
              cosmo_params=None,
              astro_params=None,
              flag_options=None,
              ):
    """Compute the UV magnitude of a halo of a given halo mass array at a given redshift.
    """

    SFR = compute_wSFR(Mh = Mh,redshifts = redshifts, user_params=user_params, cosmo_params=cosmo_params, astro_params=astro_params,flag_options=flag_options)

    '''
    # !!! MUN21
    if astro_params.SFR_MODEL == 0:

        Fstar = pow(10,astro_params.F_STAR10) * pow(Mh/1e10,astro_params.ALPHA_STAR)

        Fstar[Fstar >= 1.] = 1.
        
        M_star = cosmo_params.OMb / cosmo_params.OMm  * Fstar* Mh 

        H_z = cosmo_params.hlittle * np.sqrt(
        cosmo_params.OMm * np.power(1 + redshifts, 3) + global_params.OMr * np.power(1 + redshifts,4) + cosmo_params.OMl) * 3.2407e-18  # units of 1/s
        H_z *= 3.1536e7  # convert from 1/s to 1/year

        H_z.shape = (len(redshifts), 1)

        SFR = M_star * H_z / astro_params.t_STAR

    # YUE
    elif astro_params.SFR_MODEL == 1:


        Mpivot = pow(10,astro_params.Mp)

        # in this model we re-label epsilon_0 -> Fstar10 , gamma_high -> Alpha_star this already contains the Omb/OmM factor
        epsilon = 2*pow(10,astro_params.F_STAR10) / (pow(Mh / Mpivot, astro_params.ALPHA_STAR_HIGHM) + pow(Mh / Mpivot, astro_params.ALPHA_STAR))

        epsilon[epsilon >= 1.] = 1.

        amplitude_s = astro_params.Mdot12_YUE 
        redshifts.shape = (len(redshifts), 1)

        Mdot = amplitude_s * pow(Mh/1e12,astro_params.Alpha_accrYUE) * (1. + astro_params.z_accrYUE * redshifts)*np.sqrt(cosmo_params.OMm *pow(1+redshifts,3) + cosmo_params.OMl)

        SFR = epsilon * Mdot * cosmo_params.OMb / cosmo_params.OMm

    # GALLUMI model II 
    elif astro_params.SFR_MODEL == 2:        

        redshifts.shape = (len(redshifts), 1)
        eps_star = pow((1+redshifts)/7,astro_params.EPS_STAR_S_G) * pow(10,astro_params.F_STAR10)

        Mpivot = pow((1+redshifts)/7.,astro_params.M_C_S_G) * pow(10,astro_params.Mp)
        M_c = pow((1+redshifts)/7,astro_params.M_C_S_G) * pow(10,astro_params.Mp) 

        Fstar = eps_star / (pow(Mh/M_c,astro_params.ALPHA_STAR_HIGHM) + pow(Mh/M_c,astro_params.ALPHA_STAR))
    
        Fstar[Fstar/(cosmo_params.OMb / cosmo_params.OMm) >= 1.] = 1./(cosmo_params.OMb / cosmo_params.OMm)

        M_star = Fstar * Mh # GALLUMI already includes OMb/OMm in epsilon

        H_z = cosmo_params.hlittle * np.sqrt(
        cosmo_params.OMm * np.power(1 + redshifts, 3) + global_params.OMr * np.power(1 + redshifts,4) + cosmo_params.OMl) * 3.2407e-18  # units of 1/s
        H_z *= 3.1536e7  # convert from 1/s to 1/year

        H_z.shape = (len(redshifts), 1)

        SFR = M_star * H_z / astro_params.t_STAR

    else: 
        print('SFR MODEL not implemented yet!')
        return -1
    '''

    Kuv = 1.15e-28  # M_sun * sec /yr / erg

    Luv = SFR / Kuv

    Muv = -2.5 * np.log10(Luv) + 51.63

    return Muv

# !!! SLTK: added function to compute the HMF based on function in ps.c
def compute_HMF(*,
                redshifts,
                user_params=None,
                cosmo_params=None,
                astro_params=None,
                flag_options=None,
                nbins=100,
                ):
    """Compute the halo mass function over a given number of bins and redshifts.
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(flag_options)

    redshifts = np.array(redshifts, dtype="float32")

    HMF = np.zeros(len(redshifts) * nbins)
    Mh_HMF = np.zeros(len(redshifts) * nbins)

    HMF.shape = (len(redshifts), nbins)
    Mh_HMF.shape = (len(redshifts), nbins)

    c_Mh_HMF = ffi.cast("double *", ffi.from_buffer(Mh_HMF))
    c_HMF = ffi.cast("double *", ffi.from_buffer(HMF))

    errcode = lib.ComputeHMF_API(
        nbins,
        user_params(),
        cosmo_params(),
        astro_params(),
        flag_options(),
        len(redshifts),
        ffi.cast("float *", ffi.from_buffer(redshifts)),
        c_Mh_HMF,
        c_HMF,
    )

    _process_exitcode(
        errcode,
        lib.ComputeHMF_API,
        (
            nbins,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
            len(redshifts),
        ),
    )
    return Mh_HMF, HMF

# !!! SLTK: compute luminosity function
def compute_luminosity_function(*,
                                redshifts: np.array,
                                user_params=None,
                                cosmo_params=None,
                                astro_params=None,
                                flag_options=None,
                                nbins=100,
                                ):

    Mh_HMF, HMF = compute_HMF(redshifts=redshifts, user_params=user_params, cosmo_params=cosmo_params, astro_params=astro_params,flag_options=flag_options, nbins=nbins)

    Muv_mean = Muv_of_Mh(Mh=Mh_HMF, redshifts=redshifts, user_params=user_params, cosmo_params=cosmo_params,
                         astro_params=astro_params, flag_options=flag_options)

    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(flag_options)

    # fduty is used to estimate the minimum mass for the integration
    M_turn = pow(10,astro_params.M_TURN)

    sigma_uv = astro_params.sigma_Muv  # magnitude scatter

    LF = np.zeros(Muv_mean.shape)

    for i, m_uv_z in enumerate(Muv_mean):

        width = 0.5 if redshifts[i] < 8 else 0.8

        for j, m_uv in enumerate(m_uv_z):
            P_Muv_integrated = 0.5 * (
                    erf((m_uv_z - m_uv + width / 2.0) / (sigma_uv * np.sqrt(2)))
                    - erf((m_uv_z - m_uv - width / 2.0) / (sigma_uv * np.sqrt(2)))
            )

            if astro_params.SFR_MODEL == 0 or astro_params.SFR_MODEL == 2 or astro_params.SFR_MODEL == 3:
                f_duty = np.exp(-M_turn/ Mh_HMF[i])
            elif astro_params.SFR_MODEL == 1:
                f_duty = np.exp(-M_turn*pow((1+redshifts[i])/7.,-1.5)/Mh_HMF[i])
            else:
                print('f_duty not defined for this model, we set it to 1')
                f_duty = 1.

            # fduty is used to estimate the minimum mass for the integration
            #ind  = np.argmin(np.abs(0.9 - f_duty))
            #print(f'minimal halo mass: {np.log10(Mh_HMF[i][ind])} \n minial Muv: {Muv_mean[i][ind]} ')

            # func = HMF[i] * f_duty * P_Muv_integrated / width
            func = HMF[i] * f_duty * P_Muv_integrated / width

            LF[i, j] = np.trapz(func, Mh_HMF[i])

    return Muv_mean, Mh_HMF, LF


# !!! SLTK: this is the old luminosity function, based on computation inside ps.c that we removed
# def compute_luminosity_function(
#     *,
#     redshifts,
#     user_params=None,
#     cosmo_params=None,
#     astro_params=None,
#     flag_options=None,
#     nbins=100,
#     mturnovers=None,
#     mturnovers_mini=None,
#     component=0,
# ):
#     """Compute a the luminosity function over a given number of bins and redshifts.

#     Parameters
#     ----------
#     redshifts : array-like
#         The redshifts at which to compute the luminosity function.
#     user_params : :class:`~UserParams`, optional
#         Defines the overall options and parameters of the run.
#     cosmo_params : :class:`~CosmoParams`, optional
#         Defines the cosmological parameters used to compute initial conditions.
#     astro_params : :class:`~AstroParams`, optional
#         The astrophysical parameters defining the course of reionization.
#     flag_options : :class:`~FlagOptions`, optional
#         Some options passed to the reionization routine.
#     nbins : int, optional
#         The number of luminosity bins to produce for the luminosity function.
#     mturnovers : array-like, optional
#         The turnover mass at each redshift for massive halos (ACGs).
#         Only required when USE_MINI_HALOS is True.
#     mturnovers_mini : array-like, optional
#         The turnover mass at each redshift for minihalos (MCGs).
#         Only required when USE_MINI_HALOS is True.
#     component : int, optional
#         The component of the LF to be calculated. 0, 1 an 2 are for the total,
#         ACG and MCG LFs respectively, requiring inputs of both mturnovers and
#         mturnovers_MINI (0), only mturnovers (1) or mturnovers_MINI (2).

#     Returns
#     -------
#     Muvfunc : np.ndarray
#         Magnitude array (i.e. brightness). Shape [nredshifts, nbins]
#     Mhfunc : np.ndarray
#         Halo mass array. Shape [nredshifts, nbins]
#     lfunc : np.ndarray
#         Number density of haloes corresponding to each bin defined by `Muvfunc`.
#         Shape [nredshifts, nbins].
#     """
#     user_params = UserParams(user_params)
#     cosmo_params = CosmoParams(cosmo_params)
#     astro_params = AstroParams(astro_params)
#     flag_options = FlagOptions(
#         flag_options, USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
#     )

#     redshifts = np.array(redshifts, dtype="float32")
#     if flag_options.USE_MINI_HALOS:
#         if component in [0, 1]:
#             if mturnovers is None:
#                 logger.warning(
#                     "calculating ACG LFs with mini-halo feature requires users to "
#                     "specify mturnovers!"
#                 )
#                 return None, None, None

#             mturnovers = np.array(mturnovers, dtype="float32")
#             if len(mturnovers) != len(redshifts):
#                 logger.warning(
#                     "mturnovers(%d) does not match the length of redshifts (%d)"
#                     % (len(mturnovers), len(redshifts))
#                 )
#                 return None, None, None
#         if component in [0, 2]:
#             if mturnovers_mini is None:
#                 logger.warning(
#                     "calculating MCG LFs with mini-halo feature requires users to "
#                     "specify mturnovers_MINI!"
#                 )
#                 return None, None, None

#             mturnovers_mini = np.array(mturnovers, dtype="float32")
#             if len(mturnovers_mini) != len(redshifts):
#                 logger.warning(
#                     "mturnovers_MINI(%d) does not match the length of redshifts (%d)"
#                     % (len(mturnovers), len(redshifts))
#                 )
#                 return None, None, None

#     else:
#         mturnovers = (
#             np.zeros(len(redshifts), dtype="float32") + 10 ** astro_params.M_TURN
#         )
#         component = 1

    
#     if component == 0:
#         lfunc = np.zeros(len(redshifts) * nbins)
#         Muvfunc = np.zeros(len(redshifts) * nbins)
#         Mhfunc = np.zeros(len(redshifts) * nbins)

#         lfunc.shape = (len(redshifts), nbins)
#         Muvfunc.shape = (len(redshifts), nbins)
#         Mhfunc.shape = (len(redshifts), nbins)

#         c_Muvfunc = ffi.cast("double *", ffi.from_buffer(Muvfunc))
#         c_Mhfunc = ffi.cast("double *", ffi.from_buffer(Mhfunc))
#         c_lfunc = ffi.cast("double *", ffi.from_buffer(lfunc))

#         # Run the C code
#         errcode = lib.ComputeLF(
#             nbins,
#             user_params(),
#             cosmo_params(),
#             astro_params(),
#             flag_options(),
#             1,
#             len(redshifts),
#             ffi.cast("float *", ffi.from_buffer(redshifts)),
#             ffi.cast("float *", ffi.from_buffer(mturnovers)),
#             c_Muvfunc,
#             c_Mhfunc,
#             c_lfunc,
#         )

#         _process_exitcode(
#             errcode,
#             lib.ComputeLF,
#             (
#                 nbins,
#                 user_params,
#                 cosmo_params,
#                 astro_params,
#                 flag_options,
#                 1,
#                 len(redshifts),
#             ),
#         )

#         lfunc_MINI = np.zeros(len(redshifts) * nbins)
#         Muvfunc_MINI = np.zeros(len(redshifts) * nbins)
#         Mhfunc_MINI = np.zeros(len(redshifts) * nbins)

#         lfunc_MINI.shape = (len(redshifts), nbins)
#         Muvfunc_MINI.shape = (len(redshifts), nbins)
#         Mhfunc_MINI.shape = (len(redshifts), nbins)

#         c_Muvfunc_MINI = ffi.cast("double *", ffi.from_buffer(Muvfunc_MINI))
#         c_Mhfunc_MINI = ffi.cast("double *", ffi.from_buffer(Mhfunc_MINI))
#         c_lfunc_MINI = ffi.cast("double *", ffi.from_buffer(lfunc_MINI))

#         # Run the C code
#         errcode = lib.ComputeLF(
#             nbins,
#             user_params(),
#             cosmo_params(),
#             astro_params(),
#             flag_options(),
#             2,
#             len(redshifts),
#             ffi.cast("float *", ffi.from_buffer(redshifts)),
#             ffi.cast("float *", ffi.from_buffer(mturnovers_mini)),
#             c_Muvfunc_MINI,
#             c_Mhfunc_MINI,
#             c_lfunc_MINI,
#         )

#         _process_exitcode(
#             errcode,
#             lib.ComputeLF,
#             (
#                 nbins,
#                 user_params,
#                 cosmo_params,
#                 astro_params,
#                 flag_options,
#                 2,
#                 len(redshifts),
#             ),
#         )

#         # redo the Muv range using the faintest (most likely MINI) and the brightest (most likely massive)
#         lfunc_all = np.zeros(len(redshifts) * nbins)
#         Muvfunc_all = np.zeros(len(redshifts) * nbins)
#         Mhfunc_all = np.zeros(len(redshifts) * nbins * 2)

#         lfunc_all.shape = (len(redshifts), nbins)
#         Muvfunc_all.shape = (len(redshifts), nbins)
#         Mhfunc_all.shape = (len(redshifts), nbins, 2)
#         for iz in range(len(redshifts)):
#             Muvfunc_all[iz] = np.linspace(
#                 np.min([Muvfunc.min(), Muvfunc_MINI.min()]),
#                 np.max([Muvfunc.max(), Muvfunc_MINI.max()]),
#                 nbins,
#             )
#             lfunc_all[iz] = np.log10(
#                 10
#                 ** (
#                     interp1d(Muvfunc[iz], lfunc[iz], fill_value="extrapolate")(
#                         Muvfunc_all[iz]
#                     )
#                 )
#                 + 10
#                 ** (
#                     interp1d(
#                         Muvfunc_MINI[iz], lfunc_MINI[iz], fill_value="extrapolate"
#                     )(Muvfunc_all[iz])
#                 )
#             )
#             Mhfunc_all[iz] = np.array(
#                 [
#                     interp1d(Muvfunc[iz], Mhfunc[iz], fill_value="extrapolate")(
#                         Muvfunc_all[iz]
#                     ),
#                     interp1d(
#                         Muvfunc_MINI[iz], Mhfunc_MINI[iz], fill_value="extrapolate"
#                     )(Muvfunc_all[iz]),
#                 ],
#             ).T
#         lfunc_all[lfunc_all <= -30] = np.nan
#         return Muvfunc_all, Mhfunc_all, lfunc_all
        
#     elif component == 1:
#         lfunc = np.zeros(len(redshifts) * nbins)
#         Muvfunc = np.zeros(len(redshifts) * nbins)
#         Mhfunc = np.zeros(len(redshifts) * nbins)

#         lfunc.shape = (len(redshifts), nbins)
#         Muvfunc.shape = (len(redshifts), nbins)
#         Mhfunc.shape = (len(redshifts), nbins)

#         c_Muvfunc = ffi.cast("double *", ffi.from_buffer(Muvfunc))
#         c_Mhfunc = ffi.cast("double *", ffi.from_buffer(Mhfunc))
#         c_lfunc = ffi.cast("double *", ffi.from_buffer(lfunc))

#         #print('start c')
#         # Run the C code
#         errcode = lib.ComputeLF(
#             nbins,
#             user_params(),
#             cosmo_params(),
#             astro_params(),
#             flag_options(),
#             1,
#             len(redshifts),
#             ffi.cast("float *", ffi.from_buffer(redshifts)),
#             ffi.cast("float *", ffi.from_buffer(mturnovers)),
#             c_Muvfunc,
#             c_Mhfunc,
#             c_lfunc,
#         )
#         # print('end c')

#         _process_exitcode(
#             errcode,
#             lib.ComputeLF,
#             (
#                 nbins,
#                 user_params,
#                 cosmo_params,
#                 astro_params,
#                 flag_options,
#                 1,
#                 len(redshifts),
#             ),
#         )

#         lfunc[lfunc <= -30] = np.nan
#         return Muvfunc, Mhfunc, lfunc
    
#     elif component == 2:
#         lfunc_MINI = np.zeros(len(redshifts) * nbins)
#         Muvfunc_MINI = np.zeros(len(redshifts) * nbins)
#         Mhfunc_MINI = np.zeros(len(redshifts) * nbins)

#         lfunc_MINI.shape = (len(redshifts), nbins)
#         Muvfunc_MINI.shape = (len(redshifts), nbins)
#         Mhfunc_MINI.shape = (len(redshifts), nbins)

#         c_Muvfunc_MINI = ffi.cast("double *", ffi.from_buffer(Muvfunc_MINI))
#         c_Mhfunc_MINI = ffi.cast("double *", ffi.from_buffer(Mhfunc_MINI))
#         c_lfunc_MINI = ffi.cast("double *", ffi.from_buffer(lfunc_MINI))

#         # Run the C code
#         errcode = lib.ComputeLF(
#             nbins,
#             user_params(),
#             cosmo_params(),
#             astro_params(),
#             flag_options(),
#             2,
#             len(redshifts),
#             ffi.cast("float *", ffi.from_buffer(redshifts)),
#             ffi.cast("float *", ffi.from_buffer(mturnovers_mini)),
#             c_Muvfunc_MINI,
#             c_Mhfunc_MINI,
#             c_lfunc_MINI,
#         )

#         _process_exitcode(
#             errcode,
#             lib.ComputeLF,
#             (
#                 nbins,
#                 user_params,
#                 cosmo_params,
#                 astro_params,
#                 flag_options,
#                 2,
#                 len(redshifts),
#             ),
#         )

#         lfunc_MINI[lfunc_MINI <= -30] = np.nan
#         return Muvfunc_MINI, Mhfunc_MINI, lfunc_MINI
#     else:
#         logger.warning("What is component %d ?" % component)
#         return None, None, None


def _init_photon_conservation_correction(
    *, user_params=None, cosmo_params=None, astro_params=None, flag_options=None
):
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(
        flag_options, USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
    )

    return lib.InitialisePhotonCons(
        user_params(), cosmo_params(), astro_params(), flag_options()
    )


def _calibrate_photon_conservation_correction(
    *, redshifts_estimate, nf_estimate, NSpline
):
    # Convert the data to the right type
    redshifts_estimate = np.array(redshifts_estimate, dtype="float64")
    nf_estimate = np.array(nf_estimate, dtype="float64")

    z = ffi.cast("double *", ffi.from_buffer(redshifts_estimate))
    xHI = ffi.cast("double *", ffi.from_buffer(nf_estimate))

    logger.debug(f"PhotonCons nf estimates: {nf_estimate}")
    return lib.PhotonCons_Calibration(z, xHI, NSpline)


def _calc_zstart_photon_cons():
    # Run the C code
    return _call_c_simple(lib.ComputeZstart_PhotonCons)


def _get_photon_nonconservation_data():
    """
    Access C global data representing the photon-nonconservation corrections.

    .. note::  if not using ``PHOTON_CONS`` (in :class:`~FlagOptions`), *or* if the
               initialisation for photon conservation has not been performed yet, this
               will return None.

    Returns
    -------
    dict :
      z_analytic: array of redshifts defining the analytic ionized fraction
      Q_analytic: array of analytic  ionized fractions corresponding to `z_analytic`
      z_calibration: array of redshifts defining the ionized fraction from 21cmFAST without
      recombinations
      nf_calibration: array of calibration ionized fractions corresponding to `z_calibration`
      delta_z_photon_cons: the change in redshift required to calibrate 21cmFAST, as a function
      of z_calibration
      nf_photoncons: the neutral fraction as a function of redshift
    """
    # Check if photon conservation has been initialised at all
    if not lib.photon_cons_allocated:
        return None

    arbitrary_large_size = 2000

    data = np.zeros((6, arbitrary_large_size))

    IntVal1 = np.array(np.zeros(1), dtype="int32")
    IntVal2 = np.array(np.zeros(1), dtype="int32")
    IntVal3 = np.array(np.zeros(1), dtype="int32")

    c_z_at_Q = ffi.cast("double *", ffi.from_buffer(data[0]))
    c_Qval = ffi.cast("double *", ffi.from_buffer(data[1]))
    c_z_cal = ffi.cast("double *", ffi.from_buffer(data[2]))
    c_nf_cal = ffi.cast("double *", ffi.from_buffer(data[3]))
    c_PC_nf = ffi.cast("double *", ffi.from_buffer(data[4]))
    c_PC_deltaz = ffi.cast("double *", ffi.from_buffer(data[5]))

    c_int_NQ = ffi.cast("int *", ffi.from_buffer(IntVal1))
    c_int_NC = ffi.cast("int *", ffi.from_buffer(IntVal2))
    c_int_NP = ffi.cast("int *", ffi.from_buffer(IntVal3))

    # Run the C code
    errcode = lib.ObtainPhotonConsData(
        c_z_at_Q,
        c_Qval,
        c_int_NQ,
        c_z_cal,
        c_nf_cal,
        c_int_NC,
        c_PC_nf,
        c_PC_deltaz,
        c_int_NP,
    )

    _process_exitcode(errcode, lib.ObtainPhotonConsData, ())

    ArrayIndices = [
        IntVal1[0],
        IntVal1[0],
        IntVal2[0],
        IntVal2[0],
        IntVal3[0],
        IntVal3[0],
    ]

    data_list = [
        "z_analytic",
        "Q_analytic",
        "z_calibration",
        "nf_calibration",
        "nf_photoncons",
        "delta_z_photon_cons",
    ]

    return {name: d[:index] for name, d, index in zip(data_list, data, ArrayIndices)}


def initial_conditions(
    *,
    user_params=None,
    cosmo_params=None,
    # !!! SLTK: added astro_params and flag_options
    astro_params=None,
    flag_options=None,
    random_seed=None,
    regenerate=None,
    write=None,
    direc=None,
    hooks: Optional[Dict[Callable, Dict[str, Any]]] = None,
    **global_kwargs,
) -> InitialConditions:
    r"""
    Compute initial conditions.

    Parameters
    ----------
    user_params : :class:`~UserParams` instance, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams` instance, optional
        Defines the cosmological parameters used to compute initial conditions.
    regenerate : bool, optional
        Whether to force regeneration of data, even if matching cached data is found.
        This is applied recursively to any potential sub-calculations. It is ignored in
        the case of dependent data only if that data is explicitly passed to the function.
    write : bool, optional
        Whether to write results to file (i.e. cache). This is recursively applied to
        any potential sub-calculations.
    hooks
        Any extra functions to apply to the output object. This should be a dictionary
        where the keys are the functions, and the values are themselves dictionaries of
        parameters to pass to the function. The function signature should be
        ``(output, **params)``, where the ``output`` is the output object.
    direc : str, optional
        The directory in which to search for the boxes and write them. By default, this
        is the directory given by ``boxdir`` in the configuration file,
        ``~/.21cmfast/config.yml``. This is recursively applied to any potential
        sub-calculations.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~InitialConditions`
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        # !!! SLTK: added astro_params and flag_options
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)
        flag_options = FlagOptions(
            flag_options, USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
        )

        # Initialize memory for the boxes that will be returned.
        # !!! SLTK: add astro_params and flag_options
        boxes = InitialConditions(
            user_params=user_params, cosmo_params=cosmo_params, 
            astro_params=astro_params, flag_options=flag_options,
            random_seed=random_seed
        )

        # Construct FFTW wisdoms. Only if required
        construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

        # First check whether the boxes already exist.
        if not regenerate:
            try:
                boxes.read(direc)
                logger.info(
                    f"Existing init_boxes found and read in (seed={boxes.random_seed})."
                )
                return boxes
            except OSError:
                pass

        return boxes.compute(hooks=hooks)


def perturb_field(
    *,
    redshift,
    init_boxes=None,
    user_params=None,
    cosmo_params=None,
    # !!! SLTK: added astro_params and flag_options
    astro_params=None,
    flag_options=None,
    random_seed=None,
    regenerate=None,
    write=None,
    direc=None,
    hooks: Optional[Dict[Callable, Dict[str, Any]]] = None,
    **global_kwargs,
) -> PerturbedField:
    r"""
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.
    init_boxes : :class:`~InitialConditions`, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will
        be generated. If given,
        the user and cosmo params will be set from this object.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~PerturbedField`

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    The simplest method is just to give a redshift::

    >>> field = perturb_field(7.0)
    >>> print(field.density)

    Doing so will internally call the :func:`~initial_conditions` function. If initial conditions
    have already been
    calculated, this can be avoided by passing them:

    >>> init_boxes = initial_conditions()
    >>> field7 = perturb_field(7.0, init_boxes)
    >>> field8 = perturb_field(8.0, init_boxes)

    The user and cosmo parameter structures are by default inferred from the ``init_boxes``,
    so that the following is
    consistent::

    >>> init_boxes = initial_conditions(user_params= UserParams(HII_DIM=1000))
    >>> field7 = perturb_field(7.0, init_boxes)

    If ``init_boxes`` is not passed, then these parameters can be directly passed::

    >>> field7 = perturb_field(7.0, user_params=UserParams(HII_DIM=1000))

    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        _verify_types(init_boxes=init_boxes)

        # Configure and check input/output parameters/structs
        random_seed, user_params, cosmo_params, \
        astro_params, flag_options = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                # !!! SLTK: added astro_params and flag_options
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_boxes,
        )

        # Verify input parameter structs (need to do this after configure_inputs).
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        # !!! SLTK: added astro_params and flag_options
        flag_options = FlagOptions(flag_options,
                                   USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES)
        astro_params = AstroParams(astro_params, 
                                   INHOMO_RECO=flag_options.INHOMO_RECO)

        # Initialize perturbed boxes.
        fields = PerturbedField(
            redshift=redshift,
            user_params=user_params,
            cosmo_params=cosmo_params,
            # !!! SLTK: added astro_params and flag_options
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
        )

        # Check whether the boxes already exist
        if not regenerate:
            try:
                fields.read(direc)
                logger.info(
                    f"Existing z={redshift} perturb_field boxes found and read in "
                    f"(seed={fields.random_seed})."
                )
                return fields
            except OSError:
                pass

        # Construct FFTW wisdoms. Only if required
        construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

        # Make sure we've got computed init boxes.
        if init_boxes is None or not init_boxes.is_computed:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                # !!! SLTK: added astro_params and flag_options
                astro_params = astro_params,
                flag_options = flag_options,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update fields to have the same seed as init_boxes
            fields._random_seed = init_boxes.random_seed

        # Run the C Code
        return fields.compute(ics=init_boxes, hooks=hooks)


def determine_halo_list(
    *,
    redshift,
    init_boxes=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    random_seed=None,
    regenerate=None,
    write=None,
    direc=None,
    hooks=None,
    **global_kwargs,
):
    r"""
    Find a halo list, given a redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to determine the halo list.
    init_boxes : :class:`~InitialConditions`, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will
        be generated. If given,
        the user and cosmo params will be set from this object.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~HaloField`

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    Fill this in once finalised

    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        _verify_types(init_boxes=init_boxes)

        # Configure and check input/output parameters/structs
        (
            random_seed,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
        ) = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_boxes,
        )

        # Verify input parameter structs (need to do this after configure_inputs).
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        # !!! SLTK: changed order (astro-flag) but should be irrelevant 
        astro_params = AstroParams(astro_params,
                                   INHOMO_RECO=flag_options.INHOMO_RECO)
        flag_options = FlagOptions(
            flag_options, USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
        )

        if user_params.HMF != 1:
            raise ValueError("USE_HALO_FIELD is only valid for HMF = 1")

        # Initialize halo list boxes.
        fields = HaloField(
            redshift=redshift,
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
        )
        # Check whether the boxes already exist
        if not regenerate:
            try:
                fields.read(direc)
                logger.info(
                    f"Existing z={redshift} determine_halo_list boxes found and read in "
                    f"(seed={fields.random_seed})."
                )
                return fields
            except OSError:
                pass

        # Construct FFTW wisdoms. Only if required
        construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

        # Make sure we've got computed init boxes.
        if init_boxes is None or not init_boxes.is_computed:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                # !!! SLTK: added astro_params and flag_options
                astro_params=astro_params,
                flag_options=flag_options,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update fields to have the same seed as init_boxes
            fields._random_seed = init_boxes.random_seed

        # Run the C Code
        return fields.compute(ics=init_boxes, hooks=hooks)


def perturb_halo_list(
    *,
    redshift,
    init_boxes=None,
    halo_field=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    random_seed=None,
    regenerate=None,
    write=None,
    direc=None,
    hooks=None,
    **global_kwargs,
):
    r"""
    Given a halo list, perturb the halos for a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to determine the halo list.
    init_boxes : :class:`~InitialConditions`, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will
        be generated. If given,
        the user and cosmo params will be set from this object.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~PerturbHaloField`

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    Fill this in once finalised

    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        _verify_types(
            init_boxes=init_boxes,
            halo_field=halo_field,
        )

        # Configure and check input/output parameters/structs
        (
            random_seed,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
        ) = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_boxes,
            halo_field,
        )
        redshift = configure_redshift(redshift, halo_field)

        # Verify input parameter structs (need to do this after configure_inputs).
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        # !!! SLTK: inverted order (astro-flag) to be consistent, but it should be the same
        astro_params = AstroParams(astro_params, 
                                   INHOMO_RECO=flag_options.INHOMO_RECO)
        flag_options = FlagOptions(
            flag_options, 
            USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
        )

        if user_params.HMF != 1:
            raise ValueError("USE_HALO_FIELD is only valid for HMF = 1")

        # Initialize halo list boxes.
        fields = PerturbHaloField(
            redshift=redshift,
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
        )

        # Check whether the boxes already exist
        if not regenerate:
            try:
                fields.read(direc)
                logger.info(
                    "Existing z=%s perturb_halo_list boxes found and read in (seed=%s)."
                    % (redshift, fields.random_seed)
                )
                return fields
            except OSError:
                pass

        # Make sure we've got computed init boxes.
        if init_boxes is None or not init_boxes.is_computed:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                # !!! SLTK: added astro_params and flag_options
                astro_params=astro_params,                
                flag_options=flag_options,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update fields to have the same seed as init_boxes
            fields._random_seed = init_boxes.random_seed

        # Dynamically produce the halo list.
        if halo_field is None or not halo_field.is_computed:
            halo_field = determine_halo_list(
                init_boxes=init_boxes,
                # NOTE: this is required, rather than using cosmo_ and user_,
                # since init may have a set seed.
                redshift=redshift,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
            )

        # Run the C Code
        return fields.compute(ics=init_boxes, halo_field=halo_field, hooks=hooks)


def ionize_box(
    *,
    astro_params=None,
    flag_options=None,
    redshift=None,
    perturbed_field=None,
    previous_perturbed_field=None,
    previous_ionize_box=None,
    spin_temp=None,
    pt_halos=None,
    init_boxes=None,
    cosmo_params=None,
    user_params=None,
    regenerate=None,
    write=None,
    direc=None,
    random_seed=None,
    cleanup=True,
    hooks=None,
    **global_kwargs,
) -> IonizedBox:
    r"""
    Compute an ionized box at a given redshift.

    This function has various options for how the evolution of the ionization is computed (if at
    all). See the Notes below for details.

    Parameters
    ----------
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.
    flag_options: :class:`~FlagOptions` instance, optional
        Some options passed to the reionization routine.
    redshift : float, optional
        The redshift at which to compute the ionized box. If `perturbed_field` is given,
        its inherent redshift
        will take precedence over this argument. If not, this argument is mandatory.
    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated,
        either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`.
    previous_perturbed_field : :class:`~PerturbField`, optional
        An perturbed field at higher redshift. This is only used if mini_halo is included.
    init_boxes : :class:`~InitialConditions` , optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used
        to generate the perturbed field, otherwise initial conditions will be generated on the fly.
        If given, the user and cosmo params will be set from this object.
    previous_ionize_box: :class:`IonizedBox` or None
        An ionized box at higher redshift. This is only used if `INHOMO_RECO` and/or `do_spin_temp`
        are true. If either of these are true, and this is not given, then it will be assumed that
        this is the "first box", i.e. that it can be populated accurately without knowing source
        statistics.
    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `do_spin_temp` is True. If None, will try to read
        in a spin temp box at the current redshift, and failing that will try to automatically
        create one, using the previous ionized box redshift as the previous spin temperature
        redshift.
    pt_halos: :class:`~PerturbHaloField` or None, optional
        If passed, this contains all the dark matter haloes obtained if using the USE_HALO_FIELD.
        This is a list of halo masses and coords for the dark matter haloes.
        If not passed, it will try and automatically create them using the available initial conditions.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning. Typically,
        if `spin_temperature` is called directly, you will want this to be true, as if the next box
        to be calculate has different shape, errors will occur if memory is not cleaned. However,
        it can be useful to set it to False if scrolling through parameters for the same box shape.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~IonizedBox` :
        An object containing the ionized box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----
    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current
    one. This function provides several options for doing so. First, if neither the spin
    temperature field, nor inhomogeneous recombinations (specified in flag options) are used, no
    evolution needs to be done. Otherwise, either (in order of precedence)

    1. a specific previous :class`~IonizedBox` object is provided, which will be used directly,
    2. a previous redshift is provided, for which a cached field on disk will be sought,
    3. a step factor is provided which recursively steps through redshift, calculating previous
       fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    4. the function is instructed to treat the current field as being an initial "high-redshift"
       field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that
              redshift, the previous ionization field will be evaluated based on `z_step_factor`.

    Examples
    --------
    By default, no spin temperature is used, and neither are inhomogeneous recombinations,
    so that no evolution is required, thus the following will compute a coeval ionization box:

    >>> xHI = ionize_box(redshift=7.0)

    However, if either of those options are true, then a full evolution will be required:

    >>> xHI = ionize_box(redshift=7.0, flag_options=FlagOptions(INHOMO_RECO=True,USE_TS_FLUCT=True))

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global
    parameter), in logarithmic steps of `ZPRIME_STEP_FACTOR`. To change these:

    >>> xHI = ionize_box(redshift=7.0, zprime_step_factor=1.2, z_heat_max=15.0,
    >>>                  flag_options={"USE_TS_FLUCT":True})

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk
    cache, or evaluated:

    >>> ts_box = ionize_box(redshift=7.0, previous_ionize_box=8.0, flag_options={
    >>>                     "USE_TS_FLUCT":True})

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate
    prior boxes based on `ZPRIME_STEP_FACTOR`. Alternatively, one can pass a previous
    :class:`~IonizedBox`:

    >>> xHI_0 = ionize_box(redshift=8.0, flag_options={"USE_TS_FLUCT":True})
    >>> xHI = ionize_box(redshift=7.0, previous_ionize_box=xHI_0)

    Again, the first line here will implicitly use ``ZPRIME_STEP_FACTOR`` to evolve the field from
    ``Z_HEAT_MAX``. Note that in the second line, all of the input parameters are taken directly from
    `xHI_0` so that they are consistent, and we need not specify the ``flag_options``.

    As the function recursively evaluates previous redshift, the previous spin temperature fields
    will also be consistently recursively evaluated. Only the final ionized box will actually be
    returned and kept in memory, however intervening results will by default be cached on disk.
    One can also pass an explicit spin temperature object:

    >>> ts = spin_temperature(redshift=7.0)
    >>> xHI = ionize_box(redshift=7.0, spin_temp=ts)

    If automatic recursion is used, then it is done in such a way that no large boxes are kept
    around in memory for longer than they need to be (only two at a time are required).
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        _verify_types(
            init_boxes=init_boxes,
            perturbed_field=perturbed_field,
            previous_perturbed_field=previous_perturbed_field,
            previous_ionize_box=previous_ionize_box,
            spin_temp=spin_temp,
            pt_halos=pt_halos,
        )

        # Configure and check input/output parameters/structs
        (
            random_seed,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
        ) = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_boxes,
            spin_temp,
            init_boxes,
            perturbed_field,
            previous_perturbed_field,
            previous_ionize_box,
            pt_halos,
        )
        redshift = configure_redshift(
            redshift,
            spin_temp,
            perturbed_field,
            pt_halos,
        )

        # Verify input structs
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        # !!! SLTK: changed oreder (astro-flag) to be coherent, but should be the same
        astro_params = AstroParams(astro_params, 
                                   INHOMO_RECO=flag_options.INHOMO_RECO)
        flag_options = FlagOptions(
            flag_options, 
            USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
        )

        if spin_temp is not None and not flag_options.USE_TS_FLUCT:
            logger.warning(
                "Changing flag_options.USE_TS_FLUCT to True since spin_temp was passed."
            )
            flag_options.USE_TS_FLUCT = True

        # Get the previous redshift
        if previous_ionize_box is not None and previous_ionize_box.is_computed:
            prev_z = previous_ionize_box.redshift

            # Ensure the previous ionized box has a higher redshift than this one.
            if prev_z <= redshift:
                raise ValueError(
                    "Previous ionized box must have a higher redshift than that being evaluated."
                )
        elif flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT:
            prev_z = (1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1
            # if the previous box is before our starting point, we set it to zero,
            # which is what the C-code expects for an "initial" box
            if prev_z > global_params.Z_HEAT_MAX:
                prev_z = 0
        else:
            prev_z = 0

        box = IonizedBox(
            first_box=((1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1)
            > global_params.Z_HEAT_MAX
            and (
                not isinstance(previous_ionize_box, IonizedBox)
                or not previous_ionize_box.is_computed
            ),
            user_params=user_params,
            cosmo_params=cosmo_params,
            redshift=redshift,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
            prev_ionize_redshift=prev_z,
        )

        # Construct FFTW wisdoms. Only if required
        construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

        # Check whether the boxes already exist
        if not regenerate:
            try:
                box.read(direc)
                logger.info(
                    "Existing z=%s ionized boxes found and read in (seed=%s)."
                    % (redshift, box.random_seed)
                )
                return box
            except OSError:
                pass

        # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
        # ------------------------------------------------------------------------

        # Get init_box required.
        if init_boxes is None or not init_boxes.is_computed:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                # !!! SLTK: added astro_params and flag_options
                astro_params=astro_params,
                flag_options=flag_options,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update random seed
            box._random_seed = init_boxes.random_seed

        # Get appropriate previous ionization box
        if previous_ionize_box is None or not previous_ionize_box.is_computed:
            # If we are beyond Z_HEAT_MAX, just make an empty box
            if prev_z == 0:
                previous_ionize_box = IonizedBox(
                    redshift=0, flag_options=flag_options, initial=True
                )

            # Otherwise recursively create new previous box.
            else:
                previous_ionize_box = ionize_box(
                    astro_params=astro_params,
                    flag_options=flag_options,
                    redshift=prev_z,
                    init_boxes=init_boxes,
                    regenerate=regenerate,
                    hooks=hooks,
                    direc=direc,
                    cleanup=False,  # We *know* we're going to need the memory again.
                )

        # Dynamically produce the perturbed field.
        if perturbed_field is None or not perturbed_field.is_computed:
            perturbed_field = perturb_field(
                init_boxes=init_boxes,
                # NOTE: this is required, rather than using cosmo_ and user_,
                # since init may have a set seed.
                redshift=redshift,
                regenerate=regenerate,
                # !!! SLTK: added astro_params and flag_options specified to be coherent
                astro_params=astro_params,
                flag_options=flag_options,
                hooks=hooks,
                direc=direc,
            )

        if previous_perturbed_field is None or not previous_perturbed_field.is_computed:
            # If we are beyond Z_HEAT_MAX, just make an empty box
            if not prev_z or redshift > global_params.Z_HEAT_MAX: # JordanFlitter: added the second condition (seems to be important at high redshifts from memory considerations, I don't understand why):
                previous_perturbed_field = PerturbedField(
                    redshift=0, user_params=user_params, initial=True
                )
            else:
                previous_perturbed_field = perturb_field(
                    init_boxes=init_boxes,
                    redshift=prev_z,
                    regenerate=regenerate,
                    # !!! SLTK: added astro_params and flag_options specified to be coherent
                    astro_params=astro_params,
                    flag_options=flag_options,
                    hooks=hooks,
                    direc=direc,
                )

        # Dynamically produce the halo field.
        if not flag_options.USE_HALO_FIELD:
            # Construct an empty halo field to pass in to the function.
            pt_halos = PerturbHaloField(redshift=0, dummy=True)
        elif pt_halos is None or not pt_halos.is_computed:
            pt_halos = perturb_halo_list(
                redshift=redshift,
                init_boxes=init_boxes,
                halo_field=determine_halo_list(
                    redshift=redshift,
                    init_boxes=init_boxes,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    regenerate=regenerate,
                    hooks=hooks,
                    direc=direc,
                ),
                astro_params=astro_params,
                flag_options=flag_options,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
            )

        # Set empty spin temp box if necessary.
        if not flag_options.USE_TS_FLUCT:
            spin_temp = TsBox(redshift=0, dummy=True)
        elif spin_temp is None:
            spin_temp = spin_temperature(
                perturbed_field=perturbed_field,
                flag_options=flag_options,
                init_boxes=init_boxes,
                direc=direc,
                hooks=hooks,
                regenerate=regenerate,
                cleanup=cleanup,
            )

        # Run the C Code
        return box.compute(
            perturbed_field=perturbed_field,
            prev_perturbed_field=previous_perturbed_field,
            prev_ionize_box=previous_ionize_box,
            spin_temp=spin_temp,
            pt_halos=pt_halos,
            ics=init_boxes,
            hooks=hooks,
        )


def spin_temperature(
    *,
    astro_params=None,
    flag_options=None,
    redshift=None,
    perturbed_field=None,
    previous_spin_temp=None,
    init_boxes=None,
    next_redshift_input=None, # JordanFlitter: added next_redshift_input
    cosmo_params=None,
    user_params=None,
    regenerate=None,
    write=None,
    direc=None,
    random_seed=None,
    cleanup=True,
    hooks=None,
    **global_kwargs,
) -> TsBox:
    r"""
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    astro_params : :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    redshift : float, optional
        The redshift at which to compute the ionized box. If not given, the redshift from
        `perturbed_field` will be used. Either `redshift`, `perturbed_field`, or
        `previous_spin_temp` must be given. See notes on `perturbed_field` for how it affects the
        given redshift if both are given.
    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated,
        either `init_boxes` and `redshift` must be given, or `user_params`, `cosmo_params` and
        `redshift`. By default, this will be generated at the same redshift as the spin temperature
        box. The redshift of perturb field is allowed to be different than `redshift`. If so, it
        will be interpolated to the correct redshift, which can provide a speedup compared to
        actually computing it at the desired redshift.
    previous_spin_temp : :class:`TsBox` or None
        The previous spin temperature box.
    init_boxes : :class:`~InitialConditions`, optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used
        to generate the perturbed field, otherwise initial conditions will be generated on the fly.
        If given, the user and cosmo params will be set from this object.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. However, it can be useful to set it to False if
        scrolling through parameters for the same box shape.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----
    Typically, the spin temperature field at any redshift is dependent on the evolution of spin
    temperature up until that redshift, which necessitates providing a previous spin temperature
    field to define the current one. This function provides several options for doing so. Either
    (in order of precedence):

    1. a specific previous spin temperature object is provided, which will be used directly,
    2. a previous redshift is provided, for which a cached field on disk will be sought,
    3. a step factor is provided which recursively steps through redshift, calculating previous
       fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    4. the function is instructed to treat the current field as being an initial "high-redshift"
       field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that
              redshift, the previous spin temperature field will be evaluated based on
              ``z_step_factor``.

    Examples
    --------
    To calculate and return a fully evolved spin temperature field at a given redshift (with
    default input parameters), simply use:

    >>> ts_box = spin_temperature(redshift=7.0)

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global
    parameter), in logarithmic steps of `z_step_factor`. Thus to change these:

    >>> ts_box = spin_temperature(redshift=7.0, zprime_step_factor=1.2, z_heat_max=15.0)

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk
    cache, or evaluated:

    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=8.0)

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate
    prior boxes based on the ``z_step_factor``. Alternatively, one can pass a previous spin
    temperature box:

    >>> ts_box1 = spin_temperature(redshift=8.0)
    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=ts_box1)

    Again, the first line here will implicitly use ``z_step_factor`` to evolve the field from
    around ``Z_HEAT_MAX``. Note that in the second line, all of the input parameters are taken
    directly from `ts_box1` so that they are consistent. Finally, one can force the function to
    evaluate the current redshift as if it was beyond ``Z_HEAT_MAX`` so that it depends only on
    itself:

    >>> ts_box = spin_temperature(redshift=7.0, zprime_step_factor=None)

    This is usually a bad idea, and will give a warning, but it is possible.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        _verify_types(
            init_boxes=init_boxes,
            perturbed_field=perturbed_field,
            previous_spin_temp=previous_spin_temp,
        )

        # Configure and check input/output parameters/structs
        (
            random_seed,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
        ) = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_boxes,
            previous_spin_temp,
            init_boxes,
            perturbed_field,
        )

        # Try to determine redshift from other inputs, if required.
        # Note that perturb_field does not need to match redshift here.
        if redshift is None:
            if perturbed_field is not None:
                redshift = perturbed_field.redshift
            elif previous_spin_temp is not None:
                redshift = (
                    previous_spin_temp.redshift + 1
                ) / global_params.ZPRIME_STEP_FACTOR - 1
            else:
                raise ValueError(
                    "Either the redshift, perturbed_field or previous_spin_temp must be given."
                )
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        # !!! SLTK: inverted order (astro-flag) but should be the same
        astro_params = AstroParams(astro_params,    
                                   INHOMO_RECO=flag_options.INHOMO_RECO)
        flag_options = FlagOptions(
            flag_options, 
            USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
        )

        # Explicitly set this flag to True, though it shouldn't be required!
        flag_options.update(USE_TS_FLUCT=True)

        # Get the previous redshift
        if previous_spin_temp is not None:
            prev_z = previous_spin_temp.redshift
        else:
            prev_z = (1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1
            prev_z = min(global_params.Z_HEAT_MAX, prev_z)

        # Ensure the previous spin temperature has a higher redshift than this one.
        # TODO: there's a bit of a weird thing here where prev_z may be set to z_HEAT_MAX
        #       but `redshift` may be higher than Z_HEAT_MAX. Might need to fix this later.
        if prev_z <= redshift and prev_z < global_params.Z_HEAT_MAX:
            raise ValueError(
                "Previous spin temperature box must have a higher redshift than "
                "that being evaluated."
            )
        # JordanFlitter: added the following logic for the new boolean flag first_box_flag
        # JordanFlitterTODO: the whole first_box logic is outdated an no longer found in the public code of 21cmFAST.
        #                    In fact, the true "first_box" is when redshift > Z_HEAT_MAX (or Z_HIGH_MAX, depending on
        #                    whether or not START_AT_RECOMBINATION is on or off)
        if not user_params.START_AT_RECOMBINATION:
            first_box_flag = ((1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1) > global_params.Z_HEAT_MAX
        else:
            if (user_params.Z_HIGH_MAX > global_params.Z2_VALUE):
                if user_params.USE_TCA_COMPTON:
                    first_box_flag = (redshift + global_params.DELTA_Z1 > user_params.Z_HIGH_MAX)
                else:
                    first_box_flag = (redshift + global_params.DELTA_Z2 > user_params.Z_HIGH_MAX)
            elif (user_params.Z_HIGH_MAX > global_params.Z1_VALUE):
                first_box_flag = (redshift + global_params.DELTA_Z1 > user_params.Z_HIGH_MAX)
            else:
                first_box_flag = (redshift + global_params.DELTA_Z > user_params.Z_HIGH_MAX)
        # TODO: why is the below checking for IonizedBox??
        box = TsBox(
            first_box=first_box_flag
            and (
                not isinstance(previous_spin_temp, IonizedBox)
                or not previous_spin_temp.is_computed
            ),
            user_params=user_params,
            cosmo_params=cosmo_params,
            redshift=redshift,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
            prev_spin_redshift=prev_z,
            perturbed_field_redshift=perturbed_field.redshift
            if (perturbed_field is not None and perturbed_field.is_computed)
            else redshift,
            next_redshift_input=next_redshift_input if (next_redshift_input is not None) else global_params.Z_HEAT_MAX, # JordanFlitter: added next_redshift_input
        )

        # Construct FFTW wisdoms. Only if required
        construct_fftw_wisdoms(user_params=user_params, cosmo_params=cosmo_params)

        # Check whether the boxes already exist on disk.
        if not regenerate:
            try:
                box.read(direc)
                logger.info(
                    "Existing z=%s spin_temp boxes found and read in (seed=%s)."
                    % (redshift, box.random_seed)
                )
                return box
            except OSError:
                pass

        # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
        # ------------------------------------------------------------------------
        # Dynamically produce the initial conditions.
        if init_boxes is None or not init_boxes.is_computed:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                # !!! SLTK: added astro_params and flag_options
                astro_params=astro_params,
                flag_options=flag_options,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update random seed
            box._random_seed = init_boxes.random_seed

        # Create appropriate previous_spin_temp
        if not isinstance(previous_spin_temp, TsBox):
            if prev_z >= global_params.Z_HEAT_MAX:
                previous_spin_temp = TsBox(
                    redshift=global_params.Z_HEAT_MAX,
                    user_params=init_boxes.user_params,
                    cosmo_params=init_boxes.cosmo_params,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    initial=True,
                )
            else:
                previous_spin_temp = spin_temperature(
                    init_boxes=init_boxes,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    redshift=prev_z,
                    regenerate=regenerate,
                    hooks=hooks,
                    direc=direc,
                    cleanup=False,  # we know we'll need the memory again
                )

        # Dynamically produce the perturbed field.
        if perturbed_field is None or not perturbed_field.is_computed:
            perturbed_field = perturb_field(
                redshift=redshift,
                init_boxes=init_boxes,
                regenerate=regenerate,
                # !!! SLTK: added astro_params and flag_options specified to be coherent
                astro_params=astro_params,
                flag_options=flag_options,
                hooks=hooks,
                direc=direc,
            )

        # Run the C Code
        return box.compute(
            cleanup=cleanup,
            perturbed_field=perturbed_field,
            prev_spin_temp=previous_spin_temp,
            ics=init_boxes,
            hooks=hooks,
        )


def brightness_temperature(
    *,
    ionized_box,
    perturbed_field,
    spin_temp=None,
    write=None,
    regenerate=None,
    direc=None,
    hooks=None,
    **global_kwargs,
) -> BrightnessTemp:
    r"""
    Compute a coeval brightness temperature box.

    Parameters
    ----------
    ionized_box: :class:`IonizedBox`
        A pre-computed ionized box.
    perturbed_field: :class:`PerturbedField`
        A pre-computed perturbed field at the same redshift as `ionized_box`.
    spin_temp: :class:`TsBox`, optional
        A pre-computed spin temperature, at the same redshift as the other boxes.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`BrightnessTemp` instance.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        _verify_types(
            perturbed_field=perturbed_field,
            spin_temp=spin_temp,
            ionized_box=ionized_box,
        )

        # don't ignore redshift here
        _check_compatible_inputs(ionized_box, perturbed_field, spin_temp, ignore=[])

        # ensure ionized_box and perturbed_field aren't None, as we don't do
        # any dynamic calculations here.
        if ionized_box is None or perturbed_field is None:
            raise ValueError("both ionized_box and perturbed_field must be specified.")

        if spin_temp is None:
            if ionized_box.flag_options.USE_TS_FLUCT:
                raise ValueError(
                    "You have USE_TS_FLUCT=True, but have not provided a spin_temp!"
                )

            # Make an unused dummy box.
            spin_temp = TsBox(redshift=0, dummy=True)

        box = BrightnessTemp(
            user_params=ionized_box.user_params,
            cosmo_params=ionized_box.cosmo_params,
            astro_params=ionized_box.astro_params,
            flag_options=ionized_box.flag_options,
            redshift=ionized_box.redshift,
            random_seed=ionized_box.random_seed,
        )

        # Construct FFTW wisdoms. Only if required
        construct_fftw_wisdoms(
            user_params=ionized_box.user_params, cosmo_params=ionized_box.cosmo_params
        )

        # Check whether the boxes already exist on disk.
        if not regenerate:
            try:
                box.read(direc)
                logger.info(
                    f"Existing brightness_temp box found and read in (seed={box.random_seed})."
                )
                return box
            except OSError:
                pass

        return box.compute(
            spin_temp=spin_temp,
            ionized_box=ionized_box,
            perturbed_field=perturbed_field,
            hooks=hooks,
        )


def _logscroll_redshifts(min_redshift, z_step_factor, zmax):
    redshifts = [min_redshift]
    while redshifts[-1] < zmax:
        redshifts.append((redshifts[-1] + 1.0) * z_step_factor - 1.0)
    return redshifts[::-1]

# JordanFlitter: a similar function to above, but with linear spaceing
def _linscroll_redshifts(min_redshift, delta_z, zmax):
    redshifts = [min_redshift]
    while redshifts[-1] < zmax:
        redshifts.append(redshifts[-1] + delta_z)
    return redshifts[::-1]


def run_coeval(
    *,
    redshift=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    regenerate=None,
    write=None,
    direc=None,
    init_box=None,
    perturb=None,
    use_interp_perturb_field=False,
    pt_halos=None,
    random_seed=None,
    cleanup=True,
    hooks=None,
    always_purge: bool = False,
    **global_kwargs,
):
    r"""
    Evaluate a coeval ionized box at a given redshift, or multiple redshifts.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a
    given set of redshift. It self-consistently deals with situations in which the field needs to be
    evolved, and does this with the highest memory-efficiency, only returning the desired redshift.
    All other calculations are by default stored in the on-disk cache so they can be re-used at a
    later time.

    .. note:: User-supplied redshift are *not* used as previous redshift in any scrolling,
              so that pristine log-sampling can be maintained.

    Parameters
    ----------
    redshift: array_like
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    user_params : :class:`~inputs.UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~inputs.CosmoParams` , optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params : :class:`~inputs.AstroParams` , optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~inputs.FlagOptions` , optional
        Some options passed to the reionization routine.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not
        be re-calculated.
    perturb : list of :class:`~PerturbedField`, optional
        If given, must be compatible with init_box. It will merely negate the necessity
        of re-calculating the perturb fields.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone,
        to determine all spin temperature fields. If so, this field is interpolated in
        the underlying C-code to the correct redshift. This is less accurate (and no more
        efficient), but provides compatibility with older versions of 21cmFAST.
    pt_halos : bool, optional
        If given, must be compatible with init_box. It will merely negate the necessity
        of re-calculating the perturbed halo lists.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    coevals : :class:`~py21cmfast.outputs.Coeval`
        The full data for the Coeval class, with init boxes, perturbed fields, ionized boxes,
        brightness temperature, and potential data from the conservation of photons. If a
        single redshift was specified, it will return such a class. If multiple redshifts
        were passed, it will return a list of such classes.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.
    """
    with global_params.use(**global_kwargs):
        if redshift is None and perturb is None:
            raise ValueError("Either redshift or perturb must be given")

        direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

        singleton = False
        # Ensure perturb is a list of boxes, not just one.
        if perturb is None:
            perturb = []
        elif not hasattr(perturb, "__len__"):
            perturb = [perturb]
            singleton = True

        # Ensure perturbed halo field is a list of boxes, not just one.
        if flag_options is None or pt_halos is None:
            pt_halos = []

        elif (
            flag_options["USE_HALO_FIELD"]
            if isinstance(flag_options, dict)
            else flag_options.USE_HALO_FIELD
        ):
            pt_halos = [pt_halos] if not hasattr(pt_halos, "__len__") else []
        else:
            pt_halos = []
        # !!! SLTK: added astro_params and flag_options
        random_seed, user_params, cosmo_params 
        astro_params, flag_options = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_box,
            *perturb,
            *pt_halos,
        )

        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        # !!! SLTK: change order (astro-flag) for consistency but should be the same
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)
        flag_options = FlagOptions(
            flag_options, USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
        )

        if use_interp_perturb_field and flag_options.USE_MINI_HALOS:
            raise ValueError("Cannot use an interpolated perturb field with minihalos!")

        if init_box is None:
            init_box = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                # !!! SLTK: added astro_params and flag_options
                astro_params=astro_params,
                flag_options=flag_options,
                random_seed=random_seed,
                hooks=hooks,
                regenerate=regenerate,
                direc=direc,
            )

        # We can go ahead and purge some of the stuff in the init_box, but only if
        # it is cached -- otherwise we could be losing information.
        try:
            init_box.prepare_for_perturb(flag_options=flag_options, force=always_purge)
        except OSError:
            pass

        if perturb:
            if redshift is not None and any(
                p.redshift != z for p, z in zip(perturb, redshift)
            ):
                raise ValueError("Input redshifts do not match perturb field redshifts")
            else:
                redshift = [p.redshift for p in perturb]

        if (
            flag_options.USE_HALO_FIELD
            and pt_halos
            and any(p.redshift != z for p, z in zip(pt_halos, redshift))
        ):
            raise ValueError(
                "Input redshifts do not match the perturbed halo field redshifts"
            )

        if flag_options.PHOTON_CONS:
            calibrate_photon_cons(
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                init_box,
                regenerate,
                write,
                direc,
            )

        if not hasattr(redshift, "__len__"):
            singleton = True
            redshift = [redshift]

        # Get the list of redshift we need to scroll through.
        redshifts = _get_redshifts(flag_options, redshift)

        # Get all the perturb boxes early. We need to get the perturb at every
        # redshift, even if we are interpolating the perturb field, because the
        # ionize box needs it.

        pz = [p.redshift for p in perturb]
        perturb_ = []
        for z in redshifts:
            p = (
                perturb_field(
                    redshift=z,
                    init_boxes=init_box,
                    regenerate=regenerate,
                    # !!! SLTK: added astro_params and flag_options specified to be coherent
                    astro_params=astro_params,
                    flag_options=flag_options,
                    hooks=hooks,
                    direc=direc,
                )
                if z not in pz
                else perturb[pz.index(z)]
            )

            if user_params.MINIMIZE_MEMORY:
                try:
                    p.purge(force=always_purge)
                except OSError:
                    pass

            perturb_.append(p)

        perturb = perturb_

        # Now we can purge init_box further.
        try:
            init_box.prepare_for_spin_temp(
                flag_options=flag_options, force=always_purge
            )
        except OSError:
            pass

        if flag_options.USE_HALO_FIELD and not pt_halos:
            for z in redshift:
                pt_halos += [
                    perturb_halo_list(
                        redshift=z,
                        init_boxes=init_box,
                        user_params=user_params,
                        cosmo_params=cosmo_params,
                        astro_params=astro_params,
                        flag_options=flag_options,
                        halo_field=determine_halo_list(
                            redshift=z,
                            init_boxes=init_box,
                            astro_params=astro_params,
                            flag_options=flag_options,
                            regenerate=regenerate,
                            hooks=hooks,
                            direc=direc,
                        ),
                        regenerate=regenerate,
                        hooks=hooks,
                        direc=direc,
                    )
                ]

        if (
            flag_options.PHOTON_CONS
            and np.amin(redshifts) < global_params.PhotonConsEndCalibz
        ):
            raise ValueError(
                f"You have passed a redshift (z = {np.amin(redshifts)}) that is lower than"
                "the endpoint of the photon non-conservation correction"
                f"(global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz})."
                "If this behaviour is desired then set global_params.PhotonConsEndCalibz"
                f"to a value lower than z = {np.amin(redshifts)}."
            )

        if flag_options.PHOTON_CONS:
            calibrate_photon_cons(
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                init_box,
                regenerate,
                write,
                direc,
            )

        ib_tracker = [0] * len(redshift)
        bt = [0] * len(redshift)
        st, ib, pf = None, None, None  # At first we don't have any "previous" st or ib.

        perturb_min = perturb[np.argmin(redshift)]

        st_tracker = [None] * len(redshift)

        spin_temp_files = []
        perturb_files = []
        ionize_files = []
        brightness_files = []

        # Iterate through redshift from top to bottom
        for iz, z in enumerate(redshifts):
            pf2 = perturb[iz]
            pf2.load_all()

            if flag_options.USE_TS_FLUCT:
                logger.debug(f"Doing spin temp for z={z}.")
                st2 = spin_temperature(
                    redshift=z,
                    previous_spin_temp=st,
                    perturbed_field=perturb_min if use_interp_perturb_field else pf2,
                    # remember that perturb field is interpolated, so no need to provide exact one.
                    astro_params=astro_params,
                    flag_options=flag_options,
                    regenerate=regenerate,
                    init_boxes=init_box,
                    hooks=hooks,
                    direc=direc,
                    cleanup=(
                        cleanup and z == redshifts[-1]
                    ),  # cleanup if its the last time through
                )

                if z not in redshift:
                    st = st2

            ib2 = ionize_box(
                redshift=z,
                previous_ionize_box=ib,
                init_boxes=init_box,
                perturbed_field=pf2,
                # perturb field *not* interpolated here.
                previous_perturbed_field=pf,
                pt_halos=pt_halos[redshift.index(z)]
                if z in redshift and flag_options.USE_HALO_FIELD
                else None,
                astro_params=astro_params,
                flag_options=flag_options,
                spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                regenerate=regenerate,
                z_heat_max=global_params.Z_HEAT_MAX,
                hooks=hooks,
                direc=direc,
                cleanup=(
                    cleanup and z == redshifts[-1]
                ),  # cleanup if its the last time through
            )

            if pf is not None:
                try:
                    pf.purge(force=always_purge)
                except OSError:
                    pass

            if z in redshift:
                logger.debug(f"PID={os.getpid()} doing brightness temp for z={z}")
                ib_tracker[redshift.index(z)] = ib2
                st_tracker[redshift.index(z)] = (
                    st2 if flag_options.USE_TS_FLUCT else None
                )

                _bt = brightness_temperature(
                    ionized_box=ib2,
                    perturbed_field=pf2,
                    spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                    hooks=hooks,
                    direc=direc,
                    regenerate=regenerate,
                )

                bt[redshift.index(z)] = _bt

            else:
                ib = ib2
                pf = pf2
                _bt = None

            perturb_files.append((z, os.path.join(direc, pf2.filename)))
            if flag_options.USE_TS_FLUCT:
                spin_temp_files.append((z, os.path.join(direc, st2.filename)))
            ionize_files.append((z, os.path.join(direc, ib2.filename)))

            if _bt is not None:
                brightness_files.append((z, os.path.join(direc, _bt.filename)))

        if flag_options.PHOTON_CONS:
            photon_nonconservation_data = _get_photon_nonconservation_data()
            if photon_nonconservation_data:
                lib.FreePhotonConsMemory()
        else:
            photon_nonconservation_data = None

        if (
            flag_options.USE_TS_FLUCT
            and user_params.USE_INTERPOLATION_TABLES
            and lib.interpolation_tables_allocated
        ):
            lib.FreeTsInterpolationTables(flag_options())

        coevals = [
            Coeval(
                redshift=z,
                initial_conditions=init_box,
                perturbed_field=perturb[redshifts.index(z)],
                ionized_box=ib,
                brightness_temp=_bt,
                ts_box=st,
                photon_nonconservation_data=photon_nonconservation_data,
                cache_files={
                    "init": [(0, os.path.join(direc, init_box.filename))],
                    "perturb_field": perturb_files,
                    "ionized_box": ionize_files,
                    "brightness_temp": brightness_files,
                    "spin_temp": spin_temp_files,
                },
            )
            for z, ib, _bt, st in zip(redshift, ib_tracker, bt, st_tracker)
        ]

        # If a single redshift was passed, then pass back singletons.
        if singleton:
            coevals = coevals[0]

        logger.debug("Returning from Coeval")

        return coevals


def _get_redshifts(flag_options, redshift):
    if flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT:
        redshifts = _logscroll_redshifts(
            min(redshift),
            global_params.ZPRIME_STEP_FACTOR,
            global_params.Z_HEAT_MAX,
        )
    else:
        redshifts = [min(redshift)]
    # Add in the redshift defined by the user, and sort in order
    # Turn into a set so that exact matching user-set redshift
    # don't double-up with scrolling ones.
    redshifts += redshift
    redshifts = sorted(set(redshifts), reverse=True)
    return redshifts


def run_lightcone(
    *,
    redshift=None,
    max_redshift=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    regenerate=None,
    write=None,
    lightcone_quantities=("brightness_temp",),
    global_quantities=("brightness_temp", "xH_box"),
    direc=None,
    init_box=None,
    perturb=None,
    random_seed=None,
    coeval_callback=None,
    coeval_callback_redshifts=1,
    use_interp_perturb_field=False,
    cleanup=True,
    hooks=None,
    get_c_T=None, # JordanFlitter: added get_c_T to run_lightcone() - the median operator is not very efficient so I let the user to decide if it is needed
    k_values=None, # JordanFlitter: added k_values for the computation of the scale-dependent c_T (in the case of scale-dependent evolution)
    save_coeval_redshifts=None, # JordanFlitter: added save_coeval_redshifts to run_lightcone()
    save_coeval_quantities=None, # JordanFlitter: added save_coeval_quantities to run_lightcone()
    always_purge: bool = False,
    **global_kwargs,
):
    r"""
    Evaluate a full lightcone ending at a given redshift.

    This is generally the easiest and most efficient way to generate a lightcone, though it can
    be done manually by using the lower-level functions which are called by this function.

    Parameters
    ----------
    redshift : float
        The minimum redshift of the lightcone.
    max_redshift : float, optional
        The maximum redshift at which to keep lightcone information. By default, this is equal to
        `z_heat_max`. Note that this is not *exact*, but will be typically slightly exceeded.
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options : :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    lightcone_quantities : tuple of str, optional
        The quantities to form into a lightcone. By default, just the brightness
        temperature. Note that these quantities must exist in one of the output
        structures:

        * :class:`~InitialConditions`
        * :class:`~PerturbField`
        * :class:`~TsBox`
        * :class:`~IonizedBox`
        * :class:`BrightnessTemp`

        To get a full list of possible quantities, run :func:`get_all_fieldnames`.
    global_quantities : tuple of str, optional
        The quantities to save as globally-averaged redshift-dependent functions.
        These may be any of the quantities that can be used in ``lightcone_quantities``.
        The mean is taken over the full 3D cube at each redshift, rather than a 2D
        slice.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    perturb : list of :class:`~PerturbedField`, optional
        If given, must be compatible with init_box. It will merely negate the necessity of
        re-calculating the
        perturb fields. It will also be used to set the redshift if given.
    coeval_callback : callable, optional
        User-defined arbitrary function computed on :class:`~Coeval`, at redshifts defined in
        `coeval_callback_redshifts`.
        If given, the function returns :class:`~LightCone` and the list of `coeval_callback` outputs.
    coeval_callback_redshifts : list or int, optional
        Redshifts for `coeval_callback` computation.
        If list, computes the function on `node_redshifts` closest to the specified ones.
        If positive integer, computes the function on every n-th redshift in `node_redshifts`.
        Ignored in the case `coeval_callback is None`.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone,
        to determine all spin temperature fields. If so, this field is interpolated in the
        underlying C-code to the correct redshift. This is less accurate (and no more efficient),
        but provides compatibility with older versions of 21cmFAST.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    minimize_memory_usage
        If switched on, the routine will do all it can to minimize peak memory usage.
        This will be at the cost of disk I/O and CPU time. Recommended to only set this
        if you are running particularly large boxes, or have low RAM.
    get_c_T : bool, optional
        If switched on, the median of the c_T box, defined as c_T = delta_T/delta is
        computed at every redshift in `node_redshifts`. Similarly, the median of c_x_e,
        c_T_s and c_21 (defined very similarly to c_T) are computed. These median values
        are can then be accessed via the attribute `c_T_median` etc. Note that if
        EVOLVE_BARYONS in user_params is set to False, then delta is delta_c and c_T (and
        its friends) are scale-independent during the dark ages, where linear theory still
        applies. However, if EVOLVE_BARYONS is set to True, then c_T (and its friends) are
        scale-dependent even during the dark ages. The output then would be in the form of
        a matrix, where its columns correspond to the redshifts in `node_redshifts` while
        its rows correspond to the wavenumbers listed in `k_values`.
    k_values: list of floats
        The wavenumbers in which the median of the scale-dependent c_T (and its friends) box
        will be evaluated in the case of scale-dependent evolution.
    save_coeval_redshifts: list of floats, optional
        If not None, coeval boxes of quantities from `save_coeval_quantities` will be saved
        at redshifts that are associated with the values of `save_coeval_redshifts`.
        The redshifts in which the coeval boxes will be stored will be determined by finding
        the closest redshift in `node_redshifts. Accessing these coeval boxes then becomes
        possible via the attribute `coeval_boxes` which returns a nested dictionary of the
        form coeval_boxes[node_redshift][save_coeval_quantity].
    save_coeval_quantities: list of strings, optional
        The types of coeval boxes to store at the output. These may be any of the quantities
        that can be used in ``lightcone_quantities``.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    lightcone : :class:`~py21cmfast.LightCone`
        The lightcone object.
    coeval_callback_output : list
        Only if coeval_callback in not None.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed
        See docs of :func:`initial_conditions` for more information.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    with global_params.use(**global_kwargs):
        # !!! SLTK: added astro_params and flag_options
        random_seed, user_params, cosmo_params, \
        astro_params, flag_options = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options",flag_options),
            ],
            init_box,
            perturb,
        )
        # JordanFlitter: I added compatibility with A_s
        if ((not 'SIGMA_8' in cosmo_params) and 'A_s' in cosmo_params):
            A_s_FLAG = True
        elif (not 'A_s' in cosmo_params):
            A_s_FLAG = False
        else:
            raise ValueError("Please specify either SIGMA_8 or A_s, but not both.")

        user_params_dic = user_params
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        flag_options = FlagOptions(
            flag_options, USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES
        )
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)        

        # JordanFlitter: I added some logics to prevent conflict between inputs
        # I'm not sure if that's the best place for these logics, but it works...
        _configure_user_params(user_params,user_params_dic)

        # JordanFlitter: I added compatibility with A_s. This piece of code is required to pass 21cmFAST sigma8 in case RUN_CLASS is False
        # or alternatively, if the user wants to run CLASS with sigma8 (strange choice...) we find A_s
        if ((A_s_FLAG and (not user_params.RUN_CLASS))
            or
            ((not A_s_FLAG) and user_params.RUN_CLASS)):
            CLASS_params = {'Omega_b': cosmo_params.OMb, 'Omega_cdm': cosmo_params.OMm-cosmo_params.OMb, 'h': cosmo_params.hlittle,
                            'n_s': cosmo_params.POWER_INDEX, 'tau_reio': cosmo_params.tau_reio,
                            'm_ncdm': 0.06, 'N_ncdm': 1, 'N_ur': 2.0308, 'T_cmb': 2.728, 'output': 'mPk'}
            if user_params.FUZZY_DM:
                # Set FDM parameters
                H_0_eV = cosmo_params.hlittle * 2.1331192697532067e-33  # Hubble constant in eV
                CLASS_params['Omega_cdm'] = (1.-pow(10.,-cosmo_params.f_FDM))*(cosmo_params.OMm-cosmo_params.OMb)
                CLASS_params['Omega_scf'] = pow(10.,-cosmo_params.f_FDM) * (cosmo_params.OMm-cosmo_params.OMb)
                CLASS_params['m_axion'] = pow(10.,-cosmo_params.m_FDM) / H_0_eV # FDM mass in units of H0 (dimensionless)
                # Fix other FDM parameters/flags
                CLASS_params['f_axion'] = 1 # # Axion energy scale in units of Planck mass
                CLASS_params['n_axion'] = 1 # Power-law in the axion potential
                CLASS_params['scf_parameters'] = '2, 0' # Scalar field (scf) potential parameters: first one is lambda, second is alpha (see arXiv: astro-ph/9908085)
                CLASS_params['scf_potential'] = 'axion' # Scalar field potential type
                CLASS_params['scf_has_perturbations'] = 'yes' # Perturbations in scalar field
                CLASS_params['attractor_ic_scf'] = 'no' # Scalar field (scf) initial conditions from attractor solution
                CLASS_params['scf_tuning_index'] = 0 # Index in scf_parameters to do shooting
                # Fix other FDM parameters/flags - according to https://github.com/PoulinV/AxiCLASS/blob/master/example_axion.ipynb
                CLASS_params['do_shooting'] = 'yes'
                CLASS_params['do_shooting_scf'] = 'yes'
                CLASS_params['tol_shooting_deltax'] = 1e-4
                CLASS_params['tol_shooting_deltaF'] = 1e-4
                CLASS_params['scf_evolve_as_fluid'] = 'yes'
                CLASS_params['scf_evolve_like_axionCAMB'] = 'yes'
                CLASS_params['write background'] = 'yes'
                CLASS_params['threshold_scf_fluid_m_over_H'] = 3
                CLASS_params['include_scf_in_delta_m'] = 'yes'
                CLASS_params['include_scf_in_delta_cb'] = 'yes'
                CLASS_params['use_big_theta_scf'] = 'no'
                CLASS_params['use_delta_scf_over_1plusw'] = 'no'
                CLASS_params['background_verbose'] = 0
            if user_params.SCATTERING_DM:
                # Set SDM parameters
                CLASS_params['Omega_cdm'] = (1.-pow(10.,-cosmo_params.f_chi))*(cosmo_params.OMm-cosmo_params.OMb)
                CLASS_params['Omega_dmeff'] = pow(10.,-cosmo_params.f_chi)*(cosmo_params.OMm-cosmo_params.OMb) # ratio of SDM to total DM
                CLASS_params['m_dmeff'] = pow(10.,cosmo_params.m_chi)*1.e-9 # in GeV
                CLASS_params['sigma_dmeff'] = pow(10.,-cosmo_params.sigma_SDM) # cross section prefactor in cm^2
                CLASS_params['npow_dmeff'] = -4 # power-law of the cross section dependence on the relative velocity
                # Type of the interacting particles with the SDM (can be 'baryons', 'ionized', 'hydrogen', 'protons' or 'electrons')
                if user_params.SDM_TARGET_TYPE == 1:
                    CLASS_params['dmeff_target'] = 'baryons'
                elif user_params.SDM_TARGET_TYPE == 2:
                    CLASS_params['dmeff_target'] = 'ionized'
                elif user_params.SDM_TARGET_TYPE == 3:
                    CLASS_params['dmeff_target'] = 'hydrogen'
                elif user_params.SDM_TARGET_TYPE == 4:
                    CLASS_params['dmeff_target'] = 'protons'
                elif user_params.SDM_TARGET_TYPE == 5:
                    CLASS_params['dmeff_target'] = 'electrons'
                else:
                    raise ValueError("SDM_TARGET_TYPE must be 1 (baryons), 2 (ionized), 3 (hydrogen), 4 (protons) or 5 (electrons)")
                # Fix other SDM parameters/flags
                CLASS_params['Vrel_dmeff'] = 30 # This is the initial guess for the mean relative velocity between baryons and SDM
            # Set A_s or sigma8 based on the user input
            if A_s_FLAG:
                CLASS_params['A_s'] = cosmo_params.A_s
            else:
                CLASS_params['sigma8'] = cosmo_params.SIGMA_8

            # Import the appropriate CLASS based on the user's input
            if user_params.FUZZY_DM:
                from axiclassy import Class
            elif user_params.SCATTERING_DM:
                from dmeff_classy import Class
            else:
                from classy import Class
            # Run CLASS!
            CLASS_OUTPUT = Class()
            CLASS_OUTPUT.set(CLASS_params)
            CLASS_OUTPUT.compute()
            if A_s_FLAG:
                cosmo_params.SIGMA_8 = CLASS_OUTPUT.sigma8()
            else:
                cosmo_params.A_s = CLASS_OUTPUT.get_current_derived_parameters(['A_s'])['A_s']

        if user_params.MINIMIZE_MEMORY and not write:
            raise ValueError(
                "If trying to minimize memory usage, you must be caching. Set write=True!"
            )

        # JordanFlitter: We need lightcone boxes of delta (more precisely delta_b) and x_HI if we want to compute tau_reio from the simulation
        if user_params.EVALUATE_TAU_REIO:
            if (not user_params.EVOLVE_BARYONS) and (not 'density' in lightcone_quantities):
                lightcone_quantities += ('density',)
            if (user_params.EVOLVE_BARYONS) and (not 'baryons_density' in lightcone_quantities):
                lightcone_quantities += ('baryons_density',)
            if not 'xH_box' in lightcone_quantities:
                lightcone_quantities += ('xH_box',)

        # Ensure passed quantities are appropriate
        _fld_names = _get_interpolation_outputs(
            list(lightcone_quantities), list(global_quantities), flag_options
        )

        redshift = configure_redshift(redshift, perturb)

        max_redshift = (
            global_params.Z_HEAT_MAX
            if (
                flag_options.INHOMO_RECO
                or flag_options.USE_TS_FLUCT
                or max_redshift is None
            )
            else max_redshift
        )

        # Get the redshift through which we scroll and evaluate the ionization field.
        # JordanFlitter: I replaced scrollz with scrollz_cosmic_dawn (yes, longer name but it helps distinguishing it from scrollz_dark_ages_output)
        # JordanFlitter: added an option for many more redshift samples during cosmic dawn, below Z_HEAT_MAX. For now, it works only if START_AT_RECOMBINATION = True.
        #                This might be useful because the standard step size of 21cmFAST is insufficient for precise evolution of the temperature field if the
        #                temperature reaches low values, e.g. when SCATTERING_DM = True.
        #                If this condition is satisfied we define scrollz_cosmic_dawn_output, which is shorter than scrollz_cosmic_dawn. The reason for this is because
        #                the reionization code is not very efficient, especially when USE_MINI_HALOS is turned on, but we don't need to compute the reionization field
        #                at every iteration if we just want to have more iterations for the temperature evolution.
        if not (user_params.MANY_Z_SAMPLES_AT_COSMIC_DAWN and user_params.START_AT_RECOMBINATION):
            scrollz_cosmic_dawn = _logscroll_redshifts(
                redshift, global_params.ZPRIME_STEP_FACTOR, max_redshift
            )
            scrollz_cosmic_dawn_output = scrollz_cosmic_dawn
        else:
            scrollz_array = _logscroll_redshifts(
                redshift, global_params.ZPRIME_STEP_FACTOR, max_redshift
            )
            if global_params.Z1_VALUE > global_params.Z_HEAT_MAX:
                scrollz_cosmic_dawn = _linscroll_redshifts(redshift,global_params.DELTA_Z,max_redshift)
            else:
                scrollz_cosmic_dawn = _linscroll_redshifts(redshift,global_params.DELTA_Z1,max_redshift)

            scrollz_cosmic_dawn_output = []
            for z in scrollz_array:
                scrollz_cosmic_dawn_output.append(scrollz_cosmic_dawn[np.argmin(np.abs(np.array(scrollz_cosmic_dawn)-z))])

        if (
            flag_options.PHOTON_CONS
            and np.amin(scrollz_cosmic_dawn) < global_params.PhotonConsEndCalibz
        ):
            raise ValueError(
                f"""
                You have passed a redshift (z = {np.amin(scrollz_cosmic_dawn)}) that is lower than the endpoint
                of the photon non-conservation correction
                (global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz}).
                If this behaviour is desired then set global_params.PhotonConsEndCalibz to a value lower than
                z = {np.amin(scrollz_cosmic_dawn)}.
                """
            )

        coeval_callback_output = []
        compute_coeval_callback = _get_coeval_callbacks(
            scrollz_cosmic_dawn, coeval_callback, coeval_callback_redshifts
        )

        # JordanFlitter: Genereate initial conditions from CLASS, and get Cl data
        if user_params.RUN_CLASS:
            print("Now running CLASS...")
            time.sleep(0.1) # we pause the program for a short time just to print the above message before running CLASS
            Cl_data = run_ICs(cosmo_params,user_params,global_params)
        else:
            # JordanFlitterTODO: you can remove this call once the derived cosmological values in global_params are moved to a new structure
            _set_default_globals()
            Cl_data = "No Cl data. Need to set RUN_CLASS on True."

        # JordanFlitter: set Z_HIGH_MAX to Z_REC if it's negative (the default is -1)
        if (user_params.Z_HIGH_MAX < 0):
            user_params.Z_HIGH_MAX = global_params.Z_REC
        # JordanFlitter: defined a dark ages scrolling array at the C-level.
        #                This is the logic: between Z_HEAT_MAX and Z1_VALUE, the step size is DELTA_Z.
        #                Between Z1_VALUE and Z2_VALUE, the step size is DELTA_Z1.
        #                And between Z2_VALUE and Z_HIGH_MAX the step size is DELTA_Z2 (unless USE_TCA_COMPTON=True, in which case, it is DELTA_Z1).
        #                The if conditions deal with scenarios in which Z_HEAT_MAX<Z1_VALUE<Z2_VALUE<Z_HIGH_MAX is *not* satisfied
        if user_params.START_AT_RECOMBINATION:
            # For convenience, I round the delta's (there are floating point errors at the C-level...)
            global_params.DELTA_Z = round(global_params.DELTA_Z,2)
            global_params.DELTA_Z1 = round(global_params.DELTA_Z1,2)
            global_params.DELTA_Z2 = round(global_params.DELTA_Z2,2)

            if (scrollz_cosmic_dawn[0]+global_params.DELTA_Z < min(global_params.Z1_VALUE,user_params.Z_HIGH_MAX)):
                scrollz_dark_ages_C = (_linscroll_redshifts(scrollz_cosmic_dawn[0]+global_params.DELTA_Z
                                                    ,global_params.DELTA_Z,min(global_params.Z1_VALUE,user_params.Z_HIGH_MAX)) + scrollz_cosmic_dawn)
            else:
                scrollz_dark_ages_C = scrollz_cosmic_dawn
            if (scrollz_dark_ages_C[0]+global_params.DELTA_Z1 < min(global_params.Z2_VALUE,user_params.Z_HIGH_MAX)):
                scrollz_dark_ages_C = (_linscroll_redshifts(scrollz_dark_ages_C[0]+global_params.DELTA_Z1
                                                    ,global_params.DELTA_Z1,min(global_params.Z2_VALUE,user_params.Z_HIGH_MAX))+ scrollz_dark_ages_C)
            if user_params.USE_TCA_COMPTON:
                if (scrollz_dark_ages_C[0]+global_params.DELTA_Z1 < user_params.Z_HIGH_MAX):
                    scrollz_dark_ages_C = (_linscroll_redshifts(scrollz_dark_ages_C[0]+global_params.DELTA_Z1
                                                        ,global_params.DELTA_Z1,user_params.Z_HIGH_MAX)+ scrollz_dark_ages_C)
            else:
                if (scrollz_dark_ages_C[0]+global_params.DELTA_Z2 < user_params.Z_HIGH_MAX):
                    scrollz_dark_ages_C = (_linscroll_redshifts(scrollz_dark_ages_C[0]+global_params.DELTA_Z2
                                                        ,global_params.DELTA_Z2,user_params.Z_HIGH_MAX)+ scrollz_dark_ages_C)

            scrollz_dark_ages_C_array = np.array(scrollz_dark_ages_C)
            scrollz_dark_ages_C_array = scrollz_dark_ages_C_array[scrollz_dark_ages_C_array > scrollz_cosmic_dawn[0]]

            # JordanFlitter: I defined scrollz_2LPT_dark_ages (only relevant if REDSHIFT_2LPT > Z_HEAT_MAX)
            scrollz_2LPT_dark_ages = list(scrollz_dark_ages_C_array[scrollz_dark_ages_C_array < global_params.REDSHIFT_2LPT])
            # JordanFlitter: I defined scrollz_dark_ages and scrollz_dark_ages_output. The first is the input redshifts for the dark ages
            # loop (at the python level!), while the second is the output redshifts
            # Note that these lists are not the same because we want to do much of the scrolling at the C level (MUCH FASTER that way)
            if user_params.OUTPUT_AT_DARK_AGES:
                z_array = np.array(_logscroll_redshifts(redshift, global_params.Z_DARK_AGES_STEP_FACTOR, user_params.Z_HIGH_MAX))
                z_array = list(z_array[z_array > scrollz_cosmic_dawn[0]][1:])
                scrollz_mid_dark_ages_array = np.array([scrollz_dark_ages_C_array[scrollz_dark_ages_C_array < z][0] for z in z_array])
                try:
                    scrollz_mid_dark_ages1 = list(scrollz_mid_dark_ages_array[scrollz_mid_dark_ages_array > scrollz_2LPT_dark_ages[1]])
                except IndexError:
                    scrollz_mid_dark_ages1 = list(scrollz_mid_dark_ages_array)
                try:
                    scrollz_mid_dark_ages2 = list(scrollz_mid_dark_ages_array[scrollz_mid_dark_ages_array > scrollz_2LPT_dark_ages[0]])
                except IndexError:
                    scrollz_mid_dark_ages2 = list(scrollz_mid_dark_ages_array)
            else:
                scrollz_mid_dark_ages1 = []
                scrollz_mid_dark_ages2 = []
            scrollz_dark_ages = [scrollz_dark_ages_C[0], scrollz_dark_ages_C[1]] + scrollz_mid_dark_ages1 + scrollz_2LPT_dark_ages[1:]
            scrollz_dark_ages_output = [scrollz_dark_ages_C[0],] + scrollz_mid_dark_ages2 + scrollz_2LPT_dark_ages[:-1] + [scrollz_dark_ages_C_array[-1],]
        else:
            scrollz_dark_ages = []
            scrollz_dark_ages_output = []

        # JordanFlitter: I defined scrollz_all_output, it combines the output redshifts both at the dark ages and cosmic dawn
        if user_params.OUTPUT_AT_DARK_AGES:
            scrollz_all_output = scrollz_dark_ages_output + scrollz_cosmic_dawn_output
        else:
            scrollz_all_output = scrollz_cosmic_dawn_output

        # JordanFlitter: we find the closest redshift
        if save_coeval_redshifts is not None:
            scrollz_save_coevals = []
            scrollz_save_coevals_indices = []
            for z in save_coeval_redshifts:
                scrollz_save_coevals_indices.append(np.argmin(np.abs(np.array(scrollz_all_output)-z)))
                scrollz_save_coevals.append(scrollz_all_output[scrollz_save_coevals_indices[-1]])

        # JordanFlitter: print the following message
        print("Now generating initial boxes...")
        time.sleep(0.1) # we pause the program for a short time just to print the above message before running CLASS
        if init_box is None:  # no need to get cosmo, user params out of it.
            init_box = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                # !!! SLTK: added astro_params and flag_options
                astro_params=astro_params,
                flag_options=flag_options,
                hooks=hooks,
                regenerate=regenerate,
                write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                direc=direc,
                random_seed=random_seed,
            )

        # We can go ahead and purge some of the stuff in the init_box, but only if
        # it is cached -- otherwise we could be losing information.
        try:
            # TODO: should really check that the file at path actually contains a fully
            # working copy of the init_box.
            init_box.prepare_for_perturb(flag_options=flag_options, force=always_purge)
        except OSError:
            pass

        # JordanFlitter: Added the following condition
        # The problem with the implementation below is that the perturb_ list grows uncontrollably.
        # Because each item in that list is a coeaval box, if we START_AT_RECOMBINATION this would cause the code to crash!
        # Therefore, when we START_AT_RECOMBINATION we make sure we evaluate the density field along with the spin temperature (see dark ages loop below)
        if not user_params.DO_PERTURBS_WITH_TS:
            if perturb is None: # JordanFlitter: changed the following logic for zz
                zz = scrollz_all_output
            else:
                zz = scrollz_cosmic_dawn[:-1]

            perturb_ = []
            # JordanFlitter: Added tqdm waiting bar here to track memory issues
            for z in tqdm.tqdm(zz,
                               desc="Perturbations",
                               unit="redshift",
                               disable=False,
                               total=len(zz)):
                p = perturb_field(
                    redshift=z,
                    init_boxes=init_box,
                    regenerate=regenerate,
                    # !!! SLTK: added astro_params and flag_options specified to be coherent
                    astro_params=astro_params,
                    flag_options=flag_options,
                    direc=direc,
                    write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                    hooks=hooks,
                )
                if user_params.MINIMIZE_MEMORY:
                    try:
                        p.purge(force=always_purge)
                    except OSError:
                        pass

                perturb_.append(p)

            if perturb is not None:
                perturb_.append(perturb)
            perturb = perturb_
            perturb_min = perturb[np.argmin(scrollz_cosmic_dawn)]

        # Now that we've got all the perturb fields, we can purge init more.
        try:
            init_box.prepare_for_spin_temp(
                flag_options=flag_options, force=always_purge
            )
        except OSError:
            pass

        if flag_options.PHOTON_CONS:
            calibrate_photon_cons(
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                init_box,
                regenerate,
                write,
                direc,
            )
        # JordanFlitter: I modify the lightcone distances in case we calculate the brightness temperature at recombination
        if not user_params.OUTPUT_AT_DARK_AGES:
            d_at_redshift, lc_distances, n_lightcone = _setup_lightcone(
                cosmo_params,
                max_redshift,
                redshift,
                scrollz_cosmic_dawn_output,
                user_params,
                global_params.ZPRIME_STEP_FACTOR,
            )

        else:
            d_at_redshift, lc_distances, n_lightcone = _setup_lightcone(
                cosmo_params,
                user_params.Z_HIGH_MAX,
                redshift,
                scrollz_all_output,
                user_params,
                global_params.ZPRIME_STEP_FACTOR,
            )

        scroll_distances = (
            cosmo_params.cosmo.comoving_distance(scrollz_all_output).value - d_at_redshift
        )

        # Iterate through redshift from top to bottom
        st, ib, bt, prev_perturb = None, None, None, None
        lc_index = 0
        box_index = 0
        lc = {
            quantity: np.zeros(
                (user_params.HII_DIM, user_params.HII_DIM, n_lightcone),
                dtype=np.float32,
            )
            for quantity in lightcone_quantities
        }

        interp_functions = {
            "z_re_box": "mean_max",
        }

        # JordanFlitter: I replaced scrollz with scrollz_all_output when defining the arrays for the global quantities
        global_q = {quantity: np.zeros(len(scrollz_all_output)) for quantity in global_quantities}
        pf = None
        # JordanFlitter: added c_T and its friends to output
        if not user_params.EVOLVE_BARYONS:
            c_T_median = np.zeros(len(scrollz_all_output))
            c_x_e_median = np.zeros(len(scrollz_all_output))
            c_T_s_median = np.zeros(len(scrollz_all_output))
            c_21_median = np.zeros(len(scrollz_all_output))
        else:
            if k_values is None:
                k_values = [0.1,0.5,1.] # 1/Mpc
            c_T_median = np.zeros((len(k_values),len(scrollz_all_output)))
            c_x_e_median = np.zeros((len(k_values),len(scrollz_all_output)))
            c_T_s_median = np.zeros((len(k_values),len(scrollz_all_output)))
            c_21_median = np.zeros((len(k_values),len(scrollz_all_output)))

        perturb_files = []
        spin_temp_files = []
        ionize_files = []
        brightness_files = []
        # JordanFlitter: added coeval_boxes to output
        coeval_boxes = {}
        # JordanFlitter: Need to define a wavevector box for the c_T calculation
        if get_c_T and user_params.EVOLVE_BARYONS:
            Delta_k = 2.*np.pi/user_params.BOX_LEN
            k_samples = Delta_k*np.arange(user_params.HII_DIM)
            k_x_box, k_y_box, k_z_box = np.meshgrid(k_samples,k_samples,k_samples)
            k_mag_box = np.sqrt(k_x_box**2 + k_y_box**2 + k_z_box**2)

        # JordanFlitter: I made a copy for the redshift loop (to do the evolution during the dark ages)
        # This loop is practically the same as the loop below for cosmic dawn evolution except for the following exceptions:
        #   (1) We do not have any astrophysics during the dark ages, thereby the code runs MUCH faster per redshift iteration
        #   (2) There A LOT MORE redshift iterations in the C code (more than 10,000 compared to only ~100 iterations during cosmic dawn).
        #       To speed up the calculation significantly, we do NOT return the wrapper after each redshift iteration in the C code. Instead,
        #       the C code continutes the evolution until it reaches next_redshift_input = scrollz_dark_ages_output[iz]. At the C-level, because
        #       of floating point error, we stop at a slightly different redshift which we name as next_redshift_output. This value is then fed
        #       to the next C iteration, so it is as if we haven't left the C loop! This implementation allows both having output during the dark ages
        #       on the one hand, while going through that epoch fairly quickly.

        if not len(scrollz_dark_ages) == 0:
            # JordanFlitter: we print the following message (only if OUTPUT_AT_DARK_AGES = False, otherwise we have tqdm waiting bar)
            if not user_params.OUTPUT_AT_DARK_AGES:
                print("Now going through the dark ages...")
                time.sleep(0.1) # we pause the program for a short time just to print the above message before going through the dark ages
            # JordanFlitter: I added tqdm waiting bars
            for iz, z in tqdm.tqdm(enumerate(scrollz_dark_ages),
                                 desc="21cmFAST (dark ages)",
                                 unit="redshift",
                                 disable=not user_params.OUTPUT_AT_DARK_AGES,
                                 total=len(scrollz_dark_ages)):

                # JordanFlitter: during the dark ages we ALWAYS compute the density field at each iteration,
                #                instead of having a giant list that contains all the density boxes.
                #                See comment above for DO_PERTURBS_WITH_TS

                pf2 = perturb_field(
                      redshift=z,
                      init_boxes=init_box,
                      regenerate=regenerate,
                        # !!! SLTK: added astro_params and flag_options specified to be coherent
                        astro_params=astro_params,
                        flag_options=flag_options,
                      direc=direc,
                      write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                      hooks=hooks,
                )
                # JordanFlitter: during the dark ages we ALWAYS compute the spin temperature at each iteration,
                #                even if USE_TS_FLUCT = False
                st2 = spin_temperature(
                    redshift=z,
                    previous_spin_temp=st,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    perturbed_field=perturb_min if use_interp_perturb_field else pf2,
                    regenerate=regenerate,
                    init_boxes=init_box,
                    next_redshift_input=scrollz_dark_ages_output[iz], # JordanFlitter: added next_redshift_input
                    hooks=hooks,
                    direc=direc,
                    write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                    cleanup=(cleanup)# and iz == (len(scrollz_dark_ages) - 1)), # JordanFlitter: I want to cleanup every iteration!
                )

                # JordanFlitter: If necessary, we update the redshift after spin_temperature() did its thing
                #                Note that because of floating point error, next_redshift_output is *not* precisely next_redshift_input)
                if not (scrollz_dark_ages[iz] == scrollz_dark_ages_output[iz]):
                    if iz < len(scrollz_dark_ages)-1:
                        scrollz_dark_ages[iz+1] = st2.next_redshift_output
                    z = st2.next_redshift_output
                    st2.redshift = z
                    if user_params.OUTPUT_AT_DARK_AGES:
                        scrollz_all_output[iz] = z
                        scroll_distances[iz] = cosmo_params.cosmo.comoving_distance(z).value - d_at_redshift
                        pf2 = perturb_field(
                                  redshift=z,
                                  init_boxes=init_box,
                                  regenerate=regenerate,
                                # !!! SLTK: added astro_params and flag_options specified to be coherent
                                astro_params=astro_params,
                                flag_options=flag_options,
                                  direc=direc,
                                  write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                                  hooks=hooks,
                        )

                # JordanFlitter: added the following condition in case we are not interested in having output at the dark ages
                if user_params.OUTPUT_AT_DARK_AGES:

                    # JordanFlitter: added c_T and its friends to output
                    if get_c_T:
                        with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
                            if not user_params.EVOLVE_BARYONS:
                                c_T_Box = (st2.Tk_box-st2.Tk_box.mean())/st2.Tk_box.mean()/pf2.density
                                c_x_e_Box = (st2.x_e_box-st2.x_e_box.mean())/st2.x_e_box.mean()/pf2.density
                                c_T_s_Box = (st2.Ts_box-st2.Ts_box.mean())/st2.Ts_box.mean()/pf2.density
                                c_T_median[iz] = np.percentile(c_T_Box,50)
                                c_x_e_median[iz] = np.percentile(c_x_e_Box,50)
                                c_T_s_median[iz] = np.percentile(c_T_s_Box,50)
                            else:
                                c_T_Box = fftn((st2.Tk_box-st2.Tk_box.mean())/st2.Tk_box.mean())/fftn(pf2.baryons_density)
                                c_x_e_Box = fftn((st2.x_e_box-st2.x_e_box.mean())/st2.x_e_box.mean())/fftn(pf2.baryons_density)
                                c_T_s_Box = fftn((st2.Ts_box-st2.Ts_box.mean())/st2.Ts_box.mean())/fftn(pf2.baryons_density)
                                for k_ind,k_val in enumerate(k_values):
                                    good_inds = np.logical_and(k_mag_box < k_val + Delta_k/2., k_mag_box > k_val - Delta_k/2.)
                                    c_T_median[k_ind,iz] = np.percentile(np.real(c_T_Box[good_inds]),50)
                                    c_x_e_median[k_ind,iz] = np.percentile(np.real(c_x_e_Box[good_inds]),50)
                                    c_T_s_median[k_ind,iz] = np.percentile(np.real(c_T_s_Box[good_inds]),50)

                    ib2 = ionize_box(
                        redshift=z,
                        previous_ionize_box=ib,
                        init_boxes=init_box,
                        perturbed_field=pf2,
                        previous_perturbed_field=prev_perturb,
                        astro_params=astro_params,
                        flag_options=flag_options,
                        spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                        pt_halos=None,
                        regenerate=regenerate,
                        hooks=hooks,
                        direc=direc,
                        write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                        cleanup=(cleanup)# and iz == (len(scrollz_dark_ages) - 1)), # JordanFlitter: I want to cleanup every iteration!
                    )

                    bt2 = brightness_temperature(
                        ionized_box=ib2,
                        perturbed_field=pf2,
                        spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                        hooks=hooks,
                        direc=direc,
                        write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                        regenerate=regenerate,
                    )
                    # JordanFlitter: added c_21 to output
                    if get_c_T:
                        with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
                            if not user_params.EVOLVE_BARYONS:
                                c_21_Box = (bt2.brightness_temp-bt2.brightness_temp.mean())/bt2.brightness_temp.mean()/pf2.density
                                c_21_median[iz] = np.percentile(c_21_Box,50)
                            else:
                                c_21_Box = fftn((bt2.brightness_temp-bt2.brightness_temp.mean())/bt2.brightness_temp.mean())/fftn(pf2.baryons_density)
                                for k_ind,k_val in enumerate(k_values):
                                    good_inds = np.logical_and(k_mag_box < k_val + Delta_k/2., k_mag_box > k_val - Delta_k/2.)
                                    c_21_median[k_ind,iz] = np.percentile(np.real(c_21_Box[good_inds]),50)

                    if coeval_callback is not None and compute_coeval_callback[iz]:
                        coeval = Coeval(
                            redshift=z,
                            initial_conditions=init_box,
                            perturbed_field=pf2,
                            ionized_box=ib2,
                            brightness_temp=bt2,
                            ts_box=st2 if flag_options.USE_TS_FLUCT else None,
                            photon_nonconservation_data=_get_photon_nonconservation_data()
                            if flag_options.PHOTON_CONS
                            else None,
                            _globals=None,
                        )
                        try:
                            coeval_callback_output.append(coeval_callback(coeval))
                        except Exception as e:
                            if sum(compute_coeval_callback[: iz + 1]) == 1:
                                raise RuntimeError(
                                    f"coeval_callback computation failed on first trial, z={z}."
                                )
                            else:
                                logger.warning(
                                    f"coeval_callback computation failed on z={z}, skipping. {type(e).__name__}: {e}"
                                )

                    if write:
                        perturb_files.append((z, os.path.join(direc, pf2.filename)))
                        if flag_options.USE_TS_FLUCT:
                            spin_temp_files.append((z, os.path.join(direc, st2.filename)))
                        ionize_files.append((z, os.path.join(direc, ib2.filename)))
                        brightness_files.append((z, os.path.join(direc, bt2.filename)))

                    outs = {
                        "PerturbedField": (pf, pf2),
                        "IonizedBox": (ib, ib2),
                        "BrightnessTemp": (bt, bt2),
                    }
                    if flag_options.USE_TS_FLUCT:
                        outs["TsBox"] = (st, st2)

                    # JordanFlitter: I save coeval boxes if we are in the desired redshift range
                    #                Because we (slightly) change scrollz_dark_ages every iteration,
                    #                we cannot simply ask if "z in scrollz_save_coevals". Instead,
                    #                we ask whether "we are at the right index".
                    if (save_coeval_redshifts is not None) and iz in scrollz_save_coevals_indices:
                        if save_coeval_quantities is not None:
                            coeval_boxes[z] = {}
                            for quantity in save_coeval_quantities:
                                coeval_boxes[z][quantity] = getattr(outs[_fld_names[quantity]][1],quantity)

                    # Save mean/global quantities
                    for quantity in global_quantities:
                        computed_fields = ['PerturbedField','TsBox']
                        computed_fields.append('IonizedBox')
                        computed_fields.append('BrightnessTemp')
                        # JordanFlitter: added the following condition. If we haven't calculated certain fields, set their mean to zero!
                        if ((not quantity in dir(outs[_fld_names[quantity]][1]))):
                            global_q[quantity][iz] = 0.
                        else:
                            global_q[quantity][iz] = np.mean(
                                getattr(outs[_fld_names[quantity]][1], quantity)
                            )

                    # Interpolate the lightcone
                    if (z < user_params.Z_HIGH_MAX):
                        for quantity in lightcone_quantities:
                            data1, data2 = outs[_fld_names[quantity]]
                            fnc = interp_functions.get(quantity, "mean")
                            n = _interpolate_in_redshift(
                                iz,
                                box_index,
                                lc_index,
                                n_lightcone,
                                scroll_distances,
                                lc_distances,
                                data1,
                                data2,
                                quantity,
                                lc[quantity],
                                fnc,
                            )
                        lc_index += n
                        box_index += n

                # Save current ones as old ones.
                st = st2
                # JordanFlitter: I do not necessarily want to update the previous boxes at every iteration.
                #                Note that the only box we MUST update at each iteraion is the previous spin temperature box (st),
                #                which is why it is outside the if condition below
                if user_params.OUTPUT_AT_DARK_AGES:
                    ib = ib2
                    bt = bt2
                    # JordanFlitter: There used to be here a condition of "if flag_options.USE_MINI_HALOS:"
                    # I removed it, because if MINI_HALOS were not used, then perturb_field() would be called again
                    # from ionized_box(), even though the previous_perturbed_field box exists! (BUG?)
                    prev_perturb = pf2

                    if pf is not None:
                        try:
                            pf.purge(force=always_purge)
                        except OSError:
                            pass

                    pf = pf2

        # JordanFlitter: Now we go to the original cosmic dawn loop. It should be noted that while the global signal (derived from the coeval boxes)
        # during cosmic dawn is the same whether OUTPUT_AT_DARK_AGES is True or False, the power spectrum (derived from the lightcone box) isn't. This
        # is due to the following three FEATURES:
        #   (1) When OUTPUT_AT_DARK_AGES = False, we don’t have output at the last redshift sample prior to scrollz[0], and so we cannot interpolate
        #       between these samples. However, when OUTPUT_AT_DARK_AGES = True, we can do the interpolation, and that fills entries in the lightcone box.
        #   (2) Furthermore, the construction of the lightcone box causes an extra farthest slice between scrollz[0] and scrollz_dark_ages_output[-1]
        #       when OUTPUT_AT_DARK_AGES = True. This only happens for this interpolation, while for all the other interpolations the amount of lightcone
        #       slices is the same.
        #   (3) Finally, after each interpolation, the coeval box index (along the line-of-sight) for the next interpolation increases by the amount of slices used
        #       for the previous interpolation. This is a cumulative effect. When OUTPUT_AT_DARK_AGES = True (False) the box index is non-zero (zero) for the first
        #       cosmic dawn interpolation. Therefore, the realizations of the lightcone boxes are guaranteed to be different in the two scenarios.
        #
        # To summarize, when OUTPUT_AT_DARK_AGES = True, it is as if we can look farther into space and that causes a small deviation in the farthest cells of the
        # lightcone box (due to the above points 1 and 2). In addition, interpolation from the dark ages promotes the coeval box index, leading to different
        # realizations of the lightcone box (due to the above points 1 and 3). It is okay that the realizations are not exactly the same as long as the two-point
        # statistics of the lightcone boxes, namely the power spectrum, are the same. This has been verified

        # JordanFlitter: during cosmic dawn we need to increase "iz" by the amount of dark ages iterations. This gives us iz_CD. Note that this index is used
        #                only if we extract output from the iterated coeval boxes.

        if user_params.OUTPUT_AT_DARK_AGES:
            iz_CD = len(scrollz_dark_ages)
        else:
            iz_CD = 0

        # JordanFlitter: I added tqdm waiting bars
        for iz, z in tqdm.tqdm(enumerate(scrollz_cosmic_dawn),
                             desc="21cmFAST (cosmic dawn)",
                             unit="redshift",
                             disable=False,
                             total=len(scrollz_cosmic_dawn)):
            # JordanFlitter: Added the following condition
            if not user_params.DO_PERTURBS_WITH_TS:
                # Best to get a perturb for this redshift, to pass to brightness_temperature
                pf2 = perturb[iz_CD]

                # This ensures that all the arrays that are required for spin_temp are there,
                # in case we dumped them from memory into file.
                pf2.load_all()
            else:
                pf2 = perturb_field(
                      redshift=z,
                      init_boxes=init_box,
                      regenerate=regenerate,
                        # !!! SLTK: added astro_params and flag_options specified to be coherent
                        astro_params=astro_params,
                        flag_options=flag_options,
                      direc=direc,
                      write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                      hooks=hooks,
                )

            if (flag_options.USE_HALO_FIELD):

                halo_field = determine_halo_list(
                    redshift=z,
                    init_boxes=init_box,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    regenerate=regenerate,
                    hooks=hooks,
                    direc=direc,
                    write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                )
                pt_halos = perturb_halo_list(
                    redshift=z,
                    init_boxes=init_box,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    halo_field=halo_field,
                    regenerate=regenerate,
                    hooks=hooks,
                    direc=direc,
                    write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                )

            if flag_options.USE_TS_FLUCT:
                st2 = spin_temperature(
                    redshift=z,
                    previous_spin_temp=st,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    perturbed_field=perturb_min if use_interp_perturb_field else pf2,
                    regenerate=regenerate,
                    init_boxes=init_box,
                    next_redshift_input=z, # JordanFlitter: added next_redshift_input
                    hooks=hooks,
                    direc=direc,
                    write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                    cleanup=(cleanup)# and iz == (len(scrollz) - 1)), # JordanFlitter: I want to cleanup every iteration!
                )
                # JordanFlitter: added c_T and its friends to output
                if get_c_T and (z in scrollz_cosmic_dawn_output):
                    with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
                        if not user_params.EVOLVE_BARYONS:
                            c_T_Box = (st2.Tk_box-st2.Tk_box.mean())/st2.Tk_box.mean()/pf2.density
                            c_x_e_Box = (st2.x_e_box-st2.x_e_box.mean())/st2.x_e_box.mean()/pf2.density
                            c_T_s_Box = (st2.Ts_box-st2.Ts_box.mean())/st2.Ts_box.mean()/pf2.density
                            c_T_median[iz_CD] = np.percentile(c_T_Box,50)
                            c_x_e_median[iz_CD] = np.percentile(c_x_e_Box,50)
                            c_T_s_median[iz_CD] = np.percentile(c_T_s_Box,50)
                        else:
                            c_T_Box = fftn((st2.Tk_box-st2.Tk_box.mean())/st2.Tk_box.mean())/fftn(pf2.baryons_density)
                            c_x_e_Box = fftn((st2.x_e_box-st2.x_e_box.mean())/st2.x_e_box.mean())/fftn(pf2.baryons_density)
                            c_T_s_Box = fftn((st2.Ts_box-st2.Ts_box.mean())/st2.Ts_box.mean())/fftn(pf2.baryons_density)
                            for k_ind,k_val in enumerate(k_values):
                                good_inds = np.logical_and(k_mag_box < k_val + Delta_k/2., k_mag_box > k_val - Delta_k/2.)
                                c_T_median[k_ind,iz_CD] = np.percentile(np.real(c_T_Box[good_inds]),50)
                                c_x_e_median[k_ind,iz_CD] = np.percentile(np.real(c_x_e_Box[good_inds]),50)
                                c_T_s_median[k_ind,iz_CD] = np.percentile(np.real(c_T_s_Box[good_inds]),50)

            # JordanFlitter: I do not necessarily want to compute the reionization field (and the derived brightness temperature field) at every iteration.
            #                In fact, I do not necessarily want *output* at every iteraion. This can happen if MANY_Z_SAMPLES_AT_COSMIC_DAWN is True.
            if z in scrollz_cosmic_dawn_output:
                ib2 = ionize_box(
                    redshift=z,
                    previous_ionize_box=ib,
                    init_boxes=init_box,
                    perturbed_field=pf2,
                    previous_perturbed_field=prev_perturb,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                    pt_halos=pt_halos if flag_options.USE_HALO_FIELD else None,
                    regenerate=regenerate,
                    hooks=hooks,
                    direc=direc,
                    write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                    cleanup=(cleanup)# and iz == (len(scrollz) - 1)), # JordanFlitter: I want to cleanup every iteration!
                )

                bt2 = brightness_temperature(
                    ionized_box=ib2,
                    perturbed_field=pf2,
                    spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                    hooks=hooks,
                    direc=direc,
                    write=write, # JordanFlitter: added that input so writing to the cache can be disabled (BUG)
                    regenerate=regenerate,
                )
                # JordanFlitter: added c_21 to output
                if get_c_T:
                    with np.errstate(divide='ignore',invalid='ignore'): # Don't show division by 0 warnings
                        if not user_params.EVOLVE_BARYONS:
                            c_21_Box = (bt2.brightness_temp-bt2.brightness_temp.mean())/bt2.brightness_temp.mean()/pf2.density
                            c_21_median[iz_CD] = np.percentile(c_21_Box,50)
                        else:
                            c_21_Box = fftn((bt2.brightness_temp-bt2.brightness_temp.mean())/bt2.brightness_temp.mean())/fftn(pf2.baryons_density)
                            for k_ind,k_val in enumerate(k_values):
                                good_inds = np.logical_and(k_mag_box < k_val + Delta_k/2., k_mag_box > k_val - Delta_k/2.)
                                c_21_median[k_ind,iz_CD] = np.percentile(np.real(c_21_Box[good_inds]),50)

                if coeval_callback is not None and compute_coeval_callback[iz_CD]:
                    coeval = Coeval(
                        redshift=z,
                        initial_conditions=init_box,
                        perturbed_field=pf2,
                        ionized_box=ib2,
                        brightness_temp=bt2,
                        ts_box=st2 if flag_options.USE_TS_FLUCT else None,
                        photon_nonconservation_data=_get_photon_nonconservation_data()
                        if flag_options.PHOTON_CONS
                        else None,
                        _globals=None,
                    )
                    try:
                        coeval_callback_output.append(coeval_callback(coeval))
                    except Exception as e:
                        if sum(compute_coeval_callback[: iz_CD + 1]) == 1:
                            raise RuntimeError(
                                f"coeval_callback computation failed on first trial, z={z}."
                            )
                        else:
                            logger.warning(
                                f"coeval_callback computation failed on z={z}, skipping. {type(e).__name__}: {e}"
                            )

                if write:
                    perturb_files.append((z, os.path.join(direc, pf2.filename)))
                    if flag_options.USE_TS_FLUCT:
                        spin_temp_files.append((z, os.path.join(direc, st2.filename)))
                    ionize_files.append((z, os.path.join(direc, ib2.filename)))
                    brightness_files.append((z, os.path.join(direc, bt2.filename)))

                outs = {
                    "PerturbedField": (pf, pf2),
                    "IonizedBox": (ib, ib2),
                    "BrightnessTemp": (bt, bt2),
                }
                if flag_options.USE_TS_FLUCT:
                    outs["TsBox"] = (st, st2)
                if flag_options.USE_HALO_FIELD:
                    outs["PerturbHaloes"] = pt_halos

                # JordanFlitter: I save coeval boxes if we are in the desired redshift range
                if (save_coeval_redshifts is not None) and z in scrollz_save_coevals:
                    if save_coeval_quantities is not None:
                        coeval_boxes[z] = {}
                        for quantity in save_coeval_quantities:
                            coeval_boxes[z][quantity] = getattr(outs[_fld_names[quantity]][1],quantity)

                # Save mean/global quantities
                for quantity in global_quantities:
                    computed_fields = ['PerturbedField','TsBox']
                    computed_fields.append('IonizedBox')
                    computed_fields.append('BrightnessTemp')
                    # JordanFlitter: added the following condition. If we haven't calculated certain fields, set their mean to zero!
                    if (not quantity in dir(outs[_fld_names[quantity]][1])):
                        global_q[quantity][iz_CD] = 0.
                    else:
                        global_q[quantity][iz_CD] = np.mean(
                            getattr(outs[_fld_names[quantity]][1], quantity)
                        )

                # Interpolate the lightcone
                # JordanFlitter: if we have output during the dark ages, also do the interpolation
                if (z < max_redshift or user_params.OUTPUT_AT_DARK_AGES):
                    for quantity in lightcone_quantities:
                        data1, data2 = outs[_fld_names[quantity]]
                        fnc = interp_functions.get(quantity, "mean")
                        n = _interpolate_in_redshift(
                            iz_CD,
                            box_index,
                            lc_index,
                            n_lightcone,
                            scroll_distances,
                            lc_distances,
                            data1,
                            data2,
                            quantity,
                            lc[quantity],
                            fnc,
                        )
                    lc_index += n
                    box_index += n

            # Save current ones as old ones.
            if flag_options.USE_TS_FLUCT:
                st = st2

            # JordanFlitter: I do not necessarily want to update the previous boxes at every iteration.
            #                Note that the only box we MUST update at each iteraion is the previous spin temperature box (st),
            #                which is why it is outside the if condition below
            if z in scrollz_cosmic_dawn_output:
                ib = ib2
                bt = bt2
                # JordanFlitter: There used to be here a condition of "if flag_options.USE_MINI_HALOS:"
                #                I removed it, because if MINI_HALOS were not used, then perturb_field() would be called again
                #                from ionized_box(), even though the previous_perturbed_field box exists! (BUG?)
                prev_perturb = pf2

                if pf is not None:
                    try:
                        pf.purge(force=always_purge)
                    except OSError:
                        pass

                pf = pf2
                # JordanFlitter: we also need to increment iz_CD
                iz_CD += 1

        if flag_options.PHOTON_CONS:
            photon_nonconservation_data = _get_photon_nonconservation_data()
            if photon_nonconservation_data:
                lib.FreePhotonConsMemory()
        else:
            photon_nonconservation_data = None

        if (
            flag_options.USE_TS_FLUCT
            and user_params.USE_INTERPOLATION_TABLES
            and lib.interpolation_tables_allocated
        ):
            lib.FreeTsInterpolationTables(flag_options())

        # JordanFlitter: compute tau_reio and run CLASS with the updated value
        if user_params.EVALUATE_TAU_REIO:
            if user_params.RUN_CLASS:
                print("Now evaluating tau to reionization and re-running CLASS...")
            else:
                print("Now evaluating tau to reionization...")
            time.sleep(0.1) # we pause the program for a short time just to print the above message before running CLASS

            z_array = np.array([z_at_value(cosmo_params.cosmo.comoving_distance, d * units.Mpc, zmax=1e6) for d in lc_distances + d_at_redshift])
            if user_params.EVOLVE_BARYONS:
                cosmo_params.tau_reio = compute_tau_reio(z_array, lc['baryons_density'], lc['xH_box'], cosmo_params)
            else:
                cosmo_params.tau_reio = compute_tau_reio(z_array, lc['density'], lc['xH_box'], cosmo_params)
            # Run CLASS again, this time with the updated value of tau_reio
            if user_params.RUN_CLASS:
                CLASS_params = {'Omega_b': cosmo_params.OMb, 'Omega_cdm': cosmo_params.OMm-cosmo_params.OMb, 'h': cosmo_params.hlittle,
                                'A_s': cosmo_params.A_s, 'n_s': cosmo_params.POWER_INDEX, 'tau_reio': cosmo_params.tau_reio,
                                'm_ncdm': 0.06, 'N_ncdm': 1, 'N_ur': 2.0308, 'T_cmb': 2.728, 'output': 'tCl,pCl,lCl,mTk,vTk',
                                'l_max_scalars': 3000, 'lensing': 'yes'}
                if user_params.FUZZY_DM:
                    # Set FDM parameters
                    H_0_eV = cosmo_params.hlittle * 2.1331192697532067e-33  # Hubble constant in eV
                    CLASS_params['Omega_cdm'] = (1.-pow(10.,-cosmo_params.f_FDM))*(cosmo_params.OMm-cosmo_params.OMb)
                    CLASS_params['Omega_scf'] = pow(10.,-cosmo_params.f_FDM) * (cosmo_params.OMm-cosmo_params.OMb)
                    CLASS_params['m_axion'] = pow(10.,-cosmo_params.m_FDM) / H_0_eV # FDM mass in units of H0 (dimensionless)
                    # Fix other FDM parameters/flags
                    CLASS_params['f_axion'] = 1 # # Axion energy scale in units of Planck mass
                    CLASS_params['n_axion'] = 1 # Power-law in the axion potential
                    CLASS_params['scf_parameters'] = '2, 0' # Scalar field (scf) potential parameters: first one is lambda, second is alpha (see arXiv: astro-ph/9908085)
                    CLASS_params['scf_potential'] = 'axion' # Scalar field potential type
                    CLASS_params['scf_has_perturbations'] = 'yes' # Perturbations in scalar field
                    CLASS_params['attractor_ic_scf'] = 'no' # Scalar field (scf) initial conditions from attractor solution
                    CLASS_params['scf_tuning_index'] = 0 # Index in scf_parameters to do shooting
                    # Fix other FDM parameters/flags - according to https://github.com/PoulinV/AxiCLASS/blob/master/example_axion.ipynb
                    CLASS_params['do_shooting'] = 'yes'
                    CLASS_params['do_shooting_scf'] = 'yes'
                    CLASS_params['tol_shooting_deltax'] = 1e-4
                    CLASS_params['tol_shooting_deltaF'] = 1e-4
                    CLASS_params['scf_evolve_as_fluid'] = 'yes'
                    CLASS_params['scf_evolve_like_axionCAMB'] = 'yes'
                    CLASS_params['write background'] = 'yes'
                    CLASS_params['threshold_scf_fluid_m_over_H'] = 3
                    CLASS_params['include_scf_in_delta_m'] = 'yes'
                    CLASS_params['include_scf_in_delta_cb'] = 'yes'
                    CLASS_params['use_big_theta_scf'] = 'no'
                    CLASS_params['use_delta_scf_over_1plusw'] = 'no'
                    CLASS_params['background_verbose'] = 0
                if user_params.SCATTERING_DM:
                    # Set SDM parameters
                    CLASS_params['Omega_cdm'] = (1.-pow(10.,-cosmo_params.f_chi))*(cosmo_params.OMm-cosmo_params.OMb)
                    CLASS_params['Omega_dmeff'] = pow(10.,-cosmo_params.f_chi)*(cosmo_params.OMm-cosmo_params.OMb) # ratio of SDM to total DM
                    CLASS_params['m_dmeff'] = pow(10.,cosmo_params.m_chi)*1.e-9 # in GeV
                    CLASS_params['sigma_dmeff'] = pow(10.,-cosmo_params.sigma_SDM) # cross section prefactor in cm^2
                    CLASS_params['npow_dmeff'] = -4 # power-law of the cross section dependence on the relative velocity
                    # Type of the interacting particles with the SDM (can be 'baryons', 'ionized', 'hydrogen', 'protons' or 'electrons')
                    if user_params.SDM_TARGET_TYPE == 1:
                        CLASS_params['dmeff_target'] = 'baryons'
                    elif user_params.SDM_TARGET_TYPE == 2:
                        CLASS_params['dmeff_target'] = 'ionized'
                    elif user_params.SDM_TARGET_TYPE == 3:
                        CLASS_params['dmeff_target'] = 'hydrogen'
                    elif user_params.SDM_TARGET_TYPE == 4:
                        CLASS_params['dmeff_target'] = 'protons'
                    elif user_params.SDM_TARGET_TYPE == 5:
                        CLASS_params['dmeff_target'] = 'electrons'
                    else:
                        raise ValueError("SDM_TARGET_TYPE must be 1 (baryons), 2 (ionized), 3 (hydrogen), 4 (protons) or 5 (electrons)")
                    # Fix other SDM parameters/flags
                    CLASS_params['Vrel_dmeff'] = 30 # This is the initial guess for the mean relative velocity between baryons and SDM

                # Import the appropriate CLASS based on the user's input
                if user_params.FUZZY_DM:
                    from axiclassy import Class
                elif user_params.SCATTERING_DM:
                    from dmeff_classy import Class
                else:
                    from classy import Class
                # Run CLASS!
                CLASS_OUTPUT = Class()
                CLASS_OUTPUT.set(CLASS_params)
                CLASS_OUTPUT.compute()
                Cl_data = CLASS_OUTPUT.lensed_cl(3000)

        out = (
            LightCone(
                redshift,
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                init_box.random_seed,
                lc,
                node_redshifts=scrollz_all_output,
                global_quantities=global_q,
                photon_nonconservation_data=photon_nonconservation_data,
                _globals=dict(global_params.items()),
                cache_files={
                    "init": [(0, os.path.join(direc, init_box.filename))],
                    "perturb_field": perturb_files,
                    "ionized_box": ionize_files,
                    "brightness_temp": brightness_files,
                    "spin_temp": spin_temp_files,
                },
                Cl_data=Cl_data, # JordanFlitter: added Cl_data to lightcone structure
                c_T_median=c_T_median, # JordanFlitter: added c_T to lightcone structure
                c_x_e_median=c_x_e_median, # JordanFlitter: added c_x_e to lightcone structure
                c_T_s_median=c_T_s_median, # JordanFlitter: added c_T_s to lightcone structure
                c_21_median=c_21_median, # JordanFlitter: added c_21 to lightcone structure
                coeval_boxes=coeval_boxes, # JordanFlitter: added coeval_boxes to lightcone structure
            ),
            coeval_callback_output,
        )
        if coeval_callback is None:
            return out[0]
        else:
            return out

# JordanFlitter: new function to compute tau_reio, based on arXiv: 2305.07056
def compute_tau_reio(z_array, density_box, xH_box, cosmo_params):
    Mpc_to_meter = 3.085677581282e22
    c = 2.99792458e8 # Speed of light in m/s
    G = 6.674e-11 # Newton gravitational constant in N*m^2/kg^2
    sigma_T = 6.6524616e-29 # Thomson cross-section in m^2
    m_p = 1.6735575e-27 # Proton mass in kg
    _not4_ = 3.9715 # This is the ratio between Helium to Hydrogen mass. It is not 4!
    Y_He = global_params.Y_He
    h = cosmo_params.hlittle
    Omega_b0 = cosmo_params.OMb
    Omega_m0 = cosmo_params.OMm
    Omega_Lambda = 1. - Omega_m0 # Dark energy portion
    H_0 = 1e5*h/Mpc_to_meter # Hubble constant in 1/sec
    mu = 1./(1. - Y_He*(1.-1./_not4_))
    integrand = (1. + z_array)**2 / np.sqrt(Omega_Lambda + Omega_m0*(1. + z_array)**3) * np.mean(np.mean((1.+density_box)*(1.-xH_box),0),0)
    integral = np.trapz(x = z_array[z_array <= global_params.Z_HEAT_MAX], y = integrand[z_array <= global_params.Z_HEAT_MAX])
    z_low_array = np.linspace(0,min(z_array),100)
    integrand_low = (1. + z_low_array)**2 / np.sqrt(Omega_Lambda + Omega_m0*(1. + z_low_array)**3)
    integral += np.trapz(x = z_low_array, y = integrand_low)
    tau_reio = 3.*H_0*Omega_b0*sigma_T*c / (8.*np.pi*G*m_p*mu)*integral # (m^3/sec^2)/(N*m^2/kg) = (kg*m/sec^2)/N = dimensionless!
    return tau_reio

def _get_coeval_callbacks(
    scrollz: List[float], coeval_callback, coeval_callback_redshifts
) -> List[bool]:

    compute_coeval_callback = [False for i in range(len(scrollz))]
    if coeval_callback is not None:
        if isinstance(coeval_callback_redshifts, (list, np.ndarray)):
            for coeval_z in coeval_callback_redshifts:
                assert isinstance(coeval_z, (int, float, np.number))
                compute_coeval_callback[
                    np.argmin(np.abs(np.array(scrollz) - coeval_z))
                ] = True
            if sum(compute_coeval_callback) != len(coeval_callback_redshifts):
                logger.warning(
                    "some of the coeval_callback_redshifts refer to the same node_redshift"
                )
        elif (
            isinstance(coeval_callback_redshifts, int) and coeval_callback_redshifts > 0
        ):
            compute_coeval_callback = [
                not i % coeval_callback_redshifts for i in range(len(scrollz))
            ]
        else:
            raise ValueError("coeval_callback_redshifts has to be list or integer > 0.")

    return compute_coeval_callback


def _get_interpolation_outputs(
    lightcone_quantities: Sequence,
    global_quantities: Sequence,
    flag_options: FlagOptions,
) -> Dict[str, str]:
    _fld_names = get_all_fieldnames(arrays_only=True, lightcone_only=True, as_dict=True)

    incorrect_lc = [q for q in lightcone_quantities if q not in _fld_names.keys()]
    if incorrect_lc:
        raise ValueError(
            f"The following lightcone_quantities are not available: {incorrect_lc}"
        )

    incorrect_gl = [q for q in global_quantities if q not in _fld_names.keys()]
    if incorrect_gl:
        raise ValueError(
            f"The following global_quantities are not available: {incorrect_gl}"
        )

    if not flag_options.USE_TS_FLUCT and any(
        _fld_names[q] == "TsBox" for q in lightcone_quantities + global_quantities
    ):
        raise ValueError(
            "TsBox quantity found in lightcone_quantities or global_quantities, "
            "but not running spin_temp!"
        )

    return _fld_names


def _interpolate_in_redshift(
    z_index,
    box_index,
    lc_index,
    n_lightcone,
    scroll_distances,
    lc_distances,
    output_obj,
    output_obj2,
    quantity,
    lc,
    kind="mean",
):
    try:
        array = getattr(output_obj, quantity)
        array2 = getattr(output_obj2, quantity)
    except AttributeError:
        raise AttributeError(
            f"{quantity} is not a valid field of {output_obj.__class__.__name__}"
        )

    assert array.__class__ == array2.__class__

    # Do linear interpolation only.
    prev_d = scroll_distances[z_index - 1]
    this_d = scroll_distances[z_index]

    # Get the cells that need to be filled on this iteration.
    these_distances = lc_distances[
        np.logical_and(lc_distances < prev_d, lc_distances >= this_d)
    ]

    n = len(these_distances)
    # JordanFlitter: fill the lightcone elements only if n > 0.
    # It could be that the difference between adjacent samples in scroll_distances
    # is less than BOX_LEN/HII_DIM (the cell size) and then n=0. This can happen if either the cell
    # size is large or if we are at the highest redshifts. If this happens, we ignore
    # the current coeval box and proceed as if it has not been calculated. The lightcone box
    # will then be filled once n > 0. Afterwards, n is guaranteed to increase so we don't have
    # this problem anymore
    if n > 0:
        ind = np.arange(-(box_index + n), -box_index)

        sub_array = array.take(ind + n_lightcone, axis=2, mode="wrap")
        sub_array2 = array2.take(ind + n_lightcone, axis=2, mode="wrap")

        out = (
            np.abs(this_d - these_distances) * sub_array
            + np.abs(prev_d - these_distances) * sub_array2
        ) / (np.abs(prev_d - this_d))
        if kind == "mean_max":
            flag = sub_array * sub_array2 < 0
            out[flag] = np.maximum(sub_array, sub_array2)[flag]
        elif kind != "mean":
            raise ValueError("kind must be 'mean' or 'mean_max'")

        lc[:, :, -(lc_index + n) : n_lightcone - lc_index] = out
    return n


def _setup_lightcone(
    cosmo_params, max_redshift, redshift, scrollz, user_params, z_step_factor
):
    # Here set up the lightcone box.
    # Get a length of the lightcone (bigger than it needs to be at first).
    d_at_redshift = cosmo_params.cosmo.comoving_distance(redshift).value
    Ltotal = (
        cosmo_params.cosmo.comoving_distance(scrollz[0] * z_step_factor).value
        - d_at_redshift
    )
    lc_distances = np.arange(0, Ltotal, user_params.BOX_LEN / user_params.HII_DIM)

    # Use max_redshift to get the actual distances we require.
    Lmax = cosmo_params.cosmo.comoving_distance(max_redshift).value - d_at_redshift
    first_greater = np.argwhere(lc_distances > Lmax)[0][0]

    # Get *at least* as far as max_redshift
    lc_distances = lc_distances[: (first_greater + 1)]

    n_lightcone = len(lc_distances)
    return d_at_redshift, lc_distances, n_lightcone


def _get_lightcone_redshifts(
    cosmo_params, max_redshift, redshift, user_params, z_step_factor
):
    scrollz = _logscroll_redshifts(redshift, z_step_factor, max_redshift)
    lc_distances = _setup_lightcone(
        cosmo_params, max_redshift, redshift, scrollz, user_params, z_step_factor
    )[1]
    lc_distances += cosmo_params.cosmo.comoving_distance(redshift).value

    return np.array(
        [
            z_at_value(cosmo_params.cosmo.comoving_distance, d * units.Mpc)
            for d in lc_distances
        ]
    )


def calibrate_photon_cons(
    user_params,
    cosmo_params,
    astro_params,
    flag_options,
    init_box,
    regenerate,
    write,
    direc,
    **global_kwargs,
):
    r"""
    Set up the photon non-conservation correction.

    Scrolls through in redshift, turning off all flag_options to construct a 21cmFAST calibration
    reionisation history to be matched to the analytic expression from solving the filling factor
    ODE.


    Parameters
    ----------
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options: :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Other Parameters
    ----------------
    regenerate, write
        See docs of :func:`initial_conditions` for more information.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, {})

    if not flag_options.PHOTON_CONS:
        return

    with global_params.use(**global_kwargs):
        # Create a new astro_params and flag_options just for the photon_cons correction
        astro_params_photoncons = deepcopy(astro_params)
        astro_params_photoncons._R_BUBBLE_MAX = astro_params.R_BUBBLE_MAX

        flag_options_photoncons = FlagOptions(
            USE_MASS_DEPENDENT_ZETA=flag_options.USE_MASS_DEPENDENT_ZETA,
            M_MIN_in_Mass=flag_options.M_MIN_in_Mass,
            USE_VELS_AUX=user_params.USE_RELATIVE_VELOCITIES,
        )

        ib = None
        prev_perturb = None

        # Arrays for redshift and neutral fraction for the calibration curve
        z_for_photon_cons = []
        neutral_fraction_photon_cons = []

        # Initialise the analytic expression for the reionisation history
        logger.info("About to start photon conservation correction")
        _init_photon_conservation_correction(
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
        )

        # Determine the starting redshift to start scrolling through to create the
        # calibration reionisation history
        logger.info("Calculating photon conservation zstart")
        z = _calc_zstart_photon_cons()

        while z > global_params.PhotonConsEndCalibz:

            # Determine the ionisation box with recombinations, spin temperature etc.
            # turned off.
            this_perturb = perturb_field(
                redshift=z,
                init_boxes=init_box,
                regenerate=regenerate,
                # !!! SLTK: added astro_params and flag_options specified to be coherent
                astro_params=astro_params,
                flag_options=flag_options,
                hooks=hooks,
                direc=direc,
            )

            ib2 = ionize_box(
                redshift=z,
                previous_ionize_box=ib,
                init_boxes=init_box,
                perturbed_field=this_perturb,
                previous_perturbed_field=prev_perturb,
                astro_params=astro_params_photoncons,
                flag_options=flag_options_photoncons,
                spin_temp=None,
                regenerate=regenerate,
                hooks=hooks,
                direc=direc,
            )

            mean_nf = np.mean(ib2.xH_box)

            # Save mean/global quantities
            neutral_fraction_photon_cons.append(mean_nf)
            z_for_photon_cons.append(z)

            # Can speed up sampling in regions where the evolution is slower
            if 0.3 < mean_nf <= 0.9:
                z -= 0.15
            elif 0.01 < mean_nf <= 0.3:
                z -= 0.05
            else:
                z -= 0.5

            ib = ib2
            if flag_options.USE_MINI_HALOS:
                prev_perturb = this_perturb

        z_for_photon_cons = np.array(z_for_photon_cons[::-1])
        neutral_fraction_photon_cons = np.array(neutral_fraction_photon_cons[::-1])

        # Construct the spline for the calibration curve
        logger.info("Calibrating photon conservation correction")
        _calibrate_photon_conservation_correction(
            redshifts_estimate=z_for_photon_cons,
            nf_estimate=neutral_fraction_photon_cons,
            NSpline=len(z_for_photon_cons),
        )
