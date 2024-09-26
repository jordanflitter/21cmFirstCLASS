"""
This module is responsible for generating the initial conditions that are fed to 21cmFAST.
These initial conditions include transfer functions (e.g. matter transfer function at z=0,
relative velocity transfer function at kinematic decoupling), background quantities as
functions of redshift (e.g. T_k, x_e, D(z)) and some derived parameters (e.g. sigma8, YHe, Z_REC).
The module also supports exotic dark matter models like fuzzy dark matter (FDM)
and scattering dark matter (SDM).
"""

import tqdm
import numpy as np
import scipy.integrate as intg
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

#####################################################################################################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Define some global parameters and useful functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####################################################################################################################################################

Mpc_to_meter = 3.085677581282e22
Gyr_to_second = 3.1556952e16
Hz_to_eV = 6.582118308878102e-16
c = 2.99792458e8 # Speed of light in m/s
h_P = 6.62606896e-34 # Planck Constant in J*sec
k_B = 1.3806504e-23 # Boltzmann Constant in J/K
sigma_T = 6.6524616e-29 # Thomson cross-section in m^2
m_e = 9.10938215e-31 # Electron mass in kg
m_p = 1.6735575e-27 # Proton mass in kg
Tcmb0 = 2.728 # CMB temperature in K
_not4_ = 3.9715 # This is the ratio between Helium to Hydrogen mass. It is not 4!

# Convert redshift to time in Gyr
def z_to_time(z,cosmo_params):
    Omega_m0 = cosmo_params[2]
    Omega_Lambda = cosmo_params[8]
    H_0 = cosmo_params[9] # 1/sec
    t = (2./3./H_0)*np.sqrt(1.+Omega_m0/Omega_Lambda)*np.arcsinh(np.sqrt(Omega_Lambda/Omega_m0)*pow(1+z,-3./2.)) # sec
    # This is an approximation valid in matter domination
    # t = (2./3./H_0)/(Omega_c0+Omega_b0)**0.5/(1.+z)**1.5 # sec
    return t/Gyr_to_second

# Convert time in Gyr to redshift
def time_to_z(t,cosmo_params): # Give time in Gyr
    Omega_m0 = cosmo_params[2]
    Omega_Lambda = cosmo_params[8]
    H_0 = cosmo_params[9] # 1/sec
    z = pow(np.sinh(t*Gyr_to_second/((2./3./H_0)*np.sqrt(1.+Omega_m0/Omega_Lambda)))*np.sqrt(Omega_m0/Omega_Lambda),-2./3.) - 1.
    # This is an approximation valid in matter domination
    # z = ((2./3./H_0)/(Omega_c0+Omega_b0)**0.5/t)**(2./3.) - 1.
    return z

# Function that interpolates a CLASS transfer function.
# Extrapolation is also performed if necessary
def Interpolate_transfer(T_CLASS,k_CLASS,k_output):
    # Interpolate the trnasfer function at k_output
    Transfer_interp = interp1d(k_CLASS, T_CLASS, kind='cubic',
                                bounds_error=False,fill_value=0.)
    Transfer = Transfer_interp(k_output)
    # If necessary, extrapolate the transfer function beyond its extent
    # (in a power-law fashion)
    if max(k_output) > max(k_CLASS):
        ind = np.where(k_output > k_CLASS[-1])[0][0] - 1
        if (abs(Transfer[ind]) > 0.) and (abs(Transfer[ind-1]) > 0.):
            slope = np.log(Transfer[ind]/Transfer[ind-1])/np.log(k_output[ind]/k_output[ind-1])
            Transfer[ind:] = Transfer[ind]*pow((k_output[ind:]/k_output[ind]),slope)
    if min(k_output) < min(k_CLASS):
        ind = np.where(k_output < k_CLASS[0])[0][-1] + 1
        if (abs(Transfer[ind]) > 0.) and (abs(Transfer[ind+1]) > 0.):
            slope = np.log(Transfer[ind]/Transfer[ind+1])/np.log(k_output[ind]/k_output[ind+1])
            Transfer[:ind] = Transfer[ind]*pow((k_output[:ind]/k_output[ind]),slope)
    # Return output
    return abs(Transfer)

# Fitting function for the v_cb contribution to the matter power spectrum
def fitting_function(k, Ap, kp, sigma_p):
	return 1.-Ap*np.exp(-(np.log(k/kp))**2/(2.*sigma_p**2))

#####################################################################################################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Differential equations for vcb correction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####################################################################################################################################################

# This is the model for the differential equations for finding the vcb correction to the matter power spectrum
# Written in collaboration with Debanjan Sarkar
def model(t,y,COSMO_PARAMS,CLASS_OUTPUT,ODE_params):
    # Extract variables from y and cosmo_params
    theta_b_real = y[0]; theta_b_imag = y[1]
    theta_c_real = y[2]; theta_c_imag = y[3]
    delta_b_real = y[4]; delta_b_imag = y[5]
    delta_c_real = y[6]; delta_c_imag = y[7]
    delta_T_real = y[8]; delta_T_imag = y[9]
    h, Omega_b0, Omega_m0, Tcmb0, ns, sigma8, m_FDM = COSMO_PARAMS[0:7]
    Omega_c0, Omega_Lambda, H_0, fHe, mu_b, z_kin = COSMO_PARAMS[7:13]
    delta_m_real = (Omega_c0*delta_c_real + Omega_b0*delta_b_real)/Omega_m0
    delta_m_imag = (Omega_c0*delta_c_imag + Omega_b0*delta_b_imag)/Omega_m0
    # Extract parameters for the differential equations
    vcb_kin = ODE_params[0]*1000.*Gyr_to_second # m/Gyr
    k = ODE_params[1]/Mpc_to_meter # 1/m
    mu = ODE_params[2]
    # Calculate some time dependent quantities
    z = time_to_z(t,COSMO_PARAMS)
    a = 1./(1.+z)
    Hz = CLASS_OUTPUT.Hubble(z)*c/Mpc_to_meter*Gyr_to_second # 1/Gyr
    H0 = H_0*Gyr_to_second # 1/Gyr
    Tg = CLASS_OUTPUT.baryon_temperature(z) # K
    Tcmb = Tcmb0*(1.+z) # K
    cs_sq_b = (Tg*k_B/mu_b/m_p)*(Gyr_to_second)**2. # m^2/Gyr^2
    xe = CLASS_OUTPUT.ionization_fraction(z)
    Gamma_c = 64. * np.pi**5. * sigma_T * (k_B*Tcmb)**4.
    Gamma_c /= 45.* h_P**3. * c**4. * m_e
    Gamma_c *= xe/(1.+xe+fHe)
    Gamma_c *= Gyr_to_second # 1/Gyr
    vcb = vcb_kin*(1.+z)/(1+z_kin) # m/Gyr
    # The diffential equations to be solved, see e.g. Eq. (11) in arXiv: 1005.2416, or Eq. (4) in arXiv: 1904.07881
    theta_b_real_dt = (-2.*Hz*theta_b_real - (3./2.)*(H0**2./a**3.)*Omega_m0*delta_m_real
                       + cs_sq_b*k*k*(delta_b_real+delta_T_real)/a**2. + (vcb*k*mu/a)*theta_b_imag)

    theta_b_imag_dt = (-2.*Hz*theta_b_imag - (3./2.)*(H0**2./a**3.)*Omega_m0*delta_m_imag
                       + cs_sq_b*k*k*(delta_b_imag+delta_T_imag)/a**2. - (vcb*k*mu/a)*theta_b_real)

    theta_c_real_dt = -2.*Hz*theta_c_real - (3./2.)*(H0**2./a**3.)*Omega_m0*delta_m_real

    theta_c_imag_dt = -2.*Hz*theta_c_imag - (3./2.)*(H0**2./a**3.)*Omega_m0*delta_m_imag

    delta_b_real_dt = -theta_b_real  + (vcb*k*mu/a)*delta_b_imag

    delta_b_imag_dt = -theta_b_imag  - (vcb*k*mu/a)*delta_b_real

    delta_c_real_dt = -theta_c_real

    delta_c_imag_dt = -theta_c_imag

    delta_T_real_dt = (2./3.)*delta_b_real_dt - (Tcmb/Tg)*Gamma_c*delta_T_real

    delta_T_imag_dt = (2./3.)*delta_b_imag_dt - (Tcmb/Tg)*Gamma_c*delta_T_imag

    # This is an attempt to model FDM in the spirit of Hotinli et al. (arXiv: 2112.06943)
    # Leave it commented as it doesn't seem to give the correct matter power spectrum
    '''if FUZZY_DM == 6:
        cs_sq_a = (c*Gyr_to_second)**2 *(hbar*c*k/(2.*m_FDM*eV_to_Joules*a))**2 # m^2/Gyr^2
        if (cs_sq_a > (c*Gyr_to_second)**2):
            cs_sq_a = (c*Gyr_to_second)**2 # m^2/Gyr^2
        theta_c_real_dt += cs_sq_a*k*k*delta_c_real/a**2.
        theta_c_imag_dt += cs_sq_a*k*k*delta_c_imag/a**2.
    '''
    # Prepare output
    dy_dt = [theta_b_real_dt, theta_b_imag_dt,
             theta_c_real_dt, theta_c_imag_dt,
             delta_b_real_dt, delta_b_imag_dt,
             delta_c_real_dt, delta_c_imag_dt,
             delta_T_real_dt, delta_T_imag_dt]
    return dy_dt

#####################################################################################################################################################
#####################################################################################################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####################################################################################################################################################
#####################################################################################################################################################

# This is the main function for generating the initial conditions from CLASS
def run_ICs(cosmo_params,user_params,global_params):
    """
    Generate initial conditions for the simulation.

    This function modifies the input structures based on the output of CLASS.
    If fuzzy dark matter is considered, then AxionCAMB is called for computing
    the matter transfer function.
    The function computes interpolation tables used in the simulation, like
    transfer functions (matter density at z=0, and relative velocity between
    baryons and CDM during kinematic decoupling - vcb), background quantities (gas
    kinetic temperature, free electron fraction, scale independet growth factor)
    and derived cosmological parameters (helium density fraction, sigma8, redshift
    of kinetmatic decoupling and the average relative velocity at kinematic
    decoupling). If scale dependent growth is considered via the EVOLVE_BARYONS
    flag, the scale dependent growth factor is computed. If scattering dark matter
    is considered, then the function also computes associated relevant quantities
    (e.g. the SDM temperature). Finally, the function allows the computation of
    the best fit values for the vcb correction to the matter power spectrum if
    DO_VCB_FIT is turned on.

    Parameters
    ----------
    cosmo_params : :class:`~CosmoParams`
        Defines the cosmological parameters used to compute initial conditions.
    user_params : :class:`~UserParams`
        Defines the overall options and parameters of the run.
    global_params : :class:`~GlobalParams`
        Defines the global parameters of the run.

    Returns
    -------
    Dictionary of lensed CMB power spectrum from CLASS.
    """

    # Extract cosmological parameters
    h = cosmo_params.hlittle
    Omega_b0 = cosmo_params.OMb
    Omega_m0 = cosmo_params.OMm
    ns = cosmo_params.POWER_INDEX
    A_s = cosmo_params.A_s
    tau_reio = cosmo_params.tau_reio
    m_FDM = pow(10.,-cosmo_params.m_FDM) # eV
    f_FDM = pow(10.,-cosmo_params.f_FDM)
    m_chi = pow(10.,cosmo_params.m_chi)*1.e-9 # GeV
    f_chi = pow(10.,-cosmo_params.f_chi)
    sigma_SDM = pow(10.,-cosmo_params.sigma_SDM) # cross section prefactor in units of cm^2
    n_SDM = cosmo_params.SDM_INDEX # cross section index
    # Define these parameters as well
    Omega_c0 = Omega_m0 - Omega_b0 # CDM portion
    Omega_Lambda = 1. - Omega_m0 # Dark energy portion
    H_0 = 1e5*h/Mpc_to_meter # Hubble constant in 1/sec

    ################################################################################################################################################
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS section %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ################################################################################################################################################

    # Set CLASS parameters
    CLASS_params = {}
    CLASS_params['h'] = h
    if user_params.SCATTERING_DM:
        CLASS_params['Omega_cdm'] = (1.-f_chi)*Omega_c0
    elif user_params.FUZZY_DM:
        CLASS_params['Omega_cdm'] = (1.-f_FDM)*Omega_c0
    else:
        CLASS_params['Omega_cdm'] = Omega_c0
    CLASS_params['Omega_b'] = Omega_b0
    CLASS_params['A_s'] = A_s
    CLASS_params['n_s'] = ns
    CLASS_params['tau_reio'] = tau_reio
    CLASS_params['T_cmb'] = Tcmb0
    CLASS_params['m_ncdm'] = "0.06" # eV
    CLASS_params['N_ncdm'] = 1
    CLASS_params['N_ur'] = 2.0308
    CLASS_params['output'] = 'tCl,pCl,lCl,mTk,vTk,mPk'
    CLASS_params['lensing'] = 'yes'
    CLASS_params['z_pk'] = 1087.
    CLASS_params['l_max_scalars'] = 3000
    # We need to run CLASS for very large wavenumbers. This is required for computing sigma(M) and the HMF
    CLASS_params['P_k_max_1/Mpc'] = 1200.
    if user_params.FUZZY_DM:
        # Set FDM parameters
        H_0_eV = H_0 * Hz_to_eV  # Hubble constant in eV
        CLASS_params['Omega_scf'] = f_FDM * Omega_c0
        CLASS_params['m_axion'] = m_FDM / H_0_eV # FDM mass in units of H0 (dimensionless)
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
        CLASS_params['Omega_dmeff'] = f_chi*Omega_c0 # ratio of SDM to total DM
        CLASS_params['m_dmeff'] = m_chi # SDM mass in GeV
        CLASS_params['sigma_dmeff'] = sigma_SDM # cross section prefactor in cm^2
        CLASS_params['npow_dmeff'] = n_SDM # power-law index for the cross section dependence on the relative velocity
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

    # Extract these parameters from CLASS
    sigma8 = CLASS_OUTPUT.sigma8()
    YHe = CLASS_OUTPUT.get_current_derived_parameters(['YHe'])['YHe'] # He mass fraction (rho_He/(rho_H+rho_He)=rho_He/rho_b)
    fHe = YHe/(_not4_*(1.-YHe)) # He number fraction (n_He/n_H)
    mu_b = 1./(1.-(1.-1./_not4_)*YHe) # Mean molecular weight (assuming neutral IGM, this is rho_b/(n_b*m_H))

    # Find the redshift of kinematic decoupling (or more precisely, at recombination, where x_e = 0.1)
    z_samples = np.arange(800.,2000.+10.,10.)
    xe_samples = np.zeros(len(z_samples))
    for index, z_sample in enumerate(z_samples):
        # Note: Class returns n_e/n_H, but for 21cmFAST we need n_e/(n_H+n_He),
        # this is why we multiply by n_H/(n_H+n_He)=(1-YHe)/(1-(1.-1./_not4_)*YHe)
        xe_samples[index] = CLASS_OUTPUT.ionization_fraction(z_sample)*(1.-YHe)/(1.-(1.-1./_not4_)*YHe)
    redshift_interp = interp1d(xe_samples,z_samples,kind='cubic')
    z_kin = float(redshift_interp(0.1))

    ################################################################################################################################################
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Transfer functions section %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ################################################################################################################################################

    # Get from CLASS the matter transfer function at z=0
    Transfer_0 = CLASS_OUTPUT.get_transfer(z=0.)
    k_CLASS = Transfer_0['k (h/Mpc)'][:]*h # 1/Mpc
    delta_m_0_CLASS = Transfer_0['d_tot'][:]
    # This is the wavenumber grid for the transfer functions
    k_output = pow(10.,np.array(global_params.LOG_K_ARR_FOR_TRANSFERS)) # 1/Mpc
    # Interpolate the matter transfer function at the desired wavenumbers
    delta_m_0 = Interpolate_transfer(delta_m_0_CLASS,k_CLASS,k_output)

    # Get from CLASS the transfer functions at kinematic decoupling (as function of k)
    # Note: CLASS default gauge is the synchronous gauge, where theta_c = 0
    Transfer_kin = CLASS_OUTPUT.get_transfer(z=z_kin)
    delta_c_kin_CLASS = Transfer_kin['d_cdm'][:]
    delta_b_kin_CLASS = Transfer_kin['d_b'][:]
    theta_b_kin_CLASS = Transfer_kin['t_b'][:]*c/Mpc_to_meter # 1/sec
    theta_c_kin_CLASS = np.zeros_like(theta_b_kin_CLASS) # 1/sec

    delta_phi_CLASS = Transfer_kin['phi'][:] / h #!!! SLKF potential for non gauss

    # Interpolate transfer functions at the desired wavenumbers
    delta_c_kin = Interpolate_transfer(delta_c_kin_CLASS,k_CLASS,k_output)
    delta_b_kin = Interpolate_transfer(delta_b_kin_CLASS,k_CLASS,k_output)
    theta_c_kin = Interpolate_transfer(theta_c_kin_CLASS,k_CLASS,k_output)
    theta_b_kin = Interpolate_transfer(theta_b_kin_CLASS,k_CLASS,k_output)
    theta_b_kin = Interpolate_transfer(theta_b_kin_CLASS,k_CLASS,k_output)

    delta_phi = Interpolate_transfer(delta_phi_CLASS,k_CLASS,k_output) # !!! SLKF potential for non gauss


    # Calculate v_cb at kinematic decoupling (as a function of k)
    v_cb_kin = (theta_b_kin-theta_c_kin)/(k_output/Mpc_to_meter)/1000. # km/sec
    # Find the transfer function of v_cb (which is consistent with 21cmFAST).
    # In 21cmFAST, the transfer function is multiplied by c_KMS = c/1000
    # (the speed of light in km/sec) to obtain the v_cb box, which has units of km/sec
    T_vcb_kin = abs(v_cb_kin)/(c/1000.)

    # More transfer functions, for SDM
    # JordanFlitterTODO: we calculate the transfer functions at Z_HIGH_MAX but in fact the required initial conditions in 21cmFAST
    #                    have to be given in a slightly different redshift. Fix that in the future
    if user_params.SCATTERING_DM and user_params.USE_SDM_FLUCTS:
        # First, V_chi_b transfer function at Z_HIGH_MAX
        if user_params.Z_HIGH_MAX < 0.:
            z_high = z_kin
        else:
            z_high = user_params.Z_HIGH_MAX
        Transfer_zhigh = CLASS_OUTPUT.get_transfer(z=z_high)
        theta_b_zhigh_CLASS = Transfer_zhigh['t_b'][:]*c/Mpc_to_meter # 1/sec
        theta_chi_zhigh_CLASS = Transfer_zhigh['t_dmeff'][:]*c/Mpc_to_meter # 1/sec
        theta_b_zhigh = Interpolate_transfer(theta_b_zhigh_CLASS,k_CLASS,k_output)
        theta_chi_zhigh = Interpolate_transfer(theta_chi_zhigh_CLASS,k_CLASS,k_output)
        v_chi_b_zhigh = (theta_b_zhigh-theta_chi_zhigh)/(k_output/Mpc_to_meter)/1000. # km/sec
        T_v_chi_b_zhigh = abs(v_chi_b_zhigh)/(c/1000.)
        # JordanFlitterTODO: get the transfer functions of T_k, T_chi, and x_e at z_kin
        T_Tk_zhigh = np.zeros_like(k_output)
        T_xe_zhigh = np.zeros_like(k_output)
        T_Tchi_zhigh = np.zeros_like(k_output)

    # Calculate the rms and avg value of v_cb at kinematic decoupling
    v_cb_rms_kin = np.sqrt(intg.simps(A_s*pow(k_output/0.05,ns-1.)*(v_cb_kin**2.)/k_output, x=k_output)) # km/sec
    v_cb_avg_kin = np.sqrt(8./(3.*np.pi))*v_cb_rms_kin # km/sec

    ################################################################################################################################################
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Background values section %%%%%%%$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ################################################################################################################################################

    # JordanFlitterTODO: Most of these background values are required for interpolating the initial conditions at the first redshift iteration.
    #                    This is quite an overkill. Better implementation would be to pass 21cmFAST the initial value.

    # This is the redshift grid for our interpolation tables
    log_z_array = np.array(global_params.LOG_Z_ARR)
    # Gas kinetic temperature
    T_k = np.array([CLASS_OUTPUT.baryon_temperature(pow(10.,log_z)) for log_z in log_z_array]) # K
    # Free electron fraction. Again, we multiply by the same factor as above to convert from CLASS convention to 21cmFAST convention
    x_e = np.array([CLASS_OUTPUT.ionization_fraction(pow(10.,log_z))*(1.-YHe)/(1.-(1.-1./_not4_)*YHe) for log_z in log_z_array])
    # Scale-Independent Growth Factor (SIGF)
    D_z = np.array([CLASS_OUTPUT.scale_independent_growth_factor(pow(10.,log_z)) for log_z in log_z_array])
    # Convert to log, for better interpolation (as these quantities can change by orders of magnitude)
    log10_T_k = np.log10(T_k)
    log10_x_e = np.log10(x_e)
    log10_D_z = np.log10(D_z)

    # In case of SDM, we also need the SDM temperature
    if user_params.SCATTERING_DM:
        z_CLASS = CLASS_OUTPUT.get_thermodynamics()['z']
        Tchi_CLASS = CLASS_OUTPUT.get_thermodynamics()['T_dmeff']
        Tchi_interp = interp1d(z_CLASS, Tchi_CLASS, kind='cubic',
                               bounds_error=False,fill_value=0.)
        T_chi = np.array([Tchi_interp(pow(10.,log_z)) for log_z in log_z_array]) # K
        log10_T_chi = np.log10(T_chi)
        # If we begin with homogeneous boxes, we also need the mean relative velocity between baryons and SDM
        v_chi_b_avg = np.zeros_like(log_z_array)
        for index, z in enumerate(pow(10.,log_z_array)):
            # And lastly, we also need the relative velocity between baryons and SDM
            Transfer_z = CLASS_OUTPUT.get_transfer(z=z)
            k_CLASS = Transfer_z['k (h/Mpc)'][:]*h # 1/Mpc
            theta_b_z_CLASS = Transfer_z['t_b'][:]*c/Mpc_to_meter # 1/sec
            theta_chi_z_CLASS = Transfer_z['t_dmeff'][:]*c/Mpc_to_meter # 1/sec
            theta_chi_z = Interpolate_transfer(theta_chi_z_CLASS,k_CLASS,k_output)
            theta_b_z = Interpolate_transfer(theta_b_z_CLASS,k_CLASS,k_output)
            v_chi_b_z = (theta_b_z-theta_chi_z)/(k_output/Mpc_to_meter)/1000. # km/sec
            v_chi_b_rms = np.sqrt(intg.simps(A_s*pow(k_output/0.05,ns-1.)*(v_chi_b_z**2.)/k_output, x=k_output)) # km/sec
            v_chi_b_avg[index] = np.sqrt(8./(3.*np.pi))*v_chi_b_rms # km/sec
        log10_v_chi_b_avg = np.log10(v_chi_b_avg)

    # Scale-Dependent Growth Factor (SDGF)
    if user_params.EVOLVE_BARYONS:
        # This is the wavenumber grid for our interpolation tables
        log_k_array = np.linspace(np.log10(1e-2),np.log10(2.),300)
        # Initalization of the matrices
        log10_D_b_kz_mat = np.zeros((len(log_z_array), len(log_k_array)))
        if user_params.SCATTERING_DM:
            log10_D_chi_kz_mat = np.zeros((len(log_z_array), len(log_k_array)))
        # For each redshift entry, we evaluate the SDGF as a function of wavenumber
        for z_ind, z in enumerate(pow(10.,log_z_array)):
            # First, we start with the SDGF for baryons
            T_b_over_c = interp1d(k_CLASS,
                                  abs(CLASS_OUTPUT.get_transfer(z=z)['d_b']/CLASS_OUTPUT.get_transfer(z=z)['d_cdm']),
                                  kind='cubic',bounds_error=False)(pow(10.,log_k_array))
            D_b_kz = abs(T_b_over_c*D_z[z_ind])
            log10_D_b_kz_mat[z_ind,:] = np.log10(D_b_kz)
            # Then we do the SDGF for SDM
            if user_params.SCATTERING_DM:
                T_chi_over_c = interp1d(k_CLASS,
                                      abs(CLASS_OUTPUT.get_transfer(z=z)['d_dmeff']/CLASS_OUTPUT.get_transfer(z=z)['d_cdm']),
                                      kind='cubic',bounds_error=False)(pow(10.,log_k_array))
                D_chi_kz = abs(T_chi_over_c*D_z[z_ind])
                log10_D_chi_kz_mat[z_ind,:] = np.log10(D_chi_kz)

    ################################################################################################################################################
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% vcb correction section %%%%%%%%%%$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ################################################################################################################################################

    # Below we find the best fit values for the vcb correction to the matter power spectrum (see Eq. 14 in arXiv: 2110.13919)
    # Written in collaboration with Debanjan Sarkar
    A_p, k_p, sigma_p = (global_params.A_VCB_PM,global_params.KP_VCB_PM,global_params.SIGMAK_VCB_PM)
    if (user_params.DO_VCB_FIT and user_params.USE_RELATIVE_VELOCITIES):
        # Set interpolation tables for initial conditions
        delta_c_init = interp1d(k_output, delta_c_kin, kind='cubic')
        delta_b_init = interp1d(k_output, delta_b_kin, kind='cubic')
        theta_c_init = interp1d(k_output, theta_c_kin, kind='cubic') # 1/sec
        theta_b_init = interp1d(k_output, theta_b_kin, kind='cubic') # 1/sec
        # Define arguments for the differential equations solver
        COSMO_PARAMS = (h, Omega_b0, Omega_m0, Tcmb0, ns, sigma8, m_FDM,
                        Omega_c0, Omega_Lambda, H_0, fHe, mu_b, z_kin)
        args = (COSMO_PARAMS, CLASS_OUTPUT)
        # Set inital and final times for the differential equations solver
        z_end = 20. # Redshift of the fit
        time_kin = z_to_time(z_kin,COSMO_PARAMS) # Gyr
        time_end = z_to_time(z_end,COSMO_PARAMS) # Gyr
        # Set loop arrays
        k_arr = np.logspace(0., np.log10(3000.), 100) # 1/Mpc
        mu_arr = np.linspace(-1., 1., 21)
        T_m_vcb = np.zeros(len(k_arr))
        T_m_0 = np.zeros(len(k_arr))

        # Loop over the k array
        for k_ind, k in tqdm.tqdm(enumerate(k_arr),
                             desc="DO_VCB_FIT",
                             unit="k",
                             disable=False,
                             total=len(k_arr)):

            # Set initial conditions for the differntial equations. It is assumed that
            # the initial conditions are real, but the results do not depend on the
            # phase of the initial conditions

            # Real part
            theta_b_real_0 = theta_b_init(k)*Gyr_to_second # 1/Gyr
            theta_c_real_0 = theta_c_init(k)*Gyr_to_second # 1/Gyr
            delta_b_real_0 = delta_b_init(k)
            delta_c_real_0 = delta_c_init(k)
            delta_T_real_0 = 0.0
            # Imaginary part
            theta_b_imag_0 = 0.0 # 1/Gyr
            theta_c_imag_0 = 0.0 # 1/Gyr
            delta_b_imag_0 = 0.0
            delta_c_imag_0 = 0.0
            delta_T_imag_0 = 0.0
            # Prepare intial conditions for the solver
            IC = [theta_b_real_0, theta_b_imag_0,
                  theta_c_real_0, theta_c_imag_0,
                  delta_b_real_0, delta_b_imag_0,
                  delta_c_real_0, delta_c_imag_0,
                  delta_T_real_0, delta_T_imag_0]

            # Solve the differntial equations for v_cb=0
            ODE_params = (0., k, 0.) # km/sec, 1/Mpc, dimensionless
            Transfers = solve_ivp(model, [time_kin, time_end], IC, method='Radau',
                             dense_output=False, rtol=1.e-4, atol=1.e-6, args=args+(ODE_params,))
            T_m_0[k_ind] = abs((Omega_c0*Transfers.y[6][-1] + Omega_b0*Transfers.y[4][-1])
                          + 1j*(Omega_c0*Transfers.y[7][-1] + Omega_b0*Transfers.y[5][-1]))/Omega_m0

            # Loop over the mu array
            delta_m = np.zeros(len(mu_arr)) + 0j
            for mi, mu in enumerate(mu_arr):
                # Solve the differential equations with these parameters
                ODE_params = (v_cb_rms_kin, k, mu) # km/sec, 1/Mpc, dimensionless
                Transfers = solve_ivp(model, [time_kin, time_end], IC, method='Radau',
                                 dense_output=False, rtol=1.e-4, atol=1.e-6, args=args+(ODE_params,))
                # Extract the transfer functions
                delta_b_real = Transfers.y[4][-1]
                delta_b_imag = Transfers.y[5][-1]
                delta_c_real = Transfers.y[6][-1]
                delta_c_imag = Transfers.y[7][-1]
                # Calculate the matter transfer function
                delta_m_real = (Omega_c0*delta_c_real + Omega_b0*delta_b_real)/Omega_m0
                delta_m_imag = (Omega_c0*delta_c_imag + Omega_b0*delta_b_imag)/Omega_m0
                delta_m[mi] = delta_m_real + 1j*delta_m_imag
            # Integration over mu
            T_m_vcb[k_ind] = np.sqrt(0.5*intg.simpson(abs(delta_m)**2,x=mu_arr))

        # Calculate the matter power spectrum with and without the vcb correction
        Pk_vcb = 2.*(np.pi**2.)*A_s*pow(k_arr/0.05,ns-1.)*T_m_vcb**2./(k_arr**3.) # Mpc^3
        Pk_0 = 2.*(np.pi**2.)*A_s*pow(k_arr/0.05,ns-1.)*T_m_0**2./(k_arr**3.) # Mpc^3
        # Find best fit values
        try:
            A_p, k_p, sigma_p = curve_fit(fitting_function,k_arr, Pk_vcb/Pk_0,(A_p, k_p, sigma_p))[0]
        except RuntimeError:
            pass
        print("\n")

    ################################################################################################################################################
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Return output %%%%%%%%%%%%%%%%%%%$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ################################################################################################################################################

    # Derived cosmological parameters
    cosmo_params.SIGMA_8 = sigma8
    global_params.Y_He = YHe
    global_params.VAVG = v_cb_avg_kin
    global_params.Z_REC = z_kin

    # Best fit values for the vcb correction
    global_params.A_VCB_PM = A_p
    global_params.KP_VCB_PM = k_p
    global_params.SIGMAK_VCB_PM = sigma_p

    # Interpolation tables for background quantities
    global_params.LOG_T_k = list(log10_T_k)
    global_params.LOG_x_e = list(log10_x_e)
    global_params.LOG_SIGF = list(log10_D_z)

    # Interpolation tables for background quantities
    global_params.T_M0_TRANSFER = list(delta_m_0)
    global_params.T_VCB_KIN_TRANSFER = list(T_vcb_kin)

    # SDM quantities
    if user_params.SCATTERING_DM:
        global_params.LOG_T_chi = list(log10_T_chi)
        global_params.LOG_V_chi_b = list(log10_v_chi_b_avg)
        if user_params.USE_SDM_FLUCTS:
            global_params.T_V_CHI_B_ZHIGH_TRANSFER = list(T_v_chi_b_zhigh)

    # SLKF: NG quantities
    if user_params.NG_FIELD:
        global_params.T_phi_TRANSFER = list(delta_phi)

    # SDGF interpolation tables
    if user_params.EVOLVE_BARYONS:
        global_params.LOG_K_ARR_FOR_SDGF = list(log_k_array)
        global_params.LOG_SDGF = list(log10_D_b_kz_mat.T.flatten())
        if user_params.SCATTERING_DM:
            global_params.LOG_SDGF_SDM = list(log10_D_chi_kz_mat.T.flatten())

    # Return the lensed C_ell's
    return CLASS_OUTPUT.lensed_cl(3000)
