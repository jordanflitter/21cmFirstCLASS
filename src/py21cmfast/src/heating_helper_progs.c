
struct UserParams *user_params_hf;
struct CosmoParams *cosmo_params_hf;
struct AstroParams *astro_params_hf;
struct FlagOptions *flag_options_hf;

float determine_zpp_min, zpp_bin_width;

double BinWidth_pH,inv_BinWidth_pH,BinWidth_elec,inv_BinWidth_elec,BinWidth_10,inv_BinWidth_10,PS_ION_EFF;

double get_M_min_ion(float z);

void Broadcast_struct_global_HF(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options){

    user_params_hf = user_params;
    cosmo_params_hf = cosmo_params;
    astro_params_hf = astro_params;
    flag_options_hf = flag_options;
}


/* Returns the minimum source mass for ionizing sources, according to user specifications */
double get_M_min_ion(float z){
    double MMIN;

    if (astro_params_hf->ION_Tvir_MIN < 9.99999e3) // neutral IGM
        MMIN = TtoM(z, astro_params_hf->ION_Tvir_MIN, mu_b_neutral); //JordanFlitter: I changed the constant value to the general case
    else // ionized IGM
        MMIN = TtoM(z, astro_params_hf->ION_Tvir_MIN, mu_b_ionized); //JordanFlitter: I changed the constant value to the general case

    // check for WDM
    if (global_params.P_CUTOFF && ( MMIN < M_J_WDM()))
        MMIN = M_J_WDM();
    return MMIN;
}

// * initialization routine * //
// int init_heat();

/* destruction/deallocation routine */
void destruct_heat();

// * returns the spectral emissity * //
double spectral_emissivity(double nu_norm, int flag, int Population);

// * Ionization fraction from RECFAST. * //
double xion_RECFAST(float z, int flag);

// * IGM temperature from RECFAST; includes Compton heating and adiabatic expansion only. * //
double T_RECFAST(float z, int flag);

// * returns the spin temperature * //
float get_Ts(float z, float delta, float TK, float xe, float Jalpha, float * curr_xalpha);

//* Returns recycling fraction (=fraction of photons converted into Lyalpha for Ly-n resonance * //
double frecycle(int n);

// * Returns frequency of Lyman-n, in units of Lyman-alpha * //
double nu_n(int n);

float dfcoll_dz(float z, float sigma_min, float del_bias, float sig_bias);

// * A simple linear 2D table for quickly estimating Fcoll given a z and Tvir (for Ts.c) * //
int init_FcollTable(float zmin, float zmax);

double kappa_10_pH(double T, int flag);
double kappa_10_elec(double T, int flag);
double kappa_10(double TK, int flag);

double xcoll(double z, double TK, double delta, double xe);
double xcoll_HI(double z, double TK, double delta, double xe);
double xcoll_elec(double z, double TK, double delta, double xe);
double xcoll_prot(double z, double TK, double delta, double xe);

double xalpha_tilde(double z, double Jalpha, double TK, double TS, double delta, double xe);
double Tc_eff(double TK, double TS);
double Salpha_tilde(double TK, double TS, double tauGP);
double taugp(double z, double delta, double xe);

double species_weighted_x_ray_cross_section(double nu, double x_e);

// * Returns the maximum redshift at which a Lyn transition contributes to Lya flux at z * //
float zmax(float z, int n);

// JordanFlitter: Lyman-Alpha heating functions
double delta_E(double ah, double gammah, double etah, double delta_dopp, int ci_flag);
double interpolate_heating_efficiencies(double tk, double ts, double taugp, double *arrE);
double Energy_Lya_heating(double Tk, double Ts, double tau_gp, int flag);

// JordanFlitter: SDM functions
double T_chi_RECFAST(float z, int flag);
double V_chi_b_RECFAST(float z, int flag);
double S_alpha_correction(float xi, int flag);

// JordanFlitter: new function for HyRec
double hyrec_dx_H_dz(HYREC_DATA *rec_data, double x_H, double xe, double nH, double z, double Hz, double Tmat, double Trad);

// JordanFlitter: approximation for the adiabatic index at z=6-50 from 2302.08506
float cT_approx(float z);

// JordanFlitter: added a redshift argument
int init_heat(float redshift)
{
    kappa_10(1.0,1);
    kappa_10_elec(1.0,1);
    kappa_10_pH(1.0,1);
    if (T_RECFAST(100, 1) < 0)
        return -4;
    if (xion_RECFAST(100, 1) < 0)
        return -5;
    if (spectral_emissivity(0,1,2) < 0)
        return -6;
    if( kappa_10_elec(1.0,1) < 0)
        return -2;
    if( kappa_10_pH(1.0,1) < 0)
        return -3;
    // JordanFlitter: Initialize interpolation array for Lya heating (don't need that for z > Z_HEAT_MAX)
    if ((flag_options_hf->USE_Lya_HEATING) && (redshift <= global_params.Z_HEAT_MAX)) {
        if (Energy_Lya_heating(100, 100, 100, 1)<0)
            return -7;
    }
    // JordanFlitter: Initialize interpolation array for SDM quantities
    if (user_params_hf->SCATTERING_DM) {
        if (T_chi_RECFAST(100, 1) < 0)
            return -8;
        if (V_chi_b_RECFAST(100, 1) < 0)
            return -9;
    }
    // JordanFlitter: Initialize interpolation array for Chuzhouy & Shapiro S_alpha
    if (user_params_hf->USE_CS_S_ALPHA) {
        if (S_alpha_correction(100, 1) < 0)
            return -10;
    }
    LOG_SUPER_DEBUG("About to initialize interp arrays");

    initialize_interp_arrays();

    LOG_SUPER_DEBUG("Done initializing heat.");
    return 0;
}

void destruct_heat()
{
  T_RECFAST(100.0,2);
  xion_RECFAST(100.0,2);
  if (user_params_hf->SCATTERING_DM) {
      T_chi_RECFAST(100.0,2); //JordanFlitter: free T_chi interpolation table
      V_chi_b_RECFAST(100.0,2); //JordanFlitter: free V_chi_b interpolation table
  }
  if (user_params_hf->USE_CS_S_ALPHA) {
      S_alpha_correction(100.0,2); //JordanFlitter: free S_alpha interpolation table
  }
}

float get_Ts(float z, float delta, float TK, float xe, float Jalpha, float * curr_xalpha){
    double Trad,xc,xa_tilde;
    double TS,TSold,TSinv;
    double Tceff;

    Trad = T_cmb*(1.0+z);
    xc = xcoll(z,TK,delta,xe);
    if (Jalpha > 1.0e-20) { // * Must use WF effect * //
        TS = Trad;
        TSold = 0.0;
        while (fabs(TS-TSold)/TS > 1.0e-3) {
            TSold = TS;
            xa_tilde = xalpha_tilde(z,Jalpha,TK,TS,delta,xe);
            Tceff = Tc_eff(1./TK,1./TS);
            TS = (1.0+xa_tilde+xc)/(1.0/Trad+xa_tilde/Tceff + xc/TK);
        }
        *curr_xalpha = xa_tilde;
    } else { // * Collisions only * //
        TS = (1.0 + xc)/(1.0/Trad + xc/TK);
        // JordanFlitter: I added the tau21 correction from Eq. (31) in arXiv: astro-ph/0702600
        TS -= 0.5*(1.-TK/Trad)*xc/(TK/Trad+xc)/(1+xc)*(3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B)*((1-xe)*No*pow(1.+z,3.)*(1.+delta))/hubble(z);
        *curr_xalpha = 0;
    }

    return TS;
}



//  Redshift derivative of the conditional collapsed fraction
float dfcoll_dz(float z, float sigma_min, float del_bias, float sig_bias)
{
    double dz,z1,z2;
    //  double mu, m1, m2;
    double fc1,fc2,ans;

    dz = 0.001;
    z1 = z + dz;
    z2 = z - dz;
    fc1 = sigmaparam_FgtrM_bias(z1, sigma_min, del_bias, sig_bias);
    fc2 = sigmaparam_FgtrM_bias(z2, sigma_min, del_bias, sig_bias);
    ans = (fc1 - fc2)/(2.0*dz);
    return ans;
}


int init_FcollTable(float zmin, float zmax)
{

    int i;
    double z_table;

    zmin_1DTable = zmin;
    zmax_1DTable = 1.2*zmax;

    zbin_width_1DTable = 0.1;

    n_redshifts_1DTable = (int)ceil((zmax_1DTable - zmin_1DTable)/zbin_width_1DTable);

    FgtrM_1DTable_linear = (double *)calloc(n_redshifts_1DTable,sizeof(double));

    for(i=0;i<n_redshifts_1DTable;i++) {
        z_table = zmin_1DTable + zbin_width_1DTable*(double)i;

        if(flag_options_hf->M_MIN_in_Mass) {
            FgtrM_1DTable_linear[i] = log10(FgtrM(z_table, (astro_params_hf->M_TURN)/50.));
        }
        else {
            FgtrM_1DTable_linear[i] = log10(FgtrM(z_table, get_M_min_ion(z_table)));
        }
    }

    return 0;
}



// ******************************************************************** //
// ************************ RECFAST quantities ************************ //
// ******************************************************************** //

// IGM temperature from RECFAST; includes Compton heating and adiabatic expansion only.
// JordanFlitter: I modified this function such that we no longer read the data from a text file, but rather from global_params
double T_RECFAST(float z, int flag)
{
    double ans;
    static double log_z_arr[Z_ARRAY_NPTS];
    static double log_T_k_arr[Z_ARRAY_NPTS];
    static gsl_interp_accel *acc;
    static gsl_spline *spline;

    if (flag == 1) {

        for (size_t i = 0; i < Z_ARRAY_NPTS; i++) {
            log_z_arr[i] = global_params.LOG_Z_ARR[i];
            log_T_k_arr[i] = global_params.LOG_T_k[i];
        }

        // Set up spline table
        acc   = gsl_interp_accel_alloc ();
        spline  = gsl_spline_alloc (gsl_interp_cspline, Z_ARRAY_NPTS);
        gsl_spline_init(spline, log_z_arr, log_T_k_arr, Z_ARRAY_NPTS);

        return 0;
    }

    if (flag == 2) {
        // Free memory
        gsl_spline_free (spline);
        gsl_interp_accel_free(acc);
        return 0;
    }

    // Convert to log10
    double log10_z = log10(z);

    if (log10_z > global_params.LOG_Z_ARR[Z_ARRAY_NPTS-1]) { // Called at z>500! Bail out
        LOG_ERROR("Called T_RECFAST with z=%f.", z);
        Throw 1;
    }
    else { // Do spline
        ans = gsl_spline_eval (spline, log10_z, acc);
    }
    return pow(10.,ans);
}

// Ionization fraction from RECFAST. //
// JordanFlitter: I modified this function such that we no longer read the data from a text file, but rather from global_params
double xion_RECFAST(float z, int flag)
{
  double ans;
  static double log_z_arr[Z_ARRAY_NPTS];
  static double log_x_e_arr[Z_ARRAY_NPTS];
  static gsl_interp_accel *acc;
  static gsl_spline *spline;

    if (flag == 1) {

        for (size_t i = 0; i < Z_ARRAY_NPTS; i++) {
            log_z_arr[i] = global_params.LOG_Z_ARR[i];
            log_x_e_arr[i] = global_params.LOG_x_e[i];
        }

        // Set up spline table
        acc   = gsl_interp_accel_alloc ();
        spline  = gsl_spline_alloc (gsl_interp_cspline, Z_ARRAY_NPTS);
        gsl_spline_init(spline, log_z_arr, log_x_e_arr, Z_ARRAY_NPTS);

        return 0;
    }

    if (flag == 2) {
        gsl_spline_free (spline);
        gsl_interp_accel_free(acc);
        return 0;
    }

    // Convert to log10
    double log10_z = log10(z);

    if (log10_z > global_params.LOG_Z_ARR[Z_ARRAY_NPTS-1]) { // Called at z>500! Bail out
        LOG_ERROR("Called xion_RECFAST with z=%f.", z);
        Throw 1;
    }
    else { // Do spline
        ans = gsl_spline_eval (spline, log10_z, acc);
    }
    return pow(10.,ans);
}


//* Returns recycling fraction (=fraction of photons converted into Lyalpha for Ly-n resonance * //
double frecycle(int n)
{
    switch (n){
        case 0:
            return 1;
        case 1:
            return 1;
        case 2:
            return 1;
        case 3:
            return 0;
        case 4:
            return 0.2609;
        case 5:
            return 0.3078;
        case 6:
            return 0.3259;
        case 7:
            return 0.3353;
        case 8:
            return 0.3410;
        case 9:
            return 0.3448;
        case 10:
            return 0.3476;
        case 11:
            return 0.3496;
        case 12:
            return 0.3512;
        case 13:
            return 0.3524;
        case 14:
            return 0.3535;
        case 15:
            return 0.3543;
        case 16:
            return 0.3550;
        case 17:
            return 0.3556;
        case 18:
            return 0.3561;
        case 19:
            return 0.3565;
        case 20:
            return 0.3569;
        case 21:
            return 0.3572;
        case 22:
            return 0.3575;
        case 23:
            return 0.3578;
        case 24:
            return 0.3580;
        case 25:
            return 0.3582;
        case 26:
            return 0.3584;
        case 27:
            return 0.3586;
        case 28:
            return 0.3587;
        case 29:
            return 0.3589;
        case 30:
            return 0.3590;
        default:
            return 0;
    }
}



// * Reads in and constructs table of the piecewise power-law fits to Pop 2 and Pop 3 stellar spectra, from Barkana * //
double spectral_emissivity(double nu_norm, int flag, int Population)
{
    static int n[NSPEC_MAX];
    static float nu_n[NSPEC_MAX], alpha_S_2[NSPEC_MAX];
    static float alpha_S_3[NSPEC_MAX], N0_2[NSPEC_MAX], N0_3[NSPEC_MAX];
    double n0_fac;
    //  double ans, tot, lya;
    double result;
    int i;
    FILE *F;

    char filename[500];

    switch(flag){
        case 2:
            // For LW calculateion. New in v1.5, see...
            for (i=1;i<(NSPEC_MAX-1);i++) {
                  if ((nu_norm >= nu_n[i]) && (nu_norm < nu_n[i+1])) {
                    // We are in the correct spectral region
                    if (Population == 2){
                        // moved (1. - F_H2_SHIELD) outside
                          result = N0_2[i]  / (alpha_S_2[i] + 1) * ( pow(nu_n[i+1], alpha_S_2[i]+1) - pow(nu_norm, alpha_S_2[i]+1) );
                          return result > 0 ? result:1e-40;
                    }
                    else{
                          result = N0_3[i] / (alpha_S_3[i] + 1) * ( pow(nu_n[i+1], alpha_S_3[i]+1) - pow(nu_norm, alpha_S_3[i]+1) );
                          return result > 0 ? result:1e-40;
                    }
                  }
            }

        case 1:
            // * Read in the data * //
            sprintf(filename,"%s/%s",global_params.external_table_path,STELLAR_SPECTRA_FILENAME);
            if (!(F = fopen(filename, "r"))){
               LOG_ERROR("spectral_emissivity: Unable to open file: stellar_spectra.dat for reading.");
                Throw IOError;
            }

            for (i=1;i<NSPEC_MAX;i++) {
                fscanf(F, "%i %e %e %e %e", &n[i], &N0_2[i], &alpha_S_2[i], &N0_3[i], &alpha_S_3[i]);
                //      printf("%i\t%e\t%e\t%e\t%e\n", n[i], N0_2[i], alpha_S_2[i], N0_3[i], alpha_S_3[i]);
            }
            fclose(F);

            for (i=1;i<NSPEC_MAX;i++) {
                nu_n[i] = 4.0/3.0*(1.0-1.0/pow(n[i],2.0));
            }

            for (i=1;i<(NSPEC_MAX-1);i++) {
                n0_fac = (pow(nu_n[i+1],alpha_S_2[i]+1) - pow(nu_n[i],alpha_S_2[i]+1));
                N0_2[i] *= (alpha_S_2[i]+1)/n0_fac*global_params.Pop2_ion;
                n0_fac = (pow(nu_n[i+1],alpha_S_3[i]+1) - pow(nu_n[i],alpha_S_3[i]+1));
                N0_3[i] *= (alpha_S_3[i]+1)/n0_fac*global_params.Pop3_ion;
            }

            return 0.0;

        default:
            for (i=1;i<(NSPEC_MAX-1);i++) {
                //    printf("checking between %e and %e\n", nu_n[i], nu_n[i+1]);
                if ((nu_norm >= nu_n[i]) && (nu_norm < nu_n[i+1])) {
                    // We are in the correct spectral region
                    if (Population == 2)
                        return N0_2[i]*pow(nu_norm,alpha_S_2[i])/Ly_alpha_HZ;
                    else
                        return N0_3[i]*pow(nu_norm,alpha_S_3[i])/Ly_alpha_HZ;
                }
            }

            i= NSPEC_MAX-1;
            if (Population == 2)
                return  N0_2[i]*pow(nu_norm,alpha_S_2[i])/Ly_alpha_HZ;
            else
                return N0_3[i]*pow(nu_norm,alpha_S_3[i])/Ly_alpha_HZ;
    }
}


double xcoll(double z, double TK, double delta, double xe){
    return xcoll_HI(z,TK,delta,xe) + xcoll_elec(z,TK,delta,xe) + xcoll_prot(z,TK,delta,xe);
}

double xcoll_HI(double z, double TK, double delta, double xe)
{
    double krate,nH,Trad;
    double xcoll;

    Trad = T_cmb*(1.0+z);
    nH = (1.0-xe)*No*pow(1.0+z,3.0)*(1.0+delta);
    krate = kappa_10(TK,0);
    xcoll = T21/Trad*nH*krate/A10_HYPERFINE;
    return xcoll;
}

// Note that this assumes Helium ionized same as Hydrogen //
double xcoll_elec(double z, double TK, double delta, double xe)
{
    double krate,ne,Trad;
    double xcoll;

    Trad = T_cmb*(1.0+z);
    ne = xe*N_b0*pow(1.0+z,3.0)*(1.0+delta);
    krate = kappa_10_elec(TK,0);
    xcoll = T21/Trad*ne*krate/A10_HYPERFINE;
    return xcoll;
}

double xcoll_prot(double z, double TK, double delta, double xe)
{
    double krate,np,Trad;
    double xcoll;

    Trad = T_cmb*(1.0+z);
    np = xe*No*pow(1.0+z,3.0)*(1.0+delta);
    krate = kappa_10_pH(TK,0);
    xcoll = T21/Trad*np*krate/A10_HYPERFINE;
    return xcoll;
}

double Salpha_tilde(double TK, double TS, double tauGP)
{
    double xi;
    double ans;

    xi = pow(1.0e-7*tauGP*TK*TK, 1.0/3.0);
    ans = (1.0 - 0.0631789*TK + 0.115995*TK*TK - 0.401403*TS*TK + 0.336463*TS*TK*TK)/(1.0 + 2.98394*xi + 1.53583*xi*xi + 3.85289*xi*xi*xi);
    return ans;
}


// * Returns frequency of Lyman-n, in units of Lyman-alpha * //
double nu_n(int n)
{
    double ans;

    ans = 1.0 - pow(n, -2.0);
    ans /= 0.75;
    return ans;
}


double kappa_10(double TK, int flag)
{

    static double tkin_spline[KAPPA_10_NPTS_Spline], kap_spline[KAPPA_10_NPTS_Spline];
    double ans;
    int tkin_spline_int;

    if (flag == 1) {

        BinWidth_10 = 0.317597943861;
        inv_BinWidth_10 = 1./0.317597943861;

        tkin_spline[0] = 0.0; kap_spline[0] = -29.6115227098;
        tkin_spline[1] = 0.317597943861; kap_spline[1] = -29.6228184691;
        tkin_spline[2] = 0.635195887722; kap_spline[2] = -29.5917673123;
        tkin_spline[3] = 0.952793831584; kap_spline[3] = -29.4469989515;
        tkin_spline[4] = 1.27039177544; kap_spline[4] = -29.1171430989;
        tkin_spline[5] = 1.58798971931; kap_spline[5] = -28.5382192456;
        tkin_spline[6] = 1.90558766317; kap_spline[6] = -27.7424388865;
        tkin_spline[7] = 2.22318560703; kap_spline[7] = -26.8137036254;
        tkin_spline[8] = 2.54078355089; kap_spline[8] = -25.8749225449;
        tkin_spline[9] = 2.85838149475; kap_spline[9] = -25.0548322235;
        tkin_spline[10] = 3.17597943861; kap_spline[10] = -24.4005076336;
        tkin_spline[11] = 3.49357738247; kap_spline[11] = -23.8952335377;
        tkin_spline[12] = 3.81117532633; kap_spline[12] = -23.5075651004;
        tkin_spline[13] = 4.1287732702; kap_spline[13] = -23.201637629;
        tkin_spline[14] = 4.44637121406; kap_spline[14] = -22.9593758343;
        tkin_spline[15] = 4.76396915792; kap_spline[15] = -22.7534867461;
        tkin_spline[16] = 5.08156710178; kap_spline[16] = -22.5745752086;
        tkin_spline[17] = 5.39916504564; kap_spline[17] = -22.4195690855;
        tkin_spline[18] = 5.7167629895; kap_spline[18] = -22.2833176123;
        tkin_spline[19] = 6.03436093336; kap_spline[19] = -22.1549519419;
        tkin_spline[20] = 6.35195887722; kap_spline[20] = -22.0323282988;
        tkin_spline[21] = 6.66955682109; kap_spline[21] = -21.9149994721;
        tkin_spline[22] = 6.98715476495; kap_spline[22] = -21.800121439;
        tkin_spline[23] = 7.30475270881; kap_spline[23] = -21.6839502137;
        tkin_spline[24] = 7.62235065267; kap_spline[24] = -21.5662434981;
        tkin_spline[25] = 7.93994859653; kap_spline[25] = -21.4473595491;
        tkin_spline[26] = 8.25754654039; kap_spline[26] = -21.3279560712;
        tkin_spline[27] = 8.57514448425; kap_spline[27] = -21.2067614838;
        tkin_spline[28] = 8.89274242811; kap_spline[28] = -21.0835560288;
        tkin_spline[29] = 9.21034037198; kap_spline[29] = -20.9627928675;

    }

    TK = log(TK);

    if (TK < tkin_spline[0]) { // Below 1 K, just use that value
        ans = kap_spline[0];
    } else if (TK > tkin_spline[KAPPA_10_NPTS_Spline-1]) {
        // Power law extrapolation
        ans = log(exp(kap_spline[KAPPA_10_NPTS_Spline-1])*pow(exp(TK)/exp(tkin_spline[KAPPA_10_NPTS_Spline-1]),0.381));
    } else { // Do spline

        tkin_spline_int = (int)floor((TK - tkin_spline[0])*inv_BinWidth_10);

        ans = kap_spline[tkin_spline_int] + ( TK - (tkin_spline[0] + BinWidth_10*(float)tkin_spline_int) )*( kap_spline[tkin_spline_int+1] - kap_spline[tkin_spline_int] )*inv_BinWidth_10;
    }

    return exp(ans);
}

double kappa_10_pH(double T, int flag)
{
    static double TK_spline[KAPPA_10_pH_NPTS_Spline], kappa_spline[KAPPA_10_pH_NPTS_Spline];
    double ans;
    int TK_spline_int;

    if (flag == 1) {

        BinWidth_pH = 0.341499570777;
        inv_BinWidth_pH = 1./0.341499570777;

        TK_spline[0] = 0.0; kappa_spline[0] = -21.6395565688;
        TK_spline[1] = 0.341499570777; kappa_spline[1] = -21.5641675629;
        TK_spline[2] = 0.682999141554; kappa_spline[2] = -21.5225112028;
        TK_spline[3] = 1.02449871233; kappa_spline[3] = -21.5130514508;
        TK_spline[4] = 1.36599828311; kappa_spline[4] = -21.5342522691;
        TK_spline[5] = 1.70749785389; kappa_spline[5] = -21.5845293039;
        TK_spline[6] = 2.04899742466; kappa_spline[6] = -21.6581396414;
        TK_spline[7] = 2.39049699544; kappa_spline[7] = -21.7420392948;
        TK_spline[8] = 2.73199656622; kappa_spline[8] = -21.8221380683;
        TK_spline[9] = 3.07349613699; kappa_spline[9] = -21.8837908896;
        TK_spline[10] = 3.41499570777; kappa_spline[10] = -21.9167553997;
        TK_spline[11] = 3.75649527855; kappa_spline[11] = -21.9200173678;
        TK_spline[12] = 4.09799484933; kappa_spline[12] = -21.8938574675;
        TK_spline[13] = 4.4394944201; kappa_spline[13] = -21.8414464728;
        TK_spline[14] = 4.78099399088; kappa_spline[14] = -21.7684762963;
        TK_spline[15] = 5.12249356166; kappa_spline[15] = -21.6796222358;
        TK_spline[16] = 5.46399313243; kappa_spline[16] = -21.5784701374;
        TK_spline[17] = 5.80549270321; kappa_spline[17] = -21.4679438133;
        TK_spline[18] = 6.14699227399; kappa_spline[18] = -21.3503236936;
        TK_spline[19] = 6.48849184477; kappa_spline[19] = -21.2277666787;
        TK_spline[20] = 6.82999141554; kappa_spline[20] = -21.1017425964;
        TK_spline[21] = 7.17149098632; kappa_spline[21] = -20.9733966978;
        TK_spline[22] = 7.5129905571; kappa_spline[22] = -20.8437244283;
        TK_spline[23] = 7.85449012787; kappa_spline[23] = -20.7135746917;
        TK_spline[24] = 8.19598969865; kappa_spline[24] = -20.583135408;
        TK_spline[25] = 8.53748926943; kappa_spline[25] = -20.4523507819;
        TK_spline[26] = 8.8789888402; kappa_spline[26] = -20.3215504736;
        TK_spline[27] = 9.22048841098; kappa_spline[27] = -20.1917429161;
        TK_spline[28] = 9.56198798176; kappa_spline[28] = -20.0629513946;
        TK_spline[29] = 9.90348755254; kappa_spline[29] = -19.9343540344;
    }

    T = log(T);

    if (T < TK_spline[0]) { // Below 1 K, just use that value
        ans = kappa_spline[0];
    } else if (T > TK_spline[KAPPA_10_pH_NPTS_Spline-1]) {
        // Power law extrapolation
        ans  = kappa_spline[KAPPA_10_pH_NPTS_Spline-1] + ((kappa_spline[KAPPA_10_pH_NPTS_Spline-1] - kappa_spline[KAPPA_10_pH_NPTS_Spline-2]) / (TK_spline[KAPPA_10_pH_NPTS_Spline-1] - TK_spline[KAPPA_10_pH_NPTS_Spline-2]) * (T-TK_spline[KAPPA_10_pH_NPTS_Spline-1]));
    } else { // Do spline

        TK_spline_int = (int)floor((T - TK_spline[0])*inv_BinWidth_pH);

        ans = kappa_spline[TK_spline_int] + ( T - (TK_spline[0] + BinWidth_pH*(double)TK_spline_int))*( kappa_spline[TK_spline_int+1] - kappa_spline[TK_spline_int] )*inv_BinWidth_pH;
    }
    ans = exp(ans);
    return ans;
}


double kappa_10_elec(double T, int flag)
{

    static double TK_spline[KAPPA_10_elec_NPTS_Spline], kappa_spline[KAPPA_10_elec_NPTS_Spline];
    double ans;
    int TK_spline_int;

    if (flag == 1) {

        BinWidth_elec = 0.396997429827;
        inv_BinWidth_elec = 1./0.396997429827;

        TK_spline[0] = 0.0; kappa_spline[0] = -22.1549007191;
        TK_spline[1] = 0.396997429827; kappa_spline[1] = -21.9576919899;
        TK_spline[2] = 0.793994859653; kappa_spline[2] = -21.760758435;
        TK_spline[3] = 1.19099228948; kappa_spline[3] = -21.5641795674;
        TK_spline[4] = 1.58798971931; kappa_spline[4] = -21.3680349001;
        TK_spline[5] = 1.98498714913; kappa_spline[5] = -21.1724124486;
        TK_spline[6] = 2.38198457896; kappa_spline[6] = -20.9774403051;
        TK_spline[7] = 2.77898200879; kappa_spline[7] = -20.78327367;
        TK_spline[8] = 3.17597943861; kappa_spline[8] = -20.5901042551;
        TK_spline[9] = 3.57297686844; kappa_spline[9] = -20.3981934669;
        TK_spline[10] = 3.96997429827; kappa_spline[10] = -20.2078762485;
        TK_spline[11] = 4.36697172809; kappa_spline[11] = -20.0195787458;
        TK_spline[12] = 4.76396915792; kappa_spline[12] = -19.8339587914;
        TK_spline[13] = 5.16096658775; kappa_spline[13] = -19.6518934427;
        TK_spline[14] = 5.55796401757; kappa_spline[14] = -19.4745894649;
        TK_spline[15] = 5.9549614474; kappa_spline[15] = -19.3043925781;
        TK_spline[16] = 6.35195887722; kappa_spline[16] = -19.1444129787;
        TK_spline[17] = 6.74895630705; kappa_spline[17] = -18.9986014565;
        TK_spline[18] = 7.14595373688; kappa_spline[18] = -18.8720602784;
        TK_spline[19] = 7.5429511667; kappa_spline[19] = -18.768679825;
        TK_spline[20] = 7.93994859653; kappa_spline[20] = -18.6909581885;
        TK_spline[21] = 8.33694602636; kappa_spline[21] = -18.6387511068;
        TK_spline[22] = 8.73394345618; kappa_spline[22] = -18.6093755705;
        TK_spline[23] = 9.13094088601; kappa_spline[23] = -18.5992098958;
        TK_spline[24] = 9.52793831584; kappa_spline[24] = -18.6050625357;
        TK_spline[25] = 9.92493574566; kappa_spline[25] = -18.6319366207;
        TK_spline[26] = 10.3219331755; kappa_spline[26] = -18.7017996535;
        TK_spline[27] = 10.7189306053; kappa_spline[27] = -18.8477153986;
        TK_spline[28] = 11.1159280351; kappa_spline[28] = -19.0813436512;
        TK_spline[29] = 11.512925465; kappa_spline[29] = -19.408859606;
    }

    T = log(T);

    if (T < TK_spline[0]) { // Below 1 K, just use that value
        ans = kappa_spline[0];
    } else if (T > TK_spline[KAPPA_10_elec_NPTS_Spline-1]) {
        // Power law extrapolation
        ans  = kappa_spline[KAPPA_10_elec_NPTS_Spline-1] + ((kappa_spline[KAPPA_10_elec_NPTS_Spline-1] - kappa_spline[KAPPA_10_elec_NPTS_Spline-2]) / (TK_spline[KAPPA_10_elec_NPTS_Spline-1] - TK_spline[KAPPA_10_elec_NPTS_Spline-2]) * (T-TK_spline[KAPPA_10_elec_NPTS_Spline-1]));

    } else { // Do spline

        TK_spline_int = (int)floor((T - TK_spline[0])*inv_BinWidth_elec);

        ans = kappa_spline[TK_spline_int] + ( T - ( TK_spline[0] + BinWidth_elec*(float)TK_spline_int ) )*( kappa_spline[TK_spline_int+1] - kappa_spline[TK_spline_int] )*inv_BinWidth_elec;
    }
    return exp(ans);
}


// ******************************************************************** //
// ********************* Wouthuysen-Field Coupling ******************** //
// ******************************************************************** //

// NOTE Jalpha is by number //
double xalpha_tilde(double z, double Jalpha, double TK, double TS,
                    double delta, double xe){
    double tgp,Stilde,x;

    tgp = taugp(z,delta,xe);
    Stilde = Salpha_tilde(1./TK,1./TS,tgp);
    x = 1.66e11/(1.0+z)*Stilde*Jalpha;
    return x;
}

// Compute the Gunn-Peterson optical depth.
double taugp(double z, double delta, double xe){
    return 1.342881e-7 / hubble(z)*No*pow(1+z,3) * (1.0+delta)*(1.0-xe);
}

double Tc_eff(double TK, double TS)
{
    double ans;

    ans = 1.0/(TK + 0.405535*TK*(TS - TK));
    return ans;
}






//
//  Evaluates the frequency integral in the Tx evolution equation
//  photons starting from zpp arive at zp, with mean IGM electron
//  fraction of x_e (used to compute tau), and local electron
//  fraction local_x_e
//  FLAG = 0 for heat integral
//  FLAG = 1 for ionization integral
//  FLAG = 2 for Lya integral
//
double integrand_in_nu_heat_integral(double nu, void * params){
    double species_sum;
    float x_e = *(double *) params;

    // HI
    species_sum = interp_fheat((nu - NUIONIZATION)/NU_over_EV, x_e)
    * hplank*(nu - NUIONIZATION) * f_H * (1-x_e) * HI_ion_crosssec(nu);

    // HeI
    species_sum += interp_fheat((nu - HeI_NUIONIZATION)/NU_over_EV, x_e)
    * hplank*(nu - HeI_NUIONIZATION) * f_He * (1-x_e) * HeI_ion_crosssec(nu);

    // HeII
    species_sum += interp_fheat((nu - HeII_NUIONIZATION)/NU_over_EV, x_e)
    * hplank*(nu - HeII_NUIONIZATION) * f_He * x_e * HeII_ion_crosssec(nu);

    return species_sum * pow(nu/((astro_params_hf->NU_X_THRESH)*NU_over_EV), -(astro_params_hf->X_RAY_SPEC_INDEX)-1);
}
double integrand_in_nu_ion_integral(double nu, void * params){
    double species_sum, F_i;
    float x_e = *(double *) params;

    // photoionization of HI, prodicing e- of energy h*(nu - nu_HI)
    F_i = interp_nion_HI((nu - NUIONIZATION)/NU_over_EV, x_e) +
    interp_nion_HeI((nu - NUIONIZATION)/NU_over_EV, x_e) +
    interp_nion_HeII((nu - NUIONIZATION)/NU_over_EV, x_e) + 1;
    species_sum = F_i * f_H * (1-x_e) * HI_ion_crosssec(nu);

    // photoionization of HeI, prodicing e- of energy h*(nu - nu_HeI)
    F_i = interp_nion_HI((nu - HeI_NUIONIZATION)/NU_over_EV, x_e) +
    interp_nion_HeI((nu - HeI_NUIONIZATION)/NU_over_EV, x_e) +
    interp_nion_HeII((nu - HeI_NUIONIZATION)/NU_over_EV, x_e) + 1;
    species_sum += F_i * f_He * (1-x_e) * HeI_ion_crosssec(nu);

    // photoionization of HeII, prodicing e- of energy h*(nu - nu_HeII)
    F_i = interp_nion_HI((nu - HeII_NUIONIZATION)/NU_over_EV, x_e) +
    interp_nion_HeI((nu - HeII_NUIONIZATION)/NU_over_EV, x_e) +
    interp_nion_HeII((nu - HeII_NUIONIZATION)/NU_over_EV, x_e) + 1;
    species_sum += F_i * f_He * x_e * HeII_ion_crosssec(nu);

    return species_sum * pow(nu/((astro_params_hf->NU_X_THRESH)*NU_over_EV), -(astro_params_hf->X_RAY_SPEC_INDEX)-1);
}
double integrand_in_nu_lya_integral(double nu, void * params){
    double species_sum;
    float x_e = *(double *) params;

    // HI
    species_sum = interp_n_Lya((nu - NUIONIZATION)/NU_over_EV, x_e)
    * f_H * (double)(1-x_e) * HI_ion_crosssec(nu);

    // HeI
    species_sum += interp_n_Lya((nu - HeI_NUIONIZATION)/NU_over_EV, x_e)
    * f_He * (double)(1-x_e) * HeI_ion_crosssec(nu);

    // HeII
    species_sum += interp_n_Lya((nu - HeII_NUIONIZATION)/NU_over_EV, x_e)
    * f_He * (double)x_e * HeII_ion_crosssec(nu);

    return species_sum * pow(nu/((astro_params_hf->NU_X_THRESH)*NU_over_EV), -(astro_params_hf->X_RAY_SPEC_INDEX)-1);
}
double integrate_over_nu(double zp, double local_x_e, double lower_int_limit, int FLAG){
    double result, error;
    double rel_tol  = 0.01; //<- relative tolerance
    gsl_function F;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

    F.params = &local_x_e;

    if (FLAG==0)
        F.function = &integrand_in_nu_heat_integral;
    else if (FLAG==1)
        F.function = &integrand_in_nu_ion_integral;
    else {
        F.function = &integrand_in_nu_lya_integral;
    }

    int status;
    gsl_set_error_handler_off();
    status = gsl_integration_qag (&F, lower_int_limit, global_params.NU_X_MAX*NU_over_EV, 0, rel_tol, 1000, GSL_INTEG_GAUSS15, w, &result, &error);

    if(status!=0){
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_int_limit,global_params.NU_X_MAX*NU_over_EV,rel_tol,result,error);
        LOG_ERROR("data: zp=%e local_x_e=%e FLAG=%d",zp,local_x_e,FLAG);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    // if it is the Lya integral, add prefactor
    if (FLAG == 2)
        return result * C / FOURPI / Ly_alpha_HZ / hubble(zp);

    //       if (isnan(result))
    //     fprintf(stderr, "We have a NaN in the intergrator with calling params: %g,%g,%g,%i\n", zp, local_x_e, lower_int_limit, FLAG);

    return result;
}





// Calculates the optical depth for a photon arriving at z = zp with frequency nu,
// emitted at z = zpp.
// The filling factor of neutral IGM at zp is HI_filling_factor_zp.
//
// *** Brad Greig (22/11/2016) ***
// An approximation to evaluate this using the global averaged filling factor at that zp. Same
// approximation that uses the global averaged x_e
//
// Used to speed up Ts.c and remove parameter dependence reducing the dimensionality of the required interpolation
// table in the new version of 21CMMC (including spin-temperature fluctuations).

typedef struct{
    double nu_0, x_e, x_e_ave, ion_eff, ion_eff_MINI, log10_Mturn_MINI, LOG10_MTURN_INT;
} tauX_params;
double tauX_integrand_MINI(double zhat, void *params){

    double n, drpropdz, nuhat, sigma_tilde, fcoll, HI_filling_factor_zhat;

    int z_fcoll_int1,z_fcoll_int2;
    float z_fcoll_val1,z_fcoll_val2;
    // New in v1.4
    double log10_Mturn_MINI, LOG10_MTURN_INT;
    double fcoll_MINI_left, fcoll_MINI_right, fcoll_MINI;

    int redshift_int_fcollz, log10_Mturn_MINI_int_fcollz;
    float redshift_table_fcollz, log10_Mturn_MINI_table_fcollz;

    float Mcrit_atom_val, Mlim_Fstar, Mlim_Fesc, Mlim_Fstar_MINI, Mlim_Fesc_MINI;

    tauX_params *p = (tauX_params *) params;

    drpropdz = C * dtdz(zhat);
    n = N_b0 * pow(1+zhat, 3);
    nuhat = p->nu_0 * (1+zhat);
    log10_Mturn_MINI = p->log10_Mturn_MINI;
    LOG10_MTURN_INT = p->LOG10_MTURN_INT;

    if(user_params_hf->USE_INTERPOLATION_TABLES) {
        redshift_int_fcollz = (int)floor( ( zhat - determine_zpp_min )/zpp_bin_width );
        redshift_table_fcollz = determine_zpp_min + zpp_bin_width*(float)redshift_int_fcollz;

        fcoll = Nion_z_val[redshift_int_fcollz] + ( zhat - redshift_table_fcollz )*( Nion_z_val[redshift_int_fcollz+1] - Nion_z_val[redshift_int_fcollz] )/(zpp_bin_width);

        log10_Mturn_MINI_int_fcollz = (int)floor( ( log10_Mturn_MINI - LOG10_MTURN_MIN) / LOG10_MTURN_INT);
        log10_Mturn_MINI_table_fcollz = LOG10_MTURN_MIN + LOG10_MTURN_INT * (float)log10_Mturn_MINI_int_fcollz;

        fcoll_MINI_left =    Nion_z_val_MINI[redshift_int_fcollz  + zpp_interp_points_SFR *  log10_Mturn_MINI_int_fcollz   ] + ( zhat - redshift_table_fcollz ) / (zpp_bin_width)*\
                                      ( Nion_z_val_MINI[redshift_int_fcollz+1+ zpp_interp_points_SFR *  log10_Mturn_MINI_int_fcollz   ] -\
                                        Nion_z_val_MINI[redshift_int_fcollz  + zpp_interp_points_SFR *  log10_Mturn_MINI_int_fcollz   ] );
        fcoll_MINI_right =   Nion_z_val_MINI[redshift_int_fcollz  + zpp_interp_points_SFR * (log10_Mturn_MINI_int_fcollz+1)] + ( zhat - redshift_table_fcollz ) / (zpp_bin_width)*\
                                      ( Nion_z_val_MINI[redshift_int_fcollz+1+ zpp_interp_points_SFR * (log10_Mturn_MINI_int_fcollz+1)] -\
                                        Nion_z_val_MINI[redshift_int_fcollz  + zpp_interp_points_SFR * (log10_Mturn_MINI_int_fcollz+1)] );
        fcoll_MINI = fcoll_MINI_left + (log10_Mturn_MINI - log10_Mturn_MINI_table_fcollz) / LOG10_MTURN_INT * (fcoll_MINI_right - fcoll_MINI_left);
    }
    else {

        Mcrit_atom_val = atomic_cooling_threshold(zhat);

        Mlim_Fstar = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_hf->ALPHA_STAR, astro_params_hf->F_STAR10);
        Mlim_Fesc = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_hf->ALPHA_ESC, astro_params_hf->F_ESC10);

        fcoll = Nion_General(zhat, global_params.M_MIN_INTEGRAL, Mcrit_atom_val, astro_params_hf->ALPHA_STAR, astro_params_hf->ALPHA_ESC,
                             astro_params_hf->F_STAR10, astro_params_hf->F_ESC10, Mlim_Fstar, Mlim_Fesc);

        Mlim_Fstar_MINI = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_hf->ALPHA_STAR_MINI,
                                               astro_params_hf->F_STAR7_MINI * pow(1e3, astro_params_hf->ALPHA_STAR_MINI));
        Mlim_Fesc_MINI = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params_hf->ALPHA_ESC,
                                              astro_params_hf->F_ESC7_MINI * pow(1e3, astro_params_hf->ALPHA_ESC));

        fcoll_MINI = Nion_General_MINI(zhat, global_params.M_MIN_INTEGRAL, pow(10.,log10_Mturn_MINI), Mcrit_atom_val, astro_params_hf->ALPHA_STAR_MINI, astro_params_hf->ALPHA_ESC,
                          astro_params_hf->F_STAR7_MINI, astro_params_hf->F_ESC7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI);
    }

    if ((fcoll < 1e-20) && (fcoll_MINI < 1e-20)){
        HI_filling_factor_zhat = 1;
    }
    else{
        HI_filling_factor_zhat = 1 - (p->ion_eff * fcoll + p->ion_eff_MINI * fcoll_MINI)/(1.0 - p->x_e_ave); //simplification to use the <x_e> value at zp and not zhat.  should'nt matter much since the evolution in x_e_ave is slower than fcoll.  in principle should make an array to store past values of x_e_ave..
    }
    if (HI_filling_factor_zhat < 1e-4) HI_filling_factor_zhat = 1e-4; //set a floor for post-reionization stability

    sigma_tilde = species_weighted_x_ray_cross_section(nuhat, p->x_e);

    return drpropdz * n * HI_filling_factor_zhat * sigma_tilde;
}
double tauX_integrand(double zhat, void *params){

    double n, drpropdz, nuhat, sigma_tilde, fcoll, HI_filling_factor_zhat;

    int z_fcoll_int1,z_fcoll_int2;
    float z_fcoll_val1,z_fcoll_val2;
    // New in v1.4
    float Splined_Fcollz_mean;

    int redshift_int_fcollz;
    float redshift_table_fcollz;

    float Mlim_Fstar, Mlim_Fesc;

    tauX_params *p = (tauX_params *) params;

    drpropdz = C * dtdz(zhat);
    n = N_b0 * pow(1+zhat, 3);
    nuhat = p->nu_0 * (1+zhat);

    // New in v1.4
    if (flag_options_hf->USE_MASS_DEPENDENT_ZETA) {

        if(user_params_hf->USE_INTERPOLATION_TABLES) {
            redshift_int_fcollz = (int)floor( ( zhat - determine_zpp_min )/zpp_bin_width );
            redshift_table_fcollz = determine_zpp_min + zpp_bin_width*(float)redshift_int_fcollz;

            fcoll = Nion_z_val[redshift_int_fcollz] + ( zhat - redshift_table_fcollz )*( Nion_z_val[redshift_int_fcollz+1] - Nion_z_val[redshift_int_fcollz] )/(zpp_bin_width);
        }
        else {

            Mlim_Fstar = Mass_limit_bisection(astro_params_hf->M_TURN/50., global_params.M_MAX_INTEGRAL, astro_params_hf->ALPHA_STAR, astro_params_hf->F_STAR10);
            Mlim_Fesc = Mass_limit_bisection(astro_params_hf->M_TURN/50., global_params.M_MAX_INTEGRAL, astro_params_hf->ALPHA_ESC, astro_params_hf->F_ESC10);

            fcoll = Nion_General(zhat, astro_params_hf->M_TURN/50., astro_params_hf->M_TURN/50., astro_params_hf->ALPHA_STAR, astro_params_hf->ALPHA_ESC,
                                 astro_params_hf->F_STAR10, astro_params_hf->F_ESC10, Mlim_Fstar, Mlim_Fesc);
        }
    }
    else {

        if(user_params_hf->USE_INTERPOLATION_TABLES) {
            z_fcoll_int1 = (int)floor(( zhat - zmin_1DTable )/zbin_width_1DTable);
            z_fcoll_int2 = z_fcoll_int1 + 1;

            z_fcoll_val1 = zmin_1DTable + zbin_width_1DTable*(float)z_fcoll_int1;
            z_fcoll_val2 = zmin_1DTable + zbin_width_1DTable*(float)z_fcoll_int2;

            fcoll = FgtrM_1DTable_linear[z_fcoll_int1] + ( zhat - z_fcoll_val1 )*( FgtrM_1DTable_linear[z_fcoll_int2] - FgtrM_1DTable_linear[z_fcoll_int1] )/( z_fcoll_val2 - z_fcoll_val1 );

            fcoll = pow(10.,fcoll);
        }
        else {
            if(flag_options_hf->M_MIN_in_Mass) {
                fcoll = FgtrM(zhat, (astro_params_hf->M_TURN)/50.);
            }
            else {
                fcoll = FgtrM(zhat, get_M_min_ion(zhat));
            }
        }
    }
    if (fcoll < 1e-20)
        HI_filling_factor_zhat = 1;
    else
        HI_filling_factor_zhat = 1 - p->ion_eff * fcoll/(1.0 - p->x_e_ave); //simplification to use the <x_e> value at zp and not zhat.  should'nt matter much since the evolution in x_e_ave is slower than fcoll.  in principle should make an array to store past values of x_e_ave..
    if (HI_filling_factor_zhat < 1e-4) HI_filling_factor_zhat = 1e-4; //set a floor for post-reionization stability

    sigma_tilde = species_weighted_x_ray_cross_section(nuhat, p->x_e);

    return drpropdz * n * HI_filling_factor_zhat * sigma_tilde;
}
double tauX_MINI(double nu, double x_e, double x_e_ave, double zp, double zpp, double HI_filling_factor_zp, double log10_Mturn_MINI, double LOG10_MTURN_INT){

    double result, error, fcoll;

    gsl_function F;

    double rel_tol  = 0.005; //<- relative tolerance
    //    double rel_tol  = 0.01; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    tauX_params p;

    int z_fcoll_int1,z_fcoll_int2;
    float z_fcoll_val1,z_fcoll_val2;

    float Splined_Fcollz_mean;

    int redshift_int_fcollz;
    float redshift_table_fcollz;

    //     if (DEBUG_ON)
    //     printf("in taux, parameters are: %e, %e, %f, %f, %e\n", nu, x_e, zp, zpp, HI_filling_factor_zp);

    F.function = &tauX_integrand_MINI;
    p.nu_0 = nu/(1+zp);
    p.x_e = x_e;
    p.x_e_ave = x_e_ave;
    p.ion_eff = global_params.Pop2_ion*astro_params_hf->F_STAR10*astro_params_hf->F_ESC10;
    p.ion_eff_MINI = global_params.Pop3_ion*astro_params_hf->F_STAR7_MINI*astro_params_hf->F_ESC7_MINI;
    p.log10_Mturn_MINI = log10_Mturn_MINI;
    p.LOG10_MTURN_INT = LOG10_MTURN_INT;

    F.params = &p;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, zpp, zp, 0, rel_tol,1000, GSL_INTEG_GAUSS15, w, &result, &error);

    if(status!=0){
        LOG_ERROR("(function argument): zp=%e zpp=%e rel_tol=%e result=%e error=%e",zp,zpp,rel_tol,result,error);
        LOG_ERROR("data: nu=%e nu_0=%e x_e=%e x_e_ave=%e",nu,p.nu_0,p.x_e,p.x_e_ave);
        LOG_ERROR("data: ion_eff=%e ion_eff_MINI=%e log10_Mturn_MINI=%e LOG10_MTURN_INT=%e",p.ion_eff,p.ion_eff_MINI,p.log10_Mturn_MINI,p.LOG10_MTURN_INT);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    //     if (DEBUG_ON)
    //     printf("returning from tauX, return value=%e\n", result);

    return result;
}

double tauX(double nu, double x_e, double x_e_ave, double zp, double zpp, double HI_filling_factor_zp){

    double result, error, fcoll;

    gsl_function F;

    double rel_tol  = 0.005; //<- relative tolerance
    //    double rel_tol  = 0.01; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    tauX_params p;

    int z_fcoll_int1,z_fcoll_int2;
    float z_fcoll_val1,z_fcoll_val2;

    float Splined_Fcollz_mean;

    int redshift_int_fcollz;
    float redshift_table_fcollz;

    F.function = &tauX_integrand;
    p.nu_0 = nu/(1+zp);
    p.x_e = x_e;
    p.x_e_ave = x_e_ave;

    if(flag_options_hf->USE_MASS_DEPENDENT_ZETA) {
        p.ion_eff = global_params.Pop2_ion*astro_params_hf->F_STAR10*astro_params_hf->F_ESC10;
    }
    else {
        if (HI_filling_factor_zp > FRACT_FLOAT_ERR){

            if(user_params_hf->USE_INTERPOLATION_TABLES) {
                z_fcoll_int1 = (int)floor(( zp - zmin_1DTable )/zbin_width_1DTable);
                z_fcoll_int2 = z_fcoll_int1 + 1;

                z_fcoll_val1 = zmin_1DTable + zbin_width_1DTable*(float)z_fcoll_int1;
                z_fcoll_val2 = zmin_1DTable + zbin_width_1DTable*(float)z_fcoll_int2;

                fcoll = FgtrM_1DTable_linear[z_fcoll_int1] + ( zp - z_fcoll_val1 )*( FgtrM_1DTable_linear[z_fcoll_int2] - FgtrM_1DTable_linear[z_fcoll_int1] )/( z_fcoll_val2 - z_fcoll_val1 );

                fcoll = pow(10.,fcoll);
            }
            else {
                if(flag_options_hf->M_MIN_in_Mass) {
                    fcoll = FgtrM(zp, (astro_params_hf->M_TURN)/50.);
                }
                else {
                    fcoll = FgtrM(zp, get_M_min_ion(zp));
                }
            }


            p.ion_eff = (1.0 - HI_filling_factor_zp) / fcoll * (1.0 - x_e_ave);
            PS_ION_EFF = p.ion_eff;

        }
        else {
            p.ion_eff = PS_ION_EFF; // uses the previous one in post reionization regime
        }
    }

    F.params = &p;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, zpp, zp, 0, rel_tol,1000, GSL_INTEG_GAUSS15, w, &result, &error);

    if(status!=0){
        LOG_ERROR("(function argument): zp=%e zpp=%e rel_tol=%e result=%e error=%e",zp,zpp,rel_tol,result,error);
        LOG_ERROR("data: nu=%e nu_0=%e x_e=%e x_e_ave=%e ion_eff=%e",nu,nu/(1+zp),x_e,x_e_ave,p.ion_eff);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return result;
}

// Returns the frequency threshold where \tau_X = 1, given parameter values of
// electron fraction in the IGM outside of HII regions, x_e,
// recieved redshift, zp, and emitted redshift, zpp.
//
// *** Brad Greig (22/11/2016) ***
// An approximation to evaluate this using the global averaged filling factor at that zp. Same
// approximation that uses the global averaged x_e
//
// Used to speed up Ts.c and remove parameter dependence reducing the dimensionality of the required interpolation
// table in the new version of 21CMMC (including spin-temperature fluctuations).

typedef struct{
    double x_e, zp, zpp, HI_filling_factor_zp,log10_Mturn_MINI,LOG10_MTURN_INT;
} nu_tau_one_params;
double nu_tau_one_helper_MINI(double nu, void * params){
    nu_tau_one_params *p = (nu_tau_one_params *) params;
    return tauX_MINI(nu, p->x_e, p->x_e, p->zp, p->zpp, p->HI_filling_factor_zp,p->log10_Mturn_MINI,p->LOG10_MTURN_INT) - 1;
}
double nu_tau_one_helper(double nu, void * params){
    nu_tau_one_params *p = (nu_tau_one_params *) params;
    return tauX(nu, p->x_e, p->x_e, p->zp, p->zpp, p->HI_filling_factor_zp) - 1;
}
double nu_tau_one_MINI(double zp, double zpp, double x_e, double HI_filling_factor_zp, double log10_Mturn_MINI, double LOG10_MTURN_INT){

    int status, iter, max_iter;
    const gsl_root_fsolver_type * T;
    gsl_root_fsolver * s;
    gsl_function F;
    double x_lo, x_hi, r=0;
    double relative_error = 0.02;
    nu_tau_one_params p;

    // check if too ionized
    if (x_e > 0.9999){
//        LOG_ERROR("x_e value is too close to 1 for convergence.");
        return astro_params_hf->NU_X_THRESH;
    }

    // select solver and allocate memory
    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T); // non-derivative based Brent method
    if (!s){
        LOG_ERROR("Unable to allocate memory.");
        Throw(MemoryAllocError);
    }

    //check if lower bound has null
    if (tauX_MINI(HeI_NUIONIZATION, x_e, x_e, zp, zpp, HI_filling_factor_zp, log10_Mturn_MINI, LOG10_MTURN_INT) < 1)
        return HeI_NUIONIZATION;

    // set frequency boundary values
    x_lo= HeI_NUIONIZATION;
    x_hi = 1e6 * NU_over_EV;

    // select function we wish to solve
    p.x_e = x_e;
    p.zp = zp;
    p.zpp = zpp;
    p.HI_filling_factor_zp = HI_filling_factor_zp;
    p.log10_Mturn_MINI = log10_Mturn_MINI;
    p.LOG10_MTURN_INT = LOG10_MTURN_INT;
    F.function = &nu_tau_one_helper_MINI;
    F.params = &p;
    gsl_root_fsolver_set (s, &F, x_lo, x_hi);

    // iterate until we guess close enough
    iter = 0;
    max_iter = 100;
    do{
        iter++;
        status = gsl_root_fsolver_iterate (s);
        r = gsl_root_fsolver_root (s);
        x_lo = gsl_root_fsolver_x_lower (s);
        x_hi = gsl_root_fsolver_x_upper (s);
        status = gsl_root_test_interval (x_lo, x_hi, 0, relative_error);
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    // deallocate and return
    gsl_root_fsolver_free (s);

    if(!isfinite(r)){
        LOG_ERROR("Value for nu_tau_one_MINI is infinite or NAN");
//        Throw(ParameterError);
        Throw(InfinityorNaNError);
    }

    return r;
}

double nu_tau_one(double zp, double zpp, double x_e, double HI_filling_factor_zp){

    int status, iter, max_iter;
    const gsl_root_fsolver_type * T;
    gsl_root_fsolver * s;
    gsl_function F;
    double x_lo, x_hi, r=0;
    double relative_error = 0.02;
    nu_tau_one_params p;

    // check if too ionized
    if (x_e > 0.9999){
//        LOG_ERROR("x_e value is too close to 1 for convergence.");
        return astro_params_hf->NU_X_THRESH;
    }

    // select solver and allocate memory
    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T); // non-derivative based Brent method
    if (!s){
        LOG_ERROR("Unable to allocate memory.");
        Throw(MemoryAllocError);
    }

    //check if lower bound has null
    if (tauX(HeI_NUIONIZATION, x_e, x_e, zp, zpp, HI_filling_factor_zp) < 1)
        return HeI_NUIONIZATION;

    // set frequency boundary values
    x_lo= HeI_NUIONIZATION;
    x_hi = 1e6 * NU_over_EV;

    // select function we wish to solve
    p.x_e = x_e;
    p.zp = zp;
    p.zpp = zpp;
    p.HI_filling_factor_zp = HI_filling_factor_zp;
    F.function = &nu_tau_one_helper;
    F.params = &p;
    gsl_root_fsolver_set (s, &F, x_lo, x_hi);

    // iterate until we guess close enough
    iter = 0;
    max_iter = 100;
    do{
        iter++;
        status = gsl_root_fsolver_iterate (s);
        r = gsl_root_fsolver_root (s);
        x_lo = gsl_root_fsolver_x_lower (s);
        x_hi = gsl_root_fsolver_x_upper (s);
        status = gsl_root_test_interval (x_lo, x_hi, 0, relative_error);
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    // deallocate and return
    gsl_root_fsolver_free (s);

    if(!isfinite(r)){
        LOG_ERROR("nu_tau_one is infinite or NAN");
//        Throw(ParameterError);
        Throw(InfinityorNaNError);
    }

    return r;
}


//  The total weighted HI + HeI + HeII  cross-section in pcm^-2
//  technically, the x_e should be local, line of sight (not global) here,
//  but that would be very slow...

double species_weighted_x_ray_cross_section(double nu, double x_e){
    double HI_factor, HeI_factor, HeII_factor;

    HI_factor = f_H * (1-x_e) * HI_ion_crosssec(nu);
    HeI_factor = f_He * (1-x_e) * HeI_ion_crosssec(nu);
    HeII_factor = f_He * x_e * HeII_ion_crosssec(nu);

    return HI_factor + HeI_factor + HeII_factor;
}


// * Returns the maximum redshift at which a Lyn transition contributes to Lya flux at z * //
float zmax(float z, int n){
    double num, denom;
    num = 1 - pow(n+1, -2);
    denom = 1 - pow(n, -2);
    return (1+z)*num/denom - 1;
}


// JordanFlitter: I removed atomic_cooling_threshold() and molecular_cooling_threshold() from UsefulFunctions.c
//                to here. This allows us to use the user's astrophysical parameters in these functions
double atomic_cooling_threshold(float z){
    return TtoM(z, astro_params_hf->ION_Tvir_MIN, mu_b_ionized); // JordanFlitter: I changed the constant value to the general case
                                                                 // JordanFlitter: I also take the user's T_VIR
}

double molecular_cooling_threshold(float z){
    return TtoM(z, 600, mu_b_neutral); // JordanFlitter: I changed the constant value to the general case
}

//**************************************************************************************//
//**************************************************************************************//
// JordanFlitter: Functions for Lya-Heating, mostly written by Debanjan Sarkar

double Energy_Lya_heating(double Tk, double Ts, double tau_gp, int flag)
{
    double ans;
    static double dEC[nT * nT * ngp];
    static double dEI[nT * nT * ngp];
    double dummy_heat;
    int ii, jj, kk, index;
    FILE *F;

    char filename[500];

    if (flag == 1) {
        // Read in the Lya heating data
        sprintf(filename,"%s/%s",global_params.external_table_path,LYA_HEATING_FILENAME);
        if ( !(F=fopen(filename, "r")) ){
            LOG_ERROR("Energy_Lya_heating: Unable to open file: %s for reading.", filename);
            Throw(IOError);
        }

        for (ii=0;ii<nT;ii++){
      		for (jj=0;jj<nT;jj++){
      			for (kk=0;kk<ngp;kk++){
      			index = ii*nT*ngp + jj*ngp + kk;
      			//fscanf(F,"%lf %lf %lf %lf %lf",&dummy_heat,&dummy_heat,&dummy_heat,&dEC[index],&dEI[index]);
            fscanf(F,"%lf %lf",&dEC[index],&dEI[index]);
      			}
      		}
      	}

        fclose(F);
        return 0;
    }

    if (flag == 2) {
      ans = interpolate_heating_efficiencies(Tk, Ts, tau_gp, dEC); //Continuum
    }
    if (flag == 3) {
      ans = interpolate_heating_efficiencies(Tk, Ts, tau_gp, dEI); //Injected
    }
    return ans;
}


typedef struct{
  double ah, gammah, etah;
} h_params;

//(eq:14) of Reis, Fialkov, Barkana (2021) [2101.01777]
double Jx_red (double x, void * params) {

	h_params *p = (h_params *) params;

	double integrand (double z, void * params){
		double jx, xz;
		xz=z*z*z - x*x*x;
		jx = exp( 2.*PI*p->gammah*xz/3./p->ah + 2.*p->etah*(z-x))*z*z;
		return jx*(2.*PI*p->gammah/p->ah);
	  }


  double result, error;
  gsl_function F;
  double rel_tol  = 0.001; //<- relative tolerance
  gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc (100);

  F.function = &integrand;
  F.params = &p;

  gsl_integration_cquad(&F, -1000.0, x, 0, rel_tol, w, &result, &error, NULL);

  gsl_integration_cquad_workspace_free (w);

  return result;
}


//(eq:15) of Reis, Fialkov, Barkana (2021) [2101.01777]
double Jx_blue (double x, void * params) {

  h_params *p = (h_params *) params;

  double zeta = sqrt(16.*p->etah*p->etah*p->etah*p->ah/9./PI/p->gammah);

  return exp(-1.69*pow(zeta,2./3.))*exp(-2.*PI*p->gammah*x*x*x/3./p->ah - 2.*p->etah*x);

}


double J_continuum (double x, void * params){

	h_params *p = (h_params *) params;

	return 1. - Jx_red(x,p);

}

double J_injected (double x, void * params){

	h_params *p = (h_params *) params;

	if (x>0)
		return - Jx_blue(x,p);
	else if (x==0)
		return 0.5 - Jx_red(x,p);
	else
		return 1. - Jx_red(x,p);
}


double delta_E(double ah, double gammah, double etah, double delta_dopp, int ci_flag){

	double llim = -200.0, ulim = 200.0;

    h_params p;
    double result, error;
    gsl_function F;
    double rel_tol  = 0.001; //<- relative tolerance
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc (100);

	if (ci_flag == 0)//Continuum photons
    	F.function = &J_continuum;

	else if (ci_flag == 1)//Continuum photons
    	F.function = &J_injected;

	else{
		printf("\n Check the input flag of this function: Continuum == 0, Inject ==1; else code will be dumped;\n");
		exit(0);
	}


	p.ah = ah;
	p.gammah = gammah;
	p.etah = etah;
	F.params = &p;

	gsl_integration_cquad(&F, llim, ulim, 0, rel_tol, w, &result, &error, NULL);

    gsl_integration_cquad_workspace_free (w);

	return result * hplank * delta_dopp;

}


//Interpolation function for Lyman-alpha heating efficiencies : DS
double interpolate_heating_efficiencies(double tk, double ts, double taugp, double *arrE){
  tk = log10(tk);
  if(tk < Tk_min) {
        tk = Tk_min;
    }
  if(tk > Tk_max) {
        tk = Tk_max;
    }

  ts = log10(ts);
  if(ts < Ts_min) {
        ts = Ts_min;
    }
  if(ts > Ts_max) {
        ts = Ts_max;
    }

  taugp = log10(taugp);

  if(taugp < taugp_min) {
        taugp = taugp_min;
    }
  if(taugp > taugp_max) {
        taugp = taugp_max;
    }
	//find the nearest value
	int find_nearest_point(double min, double max, int n, double value){
	int pos=0;
  double dn = (max - min)/(n-1);
        if (value<=(min+dn)) pos=0;              // ensures we are in the first point
        else if (value>=max) pos = n-2;
	else pos = (int) ((value - min)/dn);

	//printf("%d\n",pos);
	return pos;

	}

	//find x-y-z position in an 1D array
	//x=Tk, y=Ts, z=Tau_GP
	int find_xyz_pos(int xpos, int ypos, int zpos, int len_yarr, int len_zarr){

		int pxyz = xpos*len_yarr*len_zarr + ypos*len_zarr + zpos;
		return pxyz;

	}

	int itk, its, itaugp, idec;
	itk = find_nearest_point(Tk_min, Tk_max, nT, tk);
	its = find_nearest_point(Ts_min, Ts_max, nT, ts);
	itaugp = find_nearest_point(taugp_min, taugp_max, ngp, taugp);

	idec = find_xyz_pos(itk, its, itaugp, nT, ngp);

	double x0,x1,y0,y1,z0,z1,xd, yd, zd;
	double c000, c100, c001, c101, c010, c110,c011,c111;
	double c00,c01,c10,c11,c0,c1,c;

  x0 =  Tk_min + itk*(Tk_max - Tk_min)/(nT-1);        // Making these (nT-1) ensures we reach the correct edge value
	x1 =  Tk_min + (itk+1)*(Tk_max - Tk_min)/(nT-1);

  y0 =  Ts_min + its*(Ts_max - Ts_min)/(nT-1);
  y1 =  Ts_min + (its+1)*(Ts_max - Ts_min)/(nT-1);

  z0 = taugp_min + itaugp*(taugp_max - taugp_min)/(ngp-1);
  z1 = taugp_min + (itaugp+1)*(taugp_max - taugp_min)/(ngp-1);

	xd = (tk - x0)/(x1 - x0);
	yd = (ts - y0)/(y1 - y0);
	zd = (taugp - z0)/(z1 - z0);

	c000 = arrE [find_xyz_pos(itk, its, itaugp, nT, ngp)];
	c100 = arrE [find_xyz_pos(itk+1, its, itaugp, nT, ngp)];
	c001 = arrE [find_xyz_pos(itk, its, itaugp+1, nT, ngp)];
	c101 = arrE [find_xyz_pos(itk+1, its, itaugp+1, nT, ngp)];
	c010 = arrE [find_xyz_pos(itk, its+1, itaugp, nT, ngp)];
	c110 = arrE [find_xyz_pos(itk+1, its+1, itaugp, nT, ngp)];
	c011 = arrE [find_xyz_pos(itk, its+1, itaugp+1, nT, ngp)];
	c111 = arrE [find_xyz_pos(itk+1, its+1, itaugp+1, nT, ngp)];

	c00 = c000*(1.-xd) + c100*xd;
	c01 = c001*(1.-xd) + c101*xd;
	c10 = c010*(1.-xd) + c110*xd;
	c11 = c011*(1.-xd) + c111*xd;

	c0 = c00*(1.-yd) + c10*yd;
	c1 = c01*(1.-yd) + c11*yd;

	c = c0*(1.-zd) + c1*zd;

	//printf("%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",idec,c000,c100,c001,c101,c010,c110,c011,c111,c);

	return c;
}

//*************************************************************************************************************************************
//*************************************************************************************************************************************
//*************************************************       NEW SDM FUNCTIONS!!!         ************************************************
//*************************************************************************************************************************************
//*************************************************************************************************************************************

//JordanFlitter: SDM temperature. Similar to the T_RECFAST function
double T_chi_RECFAST(float z, int flag)
{
    double ans;
    static double log_z_arr[Z_ARRAY_NPTS];
    static double log_T_chi_arr[Z_ARRAY_NPTS];
    static gsl_interp_accel *acc;
    static gsl_spline *spline;

    if (flag == 1) {

        for (size_t i = 0; i < Z_ARRAY_NPTS; i++) {
            log_z_arr[i] = global_params.LOG_Z_ARR[i];
            log_T_chi_arr[i] = global_params.LOG_T_chi[i];
        }

        // Set up spline table
        acc   = gsl_interp_accel_alloc ();
        spline  = gsl_spline_alloc (gsl_interp_cspline, Z_ARRAY_NPTS);
        gsl_spline_init(spline, log_z_arr, log_T_chi_arr, Z_ARRAY_NPTS);

        return 0;
    }

    if (flag == 2) {
        // Free memory
        gsl_spline_free (spline);
        gsl_interp_accel_free(acc);
        return 0;
    }

    // Convert to log10
    double log10_z = log10(z);

    if (log10_z > global_params.LOG_Z_ARR[Z_ARRAY_NPTS-1]) { // Called at z>500! Bail out
        LOG_ERROR("Called T_chi_RECFAST with z=%f.", z);
        Throw 1;
    }
    else { // Do spline
        ans = gsl_spline_eval (spline, log10_z, acc);
    }
    return pow(10.,ans);
}

//JordanFlitter: Relative velocity between baryons and SDM. Similar to the T_RECFAST function
double V_chi_b_RECFAST(float z, int flag)
{
    double ans;
    static double log_z_arr[Z_ARRAY_NPTS];
    static double log_V_chi_b_arr[Z_ARRAY_NPTS];
    static gsl_interp_accel *acc;
    static gsl_spline *spline;

    if (flag == 1) {

        for (size_t i = 0; i < Z_ARRAY_NPTS; i++) {
            log_z_arr[i] = global_params.LOG_Z_ARR[i];
            log_V_chi_b_arr[i] = global_params.LOG_V_chi_b[i];
        }

        // Set up spline table
        acc   = gsl_interp_accel_alloc ();
        spline  = gsl_spline_alloc (gsl_interp_cspline, Z_ARRAY_NPTS);
        gsl_spline_init(spline, log_z_arr, log_V_chi_b_arr, Z_ARRAY_NPTS);

        return 0;
    }

    if (flag == 2) {
        // Free memory
        gsl_spline_free (spline);
        gsl_interp_accel_free(acc);
        return 0;
    }

    // Convert to log10
    double log10_z = log10(z);

    if (log10_z > global_params.LOG_Z_ARR[Z_ARRAY_NPTS-1]) { // Called at z>500! Bail out
        LOG_ERROR("Called V_chi_b_RECFAST with z=%f.", z);
        Throw 1;
    }
    else { // Do spline
        ans = gsl_spline_eval (spline, log10_z, acc);
    }
    return pow(10.,ans);
}

// JordanFlitter: new function for HyRec (adopted from CLASS wrap_hyrec.c)
/**
 * Calculate the derivative of the hydrogen HII ionization fraction
 * Note: we treat xe and x_H to be the same, since after recombination x_He<<xe,x_H.
 * Therefore, we treat the output of this function to be dxe_dz
 *
 * @param rec_data   Input: pointer to HYREC_DATA structure
 * @param x_H        Input: hydrogen HII ionization fraction
 * @param xe         Input: total ionization fraction
 * @param nH         Input: comoving total number of hydrogen atoms in cm^-3
 * @param z          Input: current cosmological redshift
 * @param Hz         Input: current value of hubble parameter in 1/s
 * @param Tmat       Input: temperature of baryons in Kelvin
 * @param Trad       Input: temperature of photons in Kelvin
 * @return dx_H_dz  the derivative of the hydrogen HII ionization fraction
 */
double hyrec_dx_H_dz(HYREC_DATA *rec_data, double x_H, double xe, double nH, double z, double Hz, double Tmat, double Trad) {

  /** - define local variables */
  long iz = 0;
  int model;
  double Trad_phys;
  double dx_H_dz;

  /** - assign variables */
  rec_data->cosmo->inj_params->ion = 0.;
  rec_data->cosmo->inj_params->exclya = 0.;

  Trad_phys = Trad*kBoltz;

  if (Trad_phys <= TR_MIN || Tmat/Trad <= T_RATIO_MIN) { model = PEEBLES; }
  else { model = MODEL; }

  /** - convert to correct units, and retrieve derivative */
  // JordanFlitter: Note that I pass nH (and not nH*1e-6) to rec_dxHIIdlna since when we call to hyrec_dx_H_dz in 21cmFAST we give it nH in units of cm^-3
  dx_H_dz = -1./(1.+z)* rec_dxHIIdlna(rec_data, model, xe, x_H, nH, Hz, Tmat*kBoltz, Trad*kBoltz, iz, z);

  /** - do error management */
  if(rec_data->error != 0){
    LOG_ERROR("Error in rec_dxHIIdlna.");
    return -1.;
  }

  return dx_H_dz;
}

// JordanFlitter: New structure!
// It is useful since the function below needs to return multiple outputs (the fields of this structure)
typedef struct{
  double Q_dot_b, Q_dot_chi, D_V_chi_b, epsilon_b, epsilon_chi, T_bar_chi_b, Delta_T_b_chi;
} SDM_RATES;


// JordanFlitter: This function calculates heating rates and drag term in SDM universe, according to arXiv: 1509.00029
SDM_RATES SDM_derivatives(float zp, double x_e, double T_b, double T_chi, double V_chi_b, double delta_b, double delta_chi,
                     double H, double Gamma_C, double T_gamma, float dzp, double dT_b_2_dt_ext, double dT_chi_2_dt_ext){

  SDM_RATES SDM_rates;
  int t, t_max;
  double rho_chi, rho_b, rho_m, m_b, m_chi, n_b, n_chi, sigma_n, n_SDM, N_n, Gamma_chi_b;
  double epsilon_gamma_b, epsilon_chi_b, dV_chi_b_2_dz, dT_b_2_dt_Comp, dT_bar_chi_b_2_dt, d_Gamma_chi_b_2_dz;
  double m_t_array[2], n_t_array[2], rho_t_array[2], u_chi_t_array[2], r_t_array[2], alpha_n_array[2], beta_n_array[2], gamma_n_array[2], D_t_V_chi_b_array[2];
  float EPSILON_THRES;

  // Calculate densities
  sigma_n = pow(10.,-cosmo_params_hf->sigma_SDM); // cm^2
  n_SDM = cosmo_params_hf->SDM_INDEX;
  N_n = pow(2.,(n_SDM + 5.)/2.) * tgamma(3. + n_SDM/2.)/(3.*sqrt(PI));
  rho_chi = pow(10.,-cosmo_params_hf->f_chi)*(cosmo_params_hf->OMm-cosmo_params_hf->OMb) * RHOcrit_cgs * pow(1.+zp,3.) * (1.+delta_chi); // g/cm^3
  rho_b = cosmo_params_hf->OMb * RHOcrit_cgs * pow(1.+zp,3.) * (1.+delta_b); // g/cm^3
  rho_m = rho_b + rho_chi; // g/cm^3
  m_b = _m_H_*(No+_not4_*He_No)/((No+He_No)*(1.0+x_e)); // g
  m_chi = pow(10.,cosmo_params_hf->m_chi) * NU_over_EV*hplank /C/C; // g
  n_b = rho_b/m_b; // cm^-3
  n_chi = rho_chi/m_chi; // cm^-3
  // Define target particles
  if (user_params_hf->SDM_TARGET_TYPE == 1) { // baryons
      t_max = 1;
      m_t_array[0] = m_b; // g
      n_t_array[0] = n_b; // cm^-3
  }
  else if (user_params_hf->SDM_TARGET_TYPE == 2) { // protons and electrons
      t_max = 2;
      m_t_array[0] = m_p; // g
      n_t_array[0] = x_e*(No+He_No)*pow(1.+zp,3.); // cm^-3
      m_t_array[1] = m_e; // g
      n_t_array[1] = x_e*(No+He_No)*pow(1.+zp,3.); // cm^-3
  }
  else if (user_params_hf->SDM_TARGET_TYPE == 3) { // hydrogen nuclei
      t_max = 1;
      m_t_array[0] = _m_H_; // g
      n_t_array[0] = No*pow(1.+zp,3.); // cm^-3
  }
  else if (user_params_hf->SDM_TARGET_TYPE == 4) { // protons
      t_max = 1;
      m_t_array[0] = m_p; // g
      n_t_array[0] = x_e*(No+He_No)*pow(1.+zp,3.); // cm^-3
  }
  else if (user_params_hf->SDM_TARGET_TYPE == 5) { // electrons
      t_max = 1;
      m_t_array[0] = m_e; // g
      n_t_array[0] = x_e*(No+He_No)*pow(1.+zp,3.); // cm^-3
  }
  // Calculate Gamma_chi_b
  Gamma_chi_b = 0.;
  for(t=0; t<t_max; t++){
      rho_t_array[t] = m_t_array[t] * n_t_array[t]; // g/cm^3
      u_chi_t_array[t] = sqrt(k_B*T_chi/m_chi + k_B*T_b/m_t_array[t]); // cm/sec
      r_t_array[t] = V_chi_b/u_chi_t_array[t];
      alpha_n_array[t] = gsl_sf_hyperg_1F1(-(n_SDM+1.)/2., 5./2., -pow(r_t_array[t],2.)/2.);
      beta_n_array[t] = gsl_sf_hyperg_1F1(-(n_SDM+3.)/2., 3./2., -pow(r_t_array[t],2.)/2.);
      gamma_n_array[t] = gsl_sf_hyperg_1F1(-(n_SDM-1.)/2., 7./2., -pow(r_t_array[t],2.)/2.);
      Gamma_chi_b += rho_t_array[t]*(beta_n_array[t] - pow(r_t_array[t],2.)/3.*alpha_n_array[t])*pow(u_chi_t_array[t],n_SDM+1.)/pow(m_chi + m_t_array[t],2.); // g^-1 * cm^(-2+n_SDM) / sec^(n_SDM+1)
  }
  Gamma_chi_b *= (2.*N_n*sigma_n*rho_chi)/(n_b*pow(C,n_SDM)); // g^-1 * cm^(-2+n_SDM) / sec^(n_SDM+1) * g * cm^(2-n_SDM) * sec^n_SDM = 1/sec
  // Calculate Q_dot_b, Q_dot_chi and the drage term
  SDM_rates.Q_dot_b = 3./2. * Gamma_chi_b * k_B * (T_chi-T_b); // erg/sec
  SDM_rates.Q_dot_chi = 3./2. * (n_b/n_chi) * Gamma_chi_b * k_B * (T_b-T_chi); // erg/sec
  SDM_rates.D_V_chi_b = 0.;
  for(t=0; t<t_max; t++){
      D_t_V_chi_b_array[t] = ((rho_m * N_n * sigma_n * rho_t_array[t] * V_chi_b * pow(u_chi_t_array[t],n_SDM+1.) * alpha_n_array[t]) // g^2 * cm^{n_SDM-2} / sec^{n_SDM + 2}
                             /(rho_b * (m_t_array[t] + m_chi) * pow(C,n_SDM)) // g^2 * cm^(n_SDM-3) / sec^n_SDM
                             ); // cm/sec^2
      SDM_rates.Q_dot_b += (rho_chi/rho_m)*V_chi_b * m_chi*m_b/(m_chi+m_t_array[t]) * D_t_V_chi_b_array[t]; // g*cm^2/sec^3 = erg/sec
      SDM_rates.Q_dot_chi += (rho_b/rho_m)*V_chi_b * m_chi*m_t_array[t]/(m_chi+m_t_array[t]) * D_t_V_chi_b_array[t]; // g*cm^2/sec^3 = erg/sec
      SDM_rates.D_V_chi_b += D_t_V_chi_b_array[t]; // cm/sec^2
  }

  // Calculate epsilons
  epsilon_gamma_b = H/Gamma_C;
  SDM_rates.epsilon_b = H/Gamma_chi_b;
  SDM_rates.epsilon_chi = (n_chi/n_b)*SDM_rates.epsilon_b;
  epsilon_chi_b = SDM_rates.epsilon_b * n_chi/(n_chi+n_b);
  // Define EPSILON_THRES (for TCA-DM)
  if (zp > 100.){
      EPSILON_THRES = global_params.EPSILON_THRESH_HIGH_Z;
  }
  else {
      EPSILON_THRES = global_params.EPSILON_THRESH_LOW_Z;
  }
  // If we are in the DM-TCA, evolve the average and difference temperatures of the DM-baryons fluids
  if ((fabs(SDM_rates.epsilon_b) <= EPSILON_THRES) || (fabs(SDM_rates.epsilon_chi) <= EPSILON_THRES)){
      // Compute the relative velocity derivative
      dV_chi_b_2_dz = (V_chi_b/(1+zp) + SDM_rates.D_V_chi_b/H/(1+zp)); // cm/sec
      // Calculate Compton heating
      if ((fabs(epsilon_gamma_b) > global_params.EPSILON_THRESH_HIGH_Z) || !user_params_hf->USE_TCA_COMPTON) {
          // Calculate Compton derivative from its definition
          dT_b_2_dt_Comp = Gamma_C*(T_gamma-T_b); // K/sec
      }
      else {
          // Calculate Compton derivative in the Compton TCA
          dT_b_2_dt_Comp = H*(2.*T_b-T_gamma) - 2./(3.*k_B)*SDM_rates.Q_dot_b; // K/sec
      }
      // Evolve T_bar_chi_b
      SDM_rates.T_bar_chi_b = (T_chi+T_b)/2.; // K
      dT_bar_chi_b_2_dt = (-2.*H*SDM_rates.T_bar_chi_b
                           +n_b/(n_b+n_chi)*(dT_b_2_dt_Comp + dT_b_2_dt_ext) + n_chi/(n_b+n_chi)*dT_chi_2_dt_ext
                           +1./(n_b+n_chi)*(2./(3.*k_B))*(rho_b*rho_chi/rho_m)*V_chi_b*SDM_rates.D_V_chi_b); // K/sec
      SDM_rates.T_bar_chi_b -= dzp * dT_bar_chi_b_2_dt/H/(1+zp); // K
      // Calculate the derivative of Gamma_chi_b
      d_Gamma_chi_b_2_dz = 0.;
      for(t=0; t<t_max; t++){
        d_Gamma_chi_b_2_dz += ((rho_t_array[t]*pow(u_chi_t_array[t],n_SDM+1.))/pow(m_chi+m_t_array[t],2.) // g^-1 * cm^(n_SDM-2) / sec^(n_SDM+1)
                              *(3.*(beta_n_array[t]  - pow(r_t_array[t],2.)/3.*alpha_n_array[t])/(1.+zp)
                              + (r_t_array[t]/u_chi_t_array[t])*((n_SDM+1.)/3.)*(alpha_n_array[t] - pow(r_t_array[t],2.)/5.*gamma_n_array[t])*dV_chi_b_2_dz
                              - (n_SDM+1.)*(2.*pow(r_t_array[t],2.)/3.*alpha_n_array[t]-beta_n_array[t]- pow(r_t_array[t],4.)/15.*gamma_n_array[t])
                                          *(m_t_array[t]+m_chi)/(2.*m_t_array[t]*m_chi)*(k_B/pow(u_chi_t_array[t],2.))*dT_bar_chi_b_2_dt*(-1./H/(1+zp))) // dimensionless
                              ); // g^-1 * cm^(n_SDM-2) / sec^(n_SDM+1)
      d_Gamma_chi_b_2_dz *= (2.*N_n*sigma_n*rho_chi)/(n_b*pow(C,n_SDM)); // g^-1 * cm^(n_SDM-2) / sec^(n_SDM+1) * g * cm^(2-n_SDM) / sec^n_SDM = 1/sec
      }
      // Evolve epsilon_chi_b
      epsilon_chi_b -= (-dzp) * epsilon_chi_b * (dhubble_dz(zp)/H - d_Gamma_chi_b_2_dz/Gamma_chi_b);
      // Calculate temperature difference
      SDM_rates.Delta_T_b_chi = 0.;
      for(t=0; t<t_max; t++){
          SDM_rates.Delta_T_b_chi += (rho_b*m_t_array[t]-rho_chi*m_b)/(m_chi + m_t_array[t])*D_t_V_chi_b_array[t]; // g * cm^-2 * sec^-2
      }
      SDM_rates.Delta_T_b_chi *= (2.*V_chi_b*m_chi)/(3.*k_B*H*rho_m); // g * cm^-2 * sec^-2 * cm^4 * K / J = K
      SDM_rates.Delta_T_b_chi += (dT_b_2_dt_Comp+dT_b_2_dt_ext-dT_chi_2_dt_ext)/H; // K
      SDM_rates.Delta_T_b_chi *= epsilon_chi_b; // K
  }
  else {
      SDM_rates.T_bar_chi_b = 0.;
      SDM_rates.Delta_T_b_chi = 0.;
  }
  // Return output
  return SDM_rates;
}

// JordanFlitter: S_alpha correction. Similar to the T_RECFAST function
// This function interpolates the data in an external text file, which was generated from Eq. A4 in Mittal & Kulkarni (arXiv: 2009.10746),
// a result that was derived from the work of Chuzhouy & Shapiro (arXiv: astro-ph/0512206).
double S_alpha_correction(float xi, int flag)
{
    double ans;
    static double xi_t[RECFAST_NPTS], S_alpha[RECFAST_NPTS];
    static gsl_interp_accel *acc;
    static gsl_spline *spline;
    float currxi, currS_alpha;
    int i, ind;
    FILE *F;

    char filename[500];

    if (flag == 1) {
        // Read in the recfast data
        sprintf(filename,"%s/%s.dat",global_params.external_table_path,S_ALPHA_FILENAME);
        if ( !(F=fopen(filename, "r")) ){
            LOG_ERROR("S_alpha_correction: Unable to open file: %s for reading.", filename);
            Throw(IOError);
        }
        for (i = 0; i < RECFAST_NPTS; i++) {
            fscanf(F, "%E %E", &currxi, &currS_alpha);
            xi_t[i] = currxi;
            S_alpha[i] = currS_alpha;
        }
        fclose(F);

        // Set up spline table
        acc   = gsl_interp_accel_alloc ();
        spline  = gsl_spline_alloc (gsl_interp_cspline, RECFAST_NPTS);
        gsl_spline_init(spline, xi_t, S_alpha, RECFAST_NPTS);

        return 0;
    }
    if (flag == 2) {
        // Free memory
        gsl_spline_free (spline);
        gsl_interp_accel_free(acc);
        return 0;
    }

    if (xi > xi_t[RECFAST_NPTS-1]) { // approximate S_alpha = 1 at very large values of xi
        ans = 1.;
    }
    else if (xi < xi_t[0]) { // approximate S_alpha = 2/9 * xi at very small values of xi
        ans = 2./9. * xi;
    }
    else { // Do spline
      ans = gsl_spline_eval (spline, xi, acc);
    }
    return ans;
}

// JordanFlitter: taken the following from https://github.com/21cmfast/21cmFAST/blob/master/src/py21cmfast/src/heating_helper_progs.c
// approximation for the adiabatic index at z=6-50 from 2302.08506 (also 1506.04152). Linear only, used to initialize the Tk box at high z so it's not homogeneous. Otherwise half of the adiabatic fluctuations are missing. Definition is \delta Tk = Tk * cT * \delta (at each z).
float cT_approx(float z)
{
    return 0.58 - 0.006*(z - 10.0);
}
