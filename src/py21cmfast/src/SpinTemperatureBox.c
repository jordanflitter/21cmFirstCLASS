
// Re-write of find_HII_bubbles.c for being accessible within the MCMC

// Grids/arrays that only need to be initialised once (i.e. the lowest redshift density cube to be sampled)
double ***fcoll_R_grid, ***dfcoll_dz_grid;
double **grid_dens, **density_gridpoints;
double *Sigma_Tmin_grid, *ST_over_PS_arg_grid, *dstarlya_dt_prefactor, *zpp_edge, *sigma_atR;
double *dstarlyLW_dt_prefactor, *dstarlya_dt_prefactor_MINI, *dstarlyLW_dt_prefactor_MINI;
float **delNL0_rev,**delNL0;
float *R_values, *delNL0_bw, *delNL0_Offset, *delNL0_LL, *delNL0_UL, *delNL0_ibw, *log10delNL0_diff;
float *log10delNL0_diff_UL,*min_densities, *max_densities, *zpp_interp_table;
short **dens_grid_int_vals;
short *SingleVal_int;

float *del_fcoll_Rct, *SFR_timescale_factor;
float *del_fcoll_Rct_MINI;

double *dxheat_dt_box, *dxion_source_dt_box, *dxlya_dt_box, *dstarlya_dt_box;
double *dxheat_dt_box_MINI, *dxion_source_dt_box_MINI, *dxlya_dt_box_MINI, *dstarlya_dt_box_MINI;
double *dstarlyLW_dt_box, *dstarlyLW_dt_box_MINI;

// JordanFlitter: I need these arrays
double *dstarlya_cont_dt_box, *dstarlya_inj_dt_box, *dstarlya_cont_dt_prefactor, *dstarlya_inj_dt_prefactor, *sum_ly2, *sum_lynto2;
double *dstarlya_cont_dt_box_MINI, *dstarlya_inj_dt_box_MINI, *dstarlya_cont_dt_prefactor_MINI, *dstarlya_inj_dt_prefactor_MINI, *sum_ly2_MINI, *sum_lynto2_MINI;
fftwf_complex *delta_baryons, *delta_baryons_derivative, *delta_SDM, *delta_SDM_derivative;

double *log10_Mcrit_LW_ave_list;

float *inverse_val_box;
int *m_xHII_low_box;

// Grids/arrays that are re-evaluated for each zp
double **fcoll_interp1, **fcoll_interp2, **dfcoll_interp1, **dfcoll_interp2;
double *fcoll_R_array, *sigma_Tmin, *ST_over_PS, *sum_lyn;
float *inverse_diff, *zpp_growth, *zpp_for_evolve_list,*Mcrit_atom_interp_table;
double *ST_over_PS_MINI,*sum_lyn_MINI,*sum_lyLWn,*sum_lyLWn_MINI;

// interpolation tables for the heating/ionisation integrals
double **freq_int_heat_tbl, **freq_int_ion_tbl, **freq_int_lya_tbl, **freq_int_heat_tbl_diff;
double **freq_int_ion_tbl_diff, **freq_int_lya_tbl_diff;

bool TsInterpArraysInitialised = false;
float initialised_redshift = 0.0;
// JordanFlitter: added next_redshift_input. This will help keeping the python and C loops synchronized during the dark ages
int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options,
                  float perturbed_field_redshift, float next_redshift_input, short cleanup,
                  struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp,
                  struct InitialConditions *ini_boxes, struct TsBox *this_spin_temp) {
    int status;
    Try{ // This Try{} wraps the whole function.
LOG_DEBUG("input values:");
LOG_DEBUG("redshift=%f, prev_redshift=%f perturbed_field_redshift=%f", redshift, prev_redshift, perturbed_field_redshift);
if (LOG_LEVEL >= DEBUG_LEVEL){
    writeAstroParams(flag_options, astro_params);
}

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    // !!! SLTK: added astro_params, flag_options 
    Broadcast_struct_global_PS(user_params,cosmo_params,astro_params,flag_options);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    Broadcast_struct_global_HF(user_params,cosmo_params,astro_params, flag_options);

    // This is an entire re-write of Ts.c from 21cmFAST. You can refer back to Ts.c in 21cmFAST if this become a little obtuse. The computation has remained the same //
    omp_set_num_threads(user_params->N_THREADS);

    /////////////////// Defining variables for the computation of Ts.c //////////////

    FILE *F, *OUT;

    unsigned long long ct, FCOLL_SHORT_FACTOR, box_ct;

    int R_ct,i,ii,j,k,i_z,COMPUTE_Ts,x_e_ct,m_xHII_low,m_xHII_high,n_ct, zpp_gridpoint1_int;
    int zpp_gridpoint2_int,zpp_evolve_gridpoint1_int, zpp_evolve_gridpoint2_int,counter;

    short dens_grid_int;

    double Tk_ave, J_alpha_ave, xalpha_ave, J_alpha_tot, Xheat_ave, Xion_ave, nuprime, Ts_ave;
    double lower_int_limit,Luminosity_converstion_factor,T_inv_TS_fast_inv;
    double J_LW_ave, J_alpha_tot_MINI, J_alpha_ave_MINI, J_LW_ave_MINI,dxheat_dzp_MINI,Xheat_ave_MINI;
    double dadia_dzp, dcomp_dzp, dxheat_dt, dxion_source_dt, dxion_sink_dt, T, x_e, dxe_dzp, n_b;
    double dspec_dzp, dxheat_dzp, dxlya_dt, dstarlya_dt, fcoll_R;
    double Trad_fast,xc_fast,xc_inverse,TS_fast,TSold_fast,xa_tilde_fast,TS_prefactor,xa_tilde_prefactor;
    double T_inv,T_inv_sq,xi_power,xa_tilde_fast_arg,Trad_fast_inv,TS_fast_inv,dcomp_dzp_prefactor;

    float growth_factor_z, inverse_growth_factor_z, R, R_factor, zp, mu_for_Ts, filling_factor_of_HI_zp;
    float dzp, prev_zp, zpp, prev_zpp, prev_R, Tk_BC, xe_BC;
    float xHII_call, curr_xalpha, TK, TS, xe, deltax_highz;
    float zpp_for_evolve,dzpp_for_evolve, M_MIN;

    float determine_zpp_max, zpp_grid, zpp_gridpoint1, zpp_gridpoint2,zpp_evolve_gridpoint1;
    float zpp_evolve_gridpoint2, grad1, grad2, grad3, grad4, delNL0_bw_val;
    float OffsetValue, DensityValueLow, min_density, max_density;

    double curr_delNL0, inverse_val,prefactor_1,prefactor_2,dfcoll_dz_val, density_eval1;
    double density_eval2, grid_sigmaTmin, grid_dens_val, dens_grad, dens_width;
    double prefactor_2_MINI, dfcoll_dz_val_MINI;

    double const_zp_prefactor, dt_dzp, x_e_ave, growth_factor_zp, dgrowth_factor_dzp, fcoll_R_for_reduction;
    double const_zp_prefactor_MINI;

    int n_pts_radii;
    double trial_zpp_min,trial_zpp_max,trial_zpp, weight;
    bool first_radii, first_zero;
    first_radii = true;
    first_zero = true;
    n_pts_radii = 1000;

    float M_MIN_WDM =  M_J_WDM();

    double ave_fcoll, ave_fcoll_inv, dfcoll_dz_val_ave, ION_EFF_FACTOR;
    double ave_fcoll_MINI, ave_fcoll_inv_MINI, dfcoll_dz_val_ave_MINI, ION_EFF_FACTOR_MINI;

    float curr_dens, min_curr_dens, max_curr_dens;

    float curr_vcb;
    min_curr_dens = max_curr_dens = 0.;

    int fcoll_int_min, fcoll_int_max;

    fcoll_int_min = fcoll_int_max = 0;

    float Splined_Fcoll,Splined_Fcollzp_mean,Splined_SFRD_zpp, fcoll;
    float redshift_table_Nion_z,redshift_table_SFRD, fcoll_interp_val1, fcoll_interp_val2, dens_val;
    float fcoll_interp_min, fcoll_interp_bin_width, fcoll_interp_bin_width_inv;
    float fcoll_interp_high_min, fcoll_interp_high_bin_width, fcoll_interp_high_bin_width_inv;

    float Splined_Fcoll_MINI,Splined_Fcollzp_mean_MINI_left,Splined_Fcollzp_mean_MINI_right;
    float Splined_Fcollzp_mean_MINI,Splined_SFRD_zpp_MINI_left,Splined_SFRD_zpp_MINI_right;
    float Splined_SFRD_zpp_MINI, fcoll_MINI,fcoll_MINI_right,fcoll_MINI_left;
    float fcoll_interp_min_MINI, fcoll_interp_bin_width_MINI, fcoll_interp_bin_width_inv_MINI;
    float fcoll_interp_val1_MINI, fcoll_interp_val2_MINI;
    float fcoll_interp_high_min_MINI, fcoll_interp_high_bin_width_MINI, fcoll_interp_high_bin_width_inv_MINI;

    int fcoll_int;
    int redshift_int_Nion_z,redshift_int_SFRD;
    float zpp_integrand, Mlim_Fstar, Mlim_Fesc, Mlim_Fstar_MINI, Mlim_Fesc_MINI, Mmax, sigmaMmax;

    double log10_Mcrit_LW_ave;
    float log10_Mcrit_mol;
    float log10_Mcrit_LW_ave_table_Nion_z, log10_Mcrit_LW_ave_table_SFRD;
    int  log10_Mcrit_LW_ave_int_Nion_z, log10_Mcrit_LW_ave_int_SFRD;
    double LOG10_MTURN_INT = (double) ((LOG10_MTURN_MAX - LOG10_MTURN_MIN)) / ((double) (NMTURN - 1.));
    float **log10_Mcrit_LW;
    int log10_Mcrit_LW_int;
    float log10_Mcrit_LW_diff, log10_Mcrit_LW_val;

    int table_int_boundexceeded = 0;
    int fcoll_int_boundexceeded = 0;
    int *fcoll_int_boundexceeded_threaded = calloc(user_params->N_THREADS,sizeof(int));
    int *table_int_boundexceeded_threaded = calloc(user_params->N_THREADS,sizeof(int));
    for(i=0;i<user_params->N_THREADS;i++) {
        fcoll_int_boundexceeded_threaded[i] = 0;
        table_int_boundexceeded_threaded[i] = 0;
    }

    double total_time, total_time2, total_time3, total_time4;
    float M_MIN_at_zp;

    int NO_LIGHT = 0;

    // JordanFlitter: I need these variables
    double prev_Ts, tau21, xCMB, eps_CMB, E_continuum, E_injected, Ndot_alpha_cont, Ndot_alpha_inj, Ndot_alpha_cont_MINI, Ndot_alpha_inj_MINI;
    double dCMBheat_dzp, eps_Lya_cont, eps_Lya_inj, eps_Lya_cont_MINI, eps_Lya_inj_MINI, dstarlya_cont_dt, dstarlya_inj_dt;
    float T_chi_BC, V_chi_b_BC;
    double T_chi, V_chi_b, dSDM_b_heat_dzp, dSDM_chi_heat_dzp, D_V_chi_b_dzp;
    double dxion_source2_dt, beta_ion;
    HYREC_DATA *rec_data;
    double T_bar_gamma_b, epsilon_gamma_b, Delta_T_gamma_b, dT_b_2_dt_ext, dT_chi_2_dt_ext, dadia_dzp_SDM;
    float EPSILON_THRES;
    SDM_RATES SDM_rates;

    // JordanFlitter: added variables in order to evolve the baryons density field
    float k_x, k_y, k_z, k_mag;
    int n_x, n_y, n_z;
    float delta_baryons_local, delta_baryons_derivative_local;
    float delta_SDM_local, delta_SDM_derivative_local;

    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;
        ION_EFF_FACTOR_MINI = global_params.Pop3_ion * astro_params->F_STAR7_MINI * astro_params->F_ESC7_MINI;
    }
    else {
        ION_EFF_FACTOR = astro_params->HII_EFF_FACTOR;
        ION_EFF_FACTOR_MINI = 0.;
    }

    // Initialise arrays to be used for the Ts.c computation //
    fftwf_complex *box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *unfiltered_box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *log10_Mcrit_LW_unfiltered, *log10_Mcrit_LW_filtered;
    if (flag_options->USE_MINI_HALOS){
        log10_Mcrit_LW_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        log10_Mcrit_LW_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        log10_Mcrit_LW = (float **) calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            log10_Mcrit_LW[R_ct] = (float *) calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        }
    }

LOG_SUPER_DEBUG("initialized");

// JordanFlitter: we need init_CLASS_GROWTH_FACTOR() if the following conditions are satisfied
if (!user_params->USE_DICKE_GROWTH_FACTOR || user_params->EVOLVE_BARYONS) {
    init_CLASS_GROWTH_FACTOR();
}

// JordanFlitter: I moved the definitions for the growth factors here, and also init_heat()
init_heat(redshift); // JordanFlitter: added a redshift argument
growth_factor_z = dicke(perturbed_field_redshift);
inverse_growth_factor_z = 1./growth_factor_z;
LOG_SUPER_DEBUG("Initialised heat");

// JordanFlitter: I need to allocate memory for these boxes if we evolve the baryons (and SDM) density field
if (user_params->EVOLVE_BARYONS){
    delta_baryons = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    delta_baryons_derivative = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    if (user_params->SCATTERING_DM){
        delta_SDM = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        delta_SDM_derivative = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }
}

// JordanFlitter: We don't need these during the dark ages
if (redshift <= global_params.Z_HEAT_MAX) {

    if(!TsInterpArraysInitialised) {
LOG_SUPER_DEBUG("initalising Ts Interp Arrays");

        // Grids/arrays that only need to be initialised once (i.e. the lowest redshift density cube to be sampled)

        zpp_edge = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        sigma_atR = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        R_values = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

        if(user_params->USE_INTERPOLATION_TABLES) {
            min_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            max_densities = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

            zpp_interp_table = calloc(zpp_interp_points_SFR, sizeof(float));
        }
        // JordanFlitter: Allocate memory for Lya heating arrays
        if (flag_options->USE_Lya_HEATING){
            dstarlya_cont_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dstarlya_inj_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dstarlya_cont_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dstarlya_inj_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_ly2 = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lynto2 = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            if (flag_options->USE_MINI_HALOS){
                dstarlya_cont_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dstarlya_inj_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dstarlya_cont_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                dstarlya_inj_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                sum_ly2_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                sum_lynto2_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            }
        }

        if(flag_options->USE_MASS_DEPENDENT_ZETA) {

            SFR_timescale_factor = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

            if(user_params->MINIMIZE_MEMORY) {
                delNL0 = (float **)calloc(1,sizeof(float *));
                delNL0[0] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));
            }
            else {
                delNL0 = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                    delNL0[i] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));
                }
            }

            xi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));
            wi_SFR_Xray = calloc(NGL_SFR+1,sizeof(double));

            if(user_params->USE_INTERPOLATION_TABLES) {
                overdense_low_table = calloc(NSFR_low,sizeof(double));
                overdense_high_table = calloc(NSFR_high,sizeof(double));

                log10_SFRD_z_low_table = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    log10_SFRD_z_low_table[j] = (float *)calloc(NSFR_low,sizeof(float));
                }

                SFRD_z_high_table = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    SFRD_z_high_table[j] = (float *)calloc(NSFR_high,sizeof(float));
                }

                if(flag_options->USE_MINI_HALOS){
                    log10_SFRD_z_low_table_MINI = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                    for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                        log10_SFRD_z_low_table_MINI[j] = (float *)calloc(NSFR_low*NMTURN,sizeof(float));
                    }

                    SFRD_z_high_table_MINI = (float **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
                    for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                        SFRD_z_high_table_MINI[j] = (float *)calloc(NSFR_high*NMTURN,sizeof(float));
                    }
                }
            }

            if(flag_options->USE_MINI_HALOS){
                log10_Mcrit_LW_ave_list = (double *) calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            }

            del_fcoll_Rct = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));

            dxheat_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dxion_source_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dxlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            dstarlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            if (flag_options->USE_MINI_HALOS){
                del_fcoll_Rct_MINI = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));

                dstarlyLW_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dxheat_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dxion_source_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dxlya_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dstarlya_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                dstarlyLW_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
            }

            m_xHII_low_box = (int *)calloc(HII_TOT_NUM_PIXELS,sizeof(int));
            inverse_val_box = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));

        }
        else {

            if(user_params->USE_INTERPOLATION_TABLES) {
                Sigma_Tmin_grid = (double *)calloc(zpp_interp_points_SFR,sizeof(double));

                fcoll_R_grid = (double ***)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double **));
                dfcoll_dz_grid = (double ***)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double **));
                for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                    fcoll_R_grid[i] = (double **)calloc(zpp_interp_points_SFR,sizeof(double *));
                    dfcoll_dz_grid[i] = (double **)calloc(zpp_interp_points_SFR,sizeof(double *));
                    for(j=0;j<zpp_interp_points_SFR;j++) {
                        fcoll_R_grid[i][j] = (double *)calloc(dens_Ninterp,sizeof(double));
                        dfcoll_dz_grid[i][j] = (double *)calloc(dens_Ninterp,sizeof(double));
                    }
                }

                grid_dens = (double **)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double *));
                for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                    grid_dens[i] = (double *)calloc(dens_Ninterp,sizeof(double));
                }

                density_gridpoints = (double **)calloc(dens_Ninterp,sizeof(double *));
                for(i=0;i<dens_Ninterp;i++) {
                    density_gridpoints[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                }
                ST_over_PS_arg_grid = (double *)calloc(zpp_interp_points_SFR,sizeof(double));

                delNL0_bw = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
                delNL0_Offset = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
                delNL0_LL = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
                delNL0_UL = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
                delNL0_ibw = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
                log10delNL0_diff = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
                log10delNL0_diff_UL = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

                fcoll_interp1 = (double **)calloc(dens_Ninterp,sizeof(double *));
                fcoll_interp2 = (double **)calloc(dens_Ninterp,sizeof(double *));
                dfcoll_interp1 = (double **)calloc(dens_Ninterp,sizeof(double *));
                dfcoll_interp2 = (double **)calloc(dens_Ninterp,sizeof(double *));
                for(i=0;i<dens_Ninterp;i++) {
                    fcoll_interp1[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                    fcoll_interp2[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                    dfcoll_interp1[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                    dfcoll_interp2[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
                }

                dens_grid_int_vals = (short **)calloc(HII_TOT_NUM_PIXELS,sizeof(short *));
                for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
                    dens_grid_int_vals[i] = (short *)calloc((float)global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(short));
                }
            }

            delNL0_rev = (float **)calloc(HII_TOT_NUM_PIXELS,sizeof(float *));
            for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
                delNL0_rev[i] = (float *)calloc((float)global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            }
        }

        dstarlya_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        if (flag_options->USE_MINI_HALOS){
            dstarlya_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dstarlyLW_dt_prefactor = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dstarlyLW_dt_prefactor_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
        SingleVal_int = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(short));

        freq_int_heat_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_ion_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_lya_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_heat_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_ion_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
        freq_int_lya_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
        for(i=0;i<x_int_NXHII;i++) {
            freq_int_heat_tbl[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_ion_tbl[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_lya_tbl[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_heat_tbl_diff[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_ion_tbl_diff[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            freq_int_lya_tbl_diff[i] = (double *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }

        // Grids/arrays that are re-evaluated for each zp
        fcoll_R_array = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        inverse_diff = calloc(x_int_NXHII,sizeof(float));
        zpp_growth = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

        sigma_Tmin = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        ST_over_PS = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        sum_lyn = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        if (flag_options->USE_MINI_HALOS){
            Mcrit_atom_interp_table = (float *)calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
            ST_over_PS_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lyn_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lyLWn = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            sum_lyLWn_MINI = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
        zpp_for_evolve_list = calloc(global_params.NUM_FILTER_STEPS_FOR_Ts,sizeof(float));

        TsInterpArraysInitialised = true;
LOG_SUPER_DEBUG("initalised Ts Interp Arrays");
    }
    ///////////////////////////////  BEGIN INITIALIZATION   //////////////////////////////

    //set the minimum ionizing source mass
    // In v1.4 the miinimum ionizing source mass does not depend on redshift.
    // For the constant ionizing efficiency parameter, M_MIN is set to be M_TURN which is a sharp cut-off.
    // For the new parametrization, the number of halos hosting active galaxies (i.e. the duty cycle) is assumed to
    // exponentially decrease below M_TURNOVER Msun, : fduty \propto e^(- M_TURNOVER / M)
    // In this case, we define M_MIN = M_TURN/50, i.e. the M_MIN is integration limit to compute follapse fraction.
    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        if (flag_options->USE_MINI_HALOS){
            M_MIN = (global_params.M_MIN_INTEGRAL)/50.;

            Mlim_Fstar = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR, astro_params->F_STAR10);
            Mlim_Fesc = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC, astro_params->F_ESC10);

            Mlim_Fstar_MINI = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR_MINI,
                                                   astro_params->F_STAR7_MINI * pow(1e3, astro_params->ALPHA_STAR_MINI));
            Mlim_Fesc_MINI = Mass_limit_bisection(global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC,
                                                  astro_params->F_ESC7_MINI * pow(1e3, astro_params->ALPHA_ESC));
        }
        else{
            M_MIN = (astro_params->M_TURN)/50.;

            Mlim_Fstar = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR, astro_params->F_STAR10);
            Mlim_Fesc = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC, astro_params->F_ESC10);
        }
    }
    else {

        if(flag_options->M_MIN_in_Mass) {
            M_MIN = (astro_params->M_TURN)/50.;
        }
        else {
            //set the minimum source mass
            if (astro_params->X_RAY_Tvir_MIN < 9.99999e3) { // neutral IGM
                mu_for_Ts = mu_b_neutral; // JordanFlitter: I changed the constant value to the general case
            }
            else {  // ionized IGM
                mu_for_Ts = mu_b_ionized; // JordanFlitter: I changed the constant value to the general case
            }
        }
    }

    init_ps();

LOG_SUPER_DEBUG("Initialised PS");
LOG_SUPER_DEBUG("About to initialise heat");

    // Initialize some interpolation tables
    if(this_spin_temp->first_box || (fabs(initialised_redshift - perturbed_field_redshift) > 0.0001) ) {
        if(user_params->USE_INTERPOLATION_TABLES) {
          if(user_params->FAST_FCOLL_TABLES){
            initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
          }
          else{
            if(flag_options->M_MIN_in_Mass || flag_options->USE_MASS_DEPENDENT_ZETA) {
                if (flag_options->USE_MINI_HALOS){
                    initialiseSigmaMInterpTable(global_params.M_MIN_INTEGRAL/50.,1e20);
                }
                else{
                    initialiseSigmaMInterpTable(M_MIN,1e20);
                }
            }
            LOG_SUPER_DEBUG("Initialised sigmaM interp table");
          }
        }
    }

  } // JordanFlitter: End of cosmic dawn condition
  // JordanFlitter: define EPSILON_THRES (for TCA-DM)
  if (redshift > 100.){
      EPSILON_THRES = global_params.EPSILON_THRESH_HIGH_Z;
  }
  else {
      EPSILON_THRES = global_params.EPSILON_THRESH_LOW_Z;
  }
    // JordanFlitter: Changed the following logic for "high redshifts"
    if ((redshift > global_params.Z_HEAT_MAX && !user_params->START_AT_RECOMBINATION) ||
        (redshift > user_params->Z_HIGH_MAX && user_params->START_AT_RECOMBINATION)){
LOG_SUPER_DEBUG("redshift greater than Z_HEAT_MAX");
        // JordanFlitterTODO: It is quite an overkill to do an interpolation just for having the initial value.
        //                    This initial value can be computed at the python level and be passed to the C-code,
        //                    e.g. via global_params.
        xe = xion_RECFAST(redshift,0);
        TK = T_RECFAST(redshift,0);
        // JordanFlitter: Extract T_chi and V_chi_b at redshift
        if (user_params->SCATTERING_DM) {
            T_chi_BC = T_chi_RECFAST(redshift,0); // K
            V_chi_b_BC = V_chi_b_RECFAST(redshift,0); // km/sec
        }
        growth_factor_zp = dicke(redshift);

LOG_SUPER_DEBUG("growth factor zp = %f", growth_factor_zp);
        // read file
#pragma omp parallel shared(this_spin_temp,xe,TK,redshift,perturbed_field,inverse_growth_factor_z,growth_factor_zp) private(i,j,k,curr_xalpha) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        this_spin_temp->Tk_box[HII_R_INDEX(i,j,k)] = TK;
                        // JordanFlitter: set inhomogeneous temperature box based on arXiv: 2302.08506
                        //                Note that the automatic logic prevents of setting both USE_ADIABATIC_FLUCTUATIONS and EVOLVE_BARYONS to True,
                        //                as we currently do not have 2d fit for the temperature fluctuations
                        if (user_params->USE_ADIABATIC_FLUCTUATIONS) {
                            this_spin_temp->Tk_box[HII_R_INDEX(i,j,k)] *= 1.+perturbed_field->density[HII_R_INDEX(i,j,k)]*inverse_growth_factor_z*growth_factor_zp*cT_approx(redshift);
                        }
                        this_spin_temp->x_e_box[HII_R_INDEX(i,j,k)] = xe;
                        // compute the spin temperature
                        // JordanFlitter: set inhomogeneous temperature box based on arXiv: 2302.08506
                        if (user_params->USE_ADIABATIC_FLUCTUATIONS) {
                            this_spin_temp->Ts_box[HII_R_INDEX(i,j,k)] = get_Ts(redshift,
                                                              perturbed_field->density[HII_R_INDEX(i,j,k)]*inverse_growth_factor_z*growth_factor_zp,
                                                              TK*(1.+perturbed_field->density[HII_R_INDEX(i,j,k)]*inverse_growth_factor_z*growth_factor_zp*cT_approx(redshift)), xe, 0, &curr_xalpha);
                        }
                        else{
                            // JordanFlitter: note that we assume the baryons density field is given at z=redshift
                            if (user_params->EVOLVE_BARYONS){
                                this_spin_temp->Ts_box[HII_R_INDEX(i,j,k)] = get_Ts(redshift,
                                                                    perturbed_field->baryons_density[HII_R_INDEX(i,j,k)],
                                                                    TK, xe, 0, &curr_xalpha);
                            }
                            else{
                                this_spin_temp->Ts_box[HII_R_INDEX(i,j,k)] = get_Ts(redshift,
                                                                    perturbed_field->density[HII_R_INDEX(i,j,k)]*inverse_growth_factor_z*growth_factor_zp,
                                                                    TK, xe, 0, &curr_xalpha);
                            }
                        }
                        // JordanFlitter: I set initial values for T_chi_box and V_chi_b_box
                        if (user_params->SCATTERING_DM){
                            this_spin_temp->T_chi_box[HII_R_INDEX(i,j,k)] = T_chi_BC; // K
                            this_spin_temp->V_chi_b_box[HII_R_INDEX(i,j,k)] = V_chi_b_BC; // km/sec
                        }
                    }
                }
            }
        }

LOG_SUPER_DEBUG("read in file");
        //JordanFlitter: We don't need these during the dark ages
        if (redshift <= global_params.Z_HEAT_MAX) {

            if(!flag_options->M_MIN_in_Mass) {
                M_MIN = (float)TtoM(redshift, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
                LOG_DEBUG("Attempting to initialise sigmaM table with M_MIN=%e, Tvir_MIN=%e, mu=%e",
                          M_MIN, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
                if(user_params->USE_INTERPOLATION_TABLES) {
                  if(user_params->FAST_FCOLL_TABLES){
                    initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
                  }
                  else{
                    initialiseSigmaMInterpTable(M_MIN,1e20);
                  }
                }
            }
            LOG_SUPER_DEBUG("Initialised Sigma interp table");
      }
    }
    else {
LOG_SUPER_DEBUG("redshift less than Z_HEAT_MAX");
        // Flag is set for previous spin temperature box as a previous spin temperature box must be passed, which makes it the initial condition
        // JordanFlitterTODO: the whole first_box logic is outdated an no longer found in the public code of 21cmFAST.
        //                    when removing it, note that filling the box values of previous_spin_temp (as is done below), should be
        //                    done at the above lines. In fact, the true "first_box" is when redshift > Z_HEAT_MAX (or Z_HIGH_MAX, depending on
        //                    whether or not START_AT_RECOMBINATION is on or off)
        if (this_spin_temp->first_box) {
LOG_SUPER_DEBUG("Treating as the first box");
            // set boundary conditions for the evolution equations->  values of Tk and x_e at Z_HEAT_MAX
            if (global_params.XION_at_Z_HEAT_MAX > 0) // user has opted to use his/her own value
                xe_BC = global_params.XION_at_Z_HEAT_MAX;
            else {// will use the results obtained from recfast
                // JordanFlitter: I use prev_redshift (instead Z_HEAT_MAX) to set the proper initial conditions
                xe_BC = xion_RECFAST(prev_redshift,0);
            }
            if (global_params.TK_at_Z_HEAT_MAX > 0)
                Tk_BC = global_params.TK_at_Z_HEAT_MAX;
            else {
                Tk_BC = T_RECFAST(prev_redshift,0);
            }
            // JordanFlitter: Extract T_chi and V_chi_b at the previous redshift
            if (user_params->SCATTERING_DM) {
                T_chi_BC = T_chi_RECFAST(prev_redshift,0); // K
                if (!user_params->USE_SDM_FLUCTS){
                    V_chi_b_BC = V_chi_b_RECFAST(prev_redshift,0); // km/sec
                }
            }

            // and initialize to the boundary values at Z_HEAT_END
            // JordanFlitter: added more shared variables
            growth_factor_zp = dicke(redshift);
#pragma omp parallel shared(previous_spin_temp,Tk_BC,xe_BC,T_chi_BC,V_chi_b_BC,ini_boxes,redshift,perturbed_field,inverse_growth_factor_z,growth_factor_zp) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                    previous_spin_temp->Tk_box[ct] = Tk_BC;
                    // JordanFlitter: set inhomogeneous temperature box based on arXiv: 2302.08506
                    //                Note that the automatic logic prevents of setting both USE_ADIABATIC_FLUCTUATIONS and EVOLVE_BARYONS to True,
                    //                as we currently do not have 2d fit for the temperature fluctuations
                    if (user_params->USE_ADIABATIC_FLUCTUATIONS) {
                        previous_spin_temp->Tk_box[ct] *= 1.+perturbed_field->density[ct]*inverse_growth_factor_z*growth_factor_zp*cT_approx(redshift);
                    }
                    previous_spin_temp->x_e_box[ct] = xe_BC;
                    //JordanFlitter: I set initial values for T_chi_box and V_chi_b_box
                    if (user_params->SCATTERING_DM) {
                        previous_spin_temp->T_chi_box[ct] = T_chi_BC; // K
                        if (user_params->USE_SDM_FLUCTS) {
                            /* JordanFlitterTODO: Uncomment the following lines (unfortunately, currently CLASS is unalbe to produce the
                                                  temperature/ionization transfer functions)
                            if (ini_boxes->lowres_Tk_zhigh[ct] > -1.){
                                previous_spin_temp->Tk_box[ct] *= (1.+ini_boxes->lowres_Tk_zhigh[ct]);
                            }
                            else {
                                previous_spin_temp->Tk_box[ct] = 0.;
                            }
                            if (ini_boxes->lowres_xe_zhigh[ct] > -1.){
                                previous_spin_temp->x_e_box[ct] *= (1.+ini_boxes->lowres_xe_zhigh[ct]);
                            }
                            else {
                                previous_spin_temp->x_e_box[ct] = 0.;
                            }
                            if (ini_boxes->lowres_Tchi_zhigh[ct] > -1.){
                                previous_spin_temp->T_chi_box[ct] *= (1.+ini_boxes->lowres_Tchi_zhigh[ct]);
                            }
                            else {
                                previous_spin_temp->T_chi_box[ct] = 0.;
                            }*/
                            previous_spin_temp->V_chi_b_box[ct] = ini_boxes->lowres_V_chi_b_zhigh[ct]; // km/sec
                        }
                        else {
                            previous_spin_temp->V_chi_b_box[ct] = V_chi_b_BC; // km/sec
                        }
                    }
                    // JordanFlitter: Make sure x_e is less than 1
                    if (previous_spin_temp->x_e_box[ct] > 1.) {
                        previous_spin_temp->x_e_box[ct] = 1.;
                    }
                }
            }
            x_e_ave = xe_BC;
            Tk_ave = Tk_BC;
        }
        else {
            x_e_ave = Tk_ave = 0.0;

#pragma omp parallel shared(previous_spin_temp) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:x_e_ave,Tk_ave)
                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                    x_e_ave += previous_spin_temp->x_e_box[ct];
                    Tk_ave += previous_spin_temp->Tk_box[ct];
                }
            }
            x_e_ave /= (float)HII_TOT_NUM_PIXELS;
            Tk_ave /= (float)HII_TOT_NUM_PIXELS;
        }

        // JordanFlitter: We need to allocate memory for HyRec and fill its fields.
        // Note: I wanted to use another function (here or in another C-file) for the following lines, but the compilation failed for some reason.
        // This is why these lines are here!
        if (user_params->USE_HYREC) {
            rec_data = malloc(sizeof(HYREC_DATA));
            if (rec_data == NULL) {
                LOG_ERROR("Unable to allocate memory for Hyrec.");
                Throw(MemoryAllocError);
            }
            rec_data->path_to_hyrec = malloc(SIZE_ErrorM);
            sprintf(rec_data->path_to_hyrec,"%s/%s/",global_params.external_table_path,HYREC_DATA_PREFIX);
            hyrec_allocate(rec_data, 8000., 0.);
            if(rec_data->error != 0){
              LOG_ERROR("Unable to allocate memory for Hyrec fields.");
              printf(rec_data->error_message);
              Throw(MemoryAllocError);
            }
            rec_data->cosmo->T0 = T_cmb;
            rec_data->cosmo->obh2 = cosmo_params->OMb*cosmo_params->hlittle*cosmo_params->hlittle;
            rec_data->cosmo->ocbh2 = cosmo_params->OMm*cosmo_params->hlittle*cosmo_params->hlittle;
            rec_data->cosmo->YHe = global_params.Y_He;
            rec_data->cosmo->Neff = N_EFF;
            // JordanFlitter: 21cmFAST's f_He is f_He = n_He/(n_H + n_He), but Hyrec requires n_He/n_H.
            // Therefore, we multiply by (n_H + n_He)/n_H = (1-(1.-1./_not4_)*Y_He)/(1-Y_He)
            rec_data->cosmo->fHe = f_He*(1.-(1.-1./_not4_)*global_params.Y_He)/(1.-global_params.Y_He);
            rec_data->cosmo->fsR = 1.;
            rec_data->cosmo->meR = 1.;
            rec_data->cosmo->nH0 = No; // Hydrogen desnity today in cm^-3, as required by Hyrec
        }

        //JordanFlitter: If we are at high redshifts, we simply want to evlove x_e and T_k with no astrophysics.
        //               This makes the code to run MUCH FASTER per redshift iteration
        if (redshift > global_params.Z_HEAT_MAX) {

            // allocate memory for the nonlinear density field
            // JordanFlitter: during the dark ages, we don't need curr_delNL0 (or delta(z=0)) as we use the baryons density field at perturbed_field_redshift
            if (!user_params->NO_INI_MATTER_FLUCTS && !user_params->EVOLVE_BARYONS) {
                // JordanFlitter: Need to allocate memeory for delNL0 during the dark ages!
                delNL0 = (float **)calloc(1,sizeof(float *));
                delNL0[0] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));

                #pragma omp parallel shared(perturbed_field,delNL0,inverse_growth_factor_z) private(i,j,k,curr_delNL0) num_threads(user_params->N_THREADS)
                {
                    #pragma omp for
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<user_params->HII_DIM; k++){
                                curr_delNL0 = perturbed_field->density[HII_R_INDEX(i,j,k)];

                                if (curr_delNL0 <= -1){ // correct for aliasing in the filtering step
                                    curr_delNL0 = -1+FRACT_FLOAT_ERR;
                                }

                                // and linearly extrapolate to z=0
                                curr_delNL0 *= inverse_growth_factor_z;

                                delNL0[0][HII_R_INDEX(i,j,k)] = curr_delNL0;
                            }
                        }
                    }
                }
            }

            // Required quantities for calculating the IGM spin temperature
            zp = redshift;
            prev_zp = perturbed_field_redshift;
            dzp = 0.; // We will correct this in the lines below, just need that for the loop condition on the first iteration
            if (user_params->EVOLVE_BARYONS) {
                #pragma omp parallel shared(perturbed_field,delta_baryons,delta_SDM) private(i,j,k) num_threads(user_params->N_THREADS)
                {
                    #pragma omp for
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<user_params->HII_DIM; k++){
                                *((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->baryons_density[HII_R_INDEX(i,j,k)];

                                if (*((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                    *((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                }
                                if (user_params->SCATTERING_DM){
                                    *((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->SDM_density[HII_R_INDEX(i,j,k)];

                                    if (*((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                        *((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // JordanFlitter: We loop through the dark ages for computational efficiency
            while (zp - fabs(dzp) >= next_redshift_input){
                if ((zp > global_params.Z2_VALUE)){
                    if (user_params->USE_TCA_COMPTON){
                        dzp = -global_params.DELTA_Z1;
                    }
                    else {
                        dzp = -global_params.DELTA_Z2;
                    }
                }
                else if ((zp > global_params.Z1_VALUE)){
                    dzp = -global_params.DELTA_Z1;
                }
                else {
                    dzp = -global_params.DELTA_Z;
                }
                Trad_fast = T_cmb*(1.0+zp);
                Trad_fast_inv = 1.0/Trad_fast;
                prefactor_1 = N_b0 * pow(1+zp, 3);
                growth_factor_zp = dicke(zp);
                dgrowth_factor_dzp = ddicke_dz(zp);
                dt_dzp = dtdz(zp);
                dcomp_dzp_prefactor = (-1.51e-4)/(hubble(zp)/Ho)/(cosmo_params->hlittle)*pow(Trad_fast,4.0)/(1.0+zp);
                xc_inverse =  pow(1.0+zp,3.0)*T21/( Trad_fast*A10_HYPERFINE );
                // JordanFlitter: if we evolve the baryons density field, we need to have delta_b(z) and its redshift derivative
                if (user_params->EVOLVE_BARYONS) {
                    // We extrapolate linearly delta_baryons and its redshift derivative to zp. This requires FFT'ing the density box
                    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_baryons);
                    // Make a copy of delta_baryons at k space
                    memcpy(delta_baryons_derivative, delta_baryons, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                    if (user_params->SCATTERING_DM){
                        // Now we extrapolate linearly delta_SDM and its redshift derivative to zp. This requires FFT'ing the density box
                        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_SDM);
                        // Make a copy of delta_SDM at k space
                        memcpy(delta_SDM_derivative, delta_SDM, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                    }
                    #pragma omp parallel shared(zp,perturbed_field_redshift,delta_baryons,delta_baryons_derivative,delta_SDM,delta_SDM_derivative) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag) num_threads(user_params->N_THREADS)
                            {
                    #pragma omp for
                                for (n_x=0; n_x<user_params->HII_DIM; n_x++){
                                    if (n_x>HII_MIDDLE)
                                        k_x =(n_x-user_params->HII_DIM) * DELTA_K;  // wrap around for FFT convention
                                    else
                                        k_x = n_x * DELTA_K;
                                    for (n_y=0; n_y<user_params->HII_DIM; n_y++){
                                        if (n_y>HII_MIDDLE)
                                            k_y =(n_y-user_params->HII_DIM) * DELTA_K;
                                        else
                                            k_y = n_y * DELTA_K;
                                        for (n_z=0; n_z<=HII_MIDDLE; n_z++){
                                            k_z = n_z * DELTA_K;
                                            k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                                            *((fftwf_complex *)delta_baryons + HII_C_INDEX(n_x,n_y,n_z)) *= SDGF(zp,k_mag,0)/SDGF(prev_zp,k_mag,0)/HII_TOT_NUM_PIXELS;
                                            *((fftwf_complex *)delta_baryons_derivative + HII_C_INDEX(n_x,n_y,n_z)) *= dSDGF_dz(zp,k_mag)/SDGF(prev_zp,k_mag,0)/HII_TOT_NUM_PIXELS;
                                            if (user_params->SCATTERING_DM){
                                                *((fftwf_complex *)delta_SDM + HII_C_INDEX(n_x,n_y,n_z)) *= SDGF_SDM(zp,k_mag,0)/SDGF_SDM(prev_zp,k_mag,0)/HII_TOT_NUM_PIXELS;
                                                *((fftwf_complex *)delta_SDM_derivative + HII_C_INDEX(n_x,n_y,n_z)) *= dSDGF_SDM_dz(zp,k_mag)/SDGF_SDM(prev_zp,k_mag,0)/HII_TOT_NUM_PIXELS;
                                            }
                                        }
                                    }
                                }
                            }
                    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_baryons);
                    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_baryons_derivative);
                    if (user_params->SCATTERING_DM){
                        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_SDM);
                        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_SDM_derivative);
                    }

                    #pragma omp parallel shared(delta_baryons, delta_SDM) private(i,j,k) num_threads(user_params->N_THREADS)
                    {
                        #pragma omp for
                        for (i=0; i<user_params->HII_DIM; i++){
                            for (j=0; j<user_params->HII_DIM; j++){
                                for (k=0; k<user_params->HII_DIM; k++){
                                    if (*((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                        *((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                    }
                                }
                                if (user_params->SCATTERING_DM){
                                    if (*((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                        *((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                    }
                                }
                            }
                        }
                    }
                }

                #pragma omp parallel shared(previous_spin_temp,this_spin_temp,prefactor_1,delNL0,growth_factor_zp,dgrowth_factor_dzp,dt_dzp,zp,dcomp_dzp_prefactor,Trad_fast,dzp,\
                                            xc_inverse,Trad_fast_inv,rec_data,delta_baryons,delta_baryons_derivative,delta_SDM,delta_SDM_derivative) \
                                     private(box_ct,x_e,T,dxion_sink_dt,dxe_dzp,dadia_dzp,dspec_dzp,dcomp_dzp,T_inv,xc_fast,TS_fast,curr_delNL0,prev_Ts,tau21,xCMB,\
                                             T_chi,V_chi_b,dSDM_b_heat_dzp,dSDM_chi_heat_dzp,D_V_chi_b_dzp,\
                                             dxion_source2_dt,beta_ion,T_bar_gamma_b,epsilon_gamma_b,Delta_T_gamma_b,SDM_rates,dT_b_2_dt_ext,dT_chi_2_dt_ext,dadia_dzp_SDM,\
                                             delta_baryons_local,delta_baryons_derivative_local,delta_SDM_local,delta_SDM_derivative_local) \
                                     num_threads(user_params->N_THREADS)
                {
                    #pragma omp for
                    for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
                        // JordanFlitter: set the matter fluctuations to zero if either the user specifically asks for that (useful for comparing with CLASS)
                        if (user_params->NO_INI_MATTER_FLUCTS){
                            curr_delNL0 = 0.;
                        }
                        // JordanFlitter: if we evolve the baryons density we don't use curr_delNL0
                        if (!user_params->EVOLVE_BARYONS) {
                            curr_delNL0 = delNL0[0][box_ct];
                        }
                        // JordanFlitter: set local baryons (and SDM) density and its derivative.
                        //                Note we use box_ct_FFT to access the approporiate cell in the box
                        else {
                            delta_baryons_local = *((float *)delta_baryons + box_ct_FFT(box_ct));
                            delta_baryons_derivative_local = *((float *)delta_baryons_derivative + box_ct_FFT(box_ct));
                            if (user_params->SCATTERING_DM){
                                delta_SDM_local = *((float *)delta_SDM + box_ct_FFT(box_ct));
                                delta_SDM_derivative_local = *((float *)delta_SDM_derivative + box_ct_FFT(box_ct));
                            }
                        }
                        // JordanFlitter: If this is the first iteration, we need to take the previous box as initial conditions, otherwise we take the current one
                        if (zp == redshift) {
                            x_e = previous_spin_temp->x_e_box[box_ct];
                            T = previous_spin_temp->Tk_box[box_ct];
                            // JordanFlitter: Extract T_chi and V_chi_b from previous boxes
                            if (user_params->SCATTERING_DM) {
                                T_chi = previous_spin_temp->T_chi_box[box_ct]; // K
                                V_chi_b = 1.e5*previous_spin_temp->V_chi_b_box[box_ct]; // cm/sec
                                // Note that in the calculations below, V_chi_b is in cm/sec. However, we always save the V_chi_b_box in km/sec
                            }
                            prev_Ts = previous_spin_temp->Ts_box[box_ct];
                        }
                        else {
                            x_e = this_spin_temp->x_e_box[box_ct];
                            T = this_spin_temp->Tk_box[box_ct];
                            // JordanFlitter: Extract T_chi and V_chi_b from previous boxes
                            if (user_params->SCATTERING_DM) {
                                T_chi = this_spin_temp->T_chi_box[box_ct]; // K
                                V_chi_b = 1.e5*this_spin_temp->V_chi_b_box[box_ct]; // cm/sec
                                // Note that in the calculations below, V_chi_b is in cm/sec. However, we always save the V_chi_b_box in km/sec
                            }
                            prev_Ts = this_spin_temp->Ts_box[box_ct];
                        }
                        // JordanFlitter: we can use the baryons density field
                        if (user_params->EVOLVE_BARYONS){
                            tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)*(1.+delta_baryons_local)) /prev_Ts/hubble(zp);
                        }
                        else {
                            tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp)) /prev_Ts/hubble(zp);
                        }
                        xCMB = (1. - exp(-tau21))/tau21;


                        // First let's do dxe_dzp //
                        if (!user_params->USE_HYREC) {
                            // JordanFlitter: I added the early ionization term. Also, I multiply by the Peebles factor (and not by the clumping factor)
                            beta_ion = alpha_B(T) * pow(m_e*k_B*Trad_fast/(2.*PI*pow(hplank/(2.*PI),2.)),3./2.) * exp(-NUIONIZATION*hplank/Trad_fast/k_B); // 1/sec
                            // JordanFlitter: we can use the baryons density field
                            if (user_params->EVOLVE_BARYONS){
                              dxion_sink_dt = alpha_B(T) * x_e*x_e * f_H * prefactor_1 * \
                                              (1.+delta_baryons_local) * Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+delta_baryons_local), alpha_B(T));
                              dxion_source2_dt = beta_ion *(1.-x_e)*Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+delta_baryons_local), alpha_B(T));
                            }
                            else {
                                dxion_sink_dt = alpha_B(T) * x_e*x_e * f_H * prefactor_1 * \
                                                (1.+curr_delNL0*growth_factor_zp) * Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp), alpha_B(T));
                                dxion_source2_dt = beta_ion *(1.-x_e)*Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp), alpha_B(T));
                            }
                            dxe_dzp = dt_dzp*(- dxion_sink_dt + dxion_source2_dt);
                        }
                        else {
                            // JordanFlitter: Let's use HyRec!!
                            // Note that we treat xe and x_H to be the same, since after recombination x_He << xe,x_H.
                            // Note also that Hyrec returns the derivative of n_e/n_H. Since we are interested in the derivative of n_e/(n_H+n_He),
                            // we must multiply by n_H/(n_H+n_He)=f_H. Also, Hyrec expects an input of n_e/n_H, so we need to give it x_e/f_H=n_e/n_H
                            // JordanFlitter: we can use the baryons density field
                            if (user_params->EVOLVE_BARYONS){
                                dxe_dzp = hyrec_dx_H_dz(rec_data, x_e/f_H, x_e/f_H, No*pow(1.+zp,3)*(1.+delta_baryons_local), zp, hubble(zp), T, Trad_fast);
                            }
                            else {
                                dxe_dzp = hyrec_dx_H_dz(rec_data, x_e/f_H, x_e/f_H, No*pow(1.+zp,3)*(1.+curr_delNL0*growth_factor_zp), zp, hubble(zp), T, Trad_fast);
                            }
                            dxe_dzp *= f_H;
                        }
                        // Next, let's get the temperature components //
                        // JordanFlitter: we can use the baryons density field
                        if (user_params->EVOLVE_BARYONS){
                            dadia_dzp = 2.*T/(1.0+zp) + (2.0/3.0)*T*delta_baryons_derivative_local/(1.0+delta_baryons_local);
                        }
                        else {
                            // first, adiabatic term
                            dadia_dzp = 3/(1.0+zp);
                            if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                                dadia_dzp += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);

                            dadia_dzp *= (2.0/3.0)*T;
                        }

                        // next heating due to the changing species
                        dspec_dzp = - dxe_dzp * T / (1+x_e);

                        // next, Compton heating
                        //                dcomp_dzp = dT_comp(zp, T, x_e);
                        // JordanFlitter: there shouldn't be any f_He at the Compton heating term, as x_e=n_e/(n_H+n_He). This was verified with Mesinger
                        dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e))*( Trad_fast - T );

                        // JordanFlitter: Calculate heating rates and drag term in SDM universe, according to arXiv: 1509.00029
                        if (user_params->SCATTERING_DM) {
                            dT_b_2_dt_ext = (dspec_dzp + (dadia_dzp-2.*T/(1.0+zp)))/dtdz(zp);
                            dadia_dzp_SDM = 3/(1.0+zp);
                            if (user_params->EVOLVE_BARYONS){
                                dadia_dzp_SDM = 2.*T_chi/(1.0+zp) + (2.0/3.0)*T_chi*delta_SDM_derivative_local/(1.0+delta_SDM_local);
                            }
                            else {
                                if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                                    dadia_dzp_SDM += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);

                                dadia_dzp_SDM *= (2.0/3.0)*T_chi;
                            }
                            dT_chi_2_dt_ext = (dadia_dzp_SDM-2.*T_chi/(1.0+zp))/dtdz(zp);
                            if (user_params->EVOLVE_BARYONS){
                                SDM_rates = SDM_derivatives(zp, x_e, T, T_chi, V_chi_b, delta_baryons_local, delta_SDM_local, hubble(zp),
                                                -hubble(zp)*(1.+zp)*dcomp_dzp_prefactor*x_e/(1.+x_e), Trad_fast, dzp, dT_b_2_dt_ext,
                                                dT_chi_2_dt_ext);
                            }
                            else {
                                SDM_rates = SDM_derivatives(zp, x_e, T, T_chi, V_chi_b, curr_delNL0*growth_factor_zp, curr_delNL0*growth_factor_zp, hubble(zp),
                                                -hubble(zp)*(1.+zp)*dcomp_dzp_prefactor*x_e/(1.+x_e), Trad_fast, dzp, dT_b_2_dt_ext,
                                                dT_chi_2_dt_ext);
                            }
                            dSDM_b_heat_dzp = (2.0/3.0/k_B)*SDM_rates.Q_dot_b*dtdz(zp); // K
                            dSDM_chi_heat_dzp = (2.0/3.0/k_B)*SDM_rates.Q_dot_chi*dtdz(zp); // K
                            D_V_chi_b_dzp = SDM_rates.D_V_chi_b*dtdz(zp); // cm/sec
                        }
                        else {
                            dSDM_b_heat_dzp = 0.;
                            dSDM_chi_heat_dzp = 0.;
                            D_V_chi_b_dzp = 0.;
                        }

                        //update quantities

                        // JordanFlitter: this is epsilon_gamma_b = H/Gamma_C. If this parameter is small enough, we are in the Compton
                        // tight coupling regime
                        epsilon_gamma_b = -(1.+x_e)/x_e/(1.+zp)/dcomp_dzp_prefactor;
                        // JordanFlitter: If we are not in Compton tight coupling, and epsilon_b is not small enough, we evolve T_k with the usual ODE
                        // JordanFlitter: If epsilon_b is not small enough, or if external heating rates dominate the SDM cooling rate,
                        //                or SDM doesn't exist, we evolve T_k with the usual ODE
                        if ((fabs(epsilon_gamma_b) > global_params.EPSILON_THRESH_HIGH_Z || !user_params->USE_TCA_COMPTON) &&
                            (!user_params->SCATTERING_DM ||
                            (user_params->SCATTERING_DM && ((fabs(SDM_rates.epsilon_b) > EPSILON_THRES) || (fabs(dT_b_2_dt_ext*dtdz(zp))>fabs(dSDM_b_heat_dzp))))
                            )) {
                            // JordanFlitter: I also added the SDM heating exchange
                            if (T < MAX_TK) {
                                T += (dcomp_dzp + dspec_dzp + dadia_dzp + dSDM_b_heat_dzp) * dzp;
                            }
                        }
                        // JordanFlitter: Otherwise, if epsilon_b is not small enough (or SDM doesn't exist), we evolve T_k with Compton TCA!
                        else if (!user_params->SCATTERING_DM || (user_params->SCATTERING_DM && fabs(SDM_rates.epsilon_b) > EPSILON_THRES)) {
                            T_bar_gamma_b = (Trad_fast+T)/2.; // K
                            T_bar_gamma_b += Trad_fast/(1.+zp) * dzp; // K
                            epsilon_gamma_b += epsilon_gamma_b * (dhubble_dz(zp)/hubble(zp) - dxe_dzp/((1.+x_e)*x_e) - 4./(1.+zp)) * dzp;
                            Delta_T_gamma_b = epsilon_gamma_b*(2.*T - Trad_fast + (dspec_dzp + (dadia_dzp-2.*T/(1.0+zp)))*(1.+zp) + dSDM_b_heat_dzp*(1.+zp)); // K
                            T = T_bar_gamma_b - Delta_T_gamma_b/2.; // K
                        }
                        // JordanFlitter: Otherwise, we evolve T_k with DM TCA!
                        else {
                            T = SDM_rates.T_bar_chi_b + SDM_rates.Delta_T_b_chi/2.; // K
                        }
                        if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                            T = T_cmb*(1+zp);
                        }

                        x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                        if (x_e > 1) // can do this late in evolution if dzp is too large
                            x_e = 1 - FRACT_FLOAT_ERR;
                        else if (x_e < 0)
                            x_e = 0;
                        // JordanFlitter: evolution equations for T_chi and V_chi_b in SDM universe
                        if (user_params->SCATTERING_DM) {
                            // JordanFlitter: If epsilon_chi is not small enough, or if external heating rates dominate the baryon heating rate,
                            // we evolve T_chi with the usual ODE
                            if ((fabs(SDM_rates.epsilon_chi) > EPSILON_THRES) || (fabs(dT_chi_2_dt_ext*dtdz(zp))>fabs(dSDM_chi_heat_dzp))) {
                                T_chi += (dadia_dzp_SDM + dSDM_chi_heat_dzp)*dzp; // K
                            }
                            // JordanFlitter: Otherwise, we evolve T_chi with DM TCA!
                            else {
                                T_chi = SDM_rates.T_bar_chi_b - SDM_rates.Delta_T_b_chi/2.; // K
                            }
                            V_chi_b += (V_chi_b/(1.0+zp) - D_V_chi_b_dzp)*dzp; // cm/sec
                        }
                        // JordanFlitter: similar logic for T_chi and V_chi_b in case of spurious bahaviour
                        if (user_params->SCATTERING_DM) {
                            if (T_chi<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                T_chi = T;
                            }
                            if (V_chi_b<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                V_chi_b = 0.;
                            }
                        }

                        this_spin_temp->x_e_box[box_ct] = x_e;
                        this_spin_temp->Tk_box[box_ct] = T;
                        // JordanFlitter: update T_chi_box and V_chi_b_box
                        if (user_params->SCATTERING_DM) {
                          this_spin_temp->T_chi_box[box_ct] = T_chi; // K
                          this_spin_temp->V_chi_b_box[box_ct] = 1.e-5*V_chi_b; // km/sec
                        }

                        // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                        // Algorithm is the same, but written to be more computationally efficient
                        T_inv = expf((-1.)*logf(T));

                        // JordanFlitter: we can use the baryons density field
                        if (user_params->EVOLVE_BARYONS){
                            xc_fast = (1.0+delta_baryons_local)*xc_inverse*\
                                    ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                        }
                        else {
                            xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*\
                                    ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                        }

                        TS_fast = (xCMB + xc_fast)/(xCMB*Trad_fast_inv + xc_fast*T_inv);

                        if(TS_fast < 0.) {
                            // It can very rarely result in a negative spin temperature. If negative, it is a very small number.
                            //Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                            TS_fast = fabs(TS_fast);
                        }

                        this_spin_temp->Ts_box[box_ct] = TS_fast;
                  }
              }
            // JordanFlitter: promote redshift to next iteration
            prev_zp = zp;
            zp += dzp;
        }
        // JordanFlitter: Need to update next redshift because of floating point error (note that next_redshift_output is *not* precisely next_redshift_input)
        //                This will help keeping the python and C loops synchronized during the dark ages
        this_spin_temp->next_redshift_output = zp;

        // JordanFlitter: We need to free delNL0 during the dark ages!
        if (!user_params->NO_INI_MATTER_FLUCTS && !user_params->EVOLVE_BARYONS){
            free(delNL0[0]);
            free(delNL0);
        }

        } // JordanFlitter: End of dark ages condition.
        // JordanFlitter: We don't need all of that during the dark ages (if we enter the "else" condition below, astrophysics kicks in as we are in cosmic dawn)
        else {
        /////////////// Create the z=0 non-linear density fields smoothed on scale R to be used in computing fcoll //////////////
        R = L_FACTOR*user_params->BOX_LEN/(float)user_params->HII_DIM;
        R_factor = pow(global_params.R_XLy_MAX/R, 1/((float)global_params.NUM_FILTER_STEPS_FOR_Ts));
        //      R_factor = pow(E, log(HII_DIM)/(float)NUM_FILTER_STEPS_FOR_Ts);
LOG_SUPER_DEBUG("Looping through R");

        if(this_spin_temp->first_box || (fabs(initialised_redshift - perturbed_field_redshift) > 0.0001) ) {

            // allocate memory for the nonlinear density field
#pragma omp parallel shared(unfiltered_box,perturbed_field) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<user_params->HII_DIM; k++){
                            *((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->density[HII_R_INDEX(i,j,k)];
                        }
                    }
                }
            }
LOG_SUPER_DEBUG("Allocated unfiltered box");

            ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, unfiltered_box);
LOG_SUPER_DEBUG("Done FFT on unfiltered box");

            // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
            // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
#pragma omp parallel shared(unfiltered_box) private(ct) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                    unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
                }
            }

LOG_SUPER_DEBUG("normalised unfiltered box");

            // Smooth the density field, at the same time store the minimum and maximum densities for their usage in the interpolation tables
            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){

                R_values[R_ct] = R;

                if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
                    sigma_atR[R_ct] = sigma_z0(RtoM(R));
                }

                // copy over unfiltered box
                memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                if (R_ct > 0){ // don't filter on cell size
                    filter_box(box, 1, global_params.HEAT_FILTER, R);
                }
                // now fft back to real space
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, box);
LOG_ULTRA_DEBUG("Executed FFT for R=%f", R);

                min_density = 0.0;
                max_density = 0.0;

                // copy over the values
#pragma omp parallel shared(box,inverse_growth_factor_z,delNL0,delNL0_rev) private(i,j,k,curr_delNL0) num_threads(user_params->N_THREADS)
                {
#pragma omp for reduction(max:max_density) reduction(min:min_density)
                    for (i=0;i<user_params->HII_DIM; i++){
                        for (j=0;j<user_params->HII_DIM; j++){
                            for (k=0;k<user_params->HII_DIM; k++){
                                curr_delNL0 = *((float *)box + HII_R_FFT_INDEX(i,j,k));

                                if (curr_delNL0 <= -1){ // correct for aliasing in the filtering step
                                    curr_delNL0 = -1+FRACT_FLOAT_ERR;
                                }

                                // and linearly extrapolate to z=0
                                curr_delNL0 *= inverse_growth_factor_z;

                                if(flag_options->USE_MASS_DEPENDENT_ZETA) {
                                    if(!user_params->MINIMIZE_MEMORY) {
                                        delNL0[R_ct][HII_R_INDEX(i,j,k)] = curr_delNL0;
                                    }
                                }
                                else {
                                    delNL0_rev[HII_R_INDEX(i,j,k)][R_ct] = curr_delNL0;
                                }

                                if(curr_delNL0 < min_density) {
                                    min_density = curr_delNL0;
                                }
                                if(curr_delNL0 > max_density) {
                                    max_density = curr_delNL0;
                                }
                            }
                        }
                    }
                }

LOG_ULTRA_DEBUG("COPIED OVER VALUES");

                if(user_params->USE_INTERPOLATION_TABLES) {
                    if(min_density < 0.0) {
                        min_density = min_density*1.01;
                        // min_density here can exceed -1. as it is always extrapolated back to the appropriate redshift
                    }
                    else {
                        min_density = min_density*0.99;
                    }
                    if(max_density < 0.0) {
                        max_density = max_density*0.99;
                    }
                    else {
                        max_density = max_density*1.01;
                    }

                    if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
                        delNL0_LL[R_ct] = min_density;
                        delNL0_Offset[R_ct] = 1.e-6 - (delNL0_LL[R_ct]);
                        delNL0_UL[R_ct] = max_density;
                    }

                    min_densities[R_ct] = min_density;
                    max_densities[R_ct] = max_density;
                }

                R *= R_factor;
LOG_ULTRA_DEBUG("FINISHED WITH THIS R, MOVING ON");
            } //end for loop through the filter scales R
        }

LOG_SUPER_DEBUG("Finished loop through filter scales R");

        zp = perturbed_field_redshift*1.0001; //higher for rounding
        if(zp > global_params.Z_HEAT_MAX) {
            prev_zp = ((1+zp)/ global_params.ZPRIME_STEP_FACTOR - 1);
        }
        else {
            while (zp < global_params.Z_HEAT_MAX)
                zp = ((1+zp)*global_params.ZPRIME_STEP_FACTOR - 1);
            prev_zp = global_params.Z_HEAT_MAX;
            zp = ((1+zp)/ global_params.ZPRIME_STEP_FACTOR - 1);
        }

        // This sets the delta_z step for determining the heating/ionisation integrals. If first box is true,
        // it is only the difference between Z_HEAT_MAX and the redshift. Otherwise it is the difference between
        // the current and previous redshift (this behaviour is chosen to mimic 21cmFAST)

        // JordanFlitter: since we take the initial conditions at prev_redshift (and not at Z_HEAT_MAX), we don't have to distinguish between
        // first_box and not first_box scenarios
        /*if(this_spin_temp->first_box) {
            dzp = zp - prev_zp;
        }
        else {*/
            dzp = redshift - prev_redshift;
        //}

        determine_zpp_min = perturbed_field_redshift*0.999;

        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            if (R_ct==0){
                prev_zpp = zp;
                prev_R = 0;
            }
            else{
                prev_zpp = zpp_edge[R_ct-1];
                prev_R = R_values[R_ct-1];
            }
            zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
            zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
        }

        determine_zpp_max = zpp*1.001;

        if(!flag_options->M_MIN_in_Mass) {
            M_MIN = (float)TtoM(determine_zpp_max, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
            if(user_params->USE_INTERPOLATION_TABLES) {
              if(user_params->FAST_FCOLL_TABLES){
                initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN),1e20);
              }
              else{
                initialiseSigmaMInterpTable(M_MIN,1e20);
              }
            }
        }

        LOG_SUPER_DEBUG("Initialised sigma interp table");

        if(user_params->USE_INTERPOLATION_TABLES) {
            zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points_SFR-1.0);

            dens_width = 1./((double)dens_Ninterp - 1.);
        }

        if(this_spin_temp->first_box || (fabs(initialised_redshift - perturbed_field_redshift) > 0.0001) ) {

            ////////////////////////////    Create and fill interpolation tables to be used by Ts.c   /////////////////////////////

            if(user_params->USE_INTERPOLATION_TABLES) {

                if(flag_options->USE_MASS_DEPENDENT_ZETA) {

                    // generates an interpolation table for redshift
                    for (i=0; i<zpp_interp_points_SFR;i++) {
                        zpp_interp_table[i] = determine_zpp_min + zpp_bin_width*(float)i;
                    }

                    /* initialise interpolation of the mean collapse fraction for global reionization.*/
                    if (!flag_options->USE_MINI_HALOS){
                        initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                                 astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->ALPHA_ESC,
                                                 astro_params->F_STAR10, astro_params->F_ESC10);

                        initialise_SFRD_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                              astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->F_STAR10);
                    }
                    else{
                        initialise_Nion_Ts_spline_MINI(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                                      astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI, astro_params->ALPHA_ESC, astro_params->F_STAR10,
                                                      astro_params->F_ESC10, astro_params->F_STAR7_MINI, astro_params->F_ESC7_MINI);

                        initialise_SFRD_spline_MINI(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                                   astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI, astro_params->F_STAR10, astro_params->F_STAR7_MINI);
                    }
                    interpolation_tables_allocated = true;
                }
                else {
                    // An interpolation table for f_coll (delta vs redshift)
                    init_FcollTable(determine_zpp_min,determine_zpp_max);

                    // Determine the sampling of the density values, for the various interpolation tables
                    for(ii=0;ii<global_params.NUM_FILTER_STEPS_FOR_Ts;ii++) {
                        log10delNL0_diff_UL[ii] = log10( delNL0_UL[ii] + delNL0_Offset[ii] );
                        log10delNL0_diff[ii] = log10( delNL0_LL[ii] + delNL0_Offset[ii] );
                        delNL0_bw[ii] = ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width;
                        delNL0_ibw[ii] = 1./delNL0_bw[ii];
                    }

                    // Gridding the density values for the interpolation tables
                    for(ii=0;ii<global_params.NUM_FILTER_STEPS_FOR_Ts;ii++) {
                        for(j=0;j<dens_Ninterp;j++) {
                            grid_dens[ii][j] = log10delNL0_diff[ii] + ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width*(double)j;
                            grid_dens[ii][j] = pow(10,grid_dens[ii][j]) - delNL0_Offset[ii];
                        }
                    }

                    // Calculate the sigma_z and Fgtr_M values for each point in the interpolation table
#pragma omp parallel shared(determine_zpp_min,determine_zpp_max,Sigma_Tmin_grid,ST_over_PS_arg_grid,\
                            mu_for_Ts,M_MIN,M_MIN_WDM) \
                    private(i,zpp_grid) num_threads(user_params->N_THREADS)
                    {
#pragma omp for
                        for(i=0;i<zpp_interp_points_SFR;i++) {
                            zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points_SFR-1.0);

                            if(flag_options->M_MIN_in_Mass) {
                                Sigma_Tmin_grid[i] = sigma_z0(fmaxf(M_MIN,  M_MIN_WDM));
                                ST_over_PS_arg_grid[i] = FgtrM_General(zpp_grid, fmaxf(M_MIN,  M_MIN_WDM));
                            }
                            else {
                                Sigma_Tmin_grid[i] = sigma_z0(fmaxf((float)TtoM(zpp_grid, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                                ST_over_PS_arg_grid[i] = FgtrM_General(zpp_grid, fmaxf((float)TtoM(zpp_grid, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                            }
                        }
                    }

                // Create the interpolation tables for the derivative of the collapsed fraction and the collapse fraction itself
#pragma omp parallel shared(fcoll_R_grid,dfcoll_dz_grid,Sigma_Tmin_grid,determine_zpp_min,determine_zpp_max,\
                            grid_dens,sigma_atR) \
                    private(ii,i,j,zpp_grid,grid_sigmaTmin,grid_dens_val) num_threads(user_params->N_THREADS)
                    {
#pragma omp for
                        for(ii=0;ii<global_params.NUM_FILTER_STEPS_FOR_Ts;ii++) {
                            for(i=0;i<zpp_interp_points_SFR;i++) {

                                zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points_SFR-1.0);
                                grid_sigmaTmin = Sigma_Tmin_grid[i];

                                for(j=0;j<dens_Ninterp;j++) {

                                    grid_dens_val = grid_dens[ii][j];
                                    fcoll_R_grid[ii][i][j] = sigmaparam_FgtrM_bias(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                                    dfcoll_dz_grid[ii][i][j] = dfcoll_dz(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                                }
                            }
                        }
                    }

                    // Determine the grid point locations for solving the interpolation tables
                    for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                        for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                            SingleVal_int[R_ct] = (short)floor( ( log10(delNL0_rev[box_ct][R_ct] + delNL0_Offset[R_ct]) - log10delNL0_diff[R_ct] )*delNL0_ibw[R_ct]);
                        }
                        memcpy(dens_grid_int_vals[box_ct],SingleVal_int,sizeof(short)*global_params.NUM_FILTER_STEPS_FOR_Ts);
                    }

                    // Evaluating the interpolated density field points (for using the interpolation tables for fcoll and dfcoll_dz)
                    for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                        OffsetValue = delNL0_Offset[R_ct];
                        DensityValueLow = delNL0_LL[R_ct];
                        delNL0_bw_val = delNL0_bw[R_ct];

                        for(i=0;i<dens_Ninterp;i++) {
                            density_gridpoints[i][R_ct] = pow(10.,( log10( DensityValueLow + OffsetValue) + delNL0_bw_val*((float)i) )) - OffsetValue;
                        }
                    }
                }
            }

            initialised_redshift = perturbed_field_redshift;
        }

LOG_SUPER_DEBUG("got density gridpoints");

        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
            /* generate a table for interpolation of the collapse fraction with respect to the X-ray heating, as functions of
             filtering scale, redshift and overdensity.
             Note that at a given zp, zpp values depends on the filtering scale R, i.e. f_coll(z(R),delta).
             Compute the conditional mass function, but assume f_{esc10} = 1 and \alpha_{esc} = 0. */

            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                if (R_ct==0){
                    prev_zpp = redshift;
                    prev_R = 0;
                }
                else{
                    prev_zpp = zpp_edge[R_ct-1];
                    prev_R = R_values[R_ct-1];
                }
                zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
                zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
                zpp_growth[R_ct] = dicke(zpp);
                if (flag_options->USE_MINI_HALOS){
                    Mcrit_atom_interp_table[R_ct] = atomic_cooling_threshold(zpp);
                }
            }

            if(user_params->USE_INTERPOLATION_TABLES) {
                if (!flag_options->USE_MINI_HALOS){
                    initialise_SFRD_Conditional_table(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,
                                                     max_densities,zpp_growth,R_values, astro_params->M_TURN,
                                                     astro_params->ALPHA_STAR, astro_params->F_STAR10, user_params->FAST_FCOLL_TABLES);
                }
                else{
                    initialise_SFRD_Conditional_table_MINI(global_params.NUM_FILTER_STEPS_FOR_Ts,min_densities,
                                                          max_densities,zpp_growth,R_values,Mcrit_atom_interp_table,
                                                          astro_params->ALPHA_STAR, astro_params->ALPHA_STAR_MINI, astro_params->F_STAR10,
                                                          astro_params->F_STAR7_MINI, user_params->FAST_FCOLL_TABLES);
                }
            }
        }

        LOG_SUPER_DEBUG("Initialised SFRD table");

        zp = redshift;
        prev_zp = prev_redshift;

        if(flag_options->USE_MASS_DEPENDENT_ZETA) {

            if(user_params->USE_INTERPOLATION_TABLES) {
                redshift_int_Nion_z = (int)floor( ( zp - determine_zpp_min )/zpp_bin_width );

                if(redshift_int_Nion_z < 0 || (redshift_int_Nion_z + 1) > (zpp_interp_points_SFR - 1)) {
                    LOG_ERROR("I have overstepped my allocated memory for the interpolation table Nion_z_val");
//                    Throw(ParameterError);
                    Throw(TableEvaluationError);
                }

                redshift_table_Nion_z = determine_zpp_min + zpp_bin_width*(float)redshift_int_Nion_z;

                Splined_Fcollzp_mean = Nion_z_val[redshift_int_Nion_z] + \
                        ( zp - redshift_table_Nion_z )*( Nion_z_val[redshift_int_Nion_z+1] - Nion_z_val[redshift_int_Nion_z] )/(zpp_bin_width);
            }
            else {

                if(flag_options->USE_MINI_HALOS) {
                    Splined_Fcollzp_mean = Nion_General(zp, global_params.M_MIN_INTEGRAL, atomic_cooling_threshold(zp), astro_params->ALPHA_STAR, astro_params->ALPHA_ESC,
                                                        astro_params->F_STAR10, astro_params->F_ESC10, Mlim_Fstar, Mlim_Fesc);
                }
                else {
                    Splined_Fcollzp_mean = Nion_General(zp, M_MIN, astro_params->M_TURN, astro_params->ALPHA_STAR, astro_params->ALPHA_ESC,
                                                    astro_params->F_STAR10, astro_params->F_ESC10, Mlim_Fstar, Mlim_Fesc);
                }
            }

            if (flag_options->USE_MINI_HALOS){
                log10_Mcrit_mol = log10(lyman_werner_threshold(zp, 0., 0.,astro_params));
                log10_Mcrit_LW_ave = 0.0;
#pragma omp parallel shared(log10_Mcrit_LW_unfiltered,previous_spin_temp,zp) private(i,j,k,curr_vcb) num_threads(user_params->N_THREADS)
                {
#pragma omp for reduction(+:log10_Mcrit_LW_ave)
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<user_params->HII_DIM; k++){

                              if (flag_options->FIX_VCB_AVG){ //with this flag we ignore reading vcb box
                                curr_vcb = global_params.VAVG;
                              }
                              else{
                                if(user_params->USE_RELATIVE_VELOCITIES){
                                  // JordanFlitter: I modified the relative velocity that enters into the calculation of M_mol.
                                  // This is essentially (1-f_chi)*v_cb_kin(x)+f_chi*v_chi_b(x,z)*(1+z_kin)/(1+z).
                                  // Remember that f_chi in the code is actually -log10(f_chi).
                                  // The reason for this modeling is the following.
                                  //    If f_chi = 0, then v=v_cb_kin(x) as for LambdaCDM.
                                  //    If f_chi = 1, then v=v_chi_b(x,z)*(1+z_kin)/(1+z).
                                  //        If sigma is small (weak interaction), SDM is essentially CDM and v_chi_b(x,z)*(1+z_kin)/(1+z)=v_cb(x,z)*(1+z_kin)/(1+z)=v_cb_kin(x)
                                  //        If sigma is large (strong interaction), then v_chi_b(x,z)*(1+z_kin)/(1+z) << v_cb_kin(x)
                                  if (user_params->SCATTERING_DM) {
                                      curr_vcb = ((1.-pow(10.,-cosmo_params->f_chi))*(ini_boxes->lowres_vcb[HII_R_INDEX(i,j,k)])
                                                  + pow(10.,-cosmo_params->f_chi)*(previous_spin_temp->V_chi_b_box[HII_R_INDEX(i,j,k)])*(1.+global_params.Z_REC)/(1.+zp));
                                  }
                                  else {
                                      curr_vcb = ini_boxes->lowres_vcb[HII_R_INDEX(i,j,k)];
                                  }
                                }
                                else{ //set vcb to a constant, either zero or vavg.
                                  curr_vcb = 0.0;
                                }
                              }

                              *((float *)log10_Mcrit_LW_unfiltered + HII_R_FFT_INDEX(i,j,k)) = \
                                              log10(lyman_werner_threshold(zp, previous_spin_temp->J_21_LW_box[HII_R_INDEX(i,j,k)],
                                              curr_vcb, astro_params) );


//This only accounts for effect 3 (only on minihaloes). Effects 1+2 also affects ACGs, but is included only on average.



                                log10_Mcrit_LW_ave += *((float *)log10_Mcrit_LW_unfiltered + HII_R_FFT_INDEX(i,j,k));
                            }
                        }
                    }
                }
                log10_Mcrit_LW_ave /= (double)HII_TOT_NUM_PIXELS;

                // NEED TO FILTER Mcrit_LW!!!
                /*** Transform unfiltered box to k-space to prepare for filtering ***/
                dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, log10_Mcrit_LW_unfiltered);

#pragma omp parallel shared(log10_Mcrit_LW_unfiltered) private(ct) num_threads(user_params->N_THREADS)
                {
#pragma omp for
                    for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++) {
                        log10_Mcrit_LW_unfiltered[ct] /= (float)HII_TOT_NUM_PIXELS;
                    }
                }

                if(user_params->USE_INTERPOLATION_TABLES) {
                    log10_Mcrit_LW_ave_int_Nion_z = (int)floor( ( log10_Mcrit_LW_ave - LOG10_MTURN_MIN) / LOG10_MTURN_INT);
                    log10_Mcrit_LW_ave_table_Nion_z = LOG10_MTURN_MIN + LOG10_MTURN_INT * (float)log10_Mcrit_LW_ave_int_Nion_z;

                    Splined_Fcollzp_mean_MINI_left = Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_Nion_z] + \
                                                ( zp - redshift_table_Nion_z ) / (zpp_bin_width)*\
                                                  ( Nion_z_val_MINI[redshift_int_Nion_z + 1 + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_Nion_z] -\
                                                    Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_Nion_z] );
                    Splined_Fcollzp_mean_MINI_right = Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_Nion_z+1)] + \
                                                ( zp - redshift_table_Nion_z ) / (zpp_bin_width)*\
                                                  ( Nion_z_val_MINI[redshift_int_Nion_z + 1 + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_Nion_z+1)] -\
                                                    Nion_z_val_MINI[redshift_int_Nion_z + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_Nion_z+1)] );
                    Splined_Fcollzp_mean_MINI = Splined_Fcollzp_mean_MINI_left + \
                                (log10_Mcrit_LW_ave - log10_Mcrit_LW_ave_table_Nion_z) / LOG10_MTURN_INT * (Splined_Fcollzp_mean_MINI_right - Splined_Fcollzp_mean_MINI_left);
                }
                else {
                    Splined_Fcollzp_mean_MINI = Nion_General_MINI(zp, global_params.M_MIN_INTEGRAL, pow(10.,log10_Mcrit_LW_ave), atomic_cooling_threshold(zp),
                                                                  astro_params->ALPHA_STAR_MINI, astro_params->ALPHA_ESC, astro_params->F_STAR7_MINI,
                                                                  astro_params->F_ESC7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI);
                }
            }
            else{
                Splined_Fcollzp_mean_MINI = 0;
            }

            if ( ( Splined_Fcollzp_mean < 1e-15 ) && (Splined_Fcollzp_mean_MINI < 1e-15))
                NO_LIGHT = 1;
            else
                NO_LIGHT = 0;


            filling_factor_of_HI_zp = 1 - ( ION_EFF_FACTOR * Splined_Fcollzp_mean + ION_EFF_FACTOR_MINI * Splined_Fcollzp_mean_MINI )/ (1.0 - x_e_ave);

        }
        else {

            if(flag_options->M_MIN_in_Mass) {

                if (FgtrM(zp, fmaxf(M_MIN,  M_MIN_WDM)) < 1e-15 )
                    NO_LIGHT = 1;
                else
                    NO_LIGHT = 0;

                M_MIN_at_zp = M_MIN;
            }
            else {

                if (FgtrM(zp, fmaxf((float)TtoM(zp, astro_params->X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM)) < 1e-15 )
                    NO_LIGHT = 1;
                else
                    NO_LIGHT = 0;

                M_MIN_at_zp = get_M_min_ion(zp);
            }
            filling_factor_of_HI_zp = 1 - ION_EFF_FACTOR * FgtrM_General(zp, M_MIN_at_zp) / (1.0 - x_e_ave);
        }

        if (filling_factor_of_HI_zp > 1) filling_factor_of_HI_zp=1;

        // let's initialize an array of redshifts (z'') corresponding to the
        // far edge of the dz'' filtering shells
        // and the corresponding minimum halo scale, sigma_Tmin,
        // as well as an array of the frequency integrals
LOG_SUPER_DEBUG("beginning loop over R_ct");

        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            if (R_ct==0){
                prev_zpp = zp;
                prev_R = 0;
            }
            else{
                prev_zpp = zpp_edge[R_ct-1];
                prev_R = R_values[R_ct-1];
            }

            zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
            zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''

            zpp_for_evolve_list[R_ct] = zpp;
            if (R_ct==0){
                dzpp_for_evolve = zp - zpp_edge[0];
            }
            else{
                dzpp_for_evolve = zpp_edge[R_ct-1] - zpp_edge[R_ct];
            }
            zpp_growth[R_ct] = dicke(zpp);
            if (flag_options->USE_MINI_HALOS){
                Mcrit_atom_interp_table[R_ct] = atomic_cooling_threshold(zpp);
            }

            fcoll_R_array[R_ct] = 0.0;

            // let's now normalize the total collapse fraction so that the mean is the
            // Sheth-Torman collapse fraction
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation

                if(user_params->USE_INTERPOLATION_TABLES) {
                    redshift_int_SFRD = (int)floor( ( zpp - determine_zpp_min )/zpp_bin_width );

                    if(redshift_int_SFRD < 0 || (redshift_int_SFRD + 1) > (zpp_interp_points_SFR - 1)) {
                        LOG_ERROR("I have overstepped my allocated memory for the interpolation table SFRD_val");
//                        Throw(ParameterError);
                        Throw(TableEvaluationError);
                    }

                    redshift_table_SFRD = determine_zpp_min + zpp_bin_width*(float)redshift_int_SFRD;

                    Splined_SFRD_zpp = SFRD_val[redshift_int_SFRD] + \
                                    ( zpp - redshift_table_SFRD )*( SFRD_val[redshift_int_SFRD+1] - SFRD_val[redshift_int_SFRD] )/(zpp_bin_width);

                    ST_over_PS[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve);
                    ST_over_PS[R_ct] *= Splined_SFRD_zpp;
                }
                else {
                    ST_over_PS[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve); // Multiplied by Nion later
                }

                if(flag_options->USE_MINI_HALOS){
                    memcpy(log10_Mcrit_LW_filtered, log10_Mcrit_LW_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                    if (R_ct > 0){// don't filter on cell size
                        filter_box(log10_Mcrit_LW_filtered, 1, global_params.HEAT_FILTER, R_values[R_ct]);
                    }
                    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, log10_Mcrit_LW_filtered);

                    log10_Mcrit_LW_ave = 0; //recalculate it at this filtering scale
#pragma omp parallel shared(log10_Mcrit_LW,log10_Mcrit_LW_filtered,log10_Mcrit_mol) private(i,j,k) num_threads(user_params->N_THREADS)
                    {
#pragma omp for reduction(+:log10_Mcrit_LW_ave)
                        for (i=0; i<user_params->HII_DIM; i++){
                            for (j=0; j<user_params->HII_DIM; j++){
                                for (k=0; k<user_params->HII_DIM; k++){
                                    log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = *((float *) log10_Mcrit_LW_filtered + HII_R_FFT_INDEX(i,j,k));
                                    if(log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] < log10_Mcrit_mol)
                                        log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = log10_Mcrit_mol;
                                    if (log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] > LOG10_MTURN_MAX)
                                        log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = LOG10_MTURN_MAX;
                                    log10_Mcrit_LW_ave += log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)];
                                }
                            }
                        }
                    }
                    log10_Mcrit_LW_ave /= (double)HII_TOT_NUM_PIXELS;

                    log10_Mcrit_LW_ave_list[R_ct] = log10_Mcrit_LW_ave;

                    if(user_params->USE_INTERPOLATION_TABLES) {
                        log10_Mcrit_LW_ave_int_SFRD = (int)floor( ( log10_Mcrit_LW_ave - LOG10_MTURN_MIN) / LOG10_MTURN_INT);
                        log10_Mcrit_LW_ave_table_SFRD = LOG10_MTURN_MIN + LOG10_MTURN_INT * (float)log10_Mcrit_LW_ave_int_SFRD;

                        Splined_SFRD_zpp_MINI_left = SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_SFRD] + \
                                                ( zpp - redshift_table_SFRD ) / (zpp_bin_width)*\
                                                  ( SFRD_val_MINI[redshift_int_SFRD + 1 + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_SFRD] -\
                                                    SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_SFRD] );
                        Splined_SFRD_zpp_MINI_right = SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_SFRD+1)] + \
                                                ( zpp - redshift_table_SFRD ) / (zpp_bin_width)*\
                                                  ( SFRD_val_MINI[redshift_int_SFRD + 1 + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_SFRD+1)] -\
                                                    SFRD_val_MINI[redshift_int_SFRD + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_SFRD+1)] );
                        Splined_SFRD_zpp_MINI = Splined_SFRD_zpp_MINI_left + \
                            (log10_Mcrit_LW_ave - log10_Mcrit_LW_ave_table_SFRD) / LOG10_MTURN_INT * (Splined_SFRD_zpp_MINI_right - Splined_SFRD_zpp_MINI_left);

                        ST_over_PS_MINI[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve);
                        ST_over_PS_MINI[R_ct] *= Splined_SFRD_zpp_MINI;
                    }
                    else {
                        ST_over_PS_MINI[R_ct] = pow(1+zpp, -astro_params->X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve); // Multiplied by Nion later
                    }
                }

                SFR_timescale_factor[R_ct] = hubble(zpp)*fabs(dtdz(zpp));

            }
            else {

                if(user_params->USE_INTERPOLATION_TABLES) {
                    // Determining values for the evaluating the interpolation table
                    zpp_gridpoint1_int = (int)floor((zpp - determine_zpp_min)/zpp_bin_width);
                    zpp_gridpoint2_int = zpp_gridpoint1_int + 1;

                    if(zpp_gridpoint1_int < 0 || (zpp_gridpoint1_int + 1) > (zpp_interp_points_SFR - 1)) {
                        LOG_ERROR("I have overstepped my allocated memory for the interpolation table fcoll_R_grid");
//                        Throw(ParameterError);
                        Throw(TableEvaluationError);
                    }

                    zpp_gridpoint1 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint1_int;
                    zpp_gridpoint2 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint2_int;

                    grad1 = ( zpp_gridpoint2 - zpp )/( zpp_gridpoint2 - zpp_gridpoint1 );
                    grad2 = ( zpp - zpp_gridpoint1 )/( zpp_gridpoint2 - zpp_gridpoint1 );

                    sigma_Tmin[R_ct] = Sigma_Tmin_grid[zpp_gridpoint1_int] + grad2*( Sigma_Tmin_grid[zpp_gridpoint2_int] - Sigma_Tmin_grid[zpp_gridpoint1_int] );

                    // Evaluating the interpolation table for the collapse fraction and its derivative
                    for(i=0;i<(dens_Ninterp-1);i++) {
                        dens_grad = 1./( density_gridpoints[i+1][R_ct] - density_gridpoints[i][R_ct] );

                        fcoll_interp1[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + \
                                              ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                        fcoll_interp2[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + \
                                              ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;

                        dfcoll_interp1[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + \
                                               ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                        dfcoll_interp2[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + \
                                               ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;

                    }

                    // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation
                    ST_over_PS[R_ct] = dzpp_for_evolve * pow(1+zpp, -(astro_params->X_RAY_SPEC_INDEX));
                    ST_over_PS[R_ct] *= ( ST_over_PS_arg_grid[zpp_gridpoint1_int] + \
                                         grad2*( ST_over_PS_arg_grid[zpp_gridpoint2_int] - ST_over_PS_arg_grid[zpp_gridpoint1_int] ) );
                }
                else {
                    if(flag_options->M_MIN_in_Mass) {
                        sigma_Tmin[R_ct] = sigma_z0(fmaxf(M_MIN, M_MIN_WDM));
                    }
                    else {
                        sigma_Tmin[R_ct] = sigma_z0(fmaxf((float)TtoM(zpp, astro_params->X_RAY_Tvir_MIN, mu_for_Ts), M_MIN_WDM));
                    }

                    ST_over_PS[R_ct] = dzpp_for_evolve * pow(1+zpp, -(astro_params->X_RAY_SPEC_INDEX));
                }

            }

            if(user_params->USE_INTERPOLATION_TABLES) {
                if(flag_options->USE_MINI_HALOS){
                    lower_int_limit = fmax(nu_tau_one_MINI(zp, zpp, x_e_ave, filling_factor_of_HI_zp,
                                                              log10_Mcrit_LW_ave,LOG10_MTURN_INT), (astro_params->NU_X_THRESH)*NU_over_EV);
                }
                else{
                    lower_int_limit = fmax(nu_tau_one(zp, zpp, x_e_ave, filling_factor_of_HI_zp), (astro_params->NU_X_THRESH)*NU_over_EV);
                }

                if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly

                // set up frequency integral table for later interpolation for the cell's x_e value
                for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                    freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                    freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                    freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);

                    if(isfinite(freq_int_heat_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_ion_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_lya_tbl[x_e_ct][R_ct])==0) {
                        LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
//                        Throw(ParameterError);
                        Throw(TableGenerationError);
                    }
                }
            }

            // and create the sum over Lya transitions from direct Lyn flux
            sum_lyn[R_ct] = 0;
            // JordanFlitter: Lya flux for Lya heating
            if (flag_options->USE_Lya_HEATING){
                sum_ly2[R_ct] = 0;
                sum_lynto2[R_ct] = 0;
                if (flag_options->USE_MINI_HALOS) {
                    sum_ly2_MINI[R_ct] = 0;
                    sum_lynto2_MINI[R_ct] = 0;
                }
            }
            if (flag_options->USE_MINI_HALOS){
                sum_lyn_MINI[R_ct] = 0;
                sum_lyLWn[R_ct] = 0;
                sum_lyLWn_MINI[R_ct] = 0;
            }
            for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                if (zpp > zmax(zp, n_ct))
                    continue;

                nuprime = nu_n(n_ct)*(1+zpp)/(1.0+zp);
                if (flag_options->USE_MINI_HALOS){
                    sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
                    sum_lyn_MINI[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);
                    if (nuprime < NU_LW_THRESH / NUIONIZATION)
                        nuprime = NU_LW_THRESH / NUIONIZATION;
                    if (nuprime >= nu_n(n_ct + 1))
                        continue;
                    sum_lyLWn[R_ct]  += (1. - astro_params->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                    sum_lyLWn_MINI[R_ct] += (1. - astro_params->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
                    // JordanFlitter: Lya flux for Lya heating
                    if (flag_options->USE_Lya_HEATING && n_ct==2){
                        sum_ly2[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0,2);
                        sum_ly2_MINI[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0,3);
                    }
                    if (flag_options->USE_Lya_HEATING && n_ct>=3){
                        sum_lynto2[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0,2);
                        sum_lynto2_MINI[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0,3);
                    }
                }
                else{
                    sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, global_params.Pop);
                    if (flag_options->USE_Lya_HEATING && n_ct==2){
                        sum_ly2[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0,global_params.Pop);
                    }
                    if (flag_options->USE_Lya_HEATING && n_ct>=3){
                        sum_lynto2[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0,global_params.Pop);
                    }
                }
            }

            // Find if we need to add a partial contribution to a radii to avoid kinks in the Lyman-alpha flux
            // As we look at discrete radii (light-cone redshift, zpp) we can have two radii where one has a
            // contribution and the next (larger) radii has no contribution. However, if the number of filtering
            // steps were infinitely large, we would have contributions between these two discrete radii
            // Thus, this aims to add a weighted contribution to the first radii where this occurs to smooth out
            // kinks in the average Lyman-alpha flux.

            // Note: We do not apply this correction to the LW background as it is unaffected by this. It is only
            // the Lyn contribution that experiences the kink. Applying this correction to LW introduces kinks
            // into the otherwise smooth quantity
            if(R_ct > 1 && sum_lyn[R_ct]==0.0 && sum_lyn[R_ct-1]>0. && first_radii) {

                // The current zpp for which we are getting zero contribution
                trial_zpp_max = (prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp)+prev_zpp)*0.5;
                // The zpp for the previous radius for which we had a non-zero contribution
                trial_zpp_min = (zpp_edge[R_ct-2] - (R_values[R_ct-1] - R_values[R_ct-2])*CMperMPC / drdz(zpp_edge[R_ct-2])+zpp_edge[R_ct-2])*0.5;

                // Split the previous radii and current radii into n_pts_radii smaller radii (redshift) to have fine control of where
                // it transitions from zero to non-zero
                // This is a coarse approximation as it assumes that the linear sampling is a good representation of the different
                // volumes of the shells (from different radii).
                for(ii=0;ii<n_pts_radii;ii++) {
                    trial_zpp = trial_zpp_min + (trial_zpp_max - trial_zpp_min)*(float)ii/((float)n_pts_radii-1.);

                    counter = 0;
                    for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                        if (trial_zpp > zmax(zp, n_ct))
                            continue;

                        counter += 1;
                    }
                    if(counter==0&&first_zero) {
                        first_zero = false;
                        weight = (float)ii/(float)n_pts_radii;
                    }
                }

                // Now add a non-zero contribution to the previously zero contribution
                // The amount is the weight, multplied by the contribution from the previous radii
                sum_lyn[R_ct] = weight * sum_lyn[R_ct-1];
                // JordanFlitter: apply this also for sum_ly2 and sum_lynto2
                if (flag_options->USE_Lya_HEATING){
                    sum_ly2[R_ct] = weight * sum_ly2[R_ct-1];
                    sum_lynto2[R_ct] = weight * sum_lynto2[R_ct-1];
                    if (flag_options->USE_MINI_HALOS){
                        sum_ly2_MINI[R_ct] = weight * sum_ly2_MINI[R_ct-1];
                        sum_lynto2_MINI[R_ct] = weight * sum_lynto2_MINI[R_ct-1];
                    }
                }
                if (flag_options->USE_MINI_HALOS){
                    sum_lyn_MINI[R_ct] = weight * sum_lyn_MINI[R_ct-1];
                }
                first_radii = false;
            }


        } // end loop over R_ct filter steps


        // Throw the time intensive full calculations into a multiprocessing loop to get them evaluated faster
        if(!user_params->USE_INTERPOLATION_TABLES) {

#pragma omp parallel shared(ST_over_PS,zpp_for_evolve_list,log10_Mcrit_LW_ave_list,Mcrit_atom_interp_table,M_MIN,Mlim_Fstar,Mlim_Fstar_MINI,x_e_ave,\
                            filling_factor_of_HI_zp,x_int_XHII,freq_int_heat_tbl,freq_int_ion_tbl,freq_int_lya_tbl,LOG10_MTURN_INT) \
                    private(R_ct,x_e_ct,lower_int_limit) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                        if(flag_options->USE_MINI_HALOS){
                            ST_over_PS[R_ct] *= Nion_General(zpp_for_evolve_list[R_ct], global_params.M_MIN_INTEGRAL, Mcrit_atom_interp_table[R_ct],
                                                             astro_params->ALPHA_STAR, 0., astro_params->F_STAR10, 1.,Mlim_Fstar,0.);
                            ST_over_PS_MINI[R_ct] *= Nion_General_MINI(zpp_for_evolve_list[R_ct], global_params.M_MIN_INTEGRAL, pow(10.,log10_Mcrit_LW_ave_list[R_ct]),
                                                                Mcrit_atom_interp_table[R_ct], astro_params->ALPHA_STAR_MINI, 0.,
                                                                astro_params->F_STAR7_MINI, 1.,Mlim_Fstar_MINI,0.);
                        }
                        else {
                            ST_over_PS[R_ct] *= Nion_General(zpp_for_evolve_list[R_ct], M_MIN, astro_params->M_TURN, astro_params->ALPHA_STAR, 0., astro_params->F_STAR10, 1.,Mlim_Fstar,0.);
                        }
                    }
                    else {
                        if(flag_options->M_MIN_in_Mass) {
                            ST_over_PS[R_ct] *= FgtrM_General(zpp_for_evolve_list[R_ct], fmaxf(M_MIN, M_MIN_WDM));
                        }
                        else {
                            ST_over_PS[R_ct] *= FgtrM_General(zpp_for_evolve_list[R_ct], fmaxf((float)TtoM(zpp_for_evolve_list[R_ct], astro_params->X_RAY_Tvir_MIN, mu_for_Ts), M_MIN_WDM));
                        }
                    }

                    if(flag_options->USE_MINI_HALOS){
                        lower_int_limit = fmax(nu_tau_one_MINI(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp,
                                                               log10_Mcrit_LW_ave_list[R_ct],LOG10_MTURN_INT), (astro_params->NU_X_THRESH)*NU_over_EV);
                    }
                    else{
                        lower_int_limit = fmax(nu_tau_one(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp), (astro_params->NU_X_THRESH)*NU_over_EV);
                    }

                    if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly

                    // set up frequency integral table for later interpolation for the cell's x_e value
                    for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                        freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                        freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                        freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);
                    }
                }
            }

            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                    if(isfinite(freq_int_heat_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_ion_tbl[x_e_ct][R_ct])==0 || isfinite(freq_int_lya_tbl[x_e_ct][R_ct])==0) {
                        LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
//                        Throw(ParameterError);
                        Throw(TableGenerationError);
                    }
                }
            }
        }

LOG_SUPER_DEBUG("finished looping over R_ct filter steps");

        if(user_params->USE_INTERPOLATION_TABLES) {
            fcoll_interp_high_min = global_params.CRIT_DENS_TRANSITION;
            fcoll_interp_high_bin_width = 1./((float)NSFR_high-1.)*(Deltac - fcoll_interp_high_min);
            fcoll_interp_high_bin_width_inv = 1./fcoll_interp_high_bin_width;
        }

        // Calculate fcoll for each smoothing radius
        if(!flag_options->USE_MASS_DEPENDENT_ZETA) {
            if(user_params->N_THREADS==1) {
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                    for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){

                        if(user_params->USE_INTERPOLATION_TABLES) {
                            if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                                table_int_boundexceeded = 1;
                            }

                            fcoll_R_array[R_ct] += ( fcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                            ( density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct] ) + \
                                            fcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                            ( delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct] ) );
                        }
                        else {
                            fcoll_R_array[R_ct] += sigmaparam_FgtrM_bias(zpp_for_evolve_list[R_ct],sigma_Tmin[R_ct],delNL0_rev[box_ct][R_ct],sigma_atR[R_ct]);
                        }
                    }
                    if(table_int_boundexceeded==1) {
                        LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of fcoll");
//                        Throw(ParameterError);
                        Throw(TableEvaluationError);
                    }
                }
            }
            else {

                for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                    fcoll_R_for_reduction = 0.;

#pragma omp parallel shared(dens_grid_int_vals,R_ct,fcoll_interp1,density_gridpoints,delNL0_rev,fcoll_interp2,\
                            table_int_boundexceeded_threaded,zpp_for_evolve_list,sigma_Tmin,sigma_atR) \
                    private(box_ct) num_threads(user_params->N_THREADS)
                    {
#pragma omp for reduction(+:fcoll_R_for_reduction)
                        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                            if(user_params->USE_INTERPOLATION_TABLES) {
                                if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                                    table_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                }

                                fcoll_R_for_reduction += ( fcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                                      ( density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct] ) + \
                                                      fcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]* \
                                                      ( delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct] ) );
                            }
                            else {
                                fcoll_R_for_reduction += sigmaparam_FgtrM_bias(zpp_for_evolve_list[R_ct],sigma_Tmin[R_ct],delNL0_rev[box_ct][R_ct],sigma_atR[R_ct]);
                            }
                        }
                    }
                    fcoll_R_array[R_ct] = fcoll_R_for_reduction;
                }
                for(i=0;i<user_params->N_THREADS;i++) {
                    if(table_int_boundexceeded_threaded[i]==1) {
                        LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of fcoll");
//                        Throw(ParameterError);
                        Throw(TableEvaluationError);
                    }
                }
            }

            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                ST_over_PS[R_ct] = ST_over_PS[R_ct]/(fcoll_R_array[R_ct]/(double)HII_TOT_NUM_PIXELS);
            }
        }

        // scroll through each cell and update the temperature and residual ionization fraction
        growth_factor_zp = dicke(zp);
        dgrowth_factor_dzp = ddicke_dz(zp);
        dt_dzp = dtdz(zp);

        // Conversion of the input bolometric luminosity to a ZETA_X, as used to be used in Ts.c
        // Conversion here means the code otherwise remains the same as the original Ts.c
        if(fabs(astro_params->X_RAY_SPEC_INDEX - 1.0) < 0.000001) {
            Luminosity_converstion_factor = (astro_params->NU_X_THRESH)*NU_over_EV * log( global_params.NU_X_BAND_MAX/(astro_params->NU_X_THRESH) );
            Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
        }
        else {
            Luminosity_converstion_factor = pow( (global_params.NU_X_BAND_MAX)*NU_over_EV , 1. - (astro_params->X_RAY_SPEC_INDEX) ) - \
                                            pow( (astro_params->NU_X_THRESH)*NU_over_EV , 1. - (astro_params->X_RAY_SPEC_INDEX) ) ;
            Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
            Luminosity_converstion_factor *= pow( (astro_params->NU_X_THRESH)*NU_over_EV, - (astro_params->X_RAY_SPEC_INDEX) )*\
                                            (1 - (astro_params->X_RAY_SPEC_INDEX));
        }
        // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the definition of Luminosity)
        Luminosity_converstion_factor *= (3.1556226e7)/(hplank);

        // Leave the original 21cmFAST code for reference. Refer to Greig & Mesinger (2017) for the new parameterisation.
        const_zp_prefactor = ( (astro_params->L_X) * Luminosity_converstion_factor ) / ((astro_params->NU_X_THRESH)*NU_over_EV) \
                                * C * astro_params->F_STAR10 * cosmo_params->OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, astro_params->X_RAY_SPEC_INDEX+3);
        //          This line below is kept purely for reference w.r.t to the original 21cmFAST
        //            const_zp_prefactor = ZETA_X * X_RAY_SPEC_INDEX / NU_X_THRESH * C * F_STAR * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, X_RAY_SPEC_INDEX+3);

        if (flag_options->USE_MINI_HALOS){
            // do the same for MINI
            const_zp_prefactor_MINI = ( (astro_params->L_X_MINI) * Luminosity_converstion_factor ) / ((astro_params->NU_X_THRESH)*NU_over_EV) \
                                    * C * astro_params->F_STAR7_MINI * cosmo_params->OMb * RHOcrit * pow(CMperMPC, -3) * pow(1+zp, astro_params->X_RAY_SPEC_INDEX+3);
        }
        else{
            const_zp_prefactor_MINI = 0.;
        }
        //////////////////////////////  LOOP THROUGH BOX //////////////////////////////

        J_alpha_ave = xalpha_ave = Xheat_ave = Xion_ave = 0.;
        J_alpha_ave_MINI = J_LW_ave = J_LW_ave_MINI = Xheat_ave_MINI = 0.;

        // Extra pre-factors etc. are defined here, as they are independent of the density field,
        // and only have to be computed once per z' or R_ct, rather than each box_ct
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){

            zpp_integrand = ( pow(1+zp,2)*(1+zpp_for_evolve_list[R_ct]) )/( pow(1+zpp_for_evolve_list[R_ct], -(astro_params->X_RAY_SPEC_INDEX)) );
            dstarlya_dt_prefactor[R_ct]  = zpp_integrand * sum_lyn[R_ct];
            // JordanFlitter: Lya flux for Lya heating
            if (flag_options->USE_Lya_HEATING){
                dstarlya_cont_dt_prefactor[R_ct]  = zpp_integrand * sum_ly2[R_ct];
                dstarlya_inj_dt_prefactor[R_ct]  = zpp_integrand * sum_lynto2[R_ct];
                if (flag_options->USE_MINI_HALOS){
                    dstarlya_cont_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_ly2_MINI[R_ct];
                    dstarlya_inj_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lynto2_MINI[R_ct];
                }
            }
            if (flag_options->USE_MINI_HALOS){
                dstarlya_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lyn_MINI[R_ct];
                dstarlyLW_dt_prefactor[R_ct]  = zpp_integrand * sum_lyLWn[R_ct];
                dstarlyLW_dt_prefactor_MINI[R_ct]  = zpp_integrand * sum_lyLWn_MINI[R_ct];
            }
        }

        // Required quantities for calculating the IGM spin temperature
        // Note: These used to be determined in evolveInt (and other functions). But I moved them all here, into a single location.
        Trad_fast = T_cmb*(1.0+zp);
        Trad_fast_inv = 1.0/Trad_fast;
        TS_prefactor = pow(1.0e-7*(1.342881e-7 / hubble(zp))*No*pow(1+zp,3),1./3.);
        // JordanFlitter: I changed the following prefactor to 1.804e11 (instead 1.66e11).
        // See discussion in https://github.com/21cmfast/21cmFAST/issues/325 (Imprecise numerical factor for Lyman-alpha coupling [BUG] #325)
        xa_tilde_prefactor = 1.8038714872967493e11/(1.0+zp);

        xc_inverse =  pow(1.0+zp,3.0)*T21/( Trad_fast*A10_HYPERFINE );

        dcomp_dzp_prefactor = (-1.51e-4)/(hubble(zp)/Ho)/(cosmo_params->hlittle)*pow(Trad_fast,4.0)/(1.0+zp);

        prefactor_1 = N_b0 * pow(1+zp, 3);
        prefactor_2 = astro_params->F_STAR10 * C * N_b0 / FOURPI;
        prefactor_2_MINI = astro_params->F_STAR7_MINI * C * N_b0 / FOURPI;

        x_e_ave = 0; Tk_ave = 0; Ts_ave = 0;

        // Note: I have removed the call to evolveInt, as is default in the original Ts.c.
        // Removal of evolveInt and moving that computation below, removes unneccesary repeated computations
        // and allows for the interpolation tables that are now used to be more easily computed

        // Can precompute these quantities, independent of the density field (i.e. box_ct)
        for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            for (i=0; i<(x_int_NXHII-1); i++) {
                m_xHII_low = i;
                m_xHII_high = m_xHII_low + 1;

                inverse_diff[i] = 1./(x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
                freq_int_heat_tbl_diff[i][R_ct] = freq_int_heat_tbl[m_xHII_high][R_ct] - freq_int_heat_tbl[m_xHII_low][R_ct];
                freq_int_ion_tbl_diff[i][R_ct] = freq_int_ion_tbl[m_xHII_high][R_ct] - freq_int_ion_tbl[m_xHII_low][R_ct];
                freq_int_lya_tbl_diff[i][R_ct] = freq_int_lya_tbl[m_xHII_high][R_ct] - freq_int_lya_tbl[m_xHII_low][R_ct];

            }
        }

LOG_SUPER_DEBUG("looping over box...");

        // JordanFlitter: if we evolve the baryons density field, we need to have delta_b(z) and its redshift derivative
        if (user_params->EVOLVE_BARYONS) {
            #pragma omp parallel shared(perturbed_field,delta_baryons,delta_SDM) private(i,j,k) num_threads(user_params->N_THREADS)
            {
                #pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<user_params->HII_DIM; k++){
                            *((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->baryons_density[HII_R_INDEX(i,j,k)];

                            if (*((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                *((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                            }
                            if (user_params->SCATTERING_DM){
                                *((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) = perturbed_field->SDM_density[HII_R_INDEX(i,j,k)];

                                if (*((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                    *((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                }
                            }
                        }
                    }
                }
            }
            // Now we extrapolate linearly delta_baryons and its redshift derivative to zp. This requires FFT'ing the density box
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_baryons);
            // Make a copy of delta_baryons at k space
            memcpy(delta_baryons_derivative, delta_baryons, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            if (user_params->SCATTERING_DM){
                // Now we extrapolate linearly delta_SDM and its redshift derivative to zp. This requires FFT'ing the density box
                dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_SDM);
                // Make a copy of delta_SDM at k space
                memcpy(delta_SDM_derivative, delta_SDM, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            }

            #pragma omp parallel shared(zp,perturbed_field_redshift,delta_baryons,delta_baryons_derivative,delta_SDM,delta_SDM_derivative) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag) num_threads(user_params->N_THREADS)
                    {
            #pragma omp for
                        for (n_x=0; n_x<user_params->HII_DIM; n_x++){
                            if (n_x>HII_MIDDLE)
                                k_x =(n_x-user_params->HII_DIM) * DELTA_K;  // wrap around for FFT convention
                            else
                                k_x = n_x * DELTA_K;
                            for (n_y=0; n_y<user_params->HII_DIM; n_y++){
                                if (n_y>HII_MIDDLE)
                                    k_y =(n_y-user_params->HII_DIM) * DELTA_K;
                                else
                                    k_y = n_y * DELTA_K;
                                for (n_z=0; n_z<=HII_MIDDLE; n_z++){
                                    k_z = n_z * DELTA_K;
                                    k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                                    *((fftwf_complex *)delta_baryons + HII_C_INDEX(n_x,n_y,n_z)) *= SDGF(zp,k_mag,0)/SDGF(perturbed_field_redshift,k_mag,0)/HII_TOT_NUM_PIXELS;
                                    *((fftwf_complex *)delta_baryons_derivative + HII_C_INDEX(n_x,n_y,n_z)) *= dSDGF_dz(zp,k_mag)/SDGF(perturbed_field_redshift,k_mag,0)/HII_TOT_NUM_PIXELS;
                                    if (user_params->SCATTERING_DM){
                                        *((fftwf_complex *)delta_SDM + HII_C_INDEX(n_x,n_y,n_z)) *= SDGF_SDM(zp,k_mag,0)/SDGF_SDM(perturbed_field_redshift,k_mag,0)/HII_TOT_NUM_PIXELS;
                                        *((fftwf_complex *)delta_SDM_derivative + HII_C_INDEX(n_x,n_y,n_z)) *= dSDGF_SDM_dz(zp,k_mag)/SDGF_SDM(perturbed_field_redshift,k_mag,0)/HII_TOT_NUM_PIXELS;
                                    }
                                }
                            }
                        }
                    }
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_baryons);
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_baryons_derivative);
            if (user_params->SCATTERING_DM){
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_SDM);
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, delta_SDM_derivative);
            }

            #pragma omp parallel shared(delta_baryons,delta_SDM) private(i,j,k) num_threads(user_params->N_THREADS)
            {
                #pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<user_params->HII_DIM; k++){
                            if (*((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                *((float *)delta_baryons + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                            }
                            if (user_params->SCATTERING_DM){
                                if (*((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                    *((float *)delta_SDM + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Main loop over the entire box for the IGM spin temperature and relevant quantities.
        if(flag_options->USE_MASS_DEPENDENT_ZETA) {
// JordanFlitter: I added more shared variables
#pragma omp parallel shared(del_fcoll_Rct,dxheat_dt_box,dxion_source_dt_box,dxlya_dt_box,dstarlya_dt_box,previous_spin_temp,\
                            x_int_XHII,m_xHII_low_box,inverse_val_box,inverse_diff,dstarlyLW_dt_box,dstarlyLW_dt_box_MINI,\
                            dxheat_dt_box_MINI,dxion_source_dt_box_MINI,dxlya_dt_box_MINI,dstarlya_dt_box_MINI,\
                            dstarlya_cont_dt_box,dstarlya_inj_dt_box,dstarlya_cont_dt_box_MINI,dstarlya_inj_dt_box_MINI) \
                    private(box_ct,xHII_call) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                    del_fcoll_Rct[box_ct] = 0.;

                    dxheat_dt_box[box_ct] = 0.;
                    dxion_source_dt_box[box_ct] = 0.;
                    dxlya_dt_box[box_ct] = 0.;
                    dstarlya_dt_box[box_ct] = 0.;
                    // JordanFlitter: Initialize Lya flux for Lya heating
                    if (flag_options->USE_Lya_HEATING){
                        dstarlya_cont_dt_box[box_ct] = 0.;
                        dstarlya_inj_dt_box[box_ct] = 0.;
                        if (flag_options->USE_MINI_HALOS){
                          dstarlya_cont_dt_box_MINI[box_ct] = 0.;
                          dstarlya_inj_dt_box_MINI[box_ct] = 0.;
                        }
                    }
                    if (flag_options->USE_MINI_HALOS){
                        dstarlyLW_dt_box[box_ct] = 0.;
                        dstarlyLW_dt_box_MINI[box_ct] = 0.;
                        dxheat_dt_box_MINI[box_ct] = 0.;
                        dxion_source_dt_box_MINI[box_ct] = 0.;
                        dxlya_dt_box_MINI[box_ct] = 0.;
                        dstarlya_dt_box_MINI[box_ct] = 0.;
                    }

                    xHII_call = previous_spin_temp->x_e_box[box_ct];

                    // Check if ionized fraction is within boundaries; if not, adjust to be within
                    if (xHII_call > x_int_XHII[x_int_NXHII-1]*0.999) {
                        xHII_call = x_int_XHII[x_int_NXHII-1]*0.999;
                    } else if (xHII_call < x_int_XHII[0]) {
                        xHII_call = 1.001*x_int_XHII[0];
                    }
                    //interpolate to correct nu integral value based on the cell's ionization state

                    m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);

                    inverse_val_box[box_ct] = (xHII_call - x_int_XHII[m_xHII_low_box[box_ct]])*inverse_diff[m_xHII_low_box[box_ct]];
                }
            }

            for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){

                if(!user_params->USE_INTERPOLATION_TABLES) {
                    Mmax = RtoM(R_values[R_ct]);
                    sigmaMmax = sigma_z0(Mmax);
                }

                if(user_params->USE_INTERPOLATION_TABLES) {
                    if( min_densities[R_ct]*zpp_growth[R_ct] <= -1.) {
                        fcoll_interp_min = log10(global_params.MIN_DENSITY_LOW_LIMIT);
                    }
                    else {
                        fcoll_interp_min = log10(1. + min_densities[R_ct]*zpp_growth[R_ct]);
                    }
                    if( max_densities[R_ct]*zpp_growth[R_ct] > global_params.CRIT_DENS_TRANSITION ) {
                        fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+global_params.CRIT_DENS_TRANSITION)-fcoll_interp_min);
                    }
                    else {
                        fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+max_densities[R_ct]*zpp_growth[R_ct])-fcoll_interp_min);
                    }
                    fcoll_interp_bin_width_inv = 1./fcoll_interp_bin_width;
                }

                ave_fcoll = ave_fcoll_inv = 0.0;
                ave_fcoll_MINI = ave_fcoll_inv_MINI = 0.0;

                // If we are minimising memory usage, then we must smooth the box again
                // It's slower this way, but notably more memory efficient
                if(user_params->MINIMIZE_MEMORY) {

                    // copy over unfiltered box
                    memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

                    if (R_ct > 0){ // don't filter on cell size
                        filter_box(box, 1, global_params.HEAT_FILTER, R_values[R_ct]);
                    }
                    // now fft back to real space
                    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, box);
                    LOG_ULTRA_DEBUG("Executed FFT for R=%f", R_values[R_ct]);

                    // copy over the values
#pragma omp parallel shared(box,inverse_growth_factor_z,delNL0) private(i,j,k,curr_delNL0) num_threads(user_params->N_THREADS)
                    {
#pragma omp for
                        for (i=0;i<user_params->HII_DIM; i++){
                            for (j=0;j<user_params->HII_DIM; j++){
                                for (k=0;k<user_params->HII_DIM; k++){
                                    curr_delNL0 = *((float *)box + HII_R_FFT_INDEX(i,j,k));

                                    if (curr_delNL0 <= -1){ // correct for aliasing in the filtering step
                                        curr_delNL0 = -1+FRACT_FLOAT_ERR;
                                    }

                                    // and linearly extrapolate to z=0
                                    curr_delNL0 *= inverse_growth_factor_z;

                                    // Because we are FFT'ing again, just be careful that any rounding errors
                                    // don't cause the densities to exceed the bounds of the interpolation tables.
                                    if(user_params->USE_INTERPOLATION_TABLES) {
                                        if(curr_delNL0 > max_densities[R_ct]) {
                                            curr_delNL0 = max_densities[R_ct];
                                        }
                                        if(curr_delNL0 < min_densities[R_ct]) {
                                            curr_delNL0 = min_densities[R_ct];
                                        }
                                    }

                                    delNL0[0][HII_R_INDEX(i,j,k)] = curr_delNL0;
                                }
                            }
                        }
                    }
                }

#pragma omp parallel shared(delNL0,zpp_growth,SFRD_z_high_table,fcoll_interp_high_min,fcoll_interp_high_bin_width_inv,log10_SFRD_z_low_table,\
                            fcoll_int_boundexceeded_threaded,log10_Mcrit_LW,SFRD_z_high_table_MINI,\
                            log10_SFRD_z_low_table_MINI,del_fcoll_Rct,del_fcoll_Rct_MINI,Mmax,sigmaMmax,Mcrit_atom_interp_table,Mlim_Fstar,Mlim_Fstar_MINI) \
                    private(box_ct,curr_dens,fcoll,dens_val,fcoll_int,log10_Mcrit_LW_val,log10_Mcrit_LW_int,log10_Mcrit_LW_diff,\
                            fcoll_MINI_left,fcoll_MINI_right,fcoll_MINI) \
                    num_threads(user_params->N_THREADS)
                {
#pragma omp for reduction(+:ave_fcoll,ave_fcoll_MINI)
                    for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                        if(user_params->MINIMIZE_MEMORY) {
                            curr_dens = delNL0[0][box_ct]*zpp_growth[R_ct];
                        }
                        else {
                            curr_dens = delNL0[R_ct][box_ct]*zpp_growth[R_ct];
                        }

                        if (flag_options->USE_MINI_HALOS && user_params->USE_INTERPOLATION_TABLES){
                            log10_Mcrit_LW_val = ( log10_Mcrit_LW[R_ct][box_ct] - LOG10_MTURN_MIN) / LOG10_MTURN_INT;
                            log10_Mcrit_LW_int = (int)floorf( log10_Mcrit_LW_val );
                            log10_Mcrit_LW_diff = log10_Mcrit_LW_val - (float)log10_Mcrit_LW_int;
                        }
                        if (!NO_LIGHT){
                            // Now determine all the differentials for the heating/ionisation rate equations

                            if(user_params->USE_INTERPOLATION_TABLES) {

                                if (curr_dens < global_params.CRIT_DENS_TRANSITION){

                                    if (curr_dens <= -1.) {
                                        fcoll = 0;
                                        fcoll_MINI = 0;
                                    }
                                    else {
                                        dens_val = (log10f(curr_dens+1.) - fcoll_interp_min)*fcoll_interp_bin_width_inv;

                                        fcoll_int = (int)floorf( dens_val );

                                        if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_low - 1)) {
                                            if(fcoll_int==(NSFR_low - 1)) {
                                                if(fabs(curr_dens - global_params.CRIT_DENS_TRANSITION) < 1e-4) {
                                                    // There can be instances where the numerical rounding causes it to go in here,
                                                    // rather than the curr_dens > global_params.CRIT_DENS_TRANSITION case
                                                    // This checks for this, and calculates f_coll in this instance, rather than causing it to error
                                                    dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;

                                                    fcoll_int = (int)floorf( dens_val );

                                                    fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + \
                                                            SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );
                                                    if (flag_options->USE_MINI_HALOS){
                                                        fcoll_MINI_left = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high * log10_Mcrit_LW_int]*\
                                                                    ( 1. + (float)fcoll_int - dens_val ) +\
                                                                    SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high * log10_Mcrit_LW_int]*\
                                                                    ( dens_val - (float)fcoll_int );

                                                        fcoll_MINI_right = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high * (log10_Mcrit_LW_int + 1)]*\
                                                                    ( 1. + (float)fcoll_int - dens_val ) +\
                                                                    SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high * (log10_Mcrit_LW_int + 1)]*\
                                                                    ( dens_val - (float)fcoll_int );

                                                        fcoll_MINI = fcoll_MINI_left * (1. - log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                                    }
                                                }
                                                else {

                                                    fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int];
                                                    fcoll = expf(fcoll);
                                                    if (flag_options->USE_MINI_HALOS){
                                                        fcoll_MINI_left = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low* log10_Mcrit_LW_int];
                                                        fcoll_MINI_right = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low *(log10_Mcrit_LW_int + 1)];
                                                        fcoll_MINI = fcoll_MINI_left * (1.-log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                                        fcoll_MINI = expf(fcoll_MINI);
                                                    }
                                                }
                                            }
                                            else {
                                                fcoll_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                            }
                                        }
                                        else {

                                            fcoll = log10_SFRD_z_low_table[R_ct][fcoll_int]*( 1 + (float)fcoll_int - dens_val ) + \
                                                    log10_SFRD_z_low_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );

                                            fcoll = expf(fcoll);

                                            if (flag_options->USE_MINI_HALOS){
                                                fcoll_MINI_left = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low * log10_Mcrit_LW_int]*\
                                                                ( 1 + (float)fcoll_int - dens_val ) +\
                                                                    log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + 1 + NSFR_low * log10_Mcrit_LW_int]*\
                                                                ( dens_val - (float)fcoll_int );

                                                fcoll_MINI_right = log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + NSFR_low * (log10_Mcrit_LW_int + 1)]*\
                                                                ( 1 + (float)fcoll_int - dens_val ) +\
                                                                log10_SFRD_z_low_table_MINI[R_ct][fcoll_int + 1 + NSFR_low*(log10_Mcrit_LW_int + 1)]*\
                                                                ( dens_val - (float)fcoll_int );

                                                fcoll_MINI = fcoll_MINI_left * (1.-log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                                fcoll_MINI = expf(fcoll_MINI);
                                            }
                                        }
                                    }
                                }
                                else {

                                    if (curr_dens < 0.99*Deltac) {

                                        dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;

                                        fcoll_int = (int)floorf( dens_val );

                                        if(fcoll_int < 0 || (fcoll_int + 1) > (NSFR_high - 1)) {
                                            fcoll_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                        }

                                        fcoll = SFRD_z_high_table[R_ct][fcoll_int]*( 1. + (float)fcoll_int - dens_val ) + \
                                                SFRD_z_high_table[R_ct][fcoll_int+1]*( dens_val - (float)fcoll_int );

                                        if (flag_options->USE_MINI_HALOS){
                                            fcoll_MINI_left = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high * log10_Mcrit_LW_int]*\
                                                        ( 1. + (float)fcoll_int - dens_val ) +\
                                                            SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high * log10_Mcrit_LW_int]*\
                                                        ( dens_val - (float)fcoll_int );

                                            fcoll_MINI_right = SFRD_z_high_table_MINI[R_ct][fcoll_int + NSFR_high*(log10_Mcrit_LW_int + 1)]*\
                                                        ( 1. + (float)fcoll_int - dens_val ) +\
                                                            SFRD_z_high_table_MINI[R_ct][fcoll_int + 1 + NSFR_high*(log10_Mcrit_LW_int + 1)]*\
                                                        ( dens_val - (float)fcoll_int );

                                            fcoll_MINI = fcoll_MINI_left * (1.-log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                        }
                                    }
                                    else {
                                        fcoll = pow(10.,10.);
                                        fcoll_MINI =1e10;
                                    }
                                }
                            }
                            else {

                                if (flag_options->USE_MINI_HALOS){

                                    fcoll = Nion_ConditionalM(zpp_growth[R_ct],log(global_params.M_MIN_INTEGRAL),log(Mmax),sigmaMmax,Deltac,curr_dens,Mcrit_atom_interp_table[R_ct],
                                                              astro_params->ALPHA_STAR,0.,astro_params->F_STAR10,1.,Mlim_Fstar,0., user_params->FAST_FCOLL_TABLES);

                                    fcoll_MINI = Nion_ConditionalM_MINI(zpp_growth[R_ct],log(global_params.M_MIN_INTEGRAL),log(Mmax),sigmaMmax,Deltac,\
                                                           curr_dens,pow(10,log10_Mcrit_LW[R_ct][box_ct]),Mcrit_atom_interp_table[R_ct],\
                                                           astro_params->ALPHA_STAR_MINI,0.,astro_params->F_STAR7_MINI,1.,Mlim_Fstar_MINI, 0., user_params->FAST_FCOLL_TABLES);
                                    fcoll_MINI *= pow(10.,10.);

                                }
                                else {
                                    fcoll = Nion_ConditionalM(zpp_growth[R_ct],log(M_MIN),log(Mmax),sigmaMmax,Deltac,curr_dens,astro_params->M_TURN,
                                                              astro_params->ALPHA_STAR,0.,astro_params->F_STAR10,1.,Mlim_Fstar,0., user_params->FAST_FCOLL_TABLES);
                                }
                                fcoll *= pow(10.,10.);
                            }

                            ave_fcoll += fcoll;

                            del_fcoll_Rct[box_ct] = (1.+curr_dens)*fcoll;

                            if (flag_options->USE_MINI_HALOS){
                                ave_fcoll_MINI += fcoll_MINI;

                                del_fcoll_Rct_MINI[box_ct] = (1.+curr_dens)*fcoll_MINI;
                            }
                        }

                    }
                }

                for(i=0;i<user_params->N_THREADS;i++) {
                    if(fcoll_int_boundexceeded_threaded[omp_get_thread_num()]==1) {
                        LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables for the fcoll/nion_splines");
//                        Throw(ParameterError);
                        Throw(TableEvaluationError);
                    }
                }


                ave_fcoll /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);
                ave_fcoll_MINI /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);

                if(ave_fcoll!=0.) {
                    ave_fcoll_inv = 1./ave_fcoll;
                }

                if(ave_fcoll_MINI!=0.) {
                    ave_fcoll_inv_MINI = 1./ave_fcoll_MINI;
                }

                dfcoll_dz_val = (ave_fcoll_inv/pow(10.,10.))*ST_over_PS[R_ct]*SFR_timescale_factor[R_ct]/astro_params->t_STAR;

                dstarlya_dt_prefactor[R_ct] *= dfcoll_dz_val;
                // JordanFlitter: Lya flux for Lya heating
                if (flag_options->USE_Lya_HEATING){
                    dstarlya_cont_dt_prefactor[R_ct] *= dfcoll_dz_val;
                    dstarlya_inj_dt_prefactor[R_ct] *= dfcoll_dz_val;
                    if (flag_options->USE_MINI_HALOS){
                        dstarlya_cont_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val;
                        dstarlya_inj_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val;
                    }
                }
                if(flag_options->USE_MINI_HALOS){
                    dfcoll_dz_val_MINI = (ave_fcoll_inv_MINI/pow(10.,10.))*ST_over_PS_MINI[R_ct]*SFR_timescale_factor[R_ct]/astro_params->t_STAR;
                    dstarlya_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;
                    dstarlyLW_dt_prefactor[R_ct] *= dfcoll_dz_val;
                    dstarlyLW_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;
                }

// JordanFlitter: I added more shared and private variables
#pragma omp parallel shared(dxheat_dt_box,dxion_source_dt_box,dxlya_dt_box,dstarlya_dt_box,dfcoll_dz_val,del_fcoll_Rct,freq_int_heat_tbl_diff,\
                            m_xHII_low_box,inverse_val_box,freq_int_heat_tbl,freq_int_ion_tbl_diff,freq_int_ion_tbl,freq_int_lya_tbl_diff,\
                            freq_int_lya_tbl,dstarlya_dt_prefactor,R_ct,previous_spin_temp,this_spin_temp,const_zp_prefactor,prefactor_1,\
                            prefactor_2,delNL0,growth_factor_zp,dt_dzp,zp,dgrowth_factor_dzp,dcomp_dzp_prefactor,Trad_fast,dzp,TS_prefactor,\
                            xc_inverse,Trad_fast_inv,dstarlyLW_dt_box,dstarlyLW_dt_prefactor,dxheat_dt_box_MINI,dxion_source_dt_box_MINI,\
                            dxlya_dt_box_MINI,dstarlya_dt_box_MINI,dstarlyLW_dt_box_MINI,dfcoll_dz_val_MINI,del_fcoll_Rct_MINI,\
                            dstarlya_dt_prefactor_MINI,dstarlyLW_dt_prefactor_MINI,prefactor_2_MINI,const_zp_prefactor_MINI,\
                            dstarlya_cont_dt_box,dstarlya_inj_dt_box,dstarlya_cont_dt_prefactor,dstarlya_inj_dt_prefactor,\
                            dstarlya_cont_dt_box_MINI,dstarlya_inj_dt_box_MINI,dstarlya_cont_dt_prefactor_MINI,dstarlya_inj_dt_prefactor_MINI,rec_data,\
                            delta_baryons,delta_baryons_derivative,delta_SDM,delta_SDM_derivative) \
                    private(box_ct,x_e,T,dxion_sink_dt,dxe_dzp,dadia_dzp,dspec_dzp,dcomp_dzp,dxheat_dzp,J_alpha_tot,T_inv,T_inv_sq,\
                            xc_fast,xi_power,xa_tilde_fast_arg,TS_fast,TSold_fast,xa_tilde_fast,dxheat_dzp_MINI,J_alpha_tot_MINI,curr_delNL0,\
                            prev_Ts,tau21,xCMB,eps_CMB,dCMBheat_dzp,E_continuum,E_injected,Ndot_alpha_cont,Ndot_alpha_inj,\
                            eps_Lya_cont,eps_Lya_inj,Ndot_alpha_cont_MINI,Ndot_alpha_inj_MINI,eps_Lya_cont_MINI,eps_Lya_inj_MINI,\
                            T_chi,V_chi_b,dSDM_b_heat_dzp,dSDM_chi_heat_dzp,D_V_chi_b_dzp,SDM_rates,dT_b_2_dt_ext,dT_chi_2_dt_ext,dadia_dzp_SDM,\
                            delta_baryons_local,delta_baryons_derivative_local,delta_SDM_local,delta_SDM_derivative_local) \
                    num_threads(user_params->N_THREADS)
                {
#pragma omp for reduction(+:J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave,Ts_ave,Tk_ave,x_e_ave,J_alpha_ave_MINI,Xheat_ave_MINI,J_LW_ave,J_LW_ave_MINI)
                    for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
                        // I've added the addition of zero just in case. It should be zero anyway, but just in case there is some weird
                        // numerical thing
                        if(ave_fcoll!=0.) {
                            dxheat_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( \
                                                    (freq_int_heat_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                    freq_int_heat_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                            dxion_source_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( \
                                                    (freq_int_ion_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                    freq_int_ion_tbl[m_xHII_low_box[box_ct]][R_ct] ));

                            dxlya_dt_box[box_ct] += (dfcoll_dz_val*(double)del_fcoll_Rct[box_ct]*( \
                                                    (freq_int_lya_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                    freq_int_lya_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                            dstarlya_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_dt_prefactor[R_ct];
                            // JordanFlitter: Lya flux for Lya heating
                            if (flag_options->USE_Lya_HEATING){
                                dstarlya_cont_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_cont_dt_prefactor[R_ct];
                                dstarlya_inj_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_inj_dt_prefactor[R_ct];
                                if (flag_options->USE_MINI_HALOS){
                                    dstarlya_cont_dt_box_MINI[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_cont_dt_prefactor_MINI[R_ct];
                                    dstarlya_inj_dt_box_MINI[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlya_inj_dt_prefactor_MINI[R_ct];
                                }
                            }
                            if (flag_options->USE_MINI_HALOS){
                                dstarlyLW_dt_box[box_ct] += (double)del_fcoll_Rct[box_ct]*dstarlyLW_dt_prefactor[R_ct];
                            }
                        }
                        else {
                            dxheat_dt_box[box_ct] += 0.;
                            dxion_source_dt_box[box_ct] += 0.;

                            dxlya_dt_box[box_ct] += 0.;
                            dstarlya_dt_box[box_ct] += 0.;
                            // JordanFlitter: Lya flux for Lya heating
                            if (flag_options->USE_Lya_HEATING){
                                dstarlya_cont_dt_box[box_ct] += 0.;
                                dstarlya_inj_dt_box[box_ct] += 0.;
                                if (flag_options->USE_MINI_HALOS){
                                    dstarlya_cont_dt_box_MINI[box_ct] += 0.;
                                    dstarlya_inj_dt_box_MINI[box_ct] += 0.;
                                }
                            }

                            if (flag_options->USE_MINI_HALOS){
                                dstarlyLW_dt_box[box_ct] += 0.;
                            }
                        }

                        if (flag_options->USE_MINI_HALOS){
                            if(ave_fcoll_MINI!=0.) {
                                dxheat_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)del_fcoll_Rct_MINI[box_ct]*( \
                                                            (freq_int_heat_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                        freq_int_heat_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                                dxion_source_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)del_fcoll_Rct_MINI[box_ct]*( \
                                                            (freq_int_ion_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                        freq_int_ion_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                                dxlya_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)del_fcoll_Rct_MINI[box_ct]*( \
                                                            (freq_int_lya_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + \
                                                                        freq_int_lya_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                                dstarlya_dt_box_MINI[box_ct] += (double)del_fcoll_Rct_MINI[box_ct]*dstarlya_dt_prefactor_MINI[R_ct];
                                dstarlyLW_dt_box_MINI[box_ct] += (double)del_fcoll_Rct_MINI[box_ct]*dstarlyLW_dt_prefactor_MINI[R_ct];
                            }
                        }

                        // If R_ct == 0, as this is the final smoothing scale (i.e. it is reversed)
                        if(R_ct==0) {

                            // Note here, that by construction it doesn't matter if using MINIMIZE_MEMORY as only need the R_ct = 0 box
                            curr_delNL0 = delNL0[0][box_ct];

                            // JordanFlitter: set local baryons density and its derivative
                            //                Note we use box_ct_FFT to access the approporiate cell in the box
                            if (user_params->EVOLVE_BARYONS){
                                delta_baryons_local = *((float *)delta_baryons + box_ct_FFT(box_ct));
                                delta_baryons_derivative_local = *((float *)delta_baryons_derivative + box_ct_FFT(box_ct));
                                if (user_params->SCATTERING_DM) {
                                    delta_SDM_local = *((float *)delta_SDM + box_ct_FFT(box_ct));
                                    delta_SDM_derivative_local = *((float *)delta_SDM_derivative + box_ct_FFT(box_ct));
                                }
                            }

                            x_e = previous_spin_temp->x_e_box[box_ct];
                            T = previous_spin_temp->Tk_box[box_ct];
                            // JordanFlitter: Extract T_chi and V_chi_b from previous boxes
                            if (user_params->SCATTERING_DM) {
                                T_chi = previous_spin_temp->T_chi_box[box_ct]; // K
                                V_chi_b = 1.e5*previous_spin_temp->V_chi_b_box[box_ct]; // cm/sec
                                // Note that in the calculations below, V_chi_b is in cm/sec. However, we always save the V_chi_b_box in km/sec
                            }

                            // add prefactors
                            dxheat_dt_box[box_ct] *= const_zp_prefactor;
                            dxion_source_dt_box[box_ct] *= const_zp_prefactor;

                            // JordanFlitter: we can use the baryons density field
                            if (user_params->EVOLVE_BARYONS) {
                                dxlya_dt_box[box_ct] *= const_zp_prefactor*prefactor_1 * (1.+delta_baryons_local);
                            }
                            else{
                                dxlya_dt_box[box_ct] *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                            }
                            dstarlya_dt_box[box_ct] *= prefactor_2;
                            // JordanFlitter: Lya flux for Lya heating
                            if (flag_options->USE_Lya_HEATING){
                                dstarlya_cont_dt_box[box_ct] *= prefactor_2;
                                dstarlya_inj_dt_box[box_ct] *= prefactor_2;
                                if (flag_options->USE_MINI_HALOS){
                                    dstarlya_cont_dt_box_MINI[box_ct] *= prefactor_2_MINI;
                                    dstarlya_inj_dt_box_MINI[box_ct] *= prefactor_2_MINI;
                                }
                            }
                            if (flag_options->USE_MINI_HALOS){
                                dstarlyLW_dt_box[box_ct] *= prefactor_2 * (hplank * 1e21);

                                dxheat_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI;
                                dxion_source_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI;

                                // JordanFlitter: we can use the baryons density field
                                if (user_params->EVOLVE_BARYONS) {
                                    dxlya_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI*prefactor_1 * (1.+delta_baryons_local);
                                }
                                else{
                                    dxlya_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                                }
                                dstarlya_dt_box_MINI[box_ct] *= prefactor_2_MINI;

                                dstarlyLW_dt_box_MINI[box_ct] *= prefactor_2_MINI * (hplank * 1e21);
                            }

                            // Now we can solve the evolution equations  //

                            // JordanFlitter: added definition of x_CMB
                            prev_Ts = previous_spin_temp->Ts_box[box_ct];
                            // JordanFlitter: we can use the baryons density field
                            if (user_params->EVOLVE_BARYONS){
                                tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)*(1.+delta_baryons_local)) /prev_Ts/hubble(zp);
                            }
                            else {
                                tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp)) /prev_Ts/hubble(zp);
                            }
                            xCMB = (1. - exp(-tau21))/tau21;

                            // First let's do dxe_dzp //
                            if (user_params->USE_HYREC){
                                // JordanFlitter: Let's use HyRec!!
                                // Note that we treat xe and x_H to be the same, since after recombination x_He<<xe,x_H.
                                // Note also that Hyrec returns the derivative of n_e/n_H. Since we are interested in the derivative of n_e/(n_H+n_He),
                                // we must multiply by n_H/(n_H+n_He)=f_H. Also, Hyrec expects an input of n_e/n_H, so we need to give it x_e/f_H=n_e/n_H
                                // JordanFlitter: we can use the baryons density field
                                if (user_params->EVOLVE_BARYONS){
                                    dxion_sink_dt = hyrec_dx_H_dz(rec_data, x_e/f_H, x_e/f_H, No*pow(1.+zp,3)*(1.+delta_baryons_local), zp, hubble(zp), T, Trad_fast);
                                }
                                else {
                                    dxion_sink_dt = hyrec_dx_H_dz(rec_data, x_e/f_H, x_e/f_H, No*pow(1.+zp,3)*(1.+curr_delNL0*growth_factor_zp), zp, hubble(zp), T, Trad_fast);
                                }
                                dxion_sink_dt *= f_H;
                                dxion_sink_dt /= -dt_dzp; // Here we are only interested in the recombination rate, and not dx_e/dz (which is HyRec's output)
                            }
                            // JordanFlitter: we may want to use alpha_B recombination rate (and Peebles coefficient) at low redshifts because alpha_A diverges at low temperatures.
                            // For LCDM, both recombination rates agree to an order of percent, and the brightness temperature is not sensitive to the exact modelling of the recombination rate at these redshifts.
                            // For SDM, the brightness temperature is sensitive to the free electron fraction, even at low redshifts, becauese the temperature can reach very low values in which case Compton heating is not negligible!
                            else if (user_params->USE_ALPHA_B) {
                                // JordanFlitter: we can use the baryons density field
                                if (user_params->EVOLVE_BARYONS) {
                                    dxion_sink_dt = alpha_B(T) * x_e*x_e * f_H * prefactor_1 * \
                                                    (1.+delta_baryons_local) * Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+delta_baryons_local), alpha_B(T));
                                }
                                else{
                                    dxion_sink_dt = alpha_B(T) * x_e*x_e * f_H * prefactor_1 * \
                                                    (1.+curr_delNL0*growth_factor_zp) * Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp), alpha_B(T));
                                }
                            }
                            else{
                                // JordanFlitter: we can use the baryons density field
                                if (user_params->EVOLVE_BARYONS) {
                                    dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * \
                                                    (1.+delta_baryons_local);
                                }
                                else{
                                    dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * \
                                                    (1.+curr_delNL0*growth_factor_zp);
                                }
                            }
                            if (flag_options->USE_MINI_HALOS){
                                dxe_dzp = dt_dzp*(dxion_source_dt_box[box_ct] + dxion_source_dt_box_MINI[box_ct] - dxion_sink_dt);
                            }
                            else{
                                dxe_dzp = dt_dzp*(dxion_source_dt_box[box_ct] - dxion_sink_dt);
                            }

                            // Next, let's get the temperature components //
                            // JordanFlitter: we can use the baryons density field
                            if (user_params->EVOLVE_BARYONS){
                                dadia_dzp = 2.*T/(1.0+zp) + (2.0/3.0)*T*delta_baryons_derivative_local/(1.0+delta_baryons_local);
                            }
                            else {
                                // first, adiabatic term
                                dadia_dzp = 3/(1.0+zp);
                                if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                                    dadia_dzp += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);

                                dadia_dzp *= (2.0/3.0)*T;
                            }
                            // next heating due to the changing species
                            dspec_dzp = - dxe_dzp * T / (1+x_e);

                            // next, Compton heating
                            //                dcomp_dzp = dT_comp(zp, T, x_e);
                            // JordanFlitter: there shouldn't be any f_He at the Compton heating term, as x_e=n_e/(n_H+n_He). This was verified with Mesinger
                            dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e))*( Trad_fast - T );

                            // lastly, X-ray heating
                            dxheat_dzp = dxheat_dt_box[box_ct] * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                            if (flag_options->USE_MINI_HALOS){
                                dxheat_dzp_MINI = dxheat_dt_box_MINI[box_ct] * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                            }

                            dCMBheat_dzp = 0;
                            eps_Lya_cont = 0;
                            eps_Lya_inj = 0;
                            eps_Lya_cont_MINI = 0;
                            eps_Lya_inj_MINI = 0;

                            // JordanFlitter: defined CMB heating, following Avery Meiksin (arXiv: 2105.14516)
                            if (flag_options->USE_CMB_HEATING) {
                                eps_CMB = (3./4.) * (T_cmb*(1.+zp)/T21) * A10_HYPERFINE * f_H * (hplank*hplank/Lambda_21/Lambda_21/m_p) * (1.+2.*T/T21);
								                dCMBheat_dzp = 	-eps_CMB * (2./3./k_B/(1.+x_e))/hubble(zp)/(1.+zp);
                            }
                            // JordanFlitter: defined Lya heating
                            if (flag_options->USE_Lya_HEATING) {
                                // JordanFlitter: we can use the baryons density field
                                if (user_params->EVOLVE_BARYONS){
                                    E_continuum = Energy_Lya_heating(T, prev_Ts, taugp(zp,delta_baryons_local,x_e), 2);
                                    E_injected = Energy_Lya_heating(T, prev_Ts, taugp(zp,delta_baryons_local,x_e), 3);
                                    Ndot_alpha_cont = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+delta_baryons_local))/(1.+zp)/C * dstarlya_cont_dt_box[box_ct];
                                    Ndot_alpha_inj = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+delta_baryons_local))/(1.+zp)/C * dstarlya_inj_dt_box[box_ct];
                                }
                                else{
                                    E_continuum = Energy_Lya_heating(T, prev_Ts, taugp(zp,curr_delNL0*growth_factor_zp,x_e), 2);
                                    E_injected = Energy_Lya_heating(T, prev_Ts, taugp(zp,curr_delNL0*growth_factor_zp,x_e), 3);
                                    Ndot_alpha_cont = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_cont_dt_box[box_ct];
                                    Ndot_alpha_inj = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_inj_dt_box[box_ct];
                                }
                                if (isnan(E_continuum) || isinf(E_continuum)){
                                    E_continuum = 0.;
                                }
                                if (isnan(E_injected) || isinf(E_injected)){
                                    E_injected = 0.;
                                }
                                eps_Lya_cont = - Ndot_alpha_cont * E_continuum * (2.0 / 3.0 /k_B/ (1.0+x_e));
                                eps_Lya_inj = - Ndot_alpha_inj * E_injected * (2.0 / 3.0 /k_B/ (1.0+x_e));
                                if (flag_options->USE_MINI_HALOS) {
                                    // JordanFlitter: we can use the baryons density field
                                    if (user_params->EVOLVE_BARYONS){
                                        Ndot_alpha_cont_MINI = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+delta_baryons_local))/(1.+zp)/C * dstarlya_cont_dt_box_MINI[box_ct];
                                        Ndot_alpha_inj_MINI = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+delta_baryons_local))/(1.+zp)/C * dstarlya_inj_dt_box_MINI[box_ct];
                                    }
                                    else{
                                        Ndot_alpha_cont_MINI = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_cont_dt_box_MINI[box_ct];
                                        Ndot_alpha_inj_MINI = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_inj_dt_box_MINI[box_ct];
                                    }
                                    eps_Lya_cont_MINI = - Ndot_alpha_cont_MINI * E_continuum * (2.0 / 3.0 /k_B/ (1.0+x_e));
                                    eps_Lya_inj_MINI = - Ndot_alpha_inj_MINI * E_injected * (2.0 / 3.0 /k_B/ (1.0+x_e));
                                }
                            }
                            // JordanFlitter: Calculate heating rates and drag term in SDM universe, according to arXiv: 1509.00029
                            if (user_params->SCATTERING_DM) {
                                if (flag_options->USE_MINI_HALOS){
                                    dT_b_2_dt_ext = (dxheat_dzp + dxheat_dzp_MINI + dspec_dzp + (dadia_dzp-2.*T/(1.0+zp)) + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj + eps_Lya_cont_MINI + eps_Lya_inj_MINI)/dtdz(zp);
                                }
                                else {
                                    dT_b_2_dt_ext = (dxheat_dzp + dspec_dzp + (dadia_dzp-2.*T/(1.0+zp)) + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj)/dtdz(zp);
                                }
                                // JordanFlitter: we can use the SDM density field
                                if (user_params->EVOLVE_BARYONS){
                                    dadia_dzp_SDM = 2.*T_chi/(1.0+zp) + (2.0/3.0)*T_chi*delta_SDM_derivative_local/(1.0+delta_SDM_local);
                                }
                                else {
                                    dadia_dzp_SDM = 3/(1.0+zp);
                                    if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                                        dadia_dzp_SDM += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);

                                    dadia_dzp_SDM *= (2.0/3.0)*T_chi;
                                }
                                dT_chi_2_dt_ext = (dadia_dzp_SDM-2.*T_chi/(1.0+zp))/dtdz(zp);
                                // JordanFlitter: we can use the baryons and SDM density fields
                                if (user_params->EVOLVE_BARYONS){
                                    SDM_rates = SDM_derivatives(zp, x_e, T, T_chi, V_chi_b, delta_baryons_local, delta_SDM_local, hubble(zp),
                                                    -hubble(zp)*(1.+zp)*dcomp_dzp_prefactor*x_e/(1.+x_e), Trad_fast, dzp, dT_b_2_dt_ext,
                                                    dT_chi_2_dt_ext);
                                }
                                else {
                                    SDM_rates = SDM_derivatives(zp, x_e, T, T_chi, V_chi_b, curr_delNL0*growth_factor_zp, curr_delNL0*growth_factor_zp, hubble(zp),
                                                    -hubble(zp)*(1.+zp)*dcomp_dzp_prefactor*x_e/(1.+x_e), Trad_fast, dzp, dT_b_2_dt_ext,
                                                    dT_chi_2_dt_ext);
                                }
                                dSDM_b_heat_dzp = (2.0/3.0/k_B)*SDM_rates.Q_dot_b*dtdz(zp); // K
                                dSDM_chi_heat_dzp = (2.0/3.0/k_B)*SDM_rates.Q_dot_chi*dtdz(zp); // K
                                D_V_chi_b_dzp = SDM_rates.D_V_chi_b*dtdz(zp); // cm/sec
                            }
                            else {
                                dSDM_b_heat_dzp = 0.;
                                dSDM_chi_heat_dzp = 0.;
                                D_V_chi_b_dzp = 0.;
                            }
                            //update quantities
                            x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                            if (x_e > 1) // can do this late in evolution if dzp is too large
                                x_e = 1 - FRACT_FLOAT_ERR;
                            else if (x_e < 0)
                                x_e = 0;
                            // JordanFlitter: added CMB and Lya heating to the temperature evolution
                            // JordanFlitter: I also added the SDM heating exchange
                            // JordanFlitter: If epsilon_b is not small enough, or if external heating rates dominate the SDM cooling rate,
                            //                or SDM doesn't exist, we evolve T_k with the usual ODE
                            if (!user_params->SCATTERING_DM ||
                                (user_params->SCATTERING_DM && ((fabs(SDM_rates.epsilon_b) > EPSILON_THRES) || (fabs(dT_b_2_dt_ext*dtdz(zp))>fabs(dSDM_b_heat_dzp))))
                                ) {
                                if (T < MAX_TK) {
                                    if (flag_options->USE_MINI_HALOS){
                                        T += ( dxheat_dzp + dxheat_dzp_MINI + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj + eps_Lya_cont_MINI + eps_Lya_inj_MINI + dSDM_b_heat_dzp) * dzp;
                                    } else {
                                        T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj + dSDM_b_heat_dzp) * dzp;
                                    }
                                }
                            }
                            // JordanFlitter: Otherwise, we evolve T_k with DM TCA!
                            else {
                                T = SDM_rates.T_bar_chi_b + SDM_rates.Delta_T_b_chi/2.; // K
                            }

                            // JordanFlitter: evolution equations for T_chi and V_chi_b in SDM universe
                            if (user_params->SCATTERING_DM) {
                              // JordanFlitter: If epsilon_chi is not small enough, or if external heating rates dominate the baryon heating rate,
                              // we evolve T_chi with the usual ODE
                              if ((fabs(SDM_rates.epsilon_chi) > EPSILON_THRES) || (fabs(dT_chi_2_dt_ext*dtdz(zp))>fabs(dSDM_chi_heat_dzp))) {
                                  T_chi += (dadia_dzp_SDM + dSDM_chi_heat_dzp)*dzp; // K
                              }
                              // JordanFlitter: Otherwise, we evolve T_chi with DM TCA!
                              else {
                                  T_chi = SDM_rates.T_bar_chi_b - SDM_rates.Delta_T_b_chi/2.; // K
                              }
                              V_chi_b += (V_chi_b/(1.0+zp) - D_V_chi_b_dzp)*dzp; // cm/sec
                            }
                            // JordanFlitter: I had to change the following logic for handling negative temperatures. The reason behind this change is the following.
                            // If SDM outnumber the baryons (e.g. if f_chi~100% and the SDM mass is light), its temperature is unaffected by the interactions with the
                            // baryons and it stays cold. The rapid interactions with baryons on the other hand, cause the baryons to be tightly coupled to the SDM,
                            // i.e. they also want to stay cold. Yet, external X-ray heating causes the baryons to heat up. In this unique scenario, although the baryons
                            // are tightly coupled to the cold SDM, their temperature departs from the SDM temperature, and therefore the cooling by the SDM interactions
                            // is very strong and may lead to negative temperatures. To cure this, it is important to realize that if the SDM cooling wins the X-ray heating,
                            // then the baryons temperature approaches the SDM temperature from above (it cannot be smaller than T_chi!)
                            // This problem doesn't happen when the SDM number-density is comparable to the baryons number-density; in that case, both fluids are strongly
                            // coupled to each other, effectively behaving as a single fluid, and their common temperature is increased by the X-rays.
                            if (!user_params->SCATTERING_DM) {
                                if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                      T = T_cmb*(1+zp);
                                }
                            }
                            else {
                                if (T<T_chi){ // T should never by smaller than T_chi!
                                      if (T_chi>0){
                                          T = T_chi;
                                      }
                                      else {
                                          T = previous_spin_temp->Tk_box[box_ct]; // Don't update T in that special scenario
                                      }
                                }

                            }

                            // JordanFlitter: similar logic for T_chi and V_chi_b in case of spurious bahaviour
                            if (user_params->SCATTERING_DM) {
                                if (T_chi<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                    T_chi = previous_spin_temp->T_chi_box[box_ct]; // Don't update T_chi in that special scenario
                                }
                                if (V_chi_b<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                    V_chi_b = 0.;
                                }
                            }

                            this_spin_temp->x_e_box[box_ct] = x_e;
                            this_spin_temp->Tk_box[box_ct] = T;
                            // JordanFlitter: update T_chi_box and V_chi_b_box
                            if (user_params->SCATTERING_DM) {
                              this_spin_temp->T_chi_box[box_ct] = T_chi; // K
                              this_spin_temp->V_chi_b_box[box_ct] = 1.e-5*V_chi_b; // km/sec
                            }

                            J_alpha_tot = ( dxlya_dt_box[box_ct] + dstarlya_dt_box[box_ct] ); //not really d/dz, but the lya flux
                            if (flag_options->USE_MINI_HALOS){
                                J_alpha_tot_MINI = ( dxlya_dt_box_MINI[box_ct] + dstarlya_dt_box_MINI[box_ct] ); //not really d/dz, but the lya flux
                                this_spin_temp->J_21_LW_box[box_ct] = dstarlyLW_dt_box[box_ct] + dstarlyLW_dt_box_MINI[box_ct];
                            }

                            // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                            // Algorithm is the same, but written to be more computationally efficient
                            T_inv = expf((-1.)*logf(T));
                            T_inv_sq = expf((-2.)*logf(T));

                            // JordanFlitter: we can use the baryons density field
                            if (user_params->EVOLVE_BARYONS){
                                xc_fast = (1.0+delta_baryons_local)*xc_inverse*\
                                        ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                                xi_power = TS_prefactor * cbrt((1.0+delta_baryons_local)*(1.0-x_e)*T_inv_sq);
                            }
                            else {
                                xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*\
                                        ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                                xi_power = TS_prefactor * cbrt((1.0+curr_delNL0*growth_factor_zp)*(1.0-x_e)*T_inv_sq);
                            }


                            if (flag_options->USE_MINI_HALOS){
                                xa_tilde_fast_arg = xa_tilde_prefactor*(J_alpha_tot+J_alpha_tot_MINI)*\
                                                pow( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power, -1. );
                            }
                            else{
                                xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot*\
                                                pow( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power, -1. );
                            }

                            //if (J_alpha_tot > 1.0e-20) { // Must use WF effect
                            // New in v1.4
                            if (fabs(J_alpha_tot) > 1.0e-20) { // Must use WF effect
                                TS_fast = Trad_fast;
                                TSold_fast = 0.0;
                                while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {

                                    TSold_fast = TS_fast;

                                    // JordanFlitter: S_alpha correction, according to Eq. A4 in Mittal & Kulkarni (arXiv: 2009.10746),
                                    // a result that was derived from the work of Chuzhouy & Shapiro (arXiv: astro-ph/0512206). This method was used
                                    // in Driskell et al. (arXiv: 2209.04499) to account for low-temperature corrections which aren't captured by
                                    // Hirata's fit (arXiv: astro-ph/0507102), which is used in the public 21cmFAST
                                    if (user_params->USE_CS_S_ALPHA) {
                                        if (flag_options->USE_MINI_HALOS){
                                            xa_tilde_fast = xa_tilde_prefactor*(J_alpha_tot+J_alpha_tot_MINI);
                                        }
                                        else{
                                            xa_tilde_fast = xa_tilde_prefactor*J_alpha_tot;
                                        }
                                        /* JordanFlitter: The following commented lines are an attempt to implement Eq. (18) in arXiv:2212.08082
                                          (which was orignally introduced in Eq. 57 in arXiv:1605.04357). Unfortunately, this implementation leads
                                          to 15% maximum error in the brightness temperature (for some set of astrophysical parameters and models),
                                          which is absolutely not accepted. I don't know if that discrepancy is the result of a bug in the lines below...

                                        xa_tilde_fast *= 1./(1.-0.402/T);
                                        xa_tilde_fast *= exp(-2.06*pow(cosmo_params->OMb * cosmo_params->hlittle/0.0327,1./3.)
                                                                  *pow(cosmo_params->OMm/0.307,-1./6.)
                                                                  *pow((1.+zp)/10.,1./2.)*pow(T/0.402,-2./3.));
                                        TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+(xa_tilde_fast+xc_fast)*T_inv,-1.);*/

                                        // JordanFlitter: Instead, we use tabulated values from the work of Chuzhouy & Shapiro (arXiv: astro-ph/0512206).
                                        // The numerical prefactor in the argument of S_alpha is 24*pi^2*nu_alpha*m_H*k_B^2/(A_alhpa*gamma_alpha*c*h^3) = 6.84e13 sec/(K^2 * cm^3)
                                        // Also, in this method we have a closed analytcial formula for T_s (which is why we set TSold_fast = TS_fast)
                                        // JordanFlitter: we can use the baryons density field
                                        if (user_params->EVOLVE_BARYONS){
                                            xa_tilde_fast *= S_alpha_correction(6.84e13 * hubble(zp) * T*T / (No*pow(1.+zp,3.)*(1.+delta_baryons_local)*(1.-x_e)), 0);
                                        }
                                        else{
                                            xa_tilde_fast *= S_alpha_correction(6.84e13 * hubble(zp) * T*T / (No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp)*(1.-x_e)), 0);
                                        }
                                        //TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+xa_tilde_fast*pow(TS_fast,-1.)*(TS_fast+0.402)/(T+0.402)+xc_fast*T_inv,-1.);
                                        TS_fast = (xCMB+xc_fast+xa_tilde_fast*T/(T+0.402))*pow(xCMB*Trad_fast_inv+xa_tilde_fast*pow(T+0.402,-1.)+xc_fast*T_inv,-1.);
                                        TSold_fast = TS_fast;
                                    }
                                    else {
                                        xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - \
                                                       0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;
                                        // JordanFlitter: modified spin temperature by xCMB
                                        TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+xa_tilde_fast*( T_inv + \
                                                        0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                                    }
                                }
                            } else { // Collisions only
                                TS_fast = (xCMB + xc_fast)/(xCMB*Trad_fast_inv + xc_fast*T_inv);

                                xa_tilde_fast = 0.0;
                            }

                            if(TS_fast < 0.) {
                                // It can very rarely result in a negative spin temperature. If negative, it is a very small number.
                                //Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                                TS_fast = fabs(TS_fast);
                            }

                            this_spin_temp->Ts_box[box_ct] = TS_fast;

                            // JordanFlitter: added Lya flux to output! (because why not)
                            if (flag_options->USE_MINI_HALOS){
                                this_spin_temp->J_Lya_box[box_ct] =  J_alpha_tot+J_alpha_tot_MINI;
                            }
                            else{
                                this_spin_temp->J_Lya_box[box_ct] =  J_alpha_tot;
                            }

                            if(LOG_LEVEL >= DEBUG_LEVEL){
                                J_alpha_ave += J_alpha_tot;
                                xalpha_ave += xa_tilde_fast;
                                Xheat_ave += ( dxheat_dzp );
                                Xion_ave += ( dt_dzp*dxion_source_dt_box[box_ct] );
                                Ts_ave += TS_fast;
                                Tk_ave += T;
                                if (flag_options->USE_MINI_HALOS){
                                    J_alpha_ave_MINI += J_alpha_tot_MINI;
                                    Xheat_ave_MINI += ( dxheat_dzp_MINI );
                                    J_LW_ave += dstarlyLW_dt_box[box_ct];
                                    J_LW_ave_MINI += dstarlyLW_dt_box_MINI[box_ct];
                                }
                            }

                            x_e_ave += x_e;
                        }
                    }
                }
            }
        }
        else {
        // JordanFlitter: I added more shared and private variables
#pragma omp parallel shared(previous_spin_temp,x_int_XHII,inverse_diff,delNL0_rev,dens_grid_int_vals,ST_over_PS,zpp_growth,dfcoll_interp1,\
                            density_gridpoints,dfcoll_interp2,freq_int_heat_tbl_diff,freq_int_heat_tbl,freq_int_ion_tbl_diff,freq_int_ion_tbl,\
                            freq_int_lya_tbl_diff,freq_int_lya_tbl,dstarlya_dt_prefactor,const_zp_prefactor,prefactor_1,growth_factor_zp,dzp,\
                            dt_dzp,dgrowth_factor_dzp,dcomp_dzp_prefactor,this_spin_temp,xc_inverse,TS_prefactor,xa_tilde_prefactor,Trad_fast_inv,\
                            dstarlya_cont_dt_prefactor,dstarlya_inj_dt_prefactor,\
                            rec_data,delta_baryons,delta_baryons_derivative,delta_SDM,delta_SDM_derivative) \
                    private(box_ct,x_e,T,xHII_call,m_xHII_low,inverse_val,dxheat_dt,dxion_source_dt,dxlya_dt,dstarlya_dt,curr_delNL0,R_ct,\
                            dfcoll_dz_val,dxion_sink_dt,dxe_dzp,dadia_dzp,dspec_dzp,dcomp_dzp,J_alpha_tot,T_inv,T_inv_sq,xc_fast,xi_power,\
                            xa_tilde_fast_arg,TS_fast,TSold_fast,xa_tilde_fast,prev_Ts,tau21,xCMB,eps_CMB,dCMBheat_dzp,dstarlya_cont_dt,dstarlya_inj_dt,\
                            E_continuum,E_injected,Ndot_alpha_cont,Ndot_alpha_inj,eps_Lya_cont,eps_Lya_inj,\
                            T_chi,V_chi_b,dSDM_b_heat_dzp,dSDM_chi_heat_dzp,D_V_chi_b_dzp,SDM_rates,dT_b_2_dt_ext,dT_chi_2_dt_ext,dadia_dzp_SDM,\
                            delta_baryons_local,delta_baryons_derivative_local,delta_SDM_local,delta_SDM_derivative_local) \
                    num_threads(user_params->N_THREADS)
            {
#pragma omp for reduction(+:J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave,Ts_ave,Tk_ave,x_e_ave)
                for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){

                    x_e = previous_spin_temp->x_e_box[box_ct];
                    T = previous_spin_temp->Tk_box[box_ct];
                    // JordanFlitter: Extract T_chi and V_chi_b from previous boxes
                    if (user_params->SCATTERING_DM) {
                        T_chi = previous_spin_temp->T_chi_box[box_ct]; // K
                        V_chi_b = 1.e5*previous_spin_temp->V_chi_b_box[box_ct]; // cm/sec
                        // Note that in the calculations below, V_chi_b is in cm/sec. However, we always save the V_chi_b_box in km/sec
                    }

                    xHII_call = x_e;

                    // Check if ionized fraction is within boundaries; if not, adjust to be within
                    if (xHII_call > x_int_XHII[x_int_NXHII-1]*0.999) {
                        xHII_call = x_int_XHII[x_int_NXHII-1]*0.999;
                    } else if (xHII_call < x_int_XHII[0]) {
                        xHII_call = 1.001*x_int_XHII[0];
                    }
                    //interpolate to correct nu integral value based on the cell's ionization state

                    m_xHII_low = locate_xHII_index(xHII_call);

                    inverse_val = (xHII_call - x_int_XHII[m_xHII_low])*inverse_diff[m_xHII_low];

                    // First, let's do the trapazoidal integration over zpp
                    dxheat_dt = 0;
                    dxion_source_dt = 0;
                    dxlya_dt = 0;
                    dstarlya_dt = 0;

                    // JordanFlitter: Initialize Lya flux for Lya heating
                    if (flag_options->USE_Lya_HEATING){
                        dstarlya_cont_dt = 0;
                        dstarlya_inj_dt = 0;
                    }
                    curr_delNL0 = delNL0_rev[box_ct][0];

                    // JordanFlitter: set local baryons density and its derivative
                    //                Note we use box_ct_FFT to access the approporiate cell in the box
                    if (user_params->EVOLVE_BARYONS){
                        delta_baryons_local = *((float *)delta_baryons + box_ct_FFT(box_ct));
                        delta_baryons_derivative_local = *((float *)delta_baryons_derivative + box_ct_FFT(box_ct));
                        if (user_params->SCATTERING_DM) {
                            delta_SDM_local = *((float *)delta_SDM + box_ct_FFT(box_ct));
                            delta_SDM_derivative_local = *((float *)delta_SDM_derivative + box_ct_FFT(box_ct));
                        }
                    }

                    if (!NO_LIGHT){
                        // Now determine all the differentials for the heating/ionisation rate equations
                        for (R_ct=global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct--;){

                            if(user_params->USE_INTERPOLATION_TABLES) {
                                if( dens_grid_int_vals[box_ct][R_ct] < 0 || (dens_grid_int_vals[box_ct][R_ct] + 1) > (dens_Ninterp  - 1) ) {
                                    table_int_boundexceeded_threaded[omp_get_thread_num()] = 1;
                                }

                                dfcoll_dz_val = ST_over_PS[R_ct]*(1.+delNL0_rev[box_ct][R_ct]*zpp_growth[R_ct])*( \
                                                    dfcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*\
                                                        (density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct]) + \
                                                    dfcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*\
                                                        (delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct]) );
                            }
                            else {
                                dfcoll_dz_val = ST_over_PS[R_ct]*(1.+delNL0_rev[box_ct][R_ct]*zpp_growth[R_ct])*( \
                                                dfcoll_dz(zpp_for_evolve_list[R_ct], sigma_Tmin[R_ct], delNL0_rev[box_ct][R_ct], sigma_atR[R_ct]) );
                            }

                            dxheat_dt += dfcoll_dz_val * \
                                        ( (freq_int_heat_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_heat_tbl[m_xHII_low][R_ct] );
                            dxion_source_dt += dfcoll_dz_val * \
                                        ( (freq_int_ion_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_ion_tbl[m_xHII_low][R_ct] );

                            dxlya_dt += dfcoll_dz_val * \
                                        ( (freq_int_lya_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_lya_tbl[m_xHII_low][R_ct] );
                            dstarlya_dt += dfcoll_dz_val*dstarlya_dt_prefactor[R_ct];
                            // JordanFlitter: Lya flux for Lya heating
                            if (flag_options->USE_Lya_HEATING){
                              dstarlya_cont_dt += dfcoll_dz_val*dstarlya_cont_dt_prefactor[R_ct];
                              dstarlya_inj_dt += dfcoll_dz_val*dstarlya_inj_dt_prefactor[R_ct];
                            }
                        }
                    }

                    // add prefactors
                    dxheat_dt *= const_zp_prefactor;
                    dxion_source_dt *= const_zp_prefactor;

                    // JordanFlitter: we can use the baryons density field
                    if (user_params->EVOLVE_BARYONS){
                        dxlya_dt *= const_zp_prefactor*prefactor_1 * (1.+delta_baryons_local);
                    }
                    else {
                        dxlya_dt *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                    }

                    dstarlya_dt *= prefactor_2;

                    // Now we can solve the evolution equations  //

                    // JordanFlitter: Lya flux for Lya heating
                    if (flag_options->USE_Lya_HEATING){
                      dstarlya_cont_dt *= prefactor_2;
                      dstarlya_inj_dt *= prefactor_2;
                    }

                    // JordanFlitter: added definition of x_CMB
                    prev_Ts = previous_spin_temp->Ts_box[box_ct];
                    // JordanFlitter: we can use the baryons density field
                    if (user_params->EVOLVE_BARYONS){
                        tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)*(1.+delta_baryons_local)) /prev_Ts/hubble(zp);
                    }
                    else {
                        tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp)) /prev_Ts/hubble(zp);
                    }
                    xCMB = (1. - exp(-tau21))/tau21;

                    // First let's do dxe_dzp //
                    if (user_params->USE_HYREC){
                        // JordanFlitter: Let's use HyRec!!
                        // Note that we treat xe and x_H to be the same, since after recombination x_He<<xe,x_H.
                        // Note also that Hyrec returns the derivative of n_e/n_H. Since we are interested in the derivative of n_e/(n_H+n_He),
                        // we must multiply by n_H/(n_H+n_He)=f_H. Also, Hyrec expects an input of n_e/n_H, so we need to give it x_e/f_H=n_e/n_H
                        // JordanFlitter: we can use the baryons density field
                        if (user_params->EVOLVE_BARYONS){
                            dxion_sink_dt = hyrec_dx_H_dz(rec_data, x_e/f_H, x_e/f_H, No*pow(1.+zp,3)*(1.+delta_baryons_local), zp, hubble(zp), T, Trad_fast);
                        }
                        else {
                            dxion_sink_dt = hyrec_dx_H_dz(rec_data, x_e/f_H, x_e/f_H, No*pow(1.+zp,3)*(1.+curr_delNL0*growth_factor_zp), zp, hubble(zp), T, Trad_fast);
                        }
                        dxion_sink_dt *= f_H;
                        dxion_sink_dt /= -dt_dzp; // Here we are only interested in the recombination rate, and not dx_e/dz (which is HyRec's output)
                    }
                    // JordanFlitter: we may want to use alpha_B recombination rate (and Peebles coefficient) at low redshifts because alpha_A diverges at low temperatures.
                    // For LCDM, both recombination rates agree to an order of percent, and the brightness temperature is not sensitive to the exact modelling of the recombination rate at these redshifts.
                    // For SDM, the brightness temperature is sensitive to the free electron fraction, even at low redshifts, becauese the temperature can reach very low values in which case Compton heating is not negligible!
                    else if (user_params->USE_ALPHA_B) {
                        // JordanFlitter: we can use the baryons density field
                        if (user_params->EVOLVE_BARYONS) {
                            dxion_sink_dt = alpha_B(T) * x_e*x_e * f_H * prefactor_1 * \
                                            (1.+delta_baryons_local) * Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+delta_baryons_local), alpha_B(T));
                        }
                        else{
                            dxion_sink_dt = alpha_B(T) * x_e*x_e * f_H * prefactor_1 * \
                                            (1.+curr_delNL0*growth_factor_zp) * Peebles(x_e, Trad_fast, hubble(zp), No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp), alpha_B(T));
                        }
                    }
                    else{
                        // JordanFlitter: we can use the baryons density field
                        if (user_params->EVOLVE_BARYONS) {
                            dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * \
                                            (1.+delta_baryons_local);
                        }
                        else{
                            dxion_sink_dt = alpha_A(T) * global_params.CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * \
                                            (1.+curr_delNL0*growth_factor_zp);
                        }
                    }
                    dxe_dzp = dt_dzp*(dxion_source_dt - dxion_sink_dt );

                    // Next, let's get the temperature components //
                    // JordanFlitter: we can use the baryons density field
                    if (user_params->EVOLVE_BARYONS){
                        dadia_dzp = 2.*T/(1.0+zp) + (2.0/3.0)*T*delta_baryons_derivative_local/(1.0+delta_baryons_local);
                    }
                    else {
                        // first, adiabatic term
                        dadia_dzp = 3/(1.0+zp);
                        if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                            dadia_dzp += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);

                        dadia_dzp *= (2.0/3.0)*T;
                    }

                    // next heating due to the changing species
                    dspec_dzp = - dxe_dzp * T / (1+x_e);

                    // next, Compton heating
                    // JordanFlitter: there shouldn't be any f_He at the Compton heating term, as x_e=n_e/(n_H+n_He). This was verified with Mesinger
                    dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e))*( Trad_fast - T );

                    // lastly, X-ray heating
                    dxheat_dzp = dxheat_dt * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                    //update quantities

                    dCMBheat_dzp = 0;
                    eps_Lya_cont = 0;
                    eps_Lya_inj = 0;

                    // JordanFlitter: defined CMB heating, following Avery Meiksin (arXiv: 2105.14516)
                    if (flag_options->USE_CMB_HEATING) {
                        eps_CMB = (3./4.) * (T_cmb*(1.+zp)/T21) * A10_HYPERFINE * f_H * (hplank*hplank/Lambda_21/Lambda_21/m_p) * (1.+2.*T/T21);
                        dCMBheat_dzp = 	-eps_CMB * (2./3./k_B/(1.+x_e))/hubble(zp)/(1.+zp);
                    }
                    // JordanFlitter: defined Lya heating
                    if (flag_options->USE_Lya_HEATING) {
                      // JordanFlitter: we can use the baryons density field
                      if (user_params->EVOLVE_BARYONS){
                          E_continuum = Energy_Lya_heating(T, prev_Ts, taugp(zp,delta_baryons_local,x_e), 2);
                          E_injected = Energy_Lya_heating(T, prev_Ts, taugp(zp,delta_baryons_local,x_e), 3);
                          Ndot_alpha_cont = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+delta_baryons_local))/(1.+zp)/C * dstarlya_cont_dt_box[box_ct];
                          Ndot_alpha_inj = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+delta_baryons_local))/(1.+zp)/C * dstarlya_inj_dt_box[box_ct];
                      }
                      else{
                          E_continuum = Energy_Lya_heating(T, prev_Ts, taugp(zp,curr_delNL0*growth_factor_zp,x_e), 2);
                          E_injected = Energy_Lya_heating(T, prev_Ts, taugp(zp,curr_delNL0*growth_factor_zp,x_e), 3);
                          Ndot_alpha_cont = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_cont_dt_box[box_ct];
                          Ndot_alpha_inj = (4.*PI*Ly_alpha_HZ) / ((No+He_No)*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp))/(1.+zp)/C * dstarlya_inj_dt_box[box_ct];
                      }
                      if (isnan(E_continuum) || isinf(E_continuum)){
                          E_continuum = 0.;
                      }
                      if (isnan(E_injected) || isinf(E_injected)){
                          E_injected = 0.;
                      }
                      eps_Lya_cont = - Ndot_alpha_cont * E_continuum * (2.0 / 3.0 /k_B/ (1.0+x_e));
                        eps_Lya_inj = - Ndot_alpha_inj * E_injected * (2.0 / 3.0 /k_B/ (1.0+x_e));
                    }

                    // JordanFlitter: Calculate heating rates and drag term in SDM universe, according to arXiv: 1509.00029
                    if (user_params->SCATTERING_DM) {
                        dT_b_2_dt_ext = (dxheat_dzp + dspec_dzp + (dadia_dzp-2.*T/(1.0+zp)) + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj)/dtdz(zp);
                        // JordanFlitter: we can use the SDM density field
                        if (user_params->EVOLVE_BARYONS){
                            dadia_dzp_SDM = 2.*T_chi/(1.0+zp) + (2.0/3.0)*T_chi*delta_SDM_derivative_local/(1.0+delta_SDM_local);
                        }
                        else {
                            dadia_dzp_SDM = 3/(1.0+zp);
                            if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                                dadia_dzp_SDM += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);

                            dadia_dzp_SDM *= (2.0/3.0)*T_chi;
                        }
                        dT_chi_2_dt_ext = (dadia_dzp_SDM-2.*T_chi/(1.0+zp))/dtdz(zp);
                        // JordanFlitter: we can use the baryons and SDM density fields
                        if (user_params->EVOLVE_BARYONS){
                            SDM_rates = SDM_derivatives(zp, x_e, T, T_chi, V_chi_b, delta_baryons_local, delta_SDM_local, hubble(zp),
                                            -hubble(zp)*(1.+zp)*dcomp_dzp_prefactor*x_e/(1.+x_e), Trad_fast, dzp, dT_b_2_dt_ext,
                                            dT_chi_2_dt_ext);
                        }
                        else {
                            SDM_rates = SDM_derivatives(zp, x_e, T, T_chi, V_chi_b, curr_delNL0*growth_factor_zp, curr_delNL0*growth_factor_zp, hubble(zp),
                                            -hubble(zp)*(1.+zp)*dcomp_dzp_prefactor*x_e/(1.+x_e), Trad_fast, dzp, dT_b_2_dt_ext,
                                            dT_chi_2_dt_ext);
                        }
                        dSDM_b_heat_dzp = (2.0/3.0/k_B)*SDM_rates.Q_dot_b*dtdz(zp); // K
                        dSDM_chi_heat_dzp = (2.0/3.0/k_B)*SDM_rates.Q_dot_chi*dtdz(zp); // K
                        D_V_chi_b_dzp = SDM_rates.D_V_chi_b*dtdz(zp); // cm/sec
                    }
                    else {
                        dSDM_b_heat_dzp = 0.;
                        dSDM_chi_heat_dzp = 0.;
                        D_V_chi_b_dzp = 0.;
                    }

                    x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                    if (x_e > 1) // can do this late in evolution if dzp is too large
                        x_e = 1 - FRACT_FLOAT_ERR;
                    else if (x_e < 0)
                        x_e = 0;
                    // JordanFlitter: added CMB and Lya heating to the temperature evolution
                    // JordanFlitter: I also added the SDM heating exchange
                    // JordanFlitter: If epsilon_b is not small enough, or if external heating rates dominate the SDM cooling rate,
                    //                or SDM doesn't exist, we evolve T_k with the usual ODE
                    if (!user_params->SCATTERING_DM ||
                        (user_params->SCATTERING_DM && ((fabs(SDM_rates.epsilon_b) > EPSILON_THRES) || (fabs(dT_b_2_dt_ext*dtdz(zp))>fabs(dSDM_b_heat_dzp))))
                        ) {
                        if (T < MAX_TK) {
                            T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj + dSDM_b_heat_dzp) * dzp;
                        }
                    }
                    // JordanFlitter: Otherwise, we evolve T_k with DM TCA!
                    else {
                        T = SDM_rates.T_bar_chi_b + SDM_rates.Delta_T_b_chi/2.; // K
                    }

                    // JordanFlitter: evolution equations for T_chi and V_chi_b in SDM universe
                    if (user_params->SCATTERING_DM) {
                      // JordanFlitter: If epsilon_chi is not small enough, or if external heating rates dominate the baryon heating rate,
                      // we evolve T_chi with the usual ODE
                      if ((fabs(SDM_rates.epsilon_chi) > EPSILON_THRES) || (fabs(dT_chi_2_dt_ext*dtdz(zp))>fabs(dSDM_chi_heat_dzp))) {
                          T_chi += (dadia_dzp_SDM + dSDM_chi_heat_dzp)*dzp; // K
                      }
                      // JordanFlitter: Otherwise, we evolve T_chi with DM TCA!
                      else {
                          T_chi = SDM_rates.T_bar_chi_b - SDM_rates.Delta_T_b_chi/2.; // K
                      }
                      V_chi_b += (V_chi_b/(1.0+zp) - D_V_chi_b_dzp)*dzp; // cm/sec
                    }

                    // JordanFlitter: I had to change the following logic for handling negative temperatures. The reason behind this change is the following.
                    // If SDM outnumber the baryons (e.g. if f_chi~100% and the SDM mass is light), its temperature is unaffected by the interactions with the
                    // baryons and it stays cold. The rapid interactions with baryons on the other hand, cause the baryons to be tightly coupled to the SDM,
                    // i.e. they also want to stay cold. Yet, external X-ray heating causes the baryons to heat up. In this unique scenario, although the baryons
                    // are tightly coupled to the cold SDM, their temperature departs from the SDM temperature, and therefore the cooling by the SDM interactions
                    // is very strong and may lead to negative temperatures. To cure this, it is important to realize that if the SDM cooling wins the X-ray heating,
                    // then the baryons temperature approaches the SDM temperature from above (it cannot be smaller than T_chi!)
                    // This problem doesn't happen when the SDM number-density is comparable to the baryons number-density; in that case, both fluids are strongly
                    // coupled to each other, effectively behaving as a single fluid, and their common temperature is increased by the X-rays.
                    if (!user_params->SCATTERING_DM) {
                        if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                              T = T_cmb*(1+zp);
                        }
                    }
                    else {
                        if (T<T_chi){ // T should never by smaller than T_chi!
                              if (T_chi>0){
                                  T = T_chi;
                              }
                              else {
                                  T = previous_spin_temp->Tk_box[box_ct]; // Don't update T in that special scenario
                              }
                        }
                    }

                    // JordanFlitter: similar logic for T_chi and V_chi_b in case of spurious bahaviour
                    if (user_params->SCATTERING_DM) {
                        if (T_chi<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                            T_chi = previous_spin_temp->T_chi_box[box_ct]; // Don't update T_chi in that special scenario
                        }
                        if (V_chi_b<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                            V_chi_b = 0.;
                        }
                    }

                    this_spin_temp->x_e_box[box_ct] = x_e;
                    this_spin_temp->Tk_box[box_ct] = T;
                    // JordanFlitter: update T_chi_box and V_chi_b_box
                    if (user_params->SCATTERING_DM) {
                      this_spin_temp->T_chi_box[box_ct] = T_chi; // K
                      this_spin_temp->V_chi_b_box[box_ct] = 1.e-5*V_chi_b; // km/sec
                    }

                    J_alpha_tot = ( dxlya_dt + dstarlya_dt ); //not really d/dz, but the lya flux

                    // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                    // Algorithm is the same, but written to be more computationally efficient
                    T_inv = pow(T,-1.);
                    T_inv_sq = pow(T,-2.);

                    // JordanFlitter: we can use the baryons density field
                    if (user_params->EVOLVE_BARYONS){
                        xc_fast = (1.0+delta_baryons_local)*xc_inverse*\
                                ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                        xi_power = TS_prefactor * cbrt((1.0+delta_baryons_local)*(1.0-x_e)*T_inv_sq);
                    }
                    else {
                        xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*\
                                ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );
                        xi_power = TS_prefactor * cbrt((1.0+curr_delNL0*growth_factor_zp)*(1.0-x_e)*T_inv_sq);
                    }
                    xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot*\
                                        pow( 1.0 + 2.98394*xi_power + 1.53583*pow(xi_power,2.) + 3.85289*pow(xi_power,3.), -1. );

                    if (J_alpha_tot > 1.0e-20) { // Must use WF effect
                        TS_fast = Trad_fast;
                        TSold_fast = 0.0;
                        while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {

                            TSold_fast = TS_fast;

                            // JordanFlitter: S_alpha correction, according to Eq. A4 in Mittal & Kulkarni (arXiv: 2009.10746),
                            // a result that was derived from the work of Chuzhouy & Shapiro (arXiv: astro-ph/0512206). This method was used
                            // in Driskell et al. (arXiv: 2209.04499) to account for low-temperature corrections which aren't captured by
                            // Hirata's fit (arXiv: astro-ph/0507102), which is used in the public 21cmFAST
                            if (user_params->USE_CS_S_ALPHA) {
                                xa_tilde_fast = xa_tilde_prefactor*J_alpha_tot;

                                /* JordanFlitter: The following commented lines are an attempt to implement Eq. (18) in arXiv:2212.08082
                                  (which was orignally introduced in Eq. 57 in arXiv:1605.04357). Unfortunately, this implementation leads
                                  to 15% maximum error in the brightness temperature (for some set of astrophysical parameters and models),
                                  which is absolutely not accepted. I don't know if that discrepancy is the result of a bug in the lines below...

                                xa_tilde_fast *= 1./(1.-0.402/T);
                                xa_tilde_fast *= exp(-2.06*pow(cosmo_params->OMb * cosmo_params->hlittle/0.0327,1./3.)
                                                          *pow(cosmo_params->OMm/0.307,-1./6.)
                                                          *pow((1.+zp)/10.,1./2.)*pow(T/0.402,-2./3.));
                                TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+(xa_tilde_fast+xc_fast)*T_inv,-1.);*/

                                // JordanFlitter: Instead, we use tabulated values from the work of Chuzhouy & Shapiro (arXiv: astro-ph/0512206).
                                // The numerical prefactor in the argument of S_alpha is 24*pi^2*nu_alpha*m_H*k_B^2/(A_alhpa*gamma_alpha*c*h^3) = 6.84e13 sec/(K^2 * cm^3)
                                // Also, in this method we have a closed analytcial formula for T_s (which is why we set TSold_fast = TS_fast)
                                // JordanFlitter: we can use the baryons density field
                                if (user_params->EVOLVE_BARYONS){
                                    xa_tilde_fast *= S_alpha_correction(6.84e13 * hubble(zp) * T*T / (No*pow(1.+zp,3.)*(1.+delta_baryons_local)*(1.-x_e)), 0);
                                }
                                else{
                                    xa_tilde_fast *= S_alpha_correction(6.84e13 * hubble(zp) * T*T / (No*pow(1.+zp,3.)*(1.+curr_delNL0*growth_factor_zp)*(1.-x_e)), 0);
                                }
                                //TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+xa_tilde_fast*pow(TS_fast,-1.)*(TS_fast+0.402)/(T+0.402)+xc_fast*T_inv,-1.);
                                TS_fast = (xCMB+xc_fast+xa_tilde_fast*T/(T+0.402))*pow(xCMB*Trad_fast_inv+xa_tilde_fast*pow(T+0.402,-1.)+xc_fast*T_inv,-1.);
                                TSold_fast = TS_fast;
                            }
                            else {
                                xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - \
                                               0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;
                                // JordanFlitter: modified spin temperature by xCMB
                                TS_fast = (xCMB+xa_tilde_fast+xc_fast)*pow(xCMB*Trad_fast_inv+xa_tilde_fast*( T_inv + \
                                                0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                            }
                        }
                    } else { // Collisions only
                        TS_fast = (xCMB + xc_fast)/(xCMB*Trad_fast_inv + xc_fast*T_inv);
                        xa_tilde_fast = 0.0;
                    }

                    if(TS_fast < 0.) {
                        // It can very rarely result in a negative spin temperature. If negative, it is a very small number.
                        // Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                        TS_fast = fabs(TS_fast);
                    }

                    this_spin_temp->Ts_box[box_ct] = TS_fast;

                    // JordanFlitter: added Lya flux to output! (because why not)
                    this_spin_temp->J_Lya_box[box_ct] =  J_alpha_tot;

                    if(LOG_LEVEL >= DEBUG_LEVEL){
                        J_alpha_ave += J_alpha_tot;
                        xalpha_ave += xa_tilde_fast;
                        Xheat_ave += ( dxheat_dzp );
                        Xion_ave += ( dt_dzp*dxion_source_dt );

                        Ts_ave += TS_fast;
                        Tk_ave += T;
                    }
                    x_e_ave += x_e;
                }
            }

            for(i=0;i<user_params->N_THREADS; i++) {
                if(table_int_boundexceeded_threaded[i]==1) {
                    LOG_ERROR("I have overstepped my allocated memory for one of the interpolation tables of dfcoll_dz_val");
//                    Throw(ParameterError);
                    Throw(TableEvaluationError);
                }
            }
        }

        for (box_ct=0; box_ct<HII_TOT_NUM_PIXELS; box_ct++){
            if(isfinite(this_spin_temp->Ts_box[box_ct])==0) {
                LOG_ERROR("Estimated spin temperature is either infinite of NaN!");
//                Throw(ParameterError);
                Throw(InfinityorNaNError);
            }
        }


LOG_SUPER_DEBUG("finished loop");

        /////////////////////////////  END LOOP ////////////////////////////////////////////
        // compute new average values
        if(LOG_LEVEL >= DEBUG_LEVEL){
            x_e_ave /= (double)HII_TOT_NUM_PIXELS;

            Ts_ave /= (double)HII_TOT_NUM_PIXELS;
            Tk_ave /= (double)HII_TOT_NUM_PIXELS;
            J_alpha_ave /= (double)HII_TOT_NUM_PIXELS;
            xalpha_ave /= (double)HII_TOT_NUM_PIXELS;
            Xheat_ave /= (double)HII_TOT_NUM_PIXELS;
            Xion_ave /= (double)HII_TOT_NUM_PIXELS;

            if (flag_options->USE_MINI_HALOS){
                J_alpha_ave_MINI /= (double)HII_TOT_NUM_PIXELS;
                Xheat_ave_MINI /= (double)HII_TOT_NUM_PIXELS;
                J_LW_ave /= (double)HII_TOT_NUM_PIXELS;
                J_LW_ave_MINI /= (double)HII_TOT_NUM_PIXELS;

                LOG_DEBUG("zp = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e(%e) xalpha_ave = %e \
                          Xheat_ave = %e(%e) Xion_ave = %e J_LW_ave = %e (%e)",zp,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,\
                          J_alpha_ave_MINI,xalpha_ave,Xheat_ave,Xheat_ave_MINI,Xion_ave,J_LW_ave/1e21,J_LW_ave_MINI/1e21);
            }
            else{
                LOG_DEBUG("zp = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e xalpha_ave = %e \
                          Xheat_ave = %e Xion_ave = %e",zp,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave);
            }
        }
      } // JordanFlitter: End of cosmic dawn condition
      // JordanFlitter: We need to free HyRec memory
      if (user_params->USE_HYREC) {
          hyrec_free(rec_data);
          free(rec_data->path_to_hyrec);
          free(rec_data);
      }
    } // end main integral loop over z'

        fftwf_free(box);
        fftwf_free(unfiltered_box);

        if (flag_options->USE_MINI_HALOS){
            fftwf_free(log10_Mcrit_LW_unfiltered);
            fftwf_free(log10_Mcrit_LW_filtered);
            for (R_ct=0; R_ct<global_params.NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                free(log10_Mcrit_LW[R_ct]);
            }
            free(log10_Mcrit_LW);
        }
        //JordanFlitter: We don't need these during the dark ages
        if (redshift <= global_params.Z_HEAT_MAX) {

    //    fftwf_destroy_plan(plan);
        fftwf_forget_wisdom();
        fftwf_cleanup_threads();
        fftwf_cleanup();

        // Free all the boxes. Ideally, we wouldn't do this, as almost always
        // the *next* call to ComputeTsBox will need the same memory. However,
        // we can't be sure that a call from python will not have changed the box size
        // without freeing, and get a segfault. The only way around this would be to
        // check (probably in python) every time spin() is called, whether the boxes
        // are already initialised _and_ whether they are of the right shape. This
        // seems difficult, so we leave that as future work.
        if(cleanup) free_TsCalcBoxes(user_params,flag_options);

        }
        free(table_int_boundexceeded_threaded);
        free(fcoll_int_boundexceeded_threaded);
        // JordanFlitter: I moved destruct_heat to here
        destruct_heat();
        // JordanFlitter: we need destruct_CLASS_GROWTH_FACTOR() if the following conditions are satisfied
        if (!user_params->USE_DICKE_GROWTH_FACTOR || user_params->EVOLVE_BARYONS) {
              destruct_CLASS_GROWTH_FACTOR();
        }

    } // End of try
    Catch(status){
        return(status);
    }
    return(0);
}


void free_TsCalcBoxes(struct UserParams *user_params, struct FlagOptions *flag_options)
{
    int i,j;

    free(zpp_edge);
    free(sigma_atR);
    free(R_values);

    if(user_params->USE_INTERPOLATION_TABLES) {
        free(min_densities);
        free(max_densities);

        free(zpp_interp_table);
    }

    free(SingleVal_int);
    free(dstarlya_dt_prefactor);
    free(fcoll_R_array);
    free(zpp_growth);
    free(inverse_diff);
    free(sigma_Tmin);
    free(ST_over_PS);
    free(sum_lyn);
    free(zpp_for_evolve_list);

    if (flag_options->USE_MINI_HALOS){
        free(Mcrit_atom_interp_table);
        free(dstarlya_dt_prefactor_MINI);
        free(dstarlyLW_dt_prefactor);
        free(dstarlyLW_dt_prefactor_MINI);
        free(ST_over_PS_MINI);
        free(sum_lyn_MINI);
        free(sum_lyLWn);
        free(sum_lyLWn_MINI);
    }
    // JordanFlitter: free Lya heating arrays
    if (flag_options->USE_Lya_HEATING){
        free(dstarlya_cont_dt_box);
        free(dstarlya_inj_dt_box);
        free(dstarlya_cont_dt_prefactor);
        free(dstarlya_inj_dt_prefactor);
        free(sum_ly2);
        free(sum_lynto2);
        if (flag_options->USE_MINI_HALOS){
          free(dstarlya_cont_dt_box_MINI);
          free(dstarlya_inj_dt_box_MINI);
          free(dstarlya_cont_dt_prefactor_MINI);
          free(dstarlya_inj_dt_prefactor_MINI);
          free(sum_ly2_MINI);
          free(sum_lynto2_MINI);
        }
    }

    if(flag_options->USE_MASS_DEPENDENT_ZETA) {
        free(SFR_timescale_factor);

        if(user_params->MINIMIZE_MEMORY) {
            free(delNL0[0]);
        }
        else {
            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                free(delNL0[i]);
            }
        }

        free(delNL0);
        // JordanFlitter: need to free the baryons boxes
        if (user_params->EVOLVE_BARYONS){
            fftwf_free(delta_baryons);
            fftwf_free(delta_baryons_derivative);
            if (user_params->SCATTERING_DM){
                fftwf_free(delta_SDM);
                fftwf_free(delta_SDM_derivative);
            }
        }

        free(xi_SFR_Xray);
        free(wi_SFR_Xray);

        if(user_params->USE_INTERPOLATION_TABLES) {
            free(overdense_low_table);
            free(overdense_high_table);
            for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                free(log10_SFRD_z_low_table[j]);
            }
            free(log10_SFRD_z_low_table);

            for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                free(SFRD_z_high_table[j]);
            }
            free(SFRD_z_high_table);
        }

        free(del_fcoll_Rct);
        free(dxheat_dt_box);
        free(dxion_source_dt_box);
        free(dxlya_dt_box);
        free(dstarlya_dt_box);
        free(m_xHII_low_box);
        free(inverse_val_box);
        if(flag_options->USE_MINI_HALOS){
            if(user_params->USE_INTERPOLATION_TABLES) {
                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    free(log10_SFRD_z_low_table_MINI[j]);
                }
                free(log10_SFRD_z_low_table_MINI);

                for(j=0;j<global_params.NUM_FILTER_STEPS_FOR_Ts;j++) {
                    free(SFRD_z_high_table_MINI[j]);
                }
                free(SFRD_z_high_table_MINI);
            }
            free(log10_Mcrit_LW_ave_list); // JordanFlitter: Freed that variable (it is not freed at the public version of 21cmFAST!)
            free(del_fcoll_Rct_MINI);
            free(dstarlyLW_dt_box);
            free(dxheat_dt_box_MINI);
            free(dxion_source_dt_box_MINI);
            free(dxlya_dt_box_MINI);
            free(dstarlya_dt_box_MINI);
            free(dstarlyLW_dt_box_MINI);
        }
    }
    else {

        if(user_params->USE_INTERPOLATION_TABLES) {
            free(Sigma_Tmin_grid);
            free(ST_over_PS_arg_grid);
            free(delNL0_bw);
            free(delNL0_Offset);
            free(delNL0_LL);
            free(delNL0_UL);
            free(delNL0_ibw);
            free(log10delNL0_diff);
            free(log10delNL0_diff_UL);

            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                for(j=0;j<zpp_interp_points_SFR;j++) {
                    free(fcoll_R_grid[i][j]);
                    free(dfcoll_dz_grid[i][j]);
                }
                free(fcoll_R_grid[i]);
                free(dfcoll_dz_grid[i]);
            }
            free(fcoll_R_grid);
            free(dfcoll_dz_grid);

            for(i=0;i<global_params.NUM_FILTER_STEPS_FOR_Ts;i++) {
                free(grid_dens[i]);
            }
            free(grid_dens);

            for(i=0;i<dens_Ninterp;i++) {
                free(density_gridpoints[i]);
            }
            free(density_gridpoints);

            for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
                free(dens_grid_int_vals[i]);
            }
            free(dens_grid_int_vals);

            for(i=0;i<dens_Ninterp;i++) {
                free(fcoll_interp1[i]);
                free(fcoll_interp2[i]);
                free(dfcoll_interp1[i]);
                free(dfcoll_interp2[i]);
            }
            free(fcoll_interp1);
            free(fcoll_interp2);
            free(dfcoll_interp1);
            free(dfcoll_interp2);
        }

        for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
            free(delNL0_rev[i]);
        }
        free(delNL0_rev);

    }

    for(i=0;i<x_int_NXHII;i++) {
        free(freq_int_heat_tbl[i]);
        free(freq_int_ion_tbl[i]);
        free(freq_int_lya_tbl[i]);
        free(freq_int_heat_tbl_diff[i]);
        free(freq_int_ion_tbl_diff[i]);
        free(freq_int_lya_tbl_diff[i]);
    }
    free(freq_int_heat_tbl);
    free(freq_int_ion_tbl);
    free(freq_int_lya_tbl);
    free(freq_int_heat_tbl_diff);
    free(freq_int_ion_tbl_diff);
    free(freq_int_lya_tbl_diff);
    // JordanFlitter: added free_ps here (it is not freed in the public version of 21cmFAST!)
    free_ps();

    TsInterpArraysInitialised = false;
}
