/*
    This is the header file for the wrappable version of 21cmFAST, or 21cmMC.
    It contains function signatures, struct definitions and globals to which the Python wrapper code
    requires access.
*/

struct CosmoParams{

    float SIGMA_8;
    float hlittle;
    float OMm;
    float OMl;
    float OMb;
    float POWER_INDEX;
    float A_s; // JordanFlitter: added amplitude of primordial curvature fluctuations
    float tau_reio; // JordanFlitter: added optical depth to reionization
    float m_FDM; // JordanFlitter: added FDM mass (this is actually -log10(m_FDM/eV))
    float f_FDM; // JordanFlitter: added FDM fraction (this is actually -log10(f_FDM))
    float m_chi; // JordanFlitter: added SDM mass (this is actually log10(m_chi/eV))
    float f_chi; // JordanFlitter: added SDM fraction (this is actually -log10(f_chi))
    float sigma_SDM; // JordanFlitter: added SDM cross section prefactor (this is actually -log10(sigma/cm^2))
    float SDM_INDEX; // JordanFlitter: added SDM cross section index

};

struct UserParams{

    // Parameters taken from INIT_PARAMS.H
    int HII_DIM;
    int DIM;
    float BOX_LEN;
    bool USE_FFTW_WISDOM;
    int HMF;
    int USE_RELATIVE_VELOCITIES;
    int POWER_SPECTRUM;
    int N_THREADS;
    bool PERTURB_ON_HIGH_RES;
    bool NO_RNG;
    bool USE_INTERPOLATION_TABLES;
    bool FAST_FCOLL_TABLES; //Whether to use the fast Fcoll table approximation in EPS
    bool USE_2LPT;
    bool MINIMIZE_MEMORY;
    bool RUN_CLASS; // JordanFlitter: added flag for running CLASS
    bool DO_VCB_FIT; // JordanFlitter: added flag for performing v_cb fit to the matter power spectrum
    bool FUZZY_DM; // JordanFlitter: added flag for fuzzy DM
    bool SCATTERING_DM; // JordanFlitter: added flag for scattering DM
    int SDM_TARGET_TYPE; // JordanFlitter: added type of SDM target particles (1=baryons, 2=protons and electrons, 3=hydrogen nuclei, 4=protons, 5=electrons)
    bool USE_SDM_FLUCTS; // JordanFlitter: added flag to use initial SDM fluctuations
    bool START_AT_RECOMBINATION; // JordanFlitter: added flag to run the simulation through the dark ages
    bool USE_HYREC; // JordanFlitter: added flag to use HyRec
    bool NO_INI_MATTER_FLUCTS; // JordanFlitter: added flag to set on zero the matter fluctuations during the dark ages (useful for comparing with CLASS)
    float Z_HIGH_MAX; // JordanFlitter: added the highest redshift for the evolution (default -1 means recombination)
    bool OUTPUT_AT_DARK_AGES; // JordanFlitter: added flag to compute output during the dark ages
    bool USE_TCA_COMPTON; // JordanFlitter: added flag to use tight coupling approximation (for Compton scattering)
    bool USE_CS_S_ALPHA; // JordanFlitter: added flag to use the S_alpha correction from Chuzhouy & Shapiro (arXiv: astro-ph/0512206)
    bool MANY_Z_SAMPLES_AT_COSMIC_DAWN; // JordanFlitter: added flag to include many more redshift samples during cosmic dawn (works only if START_AT_RECOMBINATION = True)
    bool USE_ALPHA_B; // JordanFlitter: added flag to use alpha_B (with the Peebles coefficient) as the recombination rate at cosmic dawn (below Z_HEAT_MAX)
    bool DO_PERTURBS_WITH_TS; // JordanFlitter: added flag to perturb the density field while evaluating the spin temperature (to reduce required memory)
    bool USE_ADIABATIC_FLUCTUATIONS; // JordanFlitter: added flag to generate inhomogeneous temperature box at Z_HEAT_MAX, based on arXiv: 2302.08506
    bool USE_DICKE_GROWTH_FACTOR; // JordanFlitter: added flag to use 21cmFAST default Dicke growth factor even if RUN_CLASS = True
    bool CLOUD_IN_CELL; // JordanFlitter: added flag to use Bradley Greig's algorithm for cloud in cell in 2LPT calculations
    bool EVOLVE_BARYONS; // JordanFlitter: added flag to use the scale-dependent growth factor to evolve the baryons density field (see arXiv: 2309.03948)
    bool EVALUATE_TAU_REIO; // JordanFlitter: added flag to evaluate tau_reio from the simulation
};

struct AstroParams{

    // Parameters taken from INIT_PARAMS.H
    float HII_EFF_FACTOR;

    float F_STAR10;
    float ALPHA_STAR;
    float ALPHA_STAR_MINI;
    float F_ESC10;
    float ALPHA_ESC;
    float M_TURN;
    float F_STAR7_MINI;
    float F_ESC7_MINI;
    float R_BUBBLE_MAX;
    float ION_Tvir_MIN;
    double F_H2_SHIELD;
    double L_X;
    double L_X_MINI;
    float NU_X_THRESH;
    float X_RAY_SPEC_INDEX;
    float X_RAY_Tvir_MIN;

    double A_LW;
    double BETA_LW;
    double A_VCB;
    double BETA_VCB;

    float t_STAR;

    int N_RSD_STEPS;
};

struct FlagOptions{

    // Parameters taken from INIT_PARAMS.H
    bool USE_HALO_FIELD;
    bool USE_MINI_HALOS;
    bool USE_CMB_HEATING; // JordanFlitter: added CMB Heating
    bool USE_Lya_HEATING; // JordanFlitter: added Lya Heating
    bool USE_MASS_DEPENDENT_ZETA;
    bool SUBCELL_RSD;
    bool INHOMO_RECO;
    bool USE_TS_FLUCT;
    bool M_MIN_in_Mass;
    bool PHOTON_CONS;
    bool FIX_VCB_AVG;
};


struct InitialConditions{
    float *lowres_density, *lowres_vx, *lowres_vy, *lowres_vz, *lowres_vx_2LPT, *lowres_vy_2LPT, *lowres_vz_2LPT;
    float *hires_density, *hires_vx, *hires_vy, *hires_vz, *hires_vx_2LPT, *hires_vy_2LPT, *hires_vz_2LPT; //cw addition
    float *lowres_vcb;
    float *lowres_xe_zhigh, *lowres_Tk_zhigh, *lowres_Tchi_zhigh, *lowres_V_chi_b_zhigh; // JordanFlitter: added new SDM boxes to the InitialConditions structure
};

struct PerturbedField{
    float *density, *velocity;
    // JordanFlitter: added new baryons (and SDM) density box to the PerturbedField structure
    float *baryons_density;
    float *SDM_density;
};

struct HaloField{

    int n_halos;
    float *halo_masses;
    int *halo_coords;

    int n_mass_bins;
    int max_n_mass_bins;

    float *mass_bins;
    float *fgtrm;
    float *sqrt_dfgtrm;
    float *dndlm;
    float *sqrtdn_dlm;
};

struct PerturbHaloField{
    int n_halos;
    float *halo_masses;
    int *halo_coords;
};


struct TsBox{
    int first_box;
    float *Ts_box;
    float *x_e_box;
    float *Tk_box;
    float *J_21_LW_box;
    float *J_Lya_box; // JordanFlitter: added J_Lya_box to the TsBox structure (because why not)
    float *T_chi_box; // JordanFlitter: added T_chi_box to the Ts_box structure
    float *V_chi_b_box; // JordanFlitter: added V_chi_b_box to the Ts_box structure
    float next_redshift_output; // JordanFlitter: added next_redshift_output to the Ts_box structure
};

struct IonizedBox{
    int first_box;
    double mean_f_coll;
    double mean_f_coll_MINI;
    double log10_Mturnover_ave;
    double log10_Mturnover_MINI_ave;
    float *xH_box;
    float *Gamma12_box;
    float *MFP_box;
    float *z_re_box;
    float *dNrec_box;
    float *temp_kinetic_all_gas;
    float *Fcoll;
    float *Fcoll_MINI;
};

struct BrightnessTemp{
    float *brightness_temp;
};

int ComputeInitialConditions(unsigned long long random_seed, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes);

int ComputePerturbField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct InitialConditions *boxes, struct PerturbedField *perturbed_field);

int ComputeHaloField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, struct HaloField *halos);

int ComputePerturbHaloField(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                     struct AstroParams *astro_params, struct FlagOptions *flag_options,
                     struct InitialConditions *boxes, struct HaloField *halos, struct PerturbHaloField *halos_perturbed);
// JordanFlitter: added next_redshift_input
int ComputeTsBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                  struct AstroParams *astro_params, struct FlagOptions *flag_options, float perturbed_field_redshift,
                  float next_redshift_input, short cleanup,
                  struct PerturbedField *perturbed_field, struct TsBox *previous_spin_temp, struct InitialConditions *ini_boxes,
                  struct TsBox *this_spin_temp);

int ComputeIonizedBox(float redshift, float prev_redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                       struct AstroParams *astro_params, struct FlagOptions *flag_options, struct PerturbedField *perturbed_field,
                       struct PerturbedField *previous_perturbed_field, struct IonizedBox *previous_ionize_box,
                       struct TsBox *spin_temp, struct PerturbHaloField *halo, struct InitialConditions *ini_boxes,
                       struct IonizedBox *box);

int ComputeBrightnessTemp(float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
                           struct AstroParams *astro_params, struct FlagOptions *flag_options,
                           struct TsBox *spin_temp, struct IonizedBox *ionized_box,
                           struct PerturbedField *perturb_field, struct BrightnessTemp *box);

int InitialisePhotonCons(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                         struct AstroParams *astro_params, struct FlagOptions *flag_options);

int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline);
int ComputeZstart_PhotonCons(double *zstart);

int ObtainPhotonConsData(double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data, double *nf_cal_data, int *Ndata_calibration,
                         double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons);

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi);

float ComputeTau(struct UserParams *user_params, struct CosmoParams *cosmo_params, int Npoints, float *redshifts, float *global_xHI);

int CreateFFTWWisdoms(struct UserParams *user_params, struct CosmoParams *cosmo_params);

void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_UF(struct UserParams *user_params, struct CosmoParams *cosmo_params);
void Broadcast_struct_global_HF(struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options);

// JordanFlitter: I need to declare these functions here, since I moved them to heating_helper_progs (in order to use the user's astrophysical parameters),
//                but ps.c (where these functions are also called) is included prior to heating_helper_progs.
double atomic_cooling_threshold(float z);
double molecular_cooling_threshold(float z);

void free_TsCalcBoxes(struct UserParams *user_params, struct FlagOptions *flag_options);
void FreePhotonConsMemory();
void FreeTsInterpolationTables(struct FlagOptions *flag_options);
bool photon_cons_allocated = false;
bool interpolation_tables_allocated = false;
int SomethingThatCatches(bool sub_func);
int FunctionThatCatches(bool sub_func, bool pass, double* result);
void FunctionThatThrows();
int init_heat(float redshift); // JordanFlitter: added a redshift argument
void free(void *ptr);
