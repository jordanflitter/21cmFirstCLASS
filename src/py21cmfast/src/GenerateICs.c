#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
//#include <pthread.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
// JordanFlitter: added gsl 2D interpolation library
#include <gsl/gsl_interp2d.h>
// JordanFlitter: added gsl hypergeometric functions library
#include <gsl/gsl_sf_hyperg.h>

#include "21cmFAST.h"
#include "exceptions.h"
#include "logger.h"
#include "Constants.h"
#include "Globals.h"
// JordanFlitter: I include HyRec header files
#include "HyRec2020/energy_injection.h"
#include "HyRec2020/helium.h"
#include "HyRec2020/hydrogen.h"
#include "HyRec2020/history.h"
#include "HyRec2020/hyrectools.h"

#include "indexing.c"
#include "UsefulFunctions.c"
#include "ps.c"
#include "dft.c"
#include "PerturbField.c"
#include "bubble_helper_progs.c"
#include "elec_interp.c"
#include "heating_helper_progs.c"
#include "recombinations.c"
#include "IonisationBox.c"
#include "SpinTemperatureBox.c"
#include "BrightnessTemperatureBox.c"
#include "FindHaloes.c"
#include "PerturbHaloField.c"
// JordanFlitter: I include HyRec c-files
#include "HyRec2020/energy_injection.c"
#include "HyRec2020/helium.c"
#include "HyRec2020/history.c"
#include "HyRec2020/hydrogen.c"
#include "HyRec2020/hyrectools.c"

void adj_complex_conj(fftwf_complex *HIRES_box, struct UserParams *user_params, struct CosmoParams *cosmo_params){
    /*****  Adjust the complex conjugate relations for a real array  *****/

    int i, j, k;

    // corners
    HIRES_box[C_INDEX(0,0,0)] = 0;
    HIRES_box[C_INDEX(0,0,MIDDLE)] = crealf(HIRES_box[C_INDEX(0,0,MIDDLE)]);
    HIRES_box[C_INDEX(0,MIDDLE,0)] = crealf(HIRES_box[C_INDEX(0,MIDDLE,0)]);
    HIRES_box[C_INDEX(0,MIDDLE,MIDDLE)] = crealf(HIRES_box[C_INDEX(0,MIDDLE,MIDDLE)]);
    HIRES_box[C_INDEX(MIDDLE,0,0)] = crealf(HIRES_box[C_INDEX(MIDDLE,0,0)]);
    HIRES_box[C_INDEX(MIDDLE,0,MIDDLE)] = crealf(HIRES_box[C_INDEX(MIDDLE,0,MIDDLE)]);
    HIRES_box[C_INDEX(MIDDLE,MIDDLE,0)] = crealf(HIRES_box[C_INDEX(MIDDLE,MIDDLE,0)]);
    HIRES_box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)] = crealf(HIRES_box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)]);

    // do entire i except corners
#pragma omp parallel shared(HIRES_box) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=1; i<MIDDLE; i++){
            // just j corners
            for (j=0; j<=MIDDLE; j+=MIDDLE){
                for (k=0; k<=MIDDLE; k+=MIDDLE){
                    HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX((user_params->DIM)-i,j,k)]);
                }
            }

            // all of j
            for (j=1; j<MIDDLE; j++){
                for (k=0; k<=MIDDLE; k+=MIDDLE){
                    HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX((user_params->DIM)-i,(user_params->DIM)-j,k)]);
                    HIRES_box[C_INDEX(i,(user_params->DIM)-j,k)] = conjf(HIRES_box[C_INDEX((user_params->DIM)-i,j,k)]);
                }
            }
        } // end loop over i
    }

    // now the i corners
#pragma omp parallel shared(HIRES_box) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<=MIDDLE; i+=MIDDLE){
            for (j=1; j<MIDDLE; j++){
                for (k=0; k<=MIDDLE; k+=MIDDLE){
                    HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX(i,(user_params->DIM)-j,k)]);
                }
            }
        } // end loop over remaining j
    }
}

// Re-write of init.c for original 21cmFAST

int ComputeInitialConditions(
    unsigned long long random_seed, struct UserParams *user_params,
    struct CosmoParams *cosmo_params, 
    // !!! SLTK: added astro_params and flag_options
    struct AstroParams *astro_params, struct FlagOptions *flag_options,
    struct InitialConditions *boxes
){

//     Generates the initial conditions: gaussian random density field (user_params->DIM^3) as well as the equal or lower resolution velocity fields, and smoothed density field (user_params->HII_DIM^3).
//
//     Author: Andrei Mesinger
//     Date: 9/29/06

    int status;

    Try{ // This Try wraps the entire function so we don't indent.

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    // !!! SLTK: added astro_params and flag_options
    Broadcast_struct_global_PS(user_params,cosmo_params, astro_params, flag_options);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    unsigned long long ct;
    int n_x, n_y, n_z, i, j, k, ii, thread_num, dimension;
    float k_x, k_y, k_z, k_mag, p, a, b, k_sq;
    double pixel_deltax;
    float p_vcb, vcb_i;
    // JordanFlitter: new variables for SDM
    float p_SDM, delta_SDM_i;

    float f_pixel_factor;

    // !!! SLFK: parameters used only in NG conditions
    float delta_k_NG, avg_hires_potential_sq,avg_hires_potential_sq_1;
    double error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double kstart, kend;

    gsl_rng * r[user_params->N_THREADS];
    gsl_rng * rseed = gsl_rng_alloc(gsl_rng_mt19937); // An RNG for generating seeds for multithreading

    gsl_rng_set(rseed, random_seed);

    omp_set_num_threads(user_params->N_THREADS);

    switch(user_params->PERTURB_ON_HIGH_RES) {
        case 0:
            dimension = user_params->HII_DIM;
            break;
        case 1:
            dimension = user_params->DIM;
            break;
    }

    // ************  INITIALIZATION ********************** //
    unsigned int seeds[user_params->N_THREADS];

    // For multithreading, seeds for the RNGs are generated from an initial RNG (based on the input random_seed) and then shuffled (Author: Fred Davies)
    int num_int = INT_MAX/16;
    unsigned int *many_ints = (unsigned int *)malloc((size_t)(num_int*sizeof(unsigned int))); // Some large number of possible integers
    for (i=0; i<num_int; i++) {
        many_ints[i] = i;
    }

    gsl_ran_choose(rseed, seeds, user_params->N_THREADS, many_ints, num_int, sizeof(unsigned int)); // Populate the seeds array from the large list of integers
    gsl_ran_shuffle(rseed, seeds, user_params->N_THREADS, sizeof(unsigned int)); // Shuffle the randomly selected integers

    int checker;

    checker = 0;
    // seed the random number generators
    for (thread_num = 0; thread_num < user_params->N_THREADS; thread_num++){
        switch (checker){
            case 0:
                r[thread_num] = gsl_rng_alloc(gsl_rng_mt19937);
                gsl_rng_set(r[thread_num], seeds[thread_num]);
                break;
            case 1:
                r[thread_num] = gsl_rng_alloc(gsl_rng_gfsr4);
                gsl_rng_set(r[thread_num], seeds[thread_num]);
                break;
            case 2:
                r[thread_num] = gsl_rng_alloc(gsl_rng_cmrg);
                gsl_rng_set(r[thread_num], seeds[thread_num]);
                break;
            case 3:
                r[thread_num] = gsl_rng_alloc(gsl_rng_mrg);
                gsl_rng_set(r[thread_num], seeds[thread_num]);
                break;
            case 4:
                r[thread_num] = gsl_rng_alloc(gsl_rng_taus2);
                gsl_rng_set(r[thread_num], seeds[thread_num]);
                break;
        } // end switch

        checker += 1;

        if(checker==5) {
            checker = 0;
        }
    }

    free(many_ints);

    // allocate array for the k-space and real-space boxes
    fftwf_complex *HIRES_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    fftwf_complex *HIRES_box_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    // !!! SLKF: allocate array for k-space and real-space potential
    fftwf_complex *HIRES_potential_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    fftwf_complex *HIRES_NGpotential_box_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    // allocate array for the k-space and real-space boxes for vcb
    fftwf_complex *HIRES_box_vcb_saved;
    // HIRES_box_vcb_saved may be needed if FFTW_Wisdom doesn't exist -- currently unused
    // but I am not going to allocate it until I am certain I needed it.

    // find factor of HII pixel size / deltax pixel size
    f_pixel_factor = user_params->DIM/(float)user_params->HII_DIM;

    // ************  END INITIALIZATION ****************** //
    LOG_DEBUG("Finished initialization.");
    // ************ CREATE K-SPACE GAUSSIAN RANDOM FIELD *********** //

    init_ps();
    // !!! SLKF: needed to use growth in NG
    if (!user_params_ps->USE_DICKE_GROWTH_FACTOR || user_params_ps->EVOLVE_BARYONS) {
        init_CLASS_GROWTH_FACTOR();
    }

#pragma omp parallel shared(HIRES_box,r,HIRES_potential_box) \
                    private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,p,a,b,p_vcb) num_threads(user_params->N_THREADS)
    {
#pragma omp for

        for (n_x=0; n_x<user_params->DIM; n_x++){
            // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
            if (n_x>MIDDLE)
                k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y=0; n_y<user_params->DIM; n_y++){
                // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
                if (n_y>MIDDLE)
                    k_y =(n_y-user_params->DIM) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                // since physical space field is real, only half contains independent modes
                for (n_z=0; n_z<=MIDDLE; n_z++){
                    // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
                    k_z = n_z * DELTA_K;
                    
                    // now get the power spectrum; remember, only the magnitude of k counts (due to issotropy)
                    // this could be used to speed-up later maybe
                    k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                    // ok, now we can draw the values of the real and imaginary part
                    // of our k entry from a Gaussian distribution

                    // !!! SLKF: NG conditions, instead of delta first of all generate potential box
                    if(user_params->NO_RNG) {
                        a = 1.0;
                        b = -1.0;
                    }
                    else {
                        a = gsl_ran_ugaussian(r[omp_get_thread_num()]);
                        b = gsl_ran_ugaussian(r[omp_get_thread_num()]);
                    }

                    if (user_params->NG_FIELD){

                        // !!! SLKF: in this case, we are generating the FT potential, which has to be then converted 
                        // p is gravitational potential power spectrum, namely Delta^2_phi_G
                        p = power_in_k_potential(k_mag);
                        HIRES_potential_box[C_INDEX(n_x, n_y, n_z)] =  sqrt(VOLUME * p/2.) * (a + b*I); 
                    }

                    else{
                        // original 21cmFAST - see https://iopscience.iop.org/article/10.1086/521806/pdf
                        p = power_in_k(k_mag);
                        HIRES_box[C_INDEX(n_x, n_y, n_z)] = sqrt(VOLUME*p/2.0) * (a + b*I);
                    }
                    }
                }
            }
        }

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// -------------------------- BEGIN NG CASE ----------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

        if (user_params->NG_FIELD){

        // !!! SLKF: convert the phi_G box in real space
        // *****  Adjust the complex conjugate relations for a real array  ***** //
        adj_complex_conj(HIRES_potential_box,user_params,cosmo_params);

        // // FFT back to real space
        int stat = dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_potential_box);
        if(stat>0) Throw(stat);
        LOG_DEBUG("FFT'd hires potential boxes.");

        #pragma omp parallel shared(boxes,HIRES_potential_box) private(i,j,k) num_threads(user_params->N_THREADS)
            {
        #pragma omp for
                for (i=0; i<user_params->DIM; i++){
                    for (j=0; j<user_params->DIM; j++){
                        for (k=0; k<user_params->DIM; k++){
                            *((float *)boxes->hires_potential + R_INDEX(i,j,k)) = *((float *)HIRES_potential_box + R_FFT_INDEX(i,j,k))/VOLUME;
                        }
                    }
                }
            }

        // !!! SLKF: compute average Phi_G 
        avg_hires_potential_sq = 0.;
        #pragma omp parallel shared(boxes) private(i,j,k) num_threads(user_params->N_THREADS)
                {
        #pragma omp for
            for (i=0; i<user_params->DIM; i++){
            for (j=0; j<user_params->DIM; j++){
                for (k=0; k<user_params->DIM; k++){
                    avg_hires_potential_sq +=  pow(boxes->hires_potential[R_INDEX(i,j,k)],2) / TOT_NUM_PIXELS;
                }}}}

        // !!! SLKF ??? POSSIBLE CHANGE TO IMPLEMENT HERE: try to compute using the integral similar to sigma_M 
        // kstart = fmax(1.0e-99, KBOT_CLASS);
        // kend = fmin(350.0, KTOP_CLASS);
        // lower_limit = kstart;//log(kstart);
        // upper_limit = kend;//log(kend);
        // F.function = &integrate_potential;
        // F.params = &k;
        // int status;
        // gsl_set_error_handler_off();
        // status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,1000, GSL_INTEG_GAUSS61, w, &avg_hires_potential_sq, &error);
        // printf("avg_box_old = %e \n", avg_hires_potential_sq_1);
        // printf("avg_box = %e \n", avg_hires_potential_sq);
        // if(status!=0) {
        //     LOG_ERROR("gsl integration error occured!");
        //     LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,avg_hires_potential_sq,error);
        //     GSL_ERROR(status);
        // }
        // gsl_integration_workspace_free (w);


        // // !!! SLKF: compute Phi_G = phi_G + fnl*(phi_G**2 + <phi_G**2>)
        #pragma omp parallel shared(boxes) private(i,j,k) num_threads(user_params->N_THREADS)
                {
        #pragma omp for
            for (i=0; i<user_params->DIM; i++){
            for (j=0; j<user_params->DIM; j++){
                for (k=0; k<user_params->DIM; k++){
                    boxes->hires_potentialNG[R_INDEX(i,j,k)] = boxes->hires_potential[R_INDEX(i,j,k)] + cosmo_params->F_NL*(pow(boxes->hires_potential[R_INDEX(i,j,k)],2) - avg_hires_potential_sq);
        }}}
        }

        #pragma omp parallel shared(HIRES_NGpotential_box_saved,boxes) private(i,j,k) num_threads(user_params->N_THREADS)
                    {
        #pragma omp for
                        for (i=0; i<user_params->DIM; i++){
                            for (j=0; j<user_params->DIM; j++){
                                for (k=0; k<user_params->DIM; k++){
                                    *((float *)HIRES_NGpotential_box_saved + R_FFT_INDEX(i,j,k)) = boxes->hires_potentialNG[R_INDEX(i,j,k)];
                                }
                            }
                        }
                    }


        // // !!! SLKF: transform Phi_G in k space 
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS,HIRES_NGpotential_box_saved);

        // !!! SLKF: compute delta based on Phi_G
        #pragma omp parallel shared(HIRES_box,HIRES_NGpotential_box_saved) \
                            private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag) num_threads(user_params->N_THREADS)
            {
        #pragma omp for

        // !!! SLKF: include non gaussian conditions, they are estimated as delta
        // so far we have generate the FT box of the potential, now we need to move it in real space, take the avg and then convert again to FT and estimate delta
            for (n_x=0; n_x<user_params->DIM; n_x++){
                if (n_x>MIDDLE)
                    k_x =(n_x-user_params->DIM) * DELTA_K;  
                else
                    k_x = n_x * DELTA_K;

                for (n_y=0; n_y<user_params->DIM; n_y++){
                    if (n_y>MIDDLE)
                        k_y =(n_y-user_params->DIM) * DELTA_K;
                    else
                        k_y = n_y * DELTA_K;
                    for (n_z=0; n_z<=MIDDLE; n_z++){
                        
                        k_z = n_z * DELTA_K;
                        k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                        delta_k_NG = 2*pow(C_KMS,2) * pow(k_mag,2) / (3*pow(cosmo_params->hlittle*100,2)*cosmo_params->OMm) ; // to be checked: the potential in CLASS already includes the transfer function?
                        HIRES_box[C_INDEX(n_x, n_y, n_z)] =   delta_k_NG * HIRES_NGpotential_box_saved[C_INDEX(n_x, n_y, n_z)] ; 
                    }
        }}}
        }
    
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// -------------------------- END NG CASE ----------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

    LOG_DEBUG("Drawn random fields.");

    // *****  Adjust the complex conjugate relations for a real array  ***** //
    adj_complex_conj(HIRES_box,user_params,cosmo_params);

    memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    // FFT back to real space
    int stat = dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);
    if(stat>0) Throw(stat);
    LOG_DEBUG("FFT'd hires boxes.");

#pragma omp parallel shared(boxes,HIRES_box) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->DIM; i++){
            for (j=0; j<user_params->DIM; j++){
                for (k=0; k<user_params->DIM; k++){
                    *((float *)boxes->hires_density + R_INDEX(i,j,k)) = *((float *)HIRES_box + R_FFT_INDEX(i,j,k))/VOLUME;
                }
            }
        }
    }

    printf("boxes->hires_density = %e \n", boxes->hires_density[R_INDEX(0,0,0)]);
    // Throw(TableGenerationError);

    // *** If required, let's also create a lower-resolution version of the density field  *** //
    memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);


    // Only filter if we are perturbing on the low-resolution grid
    if(!user_params->PERTURB_ON_HIGH_RES) {
        if (user_params->DIM != user_params->HII_DIM) {
            filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
        }

        // FFT back to real space
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);

        // Renormalise the FFT'd box (sample the high-res box if we are perturbing on the low-res grid)
#pragma omp parallel shared(boxes,HIRES_box,f_pixel_factor) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        boxes->lowres_density[HII_R_INDEX(i,j,k)] =
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
                    }
                }
            }
        }
    }


    // ******* Relative Velocity part ******* //
  if(user_params->USE_RELATIVE_VELOCITIES){
    //JBM: We use the memory allocated to HIRES_box as it's free.

      for(ii=0;ii<3;ii++) {

        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

#pragma omp parallel shared(HIRES_box,ii) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,p,p_vcb) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (n_x=0; n_x<user_params->DIM; n_x++){
                if (n_x>MIDDLE)
                    k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                else
                    k_x = n_x * DELTA_K;

                for (n_y=0; n_y<user_params->DIM; n_y++){
                    if (n_y>MIDDLE)
                        k_y =(n_y-user_params->DIM) * DELTA_K;
                    else
                        k_y = n_y * DELTA_K;

                    for (n_z=0; n_z<=MIDDLE; n_z++){
                        k_z = n_z * DELTA_K;

                        k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                        p = power_in_k(k_mag);
                        p_vcb = power_in_vcb(k_mag);


                        // now set the velocities
                        if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                            HIRES_box[0] = 0;
                        }
                        else{
                            if(ii==0) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= I * k_x/k_mag * sqrt(p_vcb/p) * C_KMS;
                            }
                            if(ii==1) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= I * k_y/k_mag * sqrt(p_vcb/p) * C_KMS;
                            }
                            if(ii==2) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= I * k_z/k_mag * sqrt(p_vcb/p) * C_KMS;
                            }
                        }
                    }
                }
            }
        }


//we only care about the lowres vcb box, so we filter it directly.
      if (user_params->DIM != user_params->HII_DIM) {
          filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
      }

//fft each velocity component back to real space
      dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);



      #pragma omp parallel shared(boxes,HIRES_box,f_pixel_factor,ii) private(i,j,k,vcb_i) num_threads(user_params->N_THREADS)
              {
      #pragma omp for
                  for (i=0; i<user_params->HII_DIM; i++){
                      for (j=0; j<user_params->HII_DIM; j++){
                          for (k=0; k<user_params->HII_DIM; k++){
                            vcb_i = *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                             (unsigned long long)(j*f_pixel_factor+0.5),
                                                             (unsigned long long)(k*f_pixel_factor+0.5)));
                            boxes->lowres_vcb[HII_R_INDEX(i,j,k)] += vcb_i*vcb_i;
                          }
                      }
                  }
              }


    }


//now we take the sqrt of that and normalize the FFT
    for (i=0; i<user_params->HII_DIM; i++){
        for (j=0; j<user_params->HII_DIM; j++){
            for (k=0; k<user_params->HII_DIM; k++){
              boxes->lowres_vcb[HII_R_INDEX(i,j,k)] = sqrt(boxes->lowres_vcb[HII_R_INDEX(i,j,k)])/VOLUME;
            }
        }
    }

  }
    LOG_DEBUG("Completed Relative velocities.");
    // ******* End of Relative Velocity part ******* //

//*************************************************************************************************************************************************************
//*************************************************************************************************************************************************************
//********************************************************** Beginning of SDM new boxes section ***************************************************************
//*************************************************************************************************************************************************************
//*************************************************************************************************************************************************************

// JordanFlitter: In case of SDM, and we are interested in the fluctuations, we create new initial lowres boxes for x_e, T_k and T_chi (at redshift Z_HIGH_MAX).
// I follow the v_cb box generation. Note that HIRES_box is a meaningless box that is used both in real space and k-space. HIRES_box_saved on the other hand is
// the high resolution box of density at k-space. Therefore, the remaining parts of the original code shouldn't be affected by this new section.

//*************************************************************************************************************************************************************
//************************************************************* x_e, T_k, T_chi at Z_HIGH_MAX *****************************************************************
//*************************************************************************************************************************************************************

if(user_params->SCATTERING_DM && user_params->USE_SDM_FLUCTS){
    for(ii=2;ii<5;ii++) {
        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        #pragma omp parallel shared(HIRES_box,ii) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,p,p_SDM) num_threads(user_params->N_THREADS)
                {
        #pragma omp for
                    for (n_x=0; n_x<user_params->DIM; n_x++){
                        if (n_x>MIDDLE)
                            k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                        else
                            k_x = n_x * DELTA_K;

                        for (n_y=0; n_y<user_params->DIM; n_y++){
                            if (n_y>MIDDLE)
                                k_y =(n_y-user_params->DIM) * DELTA_K;
                            else
                                k_y = n_y * DELTA_K;

                            for (n_z=0; n_z<=MIDDLE; n_z++){
                                k_z = n_z * DELTA_K;

                                k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                                p = power_in_k(k_mag);
                                p_SDM = power_in_SDM(k_mag,ii);


                                // now set the "velocities"
                                if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                                    HIRES_box[0] = 0;
                                }
                                else{
                                    HIRES_box[C_INDEX(n_x,n_y,n_z)] *= sqrt(p_SDM/p);
                                }
                            }
                        }
                    }
                }
            //we only care about the lowres boxes, so we filter them directly.
            if (user_params->DIM != user_params->HII_DIM) {
                filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
            }

            //fft each velocity component back to real space
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);

            #pragma omp parallel shared(boxes,HIRES_box,f_pixel_factor,ii) private(i,j,k,vcb_i) num_threads(user_params->N_THREADS)
                    {
            #pragma omp for
                        for (i=0; i<user_params->HII_DIM; i++){
                            for (j=0; j<user_params->HII_DIM; j++){
                                for (k=0; k<user_params->HII_DIM; k++){
                                  delta_SDM_i = *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                                   (unsigned long long)(j*f_pixel_factor+0.5),
                                                                   (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
                                  if (ii == 2) {
                                      boxes->lowres_xe_zhigh[HII_R_INDEX(i,j,k)] = delta_SDM_i;
                                  }
                                  else if (ii == 3) {
                                      boxes->lowres_Tk_zhigh[HII_R_INDEX(i,j,k)] = delta_SDM_i;
                                  }
                                  else if (ii == 4) {
                                      boxes->lowres_Tchi_zhigh[HII_R_INDEX(i,j,k)] = delta_SDM_i;
                                  }
                                }
                            }
                        }
                    }

    }
}

//*************************************************************************************************************************************************************
//***************************************************************** V_chi_b at Z_HIGH_MAX *********************************************************************
//*************************************************************************************************************************************************************

//JordanFlitter: We repeat the v_cb algorithm for V_chi_b (at kinematic decoupling)
if(user_params->SCATTERING_DM && user_params->USE_SDM_FLUCTS){
    for(ii=0;ii<3;ii++) {

      memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

    #pragma omp parallel shared(HIRES_box,ii) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,p,p_vcb) num_threads(user_params->N_THREADS)
      {
    #pragma omp for
          for (n_x=0; n_x<user_params->DIM; n_x++){
              if (n_x>MIDDLE)
                  k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
              else
                  k_x = n_x * DELTA_K;

              for (n_y=0; n_y<user_params->DIM; n_y++){
                  if (n_y>MIDDLE)
                      k_y =(n_y-user_params->DIM) * DELTA_K;
                  else
                      k_y = n_y * DELTA_K;

                  for (n_z=0; n_z<=MIDDLE; n_z++){
                      k_z = n_z * DELTA_K;

                      k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                      p = power_in_k(k_mag);
                      p_vcb = power_in_SDM(k_mag,5);


                      // now set the velocities
                      if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                          HIRES_box[0] = 0;
                      }
                      else{
                          if(ii==0) {
                              HIRES_box[C_INDEX(n_x,n_y,n_z)] *= I * k_x/k_mag * sqrt(p_vcb/p) * C_KMS;
                          }
                          if(ii==1) {
                              HIRES_box[C_INDEX(n_x,n_y,n_z)] *= I * k_y/k_mag * sqrt(p_vcb/p) * C_KMS;
                          }
                          if(ii==2) {
                              HIRES_box[C_INDEX(n_x,n_y,n_z)] *= I * k_z/k_mag * sqrt(p_vcb/p) * C_KMS;
                          }
                      }
                  }
              }
          }
      }


    //we only care about the lowres vcb box, so we filter it directly.
    if (user_params->DIM != user_params->HII_DIM) {
        filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
    }

    //fft each velocity component back to real space
    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);



    #pragma omp parallel shared(boxes,HIRES_box,f_pixel_factor,ii) private(i,j,k,vcb_i) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<user_params->HII_DIM; k++){
                          vcb_i = *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)));
                          boxes->lowres_V_chi_b_zhigh[HII_R_INDEX(i,j,k)] += vcb_i*vcb_i;
                        }
                    }
                }
            }


    }


    //now we take the sqrt of that and normalize the FFT
    for (i=0; i<user_params->HII_DIM; i++){
      for (j=0; j<user_params->HII_DIM; j++){
          for (k=0; k<user_params->HII_DIM; k++){
            boxes->lowres_V_chi_b_zhigh[HII_R_INDEX(i,j,k)] = sqrt(boxes->lowres_V_chi_b_zhigh[HII_R_INDEX(i,j,k)])/VOLUME;
          }
      }
    }
}

//*************************************************************************************************************************************************************
//*************************************************************************************************************************************************************
//************************************************************ Ending of SDM new boxes section ****************************************************************
//*************************************************************************************************************************************************************
//*************************************************************************************************************************************************************

    // Now look at the velocities

    for(ii=0;ii<3;ii++) {

        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        // Now let's set the velocity field/dD/dt (in comoving Mpc)

#pragma omp parallel shared(HIRES_box,ii) private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (n_x=0; n_x<user_params->DIM; n_x++){
                if (n_x>MIDDLE)
                    k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                else
                    k_x = n_x * DELTA_K;

                for (n_y=0; n_y<user_params->DIM; n_y++){
                    if (n_y>MIDDLE)
                        k_y =(n_y-user_params->DIM) * DELTA_K;
                    else
                        k_y = n_y * DELTA_K;

                    for (n_z=0; n_z<=MIDDLE; n_z++){
                        k_z = n_z * DELTA_K;

                        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                        // now set the velocities
                        if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                            HIRES_box[0] = 0;
                        }
                        else{
                            if(ii==0) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq/VOLUME;
                            }
                            if(ii==1) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq/VOLUME;
                            }
                            if(ii==2) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq/VOLUME;
                            }
                        }
                    }
                }
            }
        }

        // Filter only if we require perturbing on the low-res grid
        if(!user_params->PERTURB_ON_HIGH_RES) {
            if (user_params->DIM != user_params->HII_DIM) {
                filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
            }
        }

        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);

        // now sample to lower res
        // now sample the filtered box
#pragma omp parallel shared(boxes,HIRES_box,f_pixel_factor,ii,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<dimension; k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            if(ii==0) {
                                boxes->hires_vx[R_INDEX(i,j,k)] =
                                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                                   (unsigned long long)(j),
                                                                   (unsigned long long)(k)));
                            }
                            if(ii==1) {
                                boxes->hires_vy[R_INDEX(i,j,k)] =
                                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                                   (unsigned long long)(j),
                                                                   (unsigned long long)(k)));
                            }
                            if(ii==2) {
                                boxes->hires_vz[R_INDEX(i,j,k)] =
                                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                                   (unsigned long long)(j),
                                                                   (unsigned long long)(k)));
                            }
                        }
                        else {
                            if(ii==0) {
                                boxes->lowres_vx[HII_R_INDEX(i,j,k)] =
                                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                                   (unsigned long long)(j*f_pixel_factor+0.5),
                                                                   (unsigned long long)(k*f_pixel_factor+0.5)));
                            }
                            if(ii==1) {
                                boxes->lowres_vy[HII_R_INDEX(i,j,k)] =
                                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                                   (unsigned long long)(j*f_pixel_factor+0.5),
                                                                   (unsigned long long)(k*f_pixel_factor+0.5)));
                            }
                            if(ii==2) {
                                boxes->lowres_vz[HII_R_INDEX(i,j,k)] =
                                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                                   (unsigned long long)(j*f_pixel_factor+0.5),
                                                                   (unsigned long long)(k*f_pixel_factor+0.5)));
                            }
                        }
                    }
                }
            }
        }
    }

    LOG_DEBUG("Done Inverse FT.");

    // * *************************************************** * //
    // *              BEGIN 2LPT PART                        * //
    // * *************************************************** * //

    // Generation of the second order Lagrangian perturbation theory (2LPT) corrections to the ZA
    // reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D

    // Parameter set in ANAL_PARAMS.H
    if(user_params->USE_2LPT){

        // use six supplementary boxes to store the gradients of phi_1 (eq. D13b)
        // Allocating the boxes
#define PHI_INDEX(i, j) ((int) ((i) - (j)) + 3*((j)) - ((int)(j))/2  )
        // ij -> INDEX
        // 00 -> 0
        // 11 -> 3
        // 22 -> 5
        // 10 -> 1
        // 20 -> 2
        // 21 -> 4

        fftwf_complex *phi_1 = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // First generate the ii,jj phi_1 boxes

        int phi_component;

        float component_ii,component_jj,component_ij;

        // Indexing for the various phy components
        int phi_directions[3][2] = {{0,1},{0,2},{1,2}};

#pragma omp parallel shared(HIRES_box,phi_1) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->DIM; i++){
                for (j=0; j<user_params->DIM; j++){
                    for (k=0; k<user_params->DIM; k++){
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                           (unsigned long long)(j),
                                                           (unsigned long long)(k)) ) = 0.;
                    }
                }
            }
        }

        // First iterate over the i = j components to phi
        // We'll also save these temporarily to the hires_vi_2LPT boxes which will get
        // overwritten later with the correct 2LPT velocities
        for(phi_component=0;phi_component<3;phi_component++) {

            i = j = phi_component;

                // generate the phi_1 boxes in Fourier transform
#pragma omp parallel shared(HIRES_box,phi_1,i,j) private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq,k) num_threads(user_params->N_THREADS)
                {
#pragma omp for
                for (n_x=0; n_x<user_params->DIM; n_x++){
                    if (n_x>MIDDLE)
                        k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                    else
                        k_x = n_x * DELTA_K;

                    for (n_y=0; n_y<user_params->DIM; n_y++){
                        if (n_y>MIDDLE)
                            k_y =(n_y-user_params->DIM) * DELTA_K;
                        else
                            k_y = n_y * DELTA_K;

                        for (n_z=0; n_z<=MIDDLE; n_z++){
                            k_z = n_z * DELTA_K;

                            k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                            float k[] = {k_x, k_y, k_z};
                            // now set the velocities
                            if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                                phi_1[0] = 0;
                            }
                            else{
                                phi_1[C_INDEX(n_x,n_y,n_z)] = -k[i]*k[j]*HIRES_box_saved[C_INDEX(n_x, n_y, n_z)]/k_sq/VOLUME;
                                // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                            }
                        }
                    }
                }
            }

            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, phi_1);

            // Temporarily store in the allocated hires_vi_2LPT boxes
#pragma omp parallel shared(boxes,phi_1,phi_component) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->DIM; i++){
                    for (j=0; j<user_params->DIM; j++){
                        for (k=0; k<user_params->DIM; k++){
                            if(phi_component==0) {
                                boxes->hires_vx_2LPT[R_INDEX(i,j,k)] = *((float *)phi_1 + R_FFT_INDEX((unsigned long long)(i),
                                                                                                      (unsigned long long)(j),
                                                                                                      (unsigned long long)(k)));
                            }
                            if(phi_component==1) {
                                boxes->hires_vy_2LPT[R_INDEX(i,j,k)] = *((float *)phi_1 + R_FFT_INDEX((unsigned long long)(i),
                                                                                                      (unsigned long long)(j),
                                                                                                      (unsigned long long)(k)));
                            }
                            if(phi_component==2) {
                                boxes->hires_vz_2LPT[R_INDEX(i,j,k)] = *((float *)phi_1 + R_FFT_INDEX((unsigned long long)(i),
                                                                                                      (unsigned long long)(j),
                                                                                                      (unsigned long long)(k)));
                            }
                        }
                    }
                }
            }
        }

        for(phi_component=0;phi_component<3;phi_component++) {
            // Now calculate the cross components and start evaluating the 2LPT field
            i = phi_directions[phi_component][0];
            j = phi_directions[phi_component][1];

            // generate the phi_1 boxes in Fourier transform
#pragma omp parallel shared(HIRES_box,phi_1) private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (n_x=0; n_x<user_params->DIM; n_x++){
                    if (n_x>MIDDLE)
                        k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                    else
                        k_x = n_x * DELTA_K;

                    for (n_y=0; n_y<user_params->DIM; n_y++){
                        if (n_y>MIDDLE)
                            k_y =(n_y-user_params->DIM) * DELTA_K;
                        else
                            k_y = n_y * DELTA_K;

                        for (n_z=0; n_z<=MIDDLE; n_z++){
                            k_z = n_z * DELTA_K;

                            k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                            float k[] = {k_x, k_y, k_z};
                            // now set the velocities
                            if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                                phi_1[0] = 0;
                            }
                            else{
                                phi_1[C_INDEX(n_x,n_y,n_z)] = -k[i]*k[j]*HIRES_box_saved[C_INDEX(n_x, n_y, n_z)]/k_sq/VOLUME;
                                // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                            }
                        }
                    }
                }
            }

            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, phi_1);

            // Then we will have the laplacian of phi_2 (eq. D13b)
            // After that we have to return in Fourier space and generate the Fourier transform of phi_2
#pragma omp parallel shared(HIRES_box,phi_1,phi_component) private(i,j,k,component_ii,component_jj,component_ij) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->DIM; i++){
                    for (j=0; j<user_params->DIM; j++){
                        for (k=0; k<user_params->DIM; k++){
                            // Note, I have temporarily stored the components into other arrays to minimise memory usage
                            // phi - {0, 1, 2} -> {hires_vx_2LPT, hires_vy_2LPT, hires_vz_2LPT}
                            // This may be opaque to the user, but this shouldn't need modification
                            if(phi_component==0) {
                                component_ii = boxes->hires_vx_2LPT[R_INDEX(i,j,k)];
                                component_jj = boxes->hires_vy_2LPT[R_INDEX(i,j,k)];
                                component_ij = *((float *)phi_1 + R_FFT_INDEX((unsigned long long)(i),
                                                                              (unsigned long long)(j),
                                                                              (unsigned long long)(k)));
                            }
                            if(phi_component==1) {
                                component_ii = boxes->hires_vx_2LPT[R_INDEX(i,j,k)];
                                component_jj = boxes->hires_vz_2LPT[R_INDEX(i,j,k)];
                                component_ij = *((float *)phi_1 + R_FFT_INDEX((unsigned long long)(i),
                                                                              (unsigned long long)(j),
                                                                              (unsigned long long)(k)));
                            }
                            if(phi_component==2) {
                                component_ii = boxes->hires_vy_2LPT[R_INDEX(i,j,k)];
                                component_jj = boxes->hires_vz_2LPT[R_INDEX(i,j,k)];
                                component_ij = *((float *)phi_1 + R_FFT_INDEX((unsigned long long)(i),
                                                                              (unsigned long long)(j),
                                                                              (unsigned long long)(k)));
                            }

                            // Kept in this form to maintain similar (possible) rounding errors
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                               (unsigned long long)(j),
                                                               (unsigned long long)(k)) ) += \
                            ( component_ii * component_jj );

                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                               (unsigned long long)(j),
                                                               (unsigned long long)(k)) ) -= \
                            ( component_ij * component_ij );
                        }
                    }
                }
            }
        }

#pragma omp parallel shared(HIRES_box,phi_1) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->DIM; i++){
                for (j=0; j<user_params->DIM; j++){
                    for (k=0; k<user_params->DIM; k++){
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) /= TOT_NUM_PIXELS;
                    }
                }
            }
        }

        // Perform FFTs
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);

        memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // Now we can store the content of box in a back-up array
        // Then we can generate the gradients of phi_2 (eq. D13b and D9)

        // ***** Store back-up k-box RHS eq. D13b ***** //

        // For each component, we generate the velocity field (same as the ZA part)

        // Now let's set the velocity field/dD/dt (in comoving Mpc)

        // read in the box
        // TODO correct free of phi_1

        for(ii=0;ii<3;ii++) {

            if(ii>0) {
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
            }

#pragma omp parallel shared(HIRES_box,ii) private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq) num_threads(user_params->N_THREADS)
            {
#pragma omp for
            // set velocities/dD/dt
                for (n_x=0; n_x<user_params->DIM; n_x++){
                    if (n_x>MIDDLE)
                        k_x =(n_x-user_params->DIM) * DELTA_K;  // wrap around for FFT convention
                    else
                        k_x = n_x * DELTA_K;

                    for (n_y=0; n_y<user_params->DIM; n_y++){
                        if (n_y>MIDDLE)
                            k_y =(n_y-user_params->DIM) * DELTA_K;
                        else
                            k_y = n_y * DELTA_K;

                        for (n_z=0; n_z<=MIDDLE; n_z++){
                            k_z = n_z * DELTA_K;

                            k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                            // now set the velocities
                            if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                                HIRES_box[0] = 0;
                            }
                            else{
                                if(ii==0) {
                                    HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq;
                                }
                                if(ii==1) {
                                    HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq;
                                }
                                if(ii==2) {
                                    HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq;
                                }
                            }
                        }
                        // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                    }
                }
            }

            // Filter only if we require perturbing on the low-res grid
            if(!user_params->PERTURB_ON_HIGH_RES) {
                if (user_params->DIM != user_params->HII_DIM) {
                    filter_box(HIRES_box, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
                }
            }

            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_box);

            // now sample to lower res
            // now sample the filtered box
#pragma omp parallel shared(boxes,HIRES_box,f_pixel_factor,ii,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                if(ii==0) {
                                    boxes->hires_vx_2LPT[R_INDEX(i,j,k)] =
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                                       (unsigned long long)(j),
                                                                       (unsigned long long)(k)));
                                }
                                if(ii==1) {
                                    boxes->hires_vy_2LPT[R_INDEX(i,j,k)] =
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                                       (unsigned long long)(j),
                                                                       (unsigned long long)(k)));
                                }
                                if(ii==2) {
                                    boxes->hires_vz_2LPT[R_INDEX(i,j,k)] =
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),
                                                                       (unsigned long long)(j),
                                                                       (unsigned long long)(k)));
                                }
                            }
                            else {
                                if(ii==0) {
                                    boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] =
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                                       (unsigned long long)(j*f_pixel_factor+0.5),
                                                                       (unsigned long long)(k*f_pixel_factor+0.5)));
                                }
                                if(ii==1) {
                                    boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] =
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                                       (unsigned long long)(j*f_pixel_factor+0.5),
                                                                       (unsigned long long)(k*f_pixel_factor+0.5)));
                                }
                                if(ii==2) {
                                    boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] =
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                                       (unsigned long long)(j*f_pixel_factor+0.5),
                                                                       (unsigned long long)(k*f_pixel_factor+0.5)));
                                }
                            }
                        }
                    }
                }
            }
        }

        // deallocate the supplementary boxes
        fftwf_free(phi_1);

    }
    LOG_DEBUG("Done 2LPT.");

    // * *********************************************** * //
    // *               END 2LPT PART                     * //
    // * *********************************************** * //
    fftwf_cleanup_threads();
    fftwf_cleanup();
    fftwf_forget_wisdom();

    // deallocate
    fftwf_free(HIRES_box);
    fftwf_free(HIRES_box_saved);
    // !!! SLKF: free memory
    fftwf_free(HIRES_potential_box);
    fftwf_free(HIRES_NGpotential_box_saved);

    free_ps();

    for (i=0; i<user_params->N_THREADS; i++) {
        gsl_rng_free (r[i]);
    }
    gsl_rng_free(rseed);
    LOG_DEBUG("Cleaned Up.");
    } // End of Try{}

    Catch(status){
        return(status);
    }
    return(0);
}
