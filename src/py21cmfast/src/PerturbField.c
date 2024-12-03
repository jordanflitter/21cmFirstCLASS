// JordanFlitter: this function performs FFT on the input box, multiplies it by a scale-dependent growth factor in Fourier space, and returns the box in real space after IFFT
/* Flags meaning:
    HIRES_FLAG - 1 if the input box is in high resolution, 0 otherwise (required for proper indexing)
    DENSITY_FLAG - if the input box is a density box or a velocity box.
                If True, the box is multiplied by D(k,z0) where z0 is global_params.INITIAL_REDSHIFT
                If False, the box is multiplied by (D(k,z)-D(k,z0))/L  where L is the box size
    SDGF_TYPE -
                0 means the scale-independent growth factor (SIGF) - D(z)
                1 means CDM scale dependent growth factor (SDGF) - D_c(k,z)
                2 means baryons scale dependent growth factor (SDGF) - D_b(k,z)
                3 means SDM scale dependent growth factor (SDGF) - D_chi(k,z)
    TWOLPT_FLAG - if the required manipulation is for 2LPT. Only relevant for velocities, the SIGF is used: the box is multiplied by -3/7*(D^2(z)-D^2(z0))
    MULT_DIV_FLAG - if positive, we want to multiply by the SDGF, otherwise we divide by it (in order to restore the original input box at the end of the calculation)
*/
void multiply_in_Fourier_space(float *box, struct fftwf_complex *FFT_dummy_box, struct UserParams *user_params, float redshift, bool HIRES_FLAG, bool DENSITY_FLAG, int SDGF_TYPE, bool TWOLPT_FLAG, int MULT_DIV_FLAG) {
      int i, j, k, n_x, n_y, n_z, dimension, switch_mid;
      float k_x, k_y, k_z, k_mag, growth_factor;

      switch(HIRES_FLAG) {
          case 0:
              dimension = user_params->HII_DIM;
              switch_mid = HII_MIDDLE;
              break;
          case 1:
              dimension = user_params->DIM;
              switch_mid = MIDDLE;
              break;
      }
      // First, we copy the content of the input box to FFT_dummy_box
      #pragma omp parallel shared(box,FFT_dummy_box,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
              {
      #pragma omp for
                  for (i=0; i<dimension; i++){
                      for (j=0; j<dimension; j++){
                          for (k=0; k<dimension; k++){
                              if(HIRES_FLAG) {
                                  *((float *)FFT_dummy_box + R_FFT_INDEX(i,j,k)) = box[R_INDEX(i,j,k)];
                              }
                              else {
                                  *((float *)FFT_dummy_box + HII_R_FFT_INDEX(i,j,k)) = box[HII_R_INDEX(i,j,k)];
                              }
                          }
                      }
                  }
              }
      // We transform FFT_dummy_box to Fourier space
      dft_r2c_cube(user_params->USE_FFTW_WISDOM, dimension, user_params->N_THREADS, FFT_dummy_box);

      // We multiply by the SDGF in Fourier space. We also divide by the total number of voxels because of FFT convention
      #pragma omp parallel shared(redshift,FFT_dummy_box,dimension,switch_mid) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,growth_factor) num_threads(user_params->N_THREADS)
              {
      #pragma omp for
                  for (n_x=0; n_x<dimension; n_x++){
                      if (n_x>switch_mid)
                          k_x =(n_x-dimension) * DELTA_K;  // wrap around for FFT convention
                      else
                          k_x = n_x * DELTA_K;
                      for (n_y=0; n_y<dimension; n_y++){
                          if (n_y>switch_mid)
                              k_y =(n_y-dimension) * DELTA_K;
                          else
                              k_y = n_y * DELTA_K;
                          for (n_z=0; n_z<=switch_mid; n_z++){
                              k_z = n_z * DELTA_K;
                              k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                              if (TWOLPT_FLAG) {
                                  growth_factor = -3./7.*(pow(dicke(redshift),2)-pow(dicke(global_params.INITIAL_REDSHIFT),2))/user_params->BOX_LEN;
                              }
                              else {
                                  if (DENSITY_FLAG) {
                                      switch(SDGF_TYPE) {
                                          case 0:
                                              growth_factor = dicke(global_params.INITIAL_REDSHIFT);
                                              break;
                                          case 1:
                                              growth_factor = SDGF_CDM(global_params.INITIAL_REDSHIFT,k_mag,0);
                                              break;
                                          case 2:
                                              growth_factor = SDGF_BARYONS(global_params.INITIAL_REDSHIFT,k_mag,0);
                                              break;
                                          case 3:
                                              growth_factor = SDGF_SDM(global_params.INITIAL_REDSHIFT,k_mag,0);
                                              break;
                                      }
                                  }
                                  else {
                                      switch(SDGF_TYPE) {
                                          case 0:
                                              growth_factor = (dicke(redshift)-dicke(global_params.INITIAL_REDSHIFT))/user_params->BOX_LEN;
                                              break;
                                          case 1:
                                              growth_factor = (SDGF_CDM(redshift,k_mag,0)-SDGF_CDM(global_params.INITIAL_REDSHIFT,k_mag,0))/user_params->BOX_LEN;
                                              break;
                                          case 2:
                                              growth_factor = (SDGF_BARYONS(redshift,k_mag,0)-SDGF_BARYONS(global_params.INITIAL_REDSHIFT,k_mag,0))/user_params->BOX_LEN;
                                              break;
                                          case 3:
                                              growth_factor = (SDGF_SDM(redshift,k_mag,0)-SDGF_SDM(global_params.INITIAL_REDSHIFT,k_mag,0))/user_params->BOX_LEN;
                                              break;
                                      }
                                  }
                              }

                              if (MULT_DIV_FLAG < 0){
                                  growth_factor = 1./growth_factor;
                              }

                              if(HIRES_FLAG) {
                                  *((fftwf_complex *)FFT_dummy_box + C_INDEX(n_x,n_y,n_z)) *= growth_factor/TOT_NUM_PIXELS;
                              }
                              else {
                                  *((fftwf_complex *)FFT_dummy_box + HII_C_INDEX(n_x,n_y,n_z)) *= growth_factor/HII_TOT_NUM_PIXELS;
                              }
                          }
                      }
                  }
              }
    // We transform FFT_dummy_box back to real space
    dft_c2r_cube(user_params->USE_FFTW_WISDOM, dimension, user_params->N_THREADS, FFT_dummy_box);

    // Finally, we copy the content of FFT_dummy_box to the input box
    #pragma omp parallel shared(box,FFT_dummy_box,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(HIRES_FLAG) {
                                *((float *)box + R_INDEX(i,j,k)) = *((float *)FFT_dummy_box + R_FFT_INDEX(i,j,k));
                                if (DENSITY_FLAG && *((float *)box + R_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                    *((float *)box + R_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                }
                            }
                            else {
                                *((float *)box + HII_R_INDEX(i,j,k)) = *((float *)FFT_dummy_box + HII_R_FFT_INDEX(i,j,k));
                                if (DENSITY_FLAG && *((float *)box + HII_R_INDEX(i,j,k)) <= -1){ // correct for aliasing in the filtering step
                                    *((float *)box + HII_R_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                                }
                            }
                        }
                    }
                }
            }
}


// Re-write of perturb_field.c for being accessible within the MCMC
int ComputePerturbField(
    float redshift, struct UserParams *user_params, struct CosmoParams *cosmo_params,
    struct InitialConditions *boxes, struct PerturbedField *perturbed_field
){
    /*
     ComputePerturbField uses the first-order Langragian displacement field to move the
     masses in the cells of the density field. The high-res density field is extrapolated
     to some high-redshift (global_params.INITIAL_REDSHIFT), then uses the zeldovich
     approximation to move the grid "particles" onto the lower-res grid we use for the
     maps. Then we recalculate the velocity fields on the perturbed grid.
    */

    int status;
    Try{  // This Try{} wraps the whole function, so we don't indent.

    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    omp_set_num_threads(user_params->N_THREADS);

    fftwf_complex *HIRES_density_perturb, *HIRES_density_perturb_saved;
    fftwf_complex *LOWRES_density_perturb, *LOWRES_density_perturb_saved;

    float growth_factor, displacement_factor_2LPT, init_growth_factor, init_displacement_factor_2LPT, xf, yf, zf;
    float mass_factor, dDdt, f_pixel_factor, velocity_displacement_factor, velocity_displacement_factor_2LPT;
    unsigned long long ct, HII_i, HII_j, HII_k;
    int i,j,k, xi, yi, zi, dimension, switch_mid;
    double ave_delta, new_ave_delta;
    // JordanFlitter: Variables to perform cloud in cell re-distribution of mass for the perturbed field
    int xp1,yp1,zp1;
    float d_x,d_y,d_z,t_x,t_y,t_z;
    // JordanFlitter: need new boxes for baryons
    fftwf_complex *FFT_HIRES_dummy_box;
    fftwf_complex *FFT_LOWRES_dummy_box;
    // JordanFlitter: added variables in order to evolve the baryons density field
    float k_mag, growth_factor_sdgf, dDdt_over_D_baryons;
    // JordanFlitter: also moved the following variables up here
    float k_x, k_y, k_z, k_sq, dDdt_over_D;
    int n_x, n_y, n_z;
    // JordanFlitter: new boolean flag
    int EVOLVE_LINEARLY;

    // Function for deciding the dimensions of loops when we could
    // use either the low or high resolution grids.
    switch(user_params->PERTURB_ON_HIGH_RES) {
        case 0:
            dimension = user_params->HII_DIM;
            switch_mid = HII_MIDDLE;
            break;
        case 1:
            dimension = user_params->DIM;
            switch_mid = MIDDLE;
            break;
    }

    // ***************   BEGIN INITIALIZATION   ************************** //

    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((user_params->BOX_LEN > user_params->DIM) && !(global_params.EVOLVE_DENSITY_LINEARLY)){
        LOG_WARNING("Resolution is likely too low for accurate evolved density fields\n \
                It is recommended that you either increase the resolution (DIM/BOX_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }
    // JordanFlitter: we need init_CLASS_GROWTH_FACTOR() if the following conditions are satisfied
    if (!user_params->USE_DICKE_GROWTH_FACTOR || user_params->EVOLVE_BARYONS || user_params->EVOLVE_MATTER) {
        Broadcast_struct_global_UF(user_params,cosmo_params);
        init_CLASS_GROWTH_FACTOR();
    }

    growth_factor = dicke(redshift);
    displacement_factor_2LPT = -(3.0/7.0) * growth_factor*growth_factor; // 2LPT eq. D8

    dDdt = ddickedt(redshift); // time derivative of the growth factor (1/s)
    init_growth_factor = dicke(global_params.INITIAL_REDSHIFT);
    init_displacement_factor_2LPT = -(3.0/7.0) * init_growth_factor*init_growth_factor; // 2LPT eq. D8
    // JordanFlitter: if we evolve the CDM density field, then we set init_growth_factor to be 1, as we already take care of evolving the
    //                CDM density field with the appropriate SDGF (in Fourier space)
    if (user_params->EVOLVE_MATTER){
        init_growth_factor = 1.;
    }
    // JordanFlitter: convenient boolean flag to include linear evolution during the dark ages
    if (global_params.EVOLVE_DENSITY_LINEARLY || (redshift > global_params.REDSHIFT_2LPT && user_params->START_AT_RECOMBINATION)) {
        EVOLVE_LINEARLY = 1;
    }
    else {
        EVOLVE_LINEARLY = 0;
    }

    // find factor of HII pixel size / deltax pixel size
    f_pixel_factor = user_params->DIM/(float)(user_params->HII_DIM);
    mass_factor = pow(f_pixel_factor, 3);

    // allocate memory for the updated density, and initialize
    LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    if(user_params->PERTURB_ON_HIGH_RES) {
        HIRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        HIRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    }
    // JordanFlitter: Need to allocate memory for dummy boxes if we do non-linear scale-dependent evolution
    if (!EVOLVE_LINEARLY && (user_params->EVOLVE_MATTER || user_params->EVOLVE_BARYONS)) {
        FFT_HIRES_dummy_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS); // Need that for the density field
        if (!user_params->PERTURB_ON_HIGH_RES){
            FFT_LOWRES_dummy_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS); // Need that for the velocity field
        }
    }

    double *resampled_box;

    debugSummarizeIC(boxes, user_params->HII_DIM, user_params->DIM);
    LOG_SUPER_DEBUG("growth_factor=%f, displacemet_factor_2LPT=%f, dDdt=%f, init_growth_factor=%f, init_displacement_factor_2LPT=%f, mass_factor=%f",
                    growth_factor, displacement_factor_2LPT, dDdt, init_growth_factor, init_displacement_factor_2LPT, mass_factor);

    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************
    /************************************************************** CDM/MATTER SECTION BEGINS *******************************************************************
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************/
    /* JordanFlitter: General comments regarding the calculations below.
       1) If we evolve the density field linearly, it is the total matter density field.
       2) Otherwise, it is the CDM density field (regardless if the SIGF or SDGF is considered), and we make the right adjustments to convert it
          to the total matter density field.
    */
    // check if the linear evolution flag was set
    // JordanFlitter: Evolve linearly if we are above REDSHIFT_2LPT
    if (EVOLVE_LINEARLY || user_params->LINEAR_DELTA_IN_EPS){

        LOG_DEBUG("Linearly evolve density field");
        // JordanFlitter: in case we evolve the CDM density field, we multiply by the scale-dependent growth factor. This requires FFT'ing the density box
        if (user_params->EVOLVE_MATTER) {
              #pragma omp parallel shared(boxes,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
                      {
              #pragma omp for
                          for (i=0; i<dimension; i++){
                              for (j=0; j<dimension; j++){
                                  for (k=0; k<dimension; k++){
                                      if(user_params->PERTURB_ON_HIGH_RES) {
                                          *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = boxes->hires_density[R_INDEX(i,j,k)];
                                      }
                                      else {
                                          *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = boxes->lowres_density[HII_R_INDEX(i,j,k)];
                                      }
                                  }
                              }
                          }
                      }
              if(user_params->PERTURB_ON_HIGH_RES) {
                  dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);
              }
              else{
                  dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);
              }
              #pragma omp parallel shared(redshift,LOWRES_density_perturb,HIRES_density_perturb,dimension,switch_mid) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,growth_factor_sdgf) num_threads(user_params->N_THREADS)
                      {
              #pragma omp for
                          for (n_x=0; n_x<dimension; n_x++){
                              if (n_x>switch_mid)
                                  k_x =(n_x-dimension) * DELTA_K;  // wrap around for FFT convention
                              else
                                  k_x = n_x * DELTA_K;
                              for (n_y=0; n_y<dimension; n_y++){
                                  if (n_y>switch_mid)
                                      k_y =(n_y-dimension) * DELTA_K;
                                  else
                                      k_y = n_y * DELTA_K;
                                  for (n_z=0; n_z<=switch_mid; n_z++){
                                      k_z = n_z * DELTA_K;
                                      k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                                      growth_factor_sdgf = SDGF_MATTER(redshift,k_mag);

                                      if(user_params->PERTURB_ON_HIGH_RES) {
                                          *((fftwf_complex *)HIRES_density_perturb + C_INDEX(n_x,n_y,n_z)) *= growth_factor_sdgf/TOT_NUM_PIXELS;
                                      }
                                      else {
                                          *((fftwf_complex *)LOWRES_density_perturb + HII_C_INDEX(n_x,n_y,n_z)) *= growth_factor_sdgf/HII_TOT_NUM_PIXELS;
                                      }
                                  }
                              }
                          }
                      }
            if(user_params->PERTURB_ON_HIGH_RES) {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);
            }
            else {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);
            }
        }
        else {
    #pragma omp parallel shared(growth_factor,boxes,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = growth_factor*boxes->hires_density[R_INDEX(i,j,k)];
                            }
                            else {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = growth_factor*boxes->lowres_density[HII_R_INDEX(i,j,k)];
                            }
                        }
                    }
                }
            }
        }
    }
    else {
        // Apply Zel'dovich/2LPT correction
        LOG_DEBUG("Apply Zel'dovich");

#pragma omp parallel shared(LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<dimension; k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = 0.;
                        }
                        else {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                        }

                    }
                }
            }
        }
        // JordanFlitter: we multiply the hires density box by D_c(k,z0) in Fourier space, and we multiply the velocity field by (D_c(k,z)-D_c(k,z0))/L
        if (user_params->EVOLVE_MATTER){
            multiply_in_Fourier_space(boxes->hires_density, FFT_HIRES_dummy_box, user_params, redshift, 1, 1, 1, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG

            if(user_params->PERTURB_ON_HIGH_RES) {
                multiply_in_Fourier_space(boxes->hires_vx, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 1, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vy, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 1, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vz, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 1, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            else{
                multiply_in_Fourier_space(boxes->lowres_vx, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 1, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vy, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 1, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vz, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 1, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
        }
        else {
            velocity_displacement_factor = (growth_factor-init_growth_factor) / user_params->BOX_LEN;

            // now add the missing factor of D
    #pragma omp parallel shared(boxes,velocity_displacement_factor,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx[R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                                boxes->hires_vy[R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                                boxes->hires_vz[R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            }
                            else {
                                boxes->lowres_vx[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                                boxes->lowres_vy[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                                boxes->lowres_vz[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
                            }
                        }
                    }
                }
            }
        }

        // * ************************************************************************* * //
        // *                           BEGIN 2LPT PART                                 * //
        // * ************************************************************************* * //
        // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        if(user_params->USE_2LPT){
            LOG_DEBUG("Apply 2LPT");

            // allocate memory for the velocity boxes and read them in
            velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;

            // now add the missing factor in eq. D9
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->hires_vy_2LPT[R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->hires_vz_2LPT[R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                            }
                            else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                            }
                        }
                    }
                }
            }
        }


        // * ************************************************************************* * //
        // *                            END 2LPT PART                                  * //
        // * ************************************************************************* * //

        // ************  END INITIALIZATION **************************** //

        // Perturbing the density field required adding over multiple cells. Store intermediate result as a double to avoid rounding errors
        if(user_params->PERTURB_ON_HIGH_RES) {
            resampled_box = (double *)calloc(TOT_NUM_PIXELS,sizeof(double));
        }
        else {
            resampled_box = (double *)calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        }

        // go through the high-res box, mapping the mass onto the low-res (updated) box
        // JordanFlitter: added new private variables for cloud-in-cell
        LOG_DEBUG("Perturb the density field");
#pragma omp parallel shared(init_growth_factor,boxes,f_pixel_factor,resampled_box,dimension) \
                        private(i,j,k,xi,xf,yi,yf,zi,zf,HII_i,HII_j,HII_k,d_x,d_y,d_z,t_x,t_y,t_z,xp1,yp1,zp1) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->DIM;i++){
                for (j=0; j<user_params->DIM;j++){
                    for (k=0; k<user_params->DIM;k++){

                        // map indeces to locations in units of box size
                        xf = (i+0.5)/((user_params->DIM)+0.0);
                        yf = (j+0.5)/((user_params->DIM)+0.0);
                        zf = (k+0.5)/((user_params->DIM)+0.0);

                        // update locations
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            xf += (boxes->hires_vx)[R_INDEX(i, j, k)];
                            yf += (boxes->hires_vy)[R_INDEX(i, j, k)];
                            zf += (boxes->hires_vz)[R_INDEX(i, j, k)];
                        }
                        else {
                            HII_i = (unsigned long long)(i/f_pixel_factor);
                            HII_j = (unsigned long long)(j/f_pixel_factor);
                            HII_k = (unsigned long long)(k/f_pixel_factor);
                            xf += (boxes->lowres_vx)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                            yf += (boxes->lowres_vy)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                            zf += (boxes->lowres_vz)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                        }

                        // 2LPT PART
                        // add second order corrections
                        if(user_params->USE_2LPT){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                xf -= (boxes->hires_vx_2LPT)[R_INDEX(i,j,k)];
                                yf -= (boxes->hires_vy_2LPT)[R_INDEX(i,j,k)];
                                zf -= (boxes->hires_vz_2LPT)[R_INDEX(i,j,k)];
                            }
                            else {
                                xf -= (boxes->lowres_vx_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                yf -= (boxes->lowres_vy_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                zf -= (boxes->lowres_vz_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                            }
                        }

                        xf *= (float)(dimension);
                        yf *= (float)(dimension);
                        zf *= (float)(dimension);
                        while (xf >= (float)(dimension)){ xf -= (dimension);}
                        while (xf < 0){ xf += (dimension);}
                        while (yf >= (float)(dimension)){ yf -= (dimension);}
                        while (yf < 0){ yf += (dimension);}
                        while (zf >= (float)(dimension)){ zf -= (dimension);}
                        while (zf < 0){ zf += (dimension);}
                        xi = xf;
                        yi = yf;
                        zi = zf;
                        if (xi >= (dimension)){ xi -= (dimension);}
                        if (xi < 0) {xi += (dimension);}
                        if (yi >= (dimension)){ yi -= (dimension);}
                        if (yi < 0) {yi += (dimension);}
                        if (zi >= (dimension)){ zi -= (dimension);}
                        if (zi < 0) {zi += (dimension);}

                        // JordanFlitter: Added Bradley Greig's new modification (cloud-in-cell)
                        if (user_params->CLOUD_IN_CELL) {
                            // Determine the fraction of the perturbed cell which overlaps with the 8 nearest grid cells,
                            // based on the grid cell which contains the centre of the perturbed cell
                            d_x = fabs(xf - (double)(xi+0.5));
                            d_y = fabs(yf - (double)(yi+0.5));
                            d_z = fabs(zf - (double)(zi+0.5));
                            if(xf < (double)(xi+0.5)) {
                                // If perturbed cell centre is less than the mid-point then update fraction
                                // of mass in the cell and determine the cell centre of neighbour to be the
                                // lowest grid point index
                                d_x = 1. - d_x;
                                xi -= 1;
                                if (xi < 0) {xi += (dimension);} // Only this critera is possible as iterate back by one (we cannot exceed DIM)
                            }
                            if(yf < (double)(yi+0.5)) {
                                d_y = 1. - d_y;
                                yi -= 1;
                                if (yi < 0) {yi += (dimension);}
                            }
                            if(zf < (double)(zi+0.5)) {
                                d_z = 1. - d_z;
                                zi -= 1;
                                if (zi < 0) {zi += (dimension);}
                            }
                            t_x = 1. - d_x;
                            t_y = 1. - d_y;
                            t_z = 1. - d_z;

                            // Determine the grid coordinates of the 8 neighbouring cells
                            // Takes into account the offset based on cell centre determined above
                            xp1 = xi + 1;
                            if(xp1 >= dimension) { xp1 -= (dimension);}
                            yp1 = yi + 1;
                            if(yp1 >= dimension) { yp1 -= (dimension);}
                            zp1 = zi + 1;
                            if(zp1 >= dimension) { zp1 -= (dimension);}

                            if(user_params->PERTURB_ON_HIGH_RES) {
                                // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
    #pragma omp atomic
                                    resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
    #pragma omp atomic
                                    resampled_box[R_INDEX(xp1,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
    #pragma omp atomic
                                    resampled_box[R_INDEX(xi,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
    #pragma omp atomic
                                    resampled_box[R_INDEX(xp1,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
    #pragma omp atomic
                                    resampled_box[R_INDEX(xi,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
    #pragma omp atomic
                                    resampled_box[R_INDEX(xp1,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
    #pragma omp atomic
                                    resampled_box[R_INDEX(xi,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
    #pragma omp atomic
                                    resampled_box[R_INDEX(xp1,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                            }
                            else {
                                // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xp1,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xi,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xp1,yp1,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xi,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xp1,yi,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xi,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
    #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xp1,yp1,zp1)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                            }
                        }
                        else {
                            if(user_params->PERTURB_ON_HIGH_RES) {
    #pragma omp atomic
                                resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)]);
                            }
                            else {
    #pragma omp atomic
                                resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + init_growth_factor*(boxes->hires_density)[R_INDEX(i,j,k)]);
                            }
                        }
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("resampled_box: ");
        debugSummarizeBoxDouble(resampled_box, dimension, "  ");

        // Resample back to a float for remaining algorithm
#pragma omp parallel shared(LOWRES_density_perturb,HIRES_density_perturb,resampled_box,dimension) \
                        private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<dimension; i++){
                for (j=0; j<dimension; j++){
                    for (k=0; k<dimension; k++){
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            *( (float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) = (float)resampled_box[R_INDEX(i,j,k)];
                        }
                        else {
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) = (float)resampled_box[HII_R_INDEX(i,j,k)];
                        }
                    }
                }
            }
        }

        // JordanFlitter: I don't want to free it yet if we do non-linear evolution for baryons
        if (!user_params->EVOLVE_BARYONS){
            free(resampled_box);
        }
        LOG_DEBUG("Finished perturbing the density field");

        LOG_SUPER_DEBUG("density_perturb: ");
        if(user_params->PERTURB_ON_HIGH_RES){
            debugSummarizeBox(HIRES_density_perturb, dimension, "  ");
        }else{
            debugSummarizeBox(LOWRES_density_perturb, dimension, "  ");
        }

        // JordanFlitter: we divide the hires density box by D_c(k,z0) in Fourier space, and we divide the velocity field by (D_c(k,z)-D_c(k,z0))/L
        if (user_params->EVOLVE_MATTER){
            multiply_in_Fourier_space(boxes->hires_density, FFT_HIRES_dummy_box, user_params, redshift, 1, 1, 1, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            if(user_params->PERTURB_ON_HIGH_RES) {
                multiply_in_Fourier_space(boxes->hires_vx, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 1, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vy, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 1, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vz, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 1, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            else{
                multiply_in_Fourier_space(boxes->lowres_vx, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 1, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vy, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 1, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vz, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 1, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
        }
        else {
            // deallocate
    #pragma omp parallel shared(boxes,velocity_displacement_factor,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx[R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                                boxes->hires_vy[R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                                boxes->hires_vz[R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            }
                            else {
                                boxes->lowres_vx[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                                boxes->lowres_vy[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                                boxes->lowres_vz[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor; // convert back to z = 0 quantity
                            }
                        }
                    }
                }
            }
        }
        // JordanFlitter: if we do baryons calculations, we want to use the same v_2LPT for CDM, so there's no need to do the division here
        if(user_params->USE_2LPT && !user_params->EVOLVE_BARYONS){
#pragma omp parallel shared(boxes,velocity_displacement_factor_2LPT,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx_2LPT[R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->hires_vy_2LPT[R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->hires_vz_2LPT[R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                            }
                            else {
                                boxes->lowres_vx_2LPT[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->lowres_vy_2LPT[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                                boxes->lowres_vz_2LPT[HII_R_INDEX(i,j,k)] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                            }
                        }
                    }
                }
            }
        }
        LOG_DEBUG("Cleanup velocities for perturb");

    } // End of non-linear evolution condition

    // Now, if I still have the high resolution density grid (HIRES_density_perturb) I need to downsample it to the low-resolution grid
    if(user_params->PERTURB_ON_HIGH_RES) {

        LOG_DEBUG("Downsample the high-res perturbed density");

        // Transform to Fourier space to sample (filter) the box
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);

        // Need to save a copy of the high-resolution unfiltered density field for the velocities
        memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

        // Now filter the box
        if (user_params->DIM != user_params->HII_DIM) {
            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
        }

        // FFT back to real space
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);

        // Renormalise the FFT'd box
#pragma omp parallel shared(HIRES_density_perturb,LOWRES_density_perturb,f_pixel_factor,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) =
                        *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)))/(float)TOT_NUM_PIXELS;

                        // JordanFlitter: We need to subtract 1 only if we evolve the density field non-linearly (BUG in 21cmFAST)
                        if (!EVOLVE_LINEARLY){
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) -= 1.;
                        }

                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1.+FRACT_FLOAT_ERR;
                        }
                    }
                }
            }
        }
    }
    else {
        // JordanFlitter: Evolve linearly if we are above REDSHIFT_2LPT
        if (!EVOLVE_LINEARLY && !user_params->LINEAR_DELTA_IN_EPS){

#pragma omp parallel shared(LOWRES_density_perturb,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
            {
#pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<user_params->HII_DIM; k++){
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                            *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1.;
                        }
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("LOWRES_density_perturb: ");
    debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

    // transform to k-space
    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);

    //smooth the field
    // JordanFlitter: Evolve linearly if we are above REDSHIFT_2LPT
    if (!EVOLVE_LINEARLY  && !user_params->LINEAR_DELTA_IN_EPS && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
        filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
    }

    LOG_SUPER_DEBUG("LOWRES_density_perturb after smoothing: ");
    debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

    // save a copy of the k-space density field
    memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);

    LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space: ");
    debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

    // normalize after FFT
    int bad_count=0;
#pragma omp parallel shared(LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS) reduction(+: bad_count)
    {
#pragma omp for
        for(i=0; i<user_params->HII_DIM; i++){
            for(j=0; j<user_params->HII_DIM; j++){
                for(k=0; k<user_params->HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;

                    if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1.0) { // shouldn't happen

                        if(bad_count<5) LOG_WARNING("LOWRES_density_perturb is <-1 for index %d %d %d (value=%f)", i,j,k, *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)));
                        if(bad_count==5) LOG_WARNING("Skipping further warnings for LOWRES_density_perturb.");
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                        bad_count++;
                    }
                }
            }
        }
    }
    if(bad_count>=5) LOG_WARNING("Total number of bad indices for LOW_density_perturb: %d", bad_count);
    LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space (normalized): ");
    debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");


#pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (i=0; i<user_params->HII_DIM; i++){
            for (j=0; j<user_params->HII_DIM; j++){
                for (k=0; k<user_params->HII_DIM; k++){
                    *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                    // JordanFlitter: if we evolve baryons, then the above is the CDM density field, and not the total matter density field.
                    //                We thus multiply by Omega_c/Omega_m (we later include the contributions of the baryons to delta_m)
                    // JordanFlitter: if we compute the linear matter density field, there is no need to take only the CDM contribution into account as we evolved
                    //                the total matter field with SDGF_MATTER (or the SIGF)
                    if (user_params->EVOLVE_BARYONS && !(EVOLVE_LINEARLY || user_params->LINEAR_DELTA_IN_EPS)){
                        if (!user_params->SCATTERING_DM){
                            *((float *)perturbed_field->density + HII_R_INDEX(i,j,k))  *= (cosmo_params->OMm-cosmo_params->OMb)/cosmo_params->OMm;
                        }
                        else {
                            *((float *)perturbed_field->density + HII_R_INDEX(i,j,k))  *= (1.-pow(10.,-cosmo_params->f_chi))*(cosmo_params->OMm-cosmo_params->OMb)/cosmo_params->OMm;
                        }
                    }
                }
            }
        }
    }
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************
    /*************************************************************** CDM/MATTER SECTION ENDS ********************************************************************
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************/

    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************
    /*************************************************************** BARYONS SECTION BEGINS *********************************************************************
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************/
    /* JordanFlitter: General comments regarding the calculations below.
       1) To minimize memory, we use LOWRES_density_perturb and HIRES_density_perturb for the baryons density, as the CDM calculation was over
          in the above lines.
       2) We overwrite LOWRES_density_perturb_saved and HIRES_density_perturb_saved, as these boxes are needed for the baryons velocity
       3) For 2LPT calculations, we take the same 2LPT velocity boxes we used for CDM
    */
    if (user_params->EVOLVE_BARYONS) {
        if (EVOLVE_LINEARLY){
            #pragma omp parallel shared(boxes,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
                    {
            #pragma omp for
                        for (i=0; i<dimension; i++){
                            for (j=0; j<dimension; j++){
                                for (k=0; k<dimension; k++){
                                    if(user_params->PERTURB_ON_HIGH_RES) {
                                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = boxes->hires_density[R_INDEX(i,j,k)];
                                    }
                                    else {
                                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = boxes->lowres_density[HII_R_INDEX(i,j,k)];
                                    }
                                }
                            }
                        }
                    }
            if(user_params->PERTURB_ON_HIGH_RES) {
                dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);
            }
            else{
                dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);
            }
            #pragma omp parallel shared(redshift,LOWRES_density_perturb,HIRES_density_perturb,dimension,switch_mid) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,growth_factor_sdgf) num_threads(user_params->N_THREADS)
                    {
            #pragma omp for
                        for (n_x=0; n_x<dimension; n_x++){
                            if (n_x>switch_mid)
                                k_x =(n_x-dimension) * DELTA_K;  // wrap around for FFT convention
                            else
                                k_x = n_x * DELTA_K;
                            for (n_y=0; n_y<dimension; n_y++){
                                if (n_y>switch_mid)
                                    k_y =(n_y-dimension) * DELTA_K;
                                else
                                    k_y = n_y * DELTA_K;
                                for (n_z=0; n_z<=switch_mid; n_z++){
                                    k_z = n_z * DELTA_K;
                                    k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                                    growth_factor_sdgf = SDGF_BARYONS(redshift,k_mag,0);
                                    if(user_params->PERTURB_ON_HIGH_RES) {
                                        *((fftwf_complex *)HIRES_density_perturb + C_INDEX(n_x,n_y,n_z)) *= growth_factor_sdgf/TOT_NUM_PIXELS;
                                    }
                                    else {
                                        *((fftwf_complex *)LOWRES_density_perturb + HII_C_INDEX(n_x,n_y,n_z)) *= growth_factor_sdgf/HII_TOT_NUM_PIXELS;
                                    }
                                }
                            }
                        }
                    }
            if(user_params->PERTURB_ON_HIGH_RES) {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);
            }
            else {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);
            }
        }
        else { // non-linear evolution
            // Apply Zel'dovich/2LPT correction
            LOG_DEBUG("Apply Zel'dovich");

            // JordanFlitter: wee need to allocate memory for resampled_box if we evolved the matter density field linearly
            if (user_params->LINEAR_DELTA_IN_EPS){
                if(user_params->PERTURB_ON_HIGH_RES) {
                    resampled_box = (double *)calloc(TOT_NUM_PIXELS,sizeof(double));
                }
                else {
                    resampled_box = (double *)calloc(HII_TOT_NUM_PIXELS,sizeof(double));
                }
            }

    #pragma omp parallel shared(resampled_box,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = 0.;
                                *((double *)resampled_box + R_INDEX(i,j,k)) = 0.;
                            }
                            else {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                                *((double *)resampled_box + HII_R_INDEX(i,j,k)) = 0.;
                            }
                        }
                    }
                }
            }

            // JordanFlitter: we multiply the hires density box by D_b(k,z0) in Fourier space, and we multiply the velocity field by (D_b(k,z)-D_b(k,z0))/L
            multiply_in_Fourier_space(boxes->hires_density, FFT_HIRES_dummy_box, user_params, redshift, 1, 1, 2, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG

            if(user_params->PERTURB_ON_HIGH_RES) {
                multiply_in_Fourier_space(boxes->hires_vx, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vy, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vz, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            else{
                multiply_in_Fourier_space(boxes->lowres_vx, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vy, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vz, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            // Note: we use for the barypns calculation the same v_2LPT boxes that were used for CDM. So there is no need to do any multiplication,
            //       unless we linearly evolved the matter density field, in which case the velocity 2LPT vectors were not multiplied, so we have to do the multiplication now!
            if(user_params->USE_2LPT && user_params->LINEAR_DELTA_IN_EPS){
                if(user_params->PERTURB_ON_HIGH_RES) {
                    multiply_in_Fourier_space(boxes->hires_vx_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vy_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vz_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
                else{
                    multiply_in_Fourier_space(boxes->lowres_vx_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vy_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vz_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
            }

            // ************  END INITIALIZATION **************************** //

            // go through the high-res box, mapping the mass onto the low-res (updated) box
            // JordanFlitter: added new private variables for cloud-in-cell
            LOG_DEBUG("Perturb the density field");
    #pragma omp parallel shared(boxes,f_pixel_factor,resampled_box,dimension) \
                            private(i,j,k,xi,xf,yi,yf,zi,zf,HII_i,HII_j,HII_k,d_x,d_y,d_z,t_x,t_y,t_z,xp1,yp1,zp1) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<user_params->DIM;i++){
                    for (j=0; j<user_params->DIM;j++){
                        for (k=0; k<user_params->DIM;k++){

                            // map indeces to locations in units of box size
                            xf = (i+0.5)/((user_params->DIM)+0.0);
                            yf = (j+0.5)/((user_params->DIM)+0.0);
                            zf = (k+0.5)/((user_params->DIM)+0.0);

                            // update locations
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                xf += (boxes->hires_vx)[R_INDEX(i, j, k)];
                                yf += (boxes->hires_vy)[R_INDEX(i, j, k)];
                                zf += (boxes->hires_vz)[R_INDEX(i, j, k)];
                            }
                            else {
                                HII_i = (unsigned long long)(i/f_pixel_factor);
                                HII_j = (unsigned long long)(j/f_pixel_factor);
                                HII_k = (unsigned long long)(k/f_pixel_factor);
                                xf += (boxes->lowres_vx)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                                yf += (boxes->lowres_vy)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                                zf += (boxes->lowres_vz)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                            }

                            // 2LPT PART
                            // add second order corrections
                            if(user_params->USE_2LPT){
                                if(user_params->PERTURB_ON_HIGH_RES) {
                                    xf -= (boxes->hires_vx_2LPT)[R_INDEX(i,j,k)];
                                    yf -= (boxes->hires_vy_2LPT)[R_INDEX(i,j,k)];
                                    zf -= (boxes->hires_vz_2LPT)[R_INDEX(i,j,k)];
                                }
                                else {
                                    xf -= (boxes->lowres_vx_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                    yf -= (boxes->lowres_vy_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                    zf -= (boxes->lowres_vz_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                }
                            }

                            xf *= (float)(dimension);
                            yf *= (float)(dimension);
                            zf *= (float)(dimension);
                            while (xf >= (float)(dimension)){ xf -= (dimension);}
                            while (xf < 0){ xf += (dimension);}
                            while (yf >= (float)(dimension)){ yf -= (dimension);}
                            while (yf < 0){ yf += (dimension);}
                            while (zf >= (float)(dimension)){ zf -= (dimension);}
                            while (zf < 0){ zf += (dimension);}
                            xi = xf;
                            yi = yf;
                            zi = zf;
                            if (xi >= (dimension)){ xi -= (dimension);}
                            if (xi < 0) {xi += (dimension);}
                            if (yi >= (dimension)){ yi -= (dimension);}
                            if (yi < 0) {yi += (dimension);}
                            if (zi >= (dimension)){ zi -= (dimension);}
                            if (zi < 0) {zi += (dimension);}


                            // JordanFlitter: Added Bradley Greig's new modification (cloud-in-cell)
                            if (user_params->CLOUD_IN_CELL) {
                                // Determine the fraction of the perturbed cell which overlaps with the 8 nearest grid cells,
                                // based on the grid cell which contains the centre of the perturbed cell
                                d_x = fabs(xf - (double)(xi+0.5));
                                d_y = fabs(yf - (double)(yi+0.5));
                                d_z = fabs(zf - (double)(zi+0.5));
                                if(xf < (double)(xi+0.5)) {
                                    // If perturbed cell centre is less than the mid-point then update fraction
                                    // of mass in the cell and determine the cell centre of neighbour to be the
                                    // lowest grid point index
                                    d_x = 1. - d_x;
                                    xi -= 1;
                                    if (xi < 0) {xi += (dimension);} // Only this critera is possible as iterate back by one (we cannot exceed DIM)
                                }
                                if(yf < (double)(yi+0.5)) {
                                    d_y = 1. - d_y;
                                    yi -= 1;
                                    if (yi < 0) {yi += (dimension);}
                                }
                                if(zf < (double)(zi+0.5)) {
                                    d_z = 1. - d_z;
                                    zi -= 1;
                                    if (zi < 0) {zi += (dimension);}
                                }
                                t_x = 1. - d_x;
                                t_y = 1. - d_y;
                                t_z = 1. - d_z;

                                // Determine the grid coordinates of the 8 neighbouring cells
                                // Takes into account the offset based on cell centre determined above
                                xp1 = xi + 1;
                                if(xp1 >= dimension) { xp1 -= (dimension);}
                                yp1 = yi + 1;
                                if(yp1 >= dimension) { yp1 -= (dimension);}
                                zp1 = zi + 1;
                                if(zp1 >= dimension) { zp1 -= (dimension);}

                                if(user_params->PERTURB_ON_HIGH_RES) {
                                    // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                                }
                                else {
                                    // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                                }
                            }
                            else {
                                if(user_params->PERTURB_ON_HIGH_RES) {
        #pragma omp atomic
                                    resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)]);
                                }
                                else {
        #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)]);
                                }
                            }
                        }
                    }
                }
            }

            LOG_SUPER_DEBUG("resampled_box: ");
            debugSummarizeBoxDouble(resampled_box, dimension, "  ");

            // Resample back to a float for remaining algorithm
    #pragma omp parallel shared(LOWRES_density_perturb,HIRES_density_perturb,resampled_box,dimension) \
                            private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                *( (float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) = (float)resampled_box[R_INDEX(i,j,k)];
                            }
                            else {
                                *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) = (float)resampled_box[HII_R_INDEX(i,j,k)];
                            }
                        }
                    }
                }
            }
            // JordanFlitter: we free resampled_box only if we don't have SDM
            if (!(user_params->SCATTERING_DM)){
                free(resampled_box);
            }

            LOG_DEBUG("Finished perturbing the density field");

            LOG_SUPER_DEBUG("density_perturb: ");
            if(user_params->PERTURB_ON_HIGH_RES){
                debugSummarizeBox(HIRES_density_perturb, dimension, "  ");
            }else{
                debugSummarizeBox(LOWRES_density_perturb, dimension, "  ");
            }
            // JordanFlitter: we divide the hires density box by D_b(k,z0) in Fourier space, and we divide the velocity field by (D_b(k,z)-D_b(k,z0))/L
            multiply_in_Fourier_space(boxes->hires_density, FFT_HIRES_dummy_box, user_params, redshift, 1, 1, 2, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            if(user_params->PERTURB_ON_HIGH_RES) {
                multiply_in_Fourier_space(boxes->hires_vx, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vy, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vz, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            else{
                multiply_in_Fourier_space(boxes->lowres_vx, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vy, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vz, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            // Note: we restore original input only if we don't have SDM, otherwise we do it later
            if(user_params->USE_2LPT && !user_params->SCATTERING_DM){
                if(user_params->PERTURB_ON_HIGH_RES) {
                    multiply_in_Fourier_space(boxes->hires_vx_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vy_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vz_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 2, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
                else{
                    multiply_in_Fourier_space(boxes->lowres_vx_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vy_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vz_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 2, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
            }

            LOG_DEBUG("Cleanup velocities for perturb");
        }
        // Now, if I still have the high resolution density grid (HIRES_density_perturb) I need to downsample it to the low-resolution grid
        if(user_params->PERTURB_ON_HIGH_RES) {

            LOG_DEBUG("Downsample the high-res perturbed baryons density");

            // Transform to Fourier space to sample (filter) the box
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);

            // Need to save a copy of the high-resolution unfiltered density field for the velocities
            // JordanFlitter: note that the following copy operation overwrites the previous copy operation to HIRES_density_perturb_saved.
            // This is because HIRES_density_perturb_saved is used only for the velocity field calculation.
            // This velocity field is associated with baryons, and so if we EVOLVE_BARYONS we are okay with overwriting the contnet of this box.
            memcpy(HIRES_density_perturb_saved, HIRES_density_perturb, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);

            // Now filter the box
            if (user_params->DIM != user_params->HII_DIM) {
                filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
            }

            // FFT back to real space
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);

            // Renormalise the FFT'd box
    #pragma omp parallel shared(HIRES_density_perturb,LOWRES_density_perturb,f_pixel_factor,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<user_params->HII_DIM; k++){
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) =
                            *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)))/(float)TOT_NUM_PIXELS;

                            // JordanFlitter: We need to subtract 1 only if we evolve the density field non-linearly (BUG in 21cmFAST)
                            if (!EVOLVE_LINEARLY){
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) -= 1.;
                            }

                            if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1.+FRACT_FLOAT_ERR;
                            }
                        }
                    }
                }
            }
        }
        else {
            // JordanFlitter: Evolve linearly if we are above REDSHIFT_2LPT
            if (!EVOLVE_LINEARLY){

    #pragma omp parallel shared(mass_factor,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
                {
    #pragma omp for
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<user_params->HII_DIM; k++){
                                *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                                *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1.;
                            }
                        }
                    }
                }
            }
        }
        LOG_SUPER_DEBUG("LOWRES_density_perturb: ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

        // transform to k-space
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);

        //smooth the field
        // JordanFlitter: Evolve linearly if we are above REDSHIFT_2LPT
        if (!EVOLVE_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
            filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
        }

        LOG_SUPER_DEBUG("LOWRES_density_perturb after smoothing: ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

        // save a copy of the k-space density field
        // JordanFlitter: note that the following copy operation overwrites the previous copy operation to LOWRES_density_perturb_saved.
        // This is because LOWRES_density_perturb_saved is used only for the velocity field calculation.
        // This velocity field is associated with baryons, and so if we EVOLVE_BARYONS we are okay with overwriting the contnet of this box.
        memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);

        LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space: ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

        // normalize after FFT
        int bad_count=0;
    #pragma omp parallel shared(LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS) reduction(+: bad_count)
        {
    #pragma omp for
            for(i=0; i<user_params->HII_DIM; i++){
                for(j=0; j<user_params->HII_DIM; j++){
                    for(k=0; k<user_params->HII_DIM; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;

                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1.0) { // shouldn't happen

                            if(bad_count<5) LOG_WARNING("LOWRES_density_perturb is <-1 for index %d %d %d (value=%f)", i,j,k, *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)));
                            if(bad_count==5) LOG_WARNING("Skipping further warnings for LOWRES_density_perturb.");
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                            bad_count++;
                        }
                    }
                }
            }
        }
        if(bad_count>=5) LOG_WARNING("Total number of bad indices for LOW_density_perturb: %d", bad_count);
        LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space (normalized): ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

    #pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
        {
    #pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)perturbed_field->baryons_density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                        // JordanFlitter: update total matter density field (only if we don't compute the linear matter density field)
                        if (!(EVOLVE_LINEARLY || user_params->LINEAR_DELTA_IN_EPS)) {
                            *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) += ((
                                                                                           cosmo_params->OMb*(*((float *)perturbed_field->baryons_density + HII_R_INDEX(i,j,k)))
                                                                                           )/cosmo_params->OMm);
                        }
                    }
                }
            }
        }
    }
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************
    /*************************************************************** BARYONS SECTION ENDS ***********************************************************************
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************/

    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************
    /***************************************************************** SDM SECTION BEGINS ***********************************************************************
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************/
    /* JordanFlitter: General comments regarding the calculations below.
       1) To minimize memory, we use LOWRES_density_perturb and HIRES_density_perturb for the SDM density, as the baryons calculation was over
          in the above lines.
       2) We do not overwrite LOWRES_density_perturb_saved and HIRES_density_perturb_saved, as these boxes are needed for the baryons velocity (SDM velocity
          is not needed)
       3) For 2LPT calculations, we take the same 2LPT velocity boxes we used for the baryons (and also for CDM)
    */
    if (user_params->SCATTERING_DM && user_params->EVOLVE_BARYONS) {
        if (EVOLVE_LINEARLY){
            #pragma omp parallel shared(boxes,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
                    {
            #pragma omp for
                        for (i=0; i<dimension; i++){
                            for (j=0; j<dimension; j++){
                                for (k=0; k<dimension; k++){
                                    if(user_params->PERTURB_ON_HIGH_RES) {
                                        *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = boxes->hires_density[R_INDEX(i,j,k)];
                                    }
                                    else {
                                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = boxes->lowres_density[HII_R_INDEX(i,j,k)];
                                    }
                                }
                            }
                        }
                    }
            if(user_params->PERTURB_ON_HIGH_RES) {
                dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);
            }
            else{
                dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);
            }
            #pragma omp parallel shared(redshift,LOWRES_density_perturb,HIRES_density_perturb,dimension,switch_mid) private(n_x,n_y,n_z,k_x,k_y,k_z,k_mag,growth_factor_sdgf) num_threads(user_params->N_THREADS)
                    {
            #pragma omp for
                        for (n_x=0; n_x<dimension; n_x++){
                            if (n_x>switch_mid)
                                k_x =(n_x-dimension) * DELTA_K;  // wrap around for FFT convention
                            else
                                k_x = n_x * DELTA_K;
                            for (n_y=0; n_y<dimension; n_y++){
                                if (n_y>switch_mid)
                                    k_y =(n_y-dimension) * DELTA_K;
                                else
                                    k_y = n_y * DELTA_K;
                                for (n_z=0; n_z<=switch_mid; n_z++){
                                    k_z = n_z * DELTA_K;
                                    k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                                    growth_factor_sdgf = SDGF_SDM(redshift,k_mag,0);
                                    if(user_params->PERTURB_ON_HIGH_RES) {
                                        *((fftwf_complex *)HIRES_density_perturb + C_INDEX(n_x,n_y,n_z)) *= growth_factor_sdgf/TOT_NUM_PIXELS;
                                    }
                                    else {
                                        *((fftwf_complex *)LOWRES_density_perturb + HII_C_INDEX(n_x,n_y,n_z)) *= growth_factor_sdgf/HII_TOT_NUM_PIXELS;
                                    }
                                }
                            }
                        }
                    }
            if(user_params->PERTURB_ON_HIGH_RES) {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);
            }
            else {
                dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);
            }
        }
        else { // non-linear evolution
            // Apply Zel'dovich/2LPT correction
            LOG_DEBUG("Apply Zel'dovich");

    #pragma omp parallel shared(resampled_box,LOWRES_density_perturb,HIRES_density_perturb,dimension) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                *((float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k)) = 0.;
                                *((double *)resampled_box + R_INDEX(i,j,k)) = 0.;
                            }
                            else {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                                *((double *)resampled_box + HII_R_INDEX(i,j,k)) = 0.;
                            }
                        }
                    }
                }
            }

            // JordanFlitter: we multiply the hires density box by D_chi(k,z0) in Fourier space, and we multiply the velocity field by (D_chi(k,z)-D_chi(k,z0))/L
            multiply_in_Fourier_space(boxes->hires_density, FFT_HIRES_dummy_box, user_params, redshift, 1, 1, 3, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG

            if(user_params->PERTURB_ON_HIGH_RES) {
                multiply_in_Fourier_space(boxes->hires_vx, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vy, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vz, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            else{
                multiply_in_Fourier_space(boxes->lowres_vx, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vy, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vz, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 0, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            // Note: we use for the SDM calculation the same v_2LPT boxes that were used for baryons. So there is no need to do any multiplication!
            /*if(user_params->USE_2LPT){
                if(user_params->PERTURB_ON_HIGH_RES) {
                    multiply_in_Fourier_space(boxes->hires_vx_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vy_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vz_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
                else{
                    multiply_in_Fourier_space(boxes->lowres_vx_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vy_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vz_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 1, 1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
            }*/

            // ************  END INITIALIZATION **************************** //

            // go through the high-res box, mapping the mass onto the low-res (updated) box
            // JordanFlitter: added new private variables for cloud-in-cell
            LOG_DEBUG("Perturb the density field");
    #pragma omp parallel shared(boxes,f_pixel_factor,resampled_box,dimension) \
                            private(i,j,k,xi,xf,yi,yf,zi,zf,HII_i,HII_j,HII_k,d_x,d_y,d_z,t_x,t_y,t_z,xp1,yp1,zp1) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<user_params->DIM;i++){
                    for (j=0; j<user_params->DIM;j++){
                        for (k=0; k<user_params->DIM;k++){

                            // map indeces to locations in units of box size
                            xf = (i+0.5)/((user_params->DIM)+0.0);
                            yf = (j+0.5)/((user_params->DIM)+0.0);
                            zf = (k+0.5)/((user_params->DIM)+0.0);

                            // update locations
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                xf += (boxes->hires_vx)[R_INDEX(i, j, k)];
                                yf += (boxes->hires_vy)[R_INDEX(i, j, k)];
                                zf += (boxes->hires_vz)[R_INDEX(i, j, k)];
                            }
                            else {
                                HII_i = (unsigned long long)(i/f_pixel_factor);
                                HII_j = (unsigned long long)(j/f_pixel_factor);
                                HII_k = (unsigned long long)(k/f_pixel_factor);
                                xf += (boxes->lowres_vx)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                                yf += (boxes->lowres_vy)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                                zf += (boxes->lowres_vz)[HII_R_INDEX(HII_i, HII_j, HII_k)];
                            }

                            // 2LPT PART
                            // add second order corrections
                            if(user_params->USE_2LPT){
                                if(user_params->PERTURB_ON_HIGH_RES) {
                                    xf -= (boxes->hires_vx_2LPT)[R_INDEX(i,j,k)];
                                    yf -= (boxes->hires_vy_2LPT)[R_INDEX(i,j,k)];
                                    zf -= (boxes->hires_vz_2LPT)[R_INDEX(i,j,k)];
                                }
                                else {
                                    xf -= (boxes->lowres_vx_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                    yf -= (boxes->lowres_vy_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                    zf -= (boxes->lowres_vz_2LPT)[HII_R_INDEX(HII_i,HII_j,HII_k)];
                                }
                            }

                            xf *= (float)(dimension);
                            yf *= (float)(dimension);
                            zf *= (float)(dimension);
                            while (xf >= (float)(dimension)){ xf -= (dimension);}
                            while (xf < 0){ xf += (dimension);}
                            while (yf >= (float)(dimension)){ yf -= (dimension);}
                            while (yf < 0){ yf += (dimension);}
                            while (zf >= (float)(dimension)){ zf -= (dimension);}
                            while (zf < 0){ zf += (dimension);}
                            xi = xf;
                            yi = yf;
                            zi = zf;
                            if (xi >= (dimension)){ xi -= (dimension);}
                            if (xi < 0) {xi += (dimension);}
                            if (yi >= (dimension)){ yi -= (dimension);}
                            if (yi < 0) {yi += (dimension);}
                            if (zi >= (dimension)){ zi -= (dimension);}
                            if (zi < 0) {zi += (dimension);}


                            // JordanFlitter: Added Bradley Greig's new modification (cloud-in-cell)
                            if (user_params->CLOUD_IN_CELL) {
                                // Determine the fraction of the perturbed cell which overlaps with the 8 nearest grid cells,
                                // based on the grid cell which contains the centre of the perturbed cell
                                d_x = fabs(xf - (double)(xi+0.5));
                                d_y = fabs(yf - (double)(yi+0.5));
                                d_z = fabs(zf - (double)(zi+0.5));
                                if(xf < (double)(xi+0.5)) {
                                    // If perturbed cell centre is less than the mid-point then update fraction
                                    // of mass in the cell and determine the cell centre of neighbour to be the
                                    // lowest grid point index
                                    d_x = 1. - d_x;
                                    xi -= 1;
                                    if (xi < 0) {xi += (dimension);} // Only this critera is possible as iterate back by one (we cannot exceed DIM)
                                }
                                if(yf < (double)(yi+0.5)) {
                                    d_y = 1. - d_y;
                                    yi -= 1;
                                    if (yi < 0) {yi += (dimension);}
                                }
                                if(zf < (double)(zi+0.5)) {
                                    d_z = 1. - d_z;
                                    zi -= 1;
                                    if (zi < 0) {zi += (dimension);}
                                }
                                t_x = 1. - d_x;
                                t_y = 1. - d_y;
                                t_z = 1. - d_z;

                                // Determine the grid coordinates of the 8 neighbouring cells
                                // Takes into account the offset based on cell centre determined above
                                xp1 = xi + 1;
                                if(xp1 >= dimension) { xp1 -= (dimension);}
                                yp1 = yi + 1;
                                if(yp1 >= dimension) { yp1 -= (dimension);}
                                zp1 = zi + 1;
                                if(zp1 >= dimension) { zp1 -= (dimension);}

                                if(user_params->PERTURB_ON_HIGH_RES) {
                                    // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xi,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
        #pragma omp atomic
                                        resampled_box[R_INDEX(xp1,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                                }
                                else {
                                    // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yp1,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*t_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yi,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*t_y*d_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xi,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(t_x*d_y*d_z);
        #pragma omp atomic
                                        resampled_box[HII_R_INDEX(xp1,yp1,zp1)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)])*(d_x*d_y*d_z);
                                }
                            }
                            else {
                                if(user_params->PERTURB_ON_HIGH_RES) {
        #pragma omp atomic
                                    resampled_box[R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)]);
                                }
                                else {
        #pragma omp atomic
                                    resampled_box[HII_R_INDEX(xi,yi,zi)] += (double)(1. + (boxes->hires_density)[R_INDEX(i,j,k)]);
                                }
                            }
                        }
                    }
                }
            }

            LOG_SUPER_DEBUG("resampled_box: ");
            debugSummarizeBoxDouble(resampled_box, dimension, "  ");

            // Resample back to a float for remaining algorithm
    #pragma omp parallel shared(LOWRES_density_perturb,HIRES_density_perturb,resampled_box,dimension) \
                            private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<dimension; i++){
                    for (j=0; j<dimension; j++){
                        for (k=0; k<dimension; k++){
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                *( (float *)HIRES_density_perturb + R_FFT_INDEX(i,j,k) ) = (float)resampled_box[R_INDEX(i,j,k)];
                            }
                            else {
                                *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) = (float)resampled_box[HII_R_INDEX(i,j,k)];
                            }
                        }
                    }
                }
            }
            free(resampled_box);

            LOG_DEBUG("Finished perturbing the density field");

            LOG_SUPER_DEBUG("density_perturb: ");
            if(user_params->PERTURB_ON_HIGH_RES){
                debugSummarizeBox(HIRES_density_perturb, dimension, "  ");
            }else{
                debugSummarizeBox(LOWRES_density_perturb, dimension, "  ");
            }
            // JordanFlitter: we divide the hires density box by D_chi(k,z0) in Fourier space, and we divide the velocity field by (D_chi(k,z)-D_chi(k,z0))/L
            multiply_in_Fourier_space(boxes->hires_density, FFT_HIRES_dummy_box, user_params, redshift, 1, 1, 3, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            if(user_params->PERTURB_ON_HIGH_RES) {
                multiply_in_Fourier_space(boxes->hires_vx, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vy, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->hires_vz, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            else{
                multiply_in_Fourier_space(boxes->lowres_vx, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vy, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                multiply_in_Fourier_space(boxes->lowres_vz, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 0, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
            }
            if(user_params->USE_2LPT){
                if(user_params->PERTURB_ON_HIGH_RES) {
                    multiply_in_Fourier_space(boxes->hires_vx_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vy_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->hires_vz_2LPT, FFT_HIRES_dummy_box, user_params, redshift, 1, 0, 3, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
                else{
                    multiply_in_Fourier_space(boxes->lowres_vx_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vy_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                    multiply_in_Fourier_space(boxes->lowres_vz_2LPT, FFT_LOWRES_dummy_box, user_params, redshift, 0, 0, 3, 1, -1); // Flags order: HIRES_FLAG, DENSITY_FLAG, SDGF_TYPE, 2LPT_FLAG, MULT_DIV_FLAG
                }
            }

            LOG_DEBUG("Cleanup velocities for perturb");
        }
        // Now, if I still have the high resolution density grid (HIRES_density_perturb) I need to downsample it to the low-resolution grid
        if(user_params->PERTURB_ON_HIGH_RES) {

            LOG_DEBUG("Downsample the high-res perturbed SDM density");

            // Transform to Fourier space to sample (filter) the box
            dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);

            // Now filter the box
            if (user_params->DIM != user_params->HII_DIM) {
                filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));
            }

            // FFT back to real space
            dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);

            // Renormalise the FFT'd box
    #pragma omp parallel shared(HIRES_density_perturb,LOWRES_density_perturb,f_pixel_factor,mass_factor) private(i,j,k) num_threads(user_params->N_THREADS)
            {
    #pragma omp for
                for (i=0; i<user_params->HII_DIM; i++){
                    for (j=0; j<user_params->HII_DIM; j++){
                        for (k=0; k<user_params->HII_DIM; k++){
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) =
                            *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)))/(float)TOT_NUM_PIXELS;

                            // JordanFlitter: We need to subtract 1 only if we evolve the density field non-linearly (BUG in 21cmFAST)
                            if (!EVOLVE_LINEARLY){
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) -= 1.;
                            }

                            if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1.+FRACT_FLOAT_ERR;
                            }
                        }
                    }
                }
            }
        }
        else {
            // JordanFlitter: Evolve linearly if we are above REDSHIFT_2LPT
            if (!EVOLVE_LINEARLY){

    #pragma omp parallel shared(mass_factor,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
                {
    #pragma omp for
                    for (i=0; i<user_params->HII_DIM; i++){
                        for (j=0; j<user_params->HII_DIM; j++){
                            for (k=0; k<user_params->HII_DIM; k++){
                                *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                                *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1.;
                            }
                        }
                    }
                }
            }
        }
        LOG_SUPER_DEBUG("LOWRES_density_perturb: ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

        // transform to k-space
        dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);

        //smooth the field
        // JordanFlitter: Evolve linearly if we are above REDSHIFT_2LPT
        if (!EVOLVE_LINEARLY && global_params.SMOOTH_EVOLVED_DENSITY_FIELD){
            filter_box(LOWRES_density_perturb, 1, 2, global_params.R_smooth_density*user_params->BOX_LEN/(float)user_params->HII_DIM);
        }

        LOG_SUPER_DEBUG("LOWRES_density_perturb after smoothing: ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);

        LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space: ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

        // normalize after FFT
        int bad_count=0;
    #pragma omp parallel shared(LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS) reduction(+: bad_count)
        {
    #pragma omp for
            for(i=0; i<user_params->HII_DIM; i++){
                for(j=0; j<user_params->HII_DIM; j++){
                    for(k=0; k<user_params->HII_DIM; k++){
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;

                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1.0) { // shouldn't happen

                            if(bad_count<5) LOG_WARNING("LOWRES_density_perturb is <-1 for index %d %d %d (value=%f)", i,j,k, *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)));
                            if(bad_count==5) LOG_WARNING("Skipping further warnings for LOWRES_density_perturb.");
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1+FRACT_FLOAT_ERR;
                            bad_count++;
                        }
                    }
                }
            }
        }
        if(bad_count>=5) LOG_WARNING("Total number of bad indices for LOW_density_perturb: %d", bad_count);
        LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space (normalized): ");
        debugSummarizeBox(LOWRES_density_perturb, user_params->HII_DIM, "  ");

    #pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
        {
    #pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)perturbed_field->SDM_density + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                        // JordanFlitter: update total matter density field (only if we don't compute the linear matter density field)
                        if (!(EVOLVE_LINEARLY || user_params->LINEAR_DELTA_IN_EPS)) {
                            *((float *)perturbed_field->density + HII_R_INDEX(i,j,k)) += ((
                                                                                           pow(10.,-cosmo_params->f_chi)*(cosmo_params->OMm-cosmo_params->OMb)*(*((float *)perturbed_field->SDM_density + HII_R_INDEX(i,j,k)))
                                                                                           )/cosmo_params->OMm);
                        }
                    }
                }
            }
        }
    }
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************
    /***************************************************************** SDM SECTION ENDS *************************************************************************
    /************************************************************************************************************************************************************
    /************************************************************************************************************************************************************/

    // ****  Convert to velocities ***** //
    LOG_DEBUG("Generate velocity fields");

    dDdt_over_D = dDdt/growth_factor;

    if(user_params->PERTURB_ON_HIGH_RES) {
        // We are going to generate the velocity field on the high-resolution perturbed density grid
        memcpy(HIRES_density_perturb, HIRES_density_perturb_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    }
    else {
        // We are going to generate the velocity field on the low-resolution perturbed density grid
        memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }
// JordanFlitter: added more shared and private variables for baryons evolution
#pragma omp parallel shared(LOWRES_density_perturb,HIRES_density_perturb,dDdt_over_D,dimension,switch_mid,redshift) \
                        private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq,k_mag,dDdt_over_D_baryons) num_threads(user_params->N_THREADS)
    {
#pragma omp for
        for (n_x=0; n_x<dimension; n_x++){
            if (n_x>switch_mid)
                k_x =(n_x-dimension) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y=0; n_y<dimension; n_y++){
                if (n_y>switch_mid)
                    k_y =(n_y-dimension) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                for (n_z=0; n_z<=switch_mid; n_z++){
                    k_z = n_z * DELTA_K;

                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;

                    // now set the velocities
                    if ((n_x==0) && (n_y==0) && (n_z==0)) { // DC mode
                        if(user_params->PERTURB_ON_HIGH_RES) {
                            HIRES_density_perturb[0] = 0;
                        }
                        else {
                            LOWRES_density_perturb[0] = 0;
                        }
                    }
                    else{
                        // JordanFlitter: in case we evolve the baryons density field, we use the scale-dependent growth factor (and its time derivative)
                        if (user_params->EVOLVE_BARYONS) {
                            k_mag = sqrt(k_sq);
                            dDdt_over_D_baryons = dSDGF_BARYONS_dt(redshift,k_mag)/SDGF_BARYONS(redshift,k_mag,0);
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                HIRES_density_perturb[C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D_baryons*k_z*I/k_sq/(TOT_NUM_PIXELS+0.0);
                            }
                            else {
                                LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D_baryons*k_z*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                            }
                        }
                        else {
                            if(user_params->PERTURB_ON_HIGH_RES) {
                                HIRES_density_perturb[C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(TOT_NUM_PIXELS+0.0);
                            }
                            else {
                                LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                            }
                        }
                    }
                }
            }
        }
    }


    if(user_params->PERTURB_ON_HIGH_RES) {

        // smooth the high resolution field ready for resampling
        if (user_params->DIM != user_params->HII_DIM)
            filter_box(HIRES_density_perturb, 0, 0, L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0));

        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->DIM, user_params->N_THREADS, HIRES_density_perturb);

#pragma omp parallel shared(perturbed_field,HIRES_density_perturb,f_pixel_factor) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)perturbed_field->velocity + HII_R_INDEX(i,j,k)) = *((float *)HIRES_density_perturb + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5), (unsigned long long)(j*f_pixel_factor+0.5), (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                }
            }
        }
    }
    else {
        dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, user_params->N_THREADS, LOWRES_density_perturb);

#pragma omp parallel shared(perturbed_field,LOWRES_density_perturb) private(i,j,k) num_threads(user_params->N_THREADS)
        {
#pragma omp for
            for (i=0; i<user_params->HII_DIM; i++){
                for (j=0; j<user_params->HII_DIM; j++){
                    for (k=0; k<user_params->HII_DIM; k++){
                        *((float *)perturbed_field->velocity + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                    }
                }
            }
        }
    }

    fftwf_cleanup_threads();
    fftwf_cleanup();
    fftwf_forget_wisdom();

    // deallocate
    fftwf_free(LOWRES_density_perturb);
    fftwf_free(LOWRES_density_perturb_saved);
    if(user_params->PERTURB_ON_HIGH_RES) {
        fftwf_free(HIRES_density_perturb);
        fftwf_free(HIRES_density_perturb_saved);
    }
    // JordanFlitter: Need to allocate memory for dummy boxes if we do non-linear scale-dependent evolution
    if (!EVOLVE_LINEARLY && (user_params->EVOLVE_MATTER || user_params->EVOLVE_BARYONS)) {
        fftwf_free(FFT_HIRES_dummy_box);
        if (!user_params->PERTURB_ON_HIGH_RES){
            fftwf_free(FFT_LOWRES_dummy_box);
        }
    }

    fftwf_cleanup();
    // JordanFlitter: we need destruct_CLASS_GROWTH_FACTOR() if the following conditions are satisfied
    if (!user_params->USE_DICKE_GROWTH_FACTOR || user_params->EVOLVE_BARYONS || user_params->EVOLVE_MATTER) {
          destruct_CLASS_GROWTH_FACTOR();
    }

    } // End of Try{}
    Catch(status){
        return(status);
    }

    return(0);
}
