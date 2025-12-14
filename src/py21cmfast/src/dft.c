int dft_c2r_cube(bool use_wisdom, int dim, int n_threads, fftw_complex *box){
    char wisdom_filename[500];
    unsigned flag = FFTW_ESTIMATE;
    int status;
    fftw_plan plan;

    Try{
        if(use_wisdom) {
            // Check to see if the wisdom exists
            sprintf(wisdom_filename,"%s/c2r_DIM%d_NTHREADS%d",global_params.wisdoms_path, dim, n_threads);

            if(fftw_import_wisdom_from_filename(wisdom_filename)!=0) {
                unsigned flag = FFTW_WISDOM_ONLY;
            }
            else {
                LOG_WARNING("Cannot locate FFTW Wisdom: %s file not found. Reverting to FFTW_ESTIMATE.", wisdom_filename);
            }
        }
        plan = fftw_plan_dft_c2r_3d(dim, dim, dim, (fftw_complex *)box, (double *)box, flag);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    Catch(status){
        return(status);
    }
    return(0);
}

int dft_r2c_cube(bool use_wisdom, int dim, int n_threads, fftw_complex *box){
    char wisdom_filename[500];
    unsigned flag = FFTW_ESTIMATE;
    int status;
    fftw_plan plan;

    Try{
        if(use_wisdom) {
            // Check to see if the wisdom exists
            sprintf(wisdom_filename,"%s/r2c_DIM%d_NTHREADS%d", global_params.wisdoms_path, dim, n_threads);

            if(fftw_import_wisdom_from_filename(wisdom_filename)!=0) {
                unsigned flag = FFTW_WISDOM_ONLY;
            }
            else {
                LOG_WARNING("Cannot locate FFTW Wisdom: %s file not found. Reverting to FFTW_ESTIMATE.", wisdom_filename);
            }
        }
        plan = fftw_plan_dft_r2c_3d(dim, dim, dim, (double *)box, (fftw_complex *)box, flag);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    Catch(status){
        return(status);
    }
    return(0);
}

int CreateFFTWWisdoms(struct UserParams *user_params, struct CosmoParams *cosmo_params) {

    int status;
    char *wisdom_string;

    Try{ // This Try wraps the entire function so we don't indent.

        Broadcast_struct_global_UF(user_params,cosmo_params);

        fftw_plan plan;

        char wisdom_filename[500];

        int i,j,k;

        omp_set_num_threads(user_params->N_THREADS);
        fftw_init_threads();
        fftw_plan_with_nthreads(user_params->N_THREADS);
        fftw_cleanup_threads();

        // allocate array for the k-space and real-space boxes
        fftw_complex *HIRES_box = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*KSPACE_NUM_PIXELS);
        fftw_complex *LOWRES_box = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*HII_KSPACE_NUM_PIXELS);

        sprintf(wisdom_filename,"%s/r2c_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->DIM,user_params->N_THREADS);
        if(fftw_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftw_plan_dft_r2c_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (double *)HIRES_box, (fftw_complex *)HIRES_box, FFTW_PATIENT);
            fftw_export_wisdom_to_filename(wisdom_filename);
            fftw_destroy_plan(plan);
        }

        sprintf(wisdom_filename,"%s/c2r_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->DIM,user_params->N_THREADS);
        if(fftw_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftw_plan_dft_c2r_3d(user_params->DIM, user_params->DIM, user_params->DIM,
                                         (fftw_complex *)HIRES_box, (double *)HIRES_box,  FFTW_PATIENT);
            fftw_export_wisdom_to_filename(wisdom_filename);
            fftw_destroy_plan(plan);
        }

        sprintf(wisdom_filename,"%s/r2c_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->HII_DIM,user_params->N_THREADS);
        if(fftw_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftw_plan_dft_r2c_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (double *)LOWRES_box, (fftw_complex *)LOWRES_box, FFTW_PATIENT);
            fftw_export_wisdom_to_filename(wisdom_filename);
            fftw_destroy_plan(plan);
        }

        sprintf(wisdom_filename,"%s/c2r_DIM%d_NTHREADS%d",global_params.wisdoms_path, user_params->HII_DIM,user_params->N_THREADS);
        if(fftw_import_wisdom_from_filename(wisdom_filename)==0) {
            plan = fftw_plan_dft_c2r_3d(user_params->HII_DIM, user_params->HII_DIM, user_params->HII_DIM,
                                         (fftw_complex *)LOWRES_box, (double *)LOWRES_box,  FFTW_PATIENT);
            fftw_export_wisdom_to_filename(wisdom_filename);
            fftw_destroy_plan(plan);
        }

        fftw_cleanup_threads();
        fftw_cleanup();
        fftw_forget_wisdom();

        // deallocate
        fftw_free(HIRES_box);
        fftw_free(LOWRES_box);


    } // End of Try{}

    Catch(status){
        return(status);
    }
    return(0);
}
