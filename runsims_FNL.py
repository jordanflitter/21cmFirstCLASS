import py21cmfast as p21c # To run 21cmFirstCLASS (21cmFAST)
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import os
import pickle
import numpy as np 
from scipy.interpolate import RectBivariateSpline

import glob 
import os
from scipy.stats import binned_statistic_2d

from matplotlib.ticker import FuncFormatter

cmap_val = plt.get_cmap("RdYlGn_r")
norm_val = colors.LogNorm(vmin=1., vmax=100.)
norm_corr_val = colors.LogNorm(vmin=2e0, vmax=1e1)

plt.rcParams.update({"text.usetex": True, "font.family": "Times new roman"}) # Use latex fonts
use_colors =  ['b', 'c', 'k', 'orange', 'r']

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=use_colors) # Set the color palette as default


main_folder = './final_test/'
if not os.path.exists(main_folder):
        os.makedirs(main_folder)

plot_folder = main_folder + 'plots/'
if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

freq_bands_boundaries = np.arange(50.,225.+8.,8.); freq_bands_boundaries[-1] = 225.

input_coeval_redshifts = [6.5,7.,8.,11.,15.]


def run_many(Lbox, Nbox, 
            fnl_arr, 
            force_mmax_arr, 
            kcut_fnl_arr, 
            use_MUSE_MINI_HALOS_list, 
            nongauss_IC = True, 
            nongauss_fcollcond = True, nongauss_fcolluncond = True, 
            use_lidz_approx = False, 
            use_ld_cond = True, 
            use_edg_uncond = True, 
            max_epsilon = 0., 
            extra_dim_fnl = 1., write_flag = False,
            nthreads = 6,):
    
    for m in force_mmax_arr:
        for kk in kcut_fnl_arr:
            for u in use_MUSE_MINI_HALOS_list:
                u_str = 'II' if u is False else 'II+III'
                print('---------------------------------')
                print('DOING ' + str(u_str) + ',' + str(kk) + ',' + str(m) + ',' + str(max_epsilon))
                print('---------------------------------')
                for fnl in fnl_arr:
                    print('\nDoing fnl = ' + str(fnl))
                    save_stuff(Lbox, Nbox, fnl, 
                    m, 
                    kk, 
                    u, 
                    nongauss_IC, 
                    nongauss_fcollcond, nongauss_fcolluncond, 
                    use_lidz_approx, 
                    use_ld_cond, 
                    use_edg_uncond, 
                    max_epsilon, 
                    extra_dim_fnl, write_flag,
                    nthreads ,)

    return 


def run_sim(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC = True, 
            nongauss_fcollcond = True, nongauss_fcolluncond = True, 
            use_lidz_approx = False, 
            use_ld_cond = True, 
            use_edg_uncond = True, 
            max_epsilon = 0., 
            extra_dim_fnl = 1., write_flag = False,
            nthreads = 6,  
            ):

    try:

        flag_options = {"USE_MINI_HALOS": USE_MINI_HALOS, } # if False, popIII stars are not included - Note: if set to True, the runtime increases significantly!

        astro_params = {"F_STAR10": -1.25, # star formation efficiency (atomic cooling galaxies) for pivot mass 1e10 Msun (log10)
                        "ALPHA_STAR": 0.5, # slope of the dependency of star formation efficiency on the host halo mass 
                        "F_ESC10": -1.35, # escape fraction of Lyman photons into the IGM for pivot mass 1e10 Msun (log10)
                        "ALPHA_ESC": 0., # slope of the dependency of escape fraction on the host halo mass 
                        "L_X": 40.5, # X-ray luminosity (log10)
                        "F_STAR7_MINI": -2.0,
                        "ALPHA_STAR_MINI": 0.5,
                        "F_ESC7_MINI": -2.0,
        }

        global_quantities = ("brightness_temp", # brightness temperature
                            "xH_box") # free electron fraction

        coeval_quantities = ["brightness_temp"]

        lightcone_quantities = ("brightness_temp",)

        user_params = {"EVALUATE_TAU_REIO":False,
                       "OUTPUT_AT_DARK_AGES":False,
                    "BOX_LEN": Lbox, # size of the simulated box (in comoving Mpc) 
                    "HII_DIM": Nbox, # number of cells along each axis of the coeval box - Note: more cells means longer runtime! 
                    "EXTRA_DIM_FNL": extra_dim_fnl,
                    "N_THREADS": nthreads, # the amount of processors the code uses in parallelization
                    "NON_GAUSS_IC": nongauss_IC, # use non gaussian field in initial conditions
                    "NON_GAUSS_FCOLL_COND": nongauss_fcollcond,                    "NON_GAUSS_FCOLL_UNCOND": nongauss_fcolluncond,# use non gaussian field in collapsed fraction
                    "NG_MODEL_APPROX": use_lidz_approx,
                    "FORCE_MMAX": force_mmax,
                    "WRITE_CGF_DIAG":write_flag, 
                    "MAX_EPSILON_NG": max_epsilon,
                    "USE_LD_cond_hmf": use_ld_cond, 
                    "USE_EDG_uncond_hmf": use_edg_uncond
                    } # 

        cosmo_params = {"hlittle": 0.6736, # hubble parameter
                        "OMb": 0.0493, # baryon density
                        "OMm": 0.3153, # matter (CDM+baryon) density
                        "A_s": 2.1e-9, # amplitude of the primordial fluctuations
                        "POWER_INDEX": 0.9649, # spectral index of the primordial spectrum
                        "tau_reio": 0.0544, # optical depth to reionization
                        "F_NL":fnl, # to test matching with old case
                        "KCUT_FNL": kcut_fnl
                        }

        lightcone = p21c.run_lightcone(redshift = 6., # minimum redshift at which the simulation will stop
                                    random_seed = 1, # numerical seed -- if None, each run will produce different initial conditions, with the same cosmological power spectrum but different spatial realization. You need to specify the random seed to coherently compare two different runs, due to cosmic variance. 
                                    regenerate = True, # create new data even if cached are found
                                    write = False, # whether or no to save cached files
                                    user_params = user_params,
                                    astro_params = astro_params, 
                                    flag_options = flag_options,
                                    cosmo_params = cosmo_params,
                                    global_quantities = global_quantities,
                                    save_coeval_quantities = coeval_quantities,
                                    save_coeval_redshifts = input_coeval_redshifts,
                                    lightcone_quantities = lightcone_quantities) ; 
    except:
        print('\n FAILED!')
        print(str(Lbox) + '_' + str(Nbox) + '_' + str(fnl) + '_' + str(force_mmax) + '_' + str(write_flag))
        print('/n')

        lightcone = -1

    power_spectrum = p21c.power_spectrum.lightcone_power_spectrum(lightcone,
                                freq_bands_boundaries = freq_bands_boundaries)#,k_bins = k_bins) 

    return lightcone, power_spectrum


def save_stuff(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC = True, 
            nongauss_fcollcond = True, nongauss_fcolluncond = True, 
            use_lidz_approx = False, 
            use_ld_cond = True, 
            use_edg_uncond = True, 
            max_epsilon = 0., 
            extra_dim_fnl = 1., write_flag = False,
            nthreads = 6,  ):

    lc, pk = run_sim(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC, 
            nongauss_fcollcond, nongauss_fcolluncond, 
            use_lidz_approx, 
            use_ld_cond, 
            use_edg_uncond, 
            max_epsilon, 
            extra_dim_fnl, write_flag,
            nthreads,)

    coeval_redshifts = list(lc.coeval_boxes.keys())
    cz_0 = coeval_redshifts[np.argmin(np.abs(input_coeval_redshifts[0] - np.array(coeval_redshifts)))]
    cz_1 = coeval_redshifts[np.argmin(np.abs(input_coeval_redshifts[1] - np.array(coeval_redshifts)))]
    cz_2 = coeval_redshifts[np.argmin(np.abs(input_coeval_redshifts[2] - np.array(coeval_redshifts)))]
    cz_3 = coeval_redshifts[np.argmin(np.abs(input_coeval_redshifts[3] - np.array(coeval_redshifts)))]
    cz_4 = coeval_redshifts[np.argmin(np.abs(input_coeval_redshifts[4] - np.array(coeval_redshifts)))]

    Mmax_label = '_Mcut' + str(force_mmax)  if force_mmax > 0. else ''
    kcut_label = '_Kcut' + str(kcut_fnl)  if kcut_fnl > 1e-3 else ''

    popIII_label = '_III' if USE_MINI_HALOS else ''

    nongauss_IC_label = '_noNGIC' if not nongauss_IC else '' 
    nongauss_fcollcond_label = '_noNGfcolCOND' if not nongauss_fcollcond else '' 
    nongauss_fcolluncond_label = '_noNGfcolUNCOND' if not nongauss_fcolluncond else '' 
    use_lidz_approx_label = '_LidzAppr' if use_lidz_approx else '' 
    use_ld_cond_label = '_CONDsaddle' if not use_ld_cond else '' 
    use_edg_uncond_label = '_UNCONDsaddle' if not use_edg_uncond else ''
    max_epsilon_label = '_maxeps' + str(max_epsilon)  if max_epsilon > 0. else ''
    extra_dim_fnl_label = '_extradimFNL' + str(extra_dim_fnl) if extra_dim_fnl > 1. else ''

    filename = main_folder + str(Lbox) + ',' + str(Nbox) + ','

    if fnl == 0.:
        filename += 'Gaus' + Mmax_label + popIII_label
    
    else:
        filename += str(fnl) + \
        Mmax_label + kcut_label + popIII_label + nongauss_IC_label + nongauss_fcollcond_label + nongauss_fcolluncond_label + use_lidz_approx_label + use_ld_cond_label + use_edg_uncond_label + max_epsilon_label + extra_dim_fnl_label 
        
    filename += '.pkl'

    with open(filename, 'wb') as handle:
        pickle.dump({'z_global': lc.node_redshifts,
                     'T21': lc.global_quantities['brightness_temp'], 
                     'xH': lc.global_quantities['xH_box'],
                     'z_pk': pk.z_values,
                     'k': pk.k_values,
                     'pk': pk.ps_values,
                     'box_' + str(input_coeval_redshifts[0]): lc.coeval_boxes[cz_0]['brightness_temp'],
                     'box_' + str(input_coeval_redshifts[1]): lc.coeval_boxes[cz_1]['brightness_temp'],
                     'box_' + str(input_coeval_redshifts[2]): lc.coeval_boxes[cz_2]['brightness_temp'],
                    'box_' + str(input_coeval_redshifts[3]): lc.coeval_boxes[cz_3]['brightness_temp'],
                    'box_' + str(input_coeval_redshifts[4]): lc.coeval_boxes[cz_4]['brightness_temp'],
                     }, handle)

    return 


def import_stuff(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC = True, 
            nongauss_fcollcond = True, nongauss_fcolluncond = True, 
            use_lidz_approx = False, 
            use_ld_cond = True, 
            use_edg_uncond = True, 
            max_epsilon = 0., 
            extra_dim_fnl = 1., ):

    Mmax_label = '_Mcut' + str(force_mmax) if force_mmax > 0. else ''
    kcut_label = '_Kcut' + str(kcut_fnl) if kcut_fnl > 1e-3 else ''

    popIII_label = '_III' if USE_MINI_HALOS else ''

    nongauss_IC_label = '_noNGIC' if not nongauss_IC else '' 
    nongauss_fcollcond_label = '_noNGfcolCOND' if not nongauss_fcollcond else '' 
    nongauss_fcolluncond_label = '_noNGfcolUNCOND' if not nongauss_fcolluncond else '' 
    use_lidz_approx_label = '_LidzAppr' if use_lidz_approx else '' 
    use_ld_cond_label = '_CONDsaddle' if not use_ld_cond else '' 
    use_edg_uncond_label = '_UNCONDsaddle' if not use_edg_uncond else ''
    max_epsilon_label = '_maxeps' + str(max_epsilon)  if max_epsilon > 0. else ''
    extra_dim_fnl_label = '_extradimFNL' + str(extra_dim_fnl) if extra_dim_fnl > 1. else ''

    filename = main_folder + str(Lbox) + ',' + str(Nbox) + ','

    if fnl == 0.:
        filename += 'Gaus' + Mmax_label + popIII_label
    
    else:
        filename += str(fnl) + \
        Mmax_label + kcut_label + popIII_label + nongauss_IC_label + nongauss_fcollcond_label + nongauss_fcolluncond_label + use_lidz_approx_label + use_ld_cond_label + use_edg_uncond_label + max_epsilon_label + extra_dim_fnl_label 
        
    filename += '.pkl'

    try: 
        with open(filename, 'rb') as handle:
            temp = pickle.load(handle)
            z_global = temp['z_global'] 
            T21 = temp['T21']
            xH = temp['xH']
            z_pk = temp['z_pk']
            k = temp['k']
            pk = temp['pk']
            box_0 = temp['box_' + str(input_coeval_redshifts[0])]
            box_1 = temp['box_' + str(input_coeval_redshifts[1])]
            box_2 = temp['box_' + str(input_coeval_redshifts[2])]
            box_3 = temp['box_' + str(input_coeval_redshifts[3])]
            box_4 = temp['box_' + str(input_coeval_redshifts[4])]

    except:
        print('File not found: ' + str(filename))
        print('Running...')

        save_stuff(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC, 
            nongauss_fcollcond, nongauss_fcolluncond, 
            use_lidz_approx, 
            use_ld_cond, 
            use_edg_uncond, 
            max_epsilon, 
            extra_dim_fnl, write_flag = False,
            nthreads = 6,  )

        with open(filename, 'rb') as handle:
            temp = pickle.load(handle)
            z_global = temp['z_global'] 
            T21 = temp['T21']
            xH = temp['xH']
            z_pk = temp['z_pk']
            k = temp['k']
            pk = temp['pk']
            box_0 = temp['box_' + str(input_coeval_redshifts[0])]
            box_1 = temp['box_' + str(input_coeval_redshifts[1])]
            box_2 = temp['box_' + str(input_coeval_redshifts[2])]
            box_3 = temp['box_' + str(input_coeval_redshifts[3])]
            box_4 = temp['box_' + str(input_coeval_redshifts[4])]

    return z_global, T21, xH, z_pk, k, pk, box_0, box_1, box_2, box_3, box_4


def plot_globals(Lbox, Nbox, fnl_array, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC = True, 
            nongauss_fcollcond = True, nongauss_fcolluncond = True, 
            use_lidz_approx = False, 
            use_ld_cond = True, 
            use_edg_uncond = True, 
            max_epsilon = 0., 
            extra_dim_fnl = 1.,):
     
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,9),sharex=True, gridspec_kw={'hspace': 0})

    for fnl in fnl_array:
        if fnl == 0.:
            z_global, T21, xH, z_pk, k, pk, box_0, box_1, box_2, box_3, box_4 = import_stuff(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC, 
            nongauss_fcollcond, nongauss_fcolluncond, 
            use_lidz_approx, 
            use_ld_cond, 
            use_edg_uncond, 
            max_epsilon, 
            extra_dim_fnl)
        else:
            z_global, T21, xH, z_pk, k, pk, box_0, box_1, box_2, box_3, box_4 = import_stuff(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC, 
            nongauss_fcollcond, nongauss_fcolluncond, 
            use_lidz_approx, 
            use_ld_cond, 
            use_edg_uncond, 
            max_epsilon, 
            extra_dim_fnl,)

        ax1.plot(z_global, T21, label=r'$f_{\rm NL}=%g$'%fnl)
        ax2.plot(z_global, 1-xH, label=r'$f_{\rm NL}=%g$'%fnl)

    ax1.set_xlabel('')
    ax2.set_xlabel(r'$z$', fontsize=15)

    ax1.set_ylabel(r'$T_{21}$', fontsize=15)
    ax2.set_ylabel(r'$x_{\rm HII}$', fontsize=15)

    ax1.legend(loc=4, fontsize=15)
    ax2.legend(loc=1, fontsize=15)

    ax1.set_xlim(6.,30)
    ax2.set_xlim(6.,30)

    ax1.set_ylim(-140,40)
    ax2.set_ylim(-0.1,1.1)

    if USE_MINI_HALOS:
         plt.title(r'$\rm Including\,popIII$')

    Mmax_label = '_Mcut' + str(force_mmax) + '_' if force_mmax > 0. else ''
    kcut_label = '_Kcut' + str(kcut_fnl) + '_' if kcut_fnl > 1e-3 else ''

    popIII_label = '_III' if USE_MINI_HALOS else ''

    nongauss_IC_label = '_noNGIC_' if not nongauss_IC else '' 
    nongauss_fcollcond_label = '_noNGfcolCOND_' if not nongauss_fcollcond else '' 
    nongauss_fcolluncond_label = '_noNGfcolUNCOND_' if not nongauss_fcolluncond else '' 
    use_lidz_approx_label = '_LidzAppr_' if use_lidz_approx else '' 
    use_ld_cond_label = '_CONDsaddle_' if not use_ld_cond else '' 
    use_edg_uncond_label = '_UNCONDsaddle_' if not use_edg_uncond else ''
    max_epsilon_label = '_maxeps' + str(max_epsilon) + '_' if max_epsilon > 0. else ''
    extra_dim_fnl_label = '_extradimFNL' + str(extra_dim_fnl) + '_' if extra_dim_fnl > 1. else ''

    filename = 'globals_' + str(Lbox) + ',' + str(Nbox) + ',' + str(fnl) + \
    Mmax_label + kcut_label + popIII_label + nongauss_IC_label + nongauss_fcollcond_label + nongauss_fcolluncond_label + use_lidz_approx_label + use_ld_cond_label + use_edg_uncond_label + max_epsilon_label + extra_dim_fnl_label \
    + '.png'

    plt.tight_layout()
    plt.savefig(plot_folder + filename)

    plt.show()

    return 


def plot_boxes(Lbox, Nbox, fnl_array, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC = True, 
            nongauss_fcollcond = True, nongauss_fcolluncond = True, 
            use_lidz_approx = False, 
            use_ld_cond = True, 
            use_edg_uncond = True, 
            max_epsilon = 0., 
            extra_dim_fnl = 1., ):

    vmin = [-30.,-60.,-100.]
    vmax = [30.,10.,-20.,]

    for fnl in fnl_array:

        fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(13, 3))

        mid_point_0 = abs(vmin[0])/(abs(vmin[0])+abs(vmax[0]))
        colors_list_0 = [(0, "cyan"),
                        (mid_point_0/2., "blue"),
                        (mid_point_0, "black"),
                        ((1.+mid_point_0)/2., "red"),
                        (1, "yellow")]
        eor_colour_0 = colors.LinearSegmentedColormap.from_list("brightness_temp",colors_list_0)

        mid_point_1 = abs(vmin[1])/(abs(vmin[1])+abs(vmax[1]))
        colors_list_1 = [(0, "cyan"),
                        (mid_point_1/2., "blue"),
                        (mid_point_1, "black"),
                        ((1.+mid_point_1)/2., "red"),
                        (1, "yellow")]
        eor_colour_1 = colors.LinearSegmentedColormap.from_list("brightness_temp",colors_list_1)

        colors_list_2 =[(0, "cyan"),
                               (0.5, "blue"),
                               (1, "black")]
        
        eor_colour_2 = colors.LinearSegmentedColormap.from_list("brightness_temp",colors_list_2)

        if fnl == 0.:
            z_global, T21, xH, z_pk, k, pk, box_0, box_1, box_2, box_3, box_4 = import_stuff(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC, 
            nongauss_fcollcond, nongauss_fcolluncond, 
            use_lidz_approx, 
            use_ld_cond, 
            use_edg_uncond, 
            max_epsilon, 
            extra_dim_fnl)
        else:
            z_global, T21, xH, z_pk, k, pk, box_0, box_1, box_2, box_3, box_4 = import_stuff(Lbox, Nbox, fnl, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC, 
            nongauss_fcollcond, nongauss_fcolluncond, 
            use_lidz_approx, 
            use_ld_cond, 
            use_edg_uncond, 
            max_epsilon, 
            extra_dim_fnl)

        ax0.imshow(box_0[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[0], vmax =vmax[0], cmap=eor_colour_0)
        ax1.imshow(box_1[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[0], vmax =vmax[0], cmap=eor_colour_0)
        ax2.imshow(box_2[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[0], vmax =vmax[0], cmap=eor_colour_0)
        ax3.imshow(box_3[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[1], vmax =vmax[1], cmap=eor_colour_1)
        ax4.imshow(box_4[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[2], vmax =vmax[2], cmap=eor_colour_2)

        ax0.set_title(r'$z = %g$'%input_coeval_redshifts[0])
        ax1.set_title(r'$z = %g$'%input_coeval_redshifts[1])
        ax2.set_title(r'$z = %g$'%input_coeval_redshifts[2])
        ax3.set_title(r'$z = %g$'%input_coeval_redshifts[3])
        ax4.set_title(r'$z = %g$'%input_coeval_redshifts[4])

        plt.suptitle(r'$f_{\rm NL} = %g$'%fnl, fontsize=15)
        if USE_MINI_HALOS:
            plt.suptitle(r'$f_{\rm NL} = %g$'%fnl + r'$,\, \rm Including\,popIII$', fontsize=15)

        Mmax_label = '_Mcut' + str(force_mmax) + '_' if force_mmax > 0. else ''
        kcut_label = '_Kcut' + str(kcut_fnl) + '_' if kcut_fnl > 1e-3 else ''

        popIII_label = '_III' if USE_MINI_HALOS else ''

        nongauss_IC_label = '_noNGIC_' if not nongauss_IC else '' 
        nongauss_fcollcond_label = '_noNGfcolCOND_' if not nongauss_fcollcond else '' 
        nongauss_fcolluncond_label = '_noNGfcolUNCOND_' if not nongauss_fcolluncond else '' 
        use_lidz_approx_label = '_LidzAppr_' if use_lidz_approx else '' 
        use_ld_cond_label = '_CONDsaddle_' if not use_ld_cond else '' 
        use_edg_uncond_label = '_UNCONDsaddle_' if not use_edg_uncond else ''
        max_epsilon_label = '_maxeps' + str(max_epsilon) + '_' if max_epsilon > 0. else ''
        extra_dim_fnl_label = '_extradimFNL' + str(extra_dim_fnl) + '_' if extra_dim_fnl > 1. else ''

        filename = 'boxes_' + str(Lbox) + ',' + str(Nbox) + ',' + str(fnl) + \
        Mmax_label + kcut_label + popIII_label + nongauss_IC_label + nongauss_fcollcond_label + nongauss_fcolluncond_label + use_lidz_approx_label + use_ld_cond_label + use_edg_uncond_label + max_epsilon_label + extra_dim_fnl_label \
        + '.png'

        plt.tight_layout()
        plt.savefig(plot_folder + filename )

    plt.show()

    return 


def plot_pk(Lbox, Nbox, fnl_array, 
            force_mmax, 
            kcut_fnl, 
            USE_MINI_HALOS, 
            nongauss_IC = True, 
            nongauss_fcollcond = True, nongauss_fcolluncond = True, 
            use_lidz_approx = False, 
            use_ld_cond = True, 
            use_edg_uncond = True, 
            max_epsilon = 0., 
            extra_dim_fnl = 1.,):
     
    k_vals = [0.1,0.3,0.5]

    for k in k_vals:
        plt.figure(figsize=(6,5))

        for fnl in fnl_array:
            if fnl == 0.:
                z_global, T21, xH, z_pk, k_pk, pk, box_0, box_1, box_2, box_3, box_4 = import_stuff(Lbox, Nbox, fnl, 
                    force_mmax, 
                    kcut_fnl, 
                    USE_MINI_HALOS, 
                    nongauss_IC, 
                    nongauss_fcollcond, nongauss_fcolluncond, 
                    use_lidz_approx, 
                    use_ld_cond, 
                    use_edg_uncond, 
                    max_epsilon, 
                    extra_dim_fnl)
            else:

                z_global, T21, xH, z_pk, k_pk, pk, box_0, box_1, box_2, box_3, box_4 = import_stuff(Lbox, Nbox, fnl, 
                    force_mmax, 
                    kcut_fnl, 
                    USE_MINI_HALOS, 
                    nongauss_IC, 
                    nongauss_fcollcond, nongauss_fcolluncond, 
                    use_lidz_approx, 
                    use_ld_cond, 
                    use_edg_uncond, 
                    max_epsilon, 
                    extra_dim_fnl)

            x_values = np.linspace(min(z_pk),max(z_pk),4*len(z_pk))
            y_values = RectBivariateSpline(k_pk, z_pk, pk.T)(k,x_values)[0]

            plt.plot(x_values, y_values, label=r'$f_{\rm NL}=%g$'%fnl)

        plt.xlabel(r'$z$', fontsize=15)
        plt.ylabel(r'$\Delta^2_{21}$', fontsize=15)

        plt.legend(loc=1, fontsize=15)

        plt.xlim(6.,30)
        if USE_MINI_HALOS:
            plt.title(r'$k=%g\,{\rm Mpc}^{-1}$'%k + r'$\, \rm Including\,popIII$', fontsize=15)
        else:
            plt.title(r'$k=%g\,{\rm Mpc}^{-1}$'%k, fontsize=15)
    
        Mmax_label = '_Mcut' + str(force_mmax) + '_' if force_mmax > 0. else ''
        kcut_label = '_Kcut' + str(kcut_fnl) + '_' if kcut_fnl > 1e-3 else ''

        popIII_label = '_III' if USE_MINI_HALOS else ''

        nongauss_IC_label = '_noNGIC_' if not nongauss_IC else '' 
        nongauss_fcollcond_label = '_noNGfcolCOND_' if not nongauss_fcollcond else '' 
        nongauss_fcolluncond_label = '_noNGfcolUNCOND_' if not nongauss_fcolluncond else '' 
        use_lidz_approx_label = '_LidzAppr_' if use_lidz_approx else '' 
        use_ld_cond_label = '_CONDsaddle_' if not use_ld_cond else '' 
        use_edg_uncond_label = '_UNCONDsaddle_' if not use_edg_uncond else ''
        max_epsilon_label = '_maxeps' + str(max_epsilon) + '_' if max_epsilon > 0. else ''
        extra_dim_fnl_label = '_extradimFNL' + str(extra_dim_fnl) + '_' if extra_dim_fnl > 1. else ''

        filename = 'pk_' + str(k) + '_' + str(Lbox) + ',' + str(Nbox) + ',' + str(fnl) + \
        Mmax_label + kcut_label + popIII_label + nongauss_IC_label + nongauss_fcollcond_label + nongauss_fcolluncond_label + use_lidz_approx_label + use_ld_cond_label + use_edg_uncond_label + max_epsilon_label + extra_dim_fnl_label \
        + '.png'

        plt.tight_layout()
        plt.savefig(plot_folder + filename)

        plt.show()

    return 



def plot_compare_multi(k_vals, 
                        list_to_compare,
                        list_colors,
                        list_ls,
                        list_labels, 
                        filename):

    list_z_global = []
    list_T21 = []
    list_xH = []
    list_z_pk = []
    list_k_pk = [] 
    list_pk = []

    for c in list_to_compare:
        list_z_global.append(c[0])
        list_T21.append(c[1])
        list_xH.append(c[2])
        list_z_pk.append(c[3])
        list_k_pk.append(c[4])
        list_pk.append(c[5])

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize=(11,9),sharex=True, gridspec_kw={'hspace': 0})


    for i in range(len(list_labels)):

        ax1.plot(list_z_global[i], list_T21[i], label= list_labels[i], color=list_colors[i], ls = list_ls[i] )

        ax3.plot(list_z_global[i], 1.-list_xH[i], label= list_labels[i], color=list_colors[i], ls = list_ls[i])

        x_values = np.linspace(min(list_z_pk[i]),max(list_z_pk[i]),4*len(list_z_pk[i]))
        y_values_1 = RectBivariateSpline(list_k_pk[i], list_z_pk[i], list_pk[i].T)(k_vals[0],x_values)[0]
        y_values_2 = RectBivariateSpline(list_k_pk[i], list_z_pk[i], list_pk[i].T)(k_vals[1],x_values)[0]

        ax2.plot(x_values, y_values_1, label= list_labels[i], color=list_colors[i], ls = list_ls[i])
        ax4.plot(x_values, y_values_2, label= list_labels[i], color=list_colors[i], ls = list_ls[i])

    ax1.set_xlabel('')
    ax3.set_xlabel(r'$z$', fontsize=15)

    ax1.set_ylabel(r'$T_{21}$', fontsize=15)
    ax3.set_ylabel(r'$x_{\rm HII}$', fontsize=15)
    
    leg =ax1.legend(loc=3, fontsize=15)
    leg.set_frame_on(False)
    
    leg = ax3.legend(loc=2, fontsize=15)
    leg.set_frame_on(False)

    ax2.set_xlabel(r'$z$', fontsize=15)
    ax2.set_ylabel(r'$\Delta^2_{21}(k=%g\,{\rm Mpc}^{-1})$'%k_vals[0], fontsize=15)
    leg = ax2.legend(loc=2, fontsize=15)
    leg.set_frame_on(False)
    
    ax4.set_xlabel(r'$z$', fontsize=15)
    ax4.set_ylabel(r'$\Delta^2_{21}(k=%g\,{\rm Mpc}^{-1})$'%k_vals[1], fontsize=15)
    leg = ax4.legend(loc=2, fontsize=15)
    leg.set_frame_on(False)
    
    ax1.set_xlim(6,20)
    ax2.set_xlim(6,20)
    ax3.set_xlim(6,20)
    ax4.set_xlim(6,20)

    ax1.set_ylim(-100,20)
    ax3.set_ylim(-0.1,1.1)

    plt.tight_layout()
    plt.savefig(plot_folder + filename)

    plt.show()

    return 


def plot_compare_boxes( idx_z, Lbox,
                        list_to_compare,
                        list_labels, 
                        filename):

    list_box = []

    for c in list_to_compare:
        list_box.append(c[6+idx_z])

    idx_zz = 1 if idx_z == 3 else 2 if idx_z == 4 else 0 

    vmin = [-30.,-60.,-100.]
    vmax = [30.,10.,-20.,]

    if idx_zz == 0:
        mid_point = abs(vmin[0])/(abs(vmin[0])+abs(vmax[0]))
        colors_list = [(0, "cyan"),
                        (mid_point/2., "blue"),
                        (mid_point, "black"),
                        ((1.+mid_point)/2., "red"),
                        (1, "yellow")]
    elif idx_zz == 1:
        mid_point = abs(vmin[1])/(abs(vmin[1])+abs(vmax[1]))
        colors_list = [(0, "cyan"),
                        (mid_point/2., "blue"),
                        (mid_point, "black"),
                        ((1.+mid_point)/2., "red"),
                        (1, "yellow")]
    else:
        colors_list =[(0, "cyan"),
                               (0.5, "blue"),
                               (1, "black")]
        
    eor_colour = colors.LinearSegmentedColormap.from_list("brightness_temp",colors_list)

    nmodels = (len(list_to_compare) - 1) // 2
        
    fig, axes = plt.subplots( nmodels, 3, figsize=(9, 3*nmodels), squeeze=False)

    Gaus_box = list_box[0]

    for i in range(nmodels):

        positive_fnl = list_box[2*i + 2]
        negative_fnl = list_box[2*i + 1]

        # left panel: A
        im = axes[i, 0].imshow(positive_fnl[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[idx_zz], vmax =vmax[idx_zz], cmap=eor_colour)
        plt.colorbar(im, ax = axes[i,0])
        axes[i,0].set_title(list_labels[2*i + 2])

        # middle panel: same val0 for all rows
        im1 = axes[i, 1].imshow(Gaus_box[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[idx_zz], vmax =vmax[idx_zz], cmap=eor_colour)
        plt.colorbar(im1, ax = axes[i,1])
        axes[i,1].set_title(list_labels[0])

        # right panel: B
        im2 = axes[i, 2].imshow(negative_fnl[0], aspect='auto',origin='lower',extent=(0, Lbox) * 2, vmin=vmin[idx_zz], vmax =vmax[idx_zz], cmap=eor_colour)
        plt.colorbar(im2, ax = axes[i,2])
        axes[i,2].set_title(list_labels[2*i + 1])

    plt.suptitle(r'$z = %g$'%input_coeval_redshifts[idx_z])

    plt.tight_layout()
    plt.savefig(plot_folder + 'z' + str(input_coeval_redshifts[idx_z]) + '_' + filename )

    plt.show()

    return 