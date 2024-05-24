"""Simple plotting functions for 21cmFAST objects."""
# JordanFlitter: I made extensive changes in this module

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un
from astropy.cosmology import z_at_value
from matplotlib import colors
from matplotlib.ticker import AutoLocator
from typing import Optional
# JordanFlitter: I import scipy to do 1d and 2d interpolation
from scipy.interpolate import interp1d, RectBivariateSpline
from matplotlib import colormaps
import warnings
import copy

from . import outputs
from .outputs import Coeval, LightCone
from .power_spectrum import POWER_SPECTRUM

eor_colour = colors.LinearSegmentedColormap.from_list(
    "EoR",
    [
        (0, "white"),
        (0.21, "yellow"),
        (0.42, "orange"),
        (0.63, "red"),
        (0.86, "black"),
        (0.9, "blue"),
        (1, "cyan"),
    ],
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore") # Don't show overwriting warnings
    colormaps.register(cmap=eor_colour,force=True)

# JordanFlitter: Frequency of the 21cm line
f_21 = 1420.40575177 # MHz

def _imshow_slice(
    cube,
    slice_axis=-1,
    slice_index=0,
    fig=None,
    ax=None,
    fig_kw=None,
    cbar=True,
    cbar_horizontal=False,
    rotate=False,
    cmap="EoR",
    log: [bool] = False,
    **imshow_kw,
):
    """
    Plot a slice of some kind of cube.

    Parameters
    ----------
    cube : nd-array
        A 3D array of some quantity.
    slice_axis : int, optional
        The axis over which to take a slice, in order to plot.
    slice_index :
        The index of the slice.
    fig : Figure object
        An optional matplotlib figure object on which to plot
    ax : Axis object
        The matplotlib axis object on which to plot (created by default).
    fig_kw :
        Optional arguments passed to the figure construction.
    cbar : bool
        Whether to plot the colorbar
    cbar_horizontal : bool
        Whether the colorbar should be horizontal underneath the plot.
    rotate : bool
        Whether to rotate the plot vertically.
    imshow_kw :
        Optional keywords to pass to :func:`maplotlib.imshow`.

    Returns
    -------
    fig, ax :
        The figure and axis objects from matplotlib.
    """
    # If no axis is passed, create a new one
    # This allows the user to add this plot into an existing grid, or alter it afterwards.
    if fig_kw is None:
        fig_kw = {}
    if ax is None and fig is None:
        fig, ax = plt.subplots(1, 1, **fig_kw)
    elif ax is None:
        ax = plt.gca()
    elif fig is None:
        fig = plt.gcf()

    plt.sca(ax)

    if slice_index >= cube.shape[slice_axis]:
        raise IndexError(
            "slice_index is too large for that axis (slice_index=%s >= %s"
            % (slice_index, cube.shape[slice_axis])
        )

    slc = np.take(cube, slice_index, axis=slice_axis)
    if not rotate:
        slc = slc.T

    if cmap == "EoR":
        imshow_kw["vmin"] = -150
        imshow_kw["vmax"] = 30

    norm = imshow_kw.get("norm", colors.LogNorm() if log else colors.Normalize())
    plt.imshow(slc, origin="lower", cmap=cmap, norm=norm, **imshow_kw)

    if cbar:
        cb = plt.colorbar(
            orientation="horizontal" if cbar_horizontal else "vertical", aspect=40
        )
        cb.outline.set_edgecolor(None)

    return fig, ax

# JordanFlitter: I modified this function
def coeval_sliceplot(
    lightcone: [LightCone],
    redshift: [float],
    kind: Optional[str] = None,
    slice_index: Optional[int] = 0,
    slice_axis: Optional[int] = -1,
    cbar: Optional[bool] = True,
    cbar_horizontal: Optional[bool] = False,
    cbar_label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Show a slice of a given coeval box.

    Parameters
    ----------
    lightcone: :class:`~py21cmfast.wrapper.Lightcone`
        The lightcone object that stores the coeval box to be plotted.
    redshift: float
        Redshift of the coeval box to be plotted. The redshift of the
        plotted coeval box has the value which is closest to the input redshift.
    kind: str, optional
        The quantity to be shown. Must be in the `coeval_boxes` dict in the lightcone.
        By default, will choose the first entry in the dict.
    slice_index: int, optional
        The index of the shown sliced. Can be as large as HII_DIM.
    slice_axis: int, optional
        The axis in which to the coeval box is sliced. Can be 0,1,2 (or -1).
    cbar: bool, optional
        Whether or not to show a colorbar.
    cbar_horizontal: bool, optional
        Whether to place the colorbar horizontally or veritcally.
    cbar_label: str, optional
        A label for the colorbar.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.imshow`.

    Returns
    -------
    fig, ax, cb :
        figure, axis and colorbar objects from matplotlib.

    """
    # Let's find first what is kind
    if kind is None:
        kind = list(lightcone.lightcones.keys())[0]

    # We find the redshift from the coeval boxes that we have that is closest to the user's redshift
    coeval_redshifts = list(lightcone.coeval_boxes.keys())
    coeval_redshift = coeval_redshifts[np.argmin(np.abs(redshift - np.array(coeval_redshifts)))]
    # We extract the coeval box from the lightcone object
    coeval_box = lightcone.coeval_boxes[coeval_redshift][kind]
    # We define the slice to be plotted
    if slice_index >= coeval_box.shape[slice_axis]:
        raise IndexError(
            "slice_index is too large for that axis (slice_index=%s >= %s"
            % (slice_index, coeval_box.shape[slice_axis]) +")"
        )
    coeval_slice = np.take(coeval_box, slice_index, axis=slice_axis)
    # We find the "minimum" and "maximum" values of the slice. We take percentile to handle outliers
    min_value = np.percentile(coeval_slice,1)
    max_value = np.percentile(coeval_slice,99)

    # Now, we set the colormap for the plot.
    # If we don't want to plot the brightness temperature, then any other quantity is positive by definition
    # and we can use the viridis colormap.
    # Otherwise, the brightness temperature slice might contain positive and negative values, and we need to
    # make sure that zero values are associated with black
    if not "cmap" in kwargs.keys():
        if kind != "brightness_temp":
            kwargs["cmap"]="viridis"
        else:
            # There are zero values! We'll have blue for negative temperatures and red for positive ones
            if min_value < 0 and max_value > 0:
                mid_point = abs(min_value)/(abs(min_value)+abs(max_value))
                colors_list = [(0, "cyan"),
                               (mid_point/2., "blue"),
                               (mid_point, "black"),
                               ((1.+mid_point)/2., "red"),
                               (1, "yellow")]
            # All values are negative! We'll have only blue colors
            elif max_value < 0:
                colors_list = [(0, "cyan"),
                               (0.5, "blue"),
                               (1, "black")]
            # All values are positive! We'll have only red colors
            else:
                colors_list = [(0, "black"),
                               (0.5, "red"),
                               (1., "yellow")]
            eor_colour = colors.LinearSegmentedColormap.from_list("brightness_temp",colors_list)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Don't show overwriting warnings
                colormaps.register(cmap=eor_colour,force=True)
            kwargs["cmap"] = "brightness_temp"

    # Now, set colorbar label and title
    if kind == 'brightness_temp':
        cbar_label_helper = '$T_{21}\,[\mathrm{mK}]$'
    elif kind == 'Tk_box':
        cbar_label_helper = '$T_k\,[\mathrm{K}]$'
    elif kind == 'Ts_box':
        cbar_label_helper = '$T_s\,[\mathrm{K}]$'
    elif kind == 'T_chi_box':
        cbar_label_helper = '$T_\chi\,[\mathrm{K}]$'
    elif kind == 'V_chi_b_box':
        cbar_label_helper = '$V_{\chi b}\,[\mathrm{km/sec}]$'
    elif kind == 'x_e_box':
        cbar_label_helper = '$x_e$'
    elif kind == 'xH_box':
        cbar_label_helper = '$x_\mathrm{HI}$'
    elif kind == 'J_Lya_box':
        cbar_label_helper = '$J_\\alpha\,[\mathrm{cm^{-2}\,Hz^{-1}\,sec^{-1}\,Str^{-1}}]$'
    elif kind == 'density':
        cbar_label_helper = '$\delta$'
    elif kind == 'baryons_density':
        cbar_label_helper = '$\delta_b$'
    digits = 3
    coeval_redshift = int(pow(10,digits)*coeval_redshift+0.5)/pow(10,digits)
    title = cbar_label_helper.split('\,')[0]
    if not title[-1] == '$':
        title += '$'
    title += f' at $z={coeval_redshift}$'
    if cbar_label is None:
        cbar_label = cbar_label_helper


    # Determine which axes are being plotted.
    if slice_axis in (2, -1):
        xax = "x"
        yax = "y"
    elif slice_axis == 1:
        xax = "x"
        yax = "z"
    elif slice_axis == 0:
        xax = "y"
        yax = "z"
    else:
        raise ValueError("slice_axis should be between -1 and 2")

    # Now, set fig and ax
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
    else:
        fig = ax.figure
    plt.sca(ax)

    # Plot the slice!
    plt.imshow(coeval_slice,aspect='auto',vmin=min_value,vmax=max_value,
               origin='lower',extent=(0, lightcone.user_params.BOX_LEN) * 2,**kwargs)

    # Add colorbar
    if cbar:
        cb = plt.colorbar(orientation="horizontal" if cbar_horizontal else "vertical")
        cb.ax.tick_params(labelsize=20)
        cb.ax.set_ylabel(cbar_label,fontsize=20)
    else:
        cb = None

    # Prettify the plot
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(xlabel=f'{xax}-axis [Mpc]',fontsize=25)
    ax.set_ylabel(ylabel=f'{yax}-axis [Mpc]',fontsize=25)
    if not title is None:
        ax.set_title(title,fontsize=25);

    # Return output
    return fig, ax, cb

# JordanFlitter: I modified this function
def lightcone_sliceplot(
    lightcone: [LightCone],
    kind: Optional[str] = None,
    slice_index: Optional[int] = 0,
    slice_axis: Optional[int] = 0,
    vertical: Optional[bool] = False,
    zticks: Optional[str] = 'redshift',
    cbar: Optional[bool] = True,
    cbar_horizontal: Optional[bool] = None,
    cbar_label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """Create a 2D plot of a slice through a lightcone.

    Parameters
    ----------
    lightcone: :class:`~py21cmfast.wrapper.Lightcone`
        The lightcone object that stores the lightcone box to be plotted.
    kind: str, optional
        The quantity to be shown. Must be in the `lightcones` dict in the lightcone.
        By default, will choose the first entry in the dict.
    slice_index: int, optional
        The index of the shown sliced. Can be as large as HII_DIM.
    slice_axis: int, optional
        The axis in which to the coeval box is sliced. Can be 0 or 1.
    vertical: bool, optional
        Whether to plot the lightcone box vertically or horizontally.
    zticks : str, optional
        Defines the co-ordinates of the ticks along the redshift axis.
        Can be "redshift" (default), "frequency" or "distance" (which starts at zero
        for the lowest redshift).
    cbar: bool, optional
        Whether or not to show a colorbar.
    cbar_horizontal: bool, optional
        Whether to place the colorbar horizontally or veritcally.
    cbar_label: str, optional
        A label for the colorbar.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.imshow`.

    Returns
    -------
    fig, ax, cb :
        figure, axis and colorbar objects from matplotlib.

    """
    # Let's find first what is kind
    if kind is None:
        kind = list(lightcone.lightcones.keys())[0]

    # Determine which axes are being plotted
    if slice_axis == 0:
        yax = "y"
    elif slice_axis == 1:
        yax = "x"
    else:
        raise ValueError(f"slice_axis should be either 0 or 1 (got {slice_axis})")

    # We extract the lightcone box from the lightcone object
    lightcone_box = lightcone.lightcones[kind]
    # We define the slice to be plotted
    if slice_index >= lightcone_box.shape[slice_axis]:
        raise IndexError(
            "slice_index is too large for that axis (slice_index=%s >= %s"
            % (slice_index, lightcone_box.shape[slice_axis]) +")"
        )
    lightcone_slice = np.take(lightcone_box, slice_index, axis=slice_axis)
    if vertical:
        lightcone_slice = lightcone_slice.T[::-1,:]
    # We find the "minimum" and "maximum" values of the slice. We take percentile to handle outliers
    min_value = np.percentile(lightcone_slice,1)
    max_value = np.percentile(lightcone_slice,99)

    # Now, we set the colormap for the plot.
    # If we don't want to plot the brightness temperature, then any other quantity is positive by definition
    # and we can use the viridis colormap.
    # Otherwise, the brightness temperature slice might contain positive and negative values, and we need to
    # make sure that zero values are associated with black
    if not "cmap" in kwargs.keys():
        if kind != "brightness_temp":
            kwargs["cmap"]="viridis"
        else:
            mid_point = abs(min_value)/(abs(min_value)+abs(max_value))
            colors_list = [(0, "cyan"),
                           (mid_point/2., "blue"),
                           (mid_point, "black"),
                           ((1.+mid_point)/2., "red"),
                           (1, "yellow")]
            eor_colour = colors.LinearSegmentedColormap.from_list("brightness_temp",colors_list)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Don't show overwriting warnings
                colormaps.register(cmap=eor_colour,force=True)
            kwargs["cmap"] = "brightness_temp"

    # Define values for the z-axis labels and define the grid of the interpolation
    if zticks == "distance":
        max_distance_value = lightcone.lightcone_coords[-1]
        distance_values_array = np.array([0, 1000, 2000, 3000, 4000, 5000])
        z_values_array = distance_values_array[distance_values_array < max_distance_value]
        z_grid = lightcone.lightcone_coords
        z_label = "Line-of-Sight Distance [Mpc]"
    elif zticks == "redshift":
        max_redshift_value = lightcone.node_redshifts[0]
        redshift_values = list(np.arange(int(lightcone.redshift)+1,11))
        redshift_values += [12,15,20,25,35,50,100,200,1000]
        redshift_values_array = np.array(redshift_values)
        z_values_array = redshift_values_array[redshift_values_array < max_redshift_value]
        z_grid = lightcone.lightcone_redshifts
        z_label = '$z$'
    elif zticks == "frequency":
        # Get labels for the z-axis
        min_frequency_value = f_21/(1.+lightcone.lightcone_redshifts[-1])
        frequency_values_array = np.array([5, 10, 20, 30, 40, 50, 70, 85, 100, 120, 150, 170, 200])
        z_values_array = frequency_values_array[frequency_values_array > min_frequency_value]
        z_grid = f_21/(1.+lightcone.lightcone_redshifts)
        z_label = '$\\nu\,[\mathrm{MHz}]$'
    else:
        raise ValueError(f"zticks should be either 'redshift', 'distance', or 'frequency'.")
    # Get ticks and labels for the z-axis
    z_axis_labels = [str(z) for z in z_values_array]
    z_ticks = interp1d(z_grid,lightcone.lightcone_coords,
                        kind='cubic',bounds_error=False)(z_values_array)
    if zticks == "redshift":
        z_axis_labels.insert(0,str(int(lightcone.redshift)))
        z_ticks = np.insert(z_ticks,0,0.)
    if vertical:
        if zticks != "frequency":
            z_axis_labels = z_axis_labels[::-1]
        z_ticks = np.sort(lightcone.lightcone_coords[-1]-z_ticks)

    # Now, set colorbar label
    if cbar_label is None:
        if kind == 'brightness_temp':
            cbar_label = '$T_{21}\,[\mathrm{mK}]$'
        elif kind == 'Tk_box':
            cbar_label = '$T_k\,[\mathrm{K}]$'
        elif kind == 'Ts_box':
            cbar_label = '$T_s\,[\mathrm{K}]$'
        elif kind == 'T_chi_box':
            cbar_label = '$T_\chi\,[\mathrm{K}]$'
        elif kind == 'V_chi_b_box':
            cbar_label = '$V_{\chi b}\,[\mathrm{km/sec}]$'
        elif kind == 'x_e_box':
            cbar_label = '$x_e$'
        elif kind == 'xH_box':
            cbar_label = '$x_\mathrm{HI}$'
        elif kind == 'J_Lya_box':
            cbar_label = '$J_\\alpha\,[\mathrm{cm^{-2}\,Hz^{-1}\,sec^{-1}\,Str^{-1}}]$'
        elif kind == 'density':
            cbar_label = '$\delta$'
        elif kind == 'baryons_density':
            cbar_label = '$\delta_b$'

    # Define the extent of the lightcone plot
    if vertical:
        extent = (0,lightcone.lightcone_dimensions[0],
                  0,lightcone.lightcone_dimensions[2])
    else:
        extent = (0,lightcone.lightcone_dimensions[2],
                  0,lightcone.lightcone_dimensions[0])

    # Now, set fig and ax
    if ax is None:
        if not vertical:
            figsize = (15,7)
        else:
            figsize = (4,15)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    plt.sca(ax)

    # Plot the slice!
    plt.imshow(lightcone_slice,aspect='auto',vmin=min_value,vmax=max_value,
               origin='lower',extent=extent,**kwargs)

    # Add colorbar
    if cbar:
        if cbar_horizontal is None:
            cbar_horizontal = not vertical
        cb = plt.colorbar(orientation="horizontal" if cbar_horizontal else "vertical")
        cb.ax.tick_params(labelsize=20)
        cb.ax.set_ylabel(cbar_label,fontsize=20)
    else:
        cb = None

    # Prettify the plot
    if not vertical:
        ax.set_xticks(ticks=z_ticks)
        ax.set_xticklabels(labels=z_axis_labels)
        ax.set_xlabel(xlabel=z_label,fontsize=25)
        ax.set_ylabel(ylabel=f'{yax}-axis [Mpc]',fontsize=25)
    else:
        ax.set_yticks(ticks=z_ticks)
        ax.set_yticklabels(labels=z_axis_labels)
        ax.set_ylabel(ylabel=z_label,fontsize=25)
        ax.set_xlabel(xlabel=f'{yax}-axis [Mpc]',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)

    # Return output
    return fig, ax, cb


def _set_zaxis_ticks(ax, lightcone, zticks, z_axis):
    if zticks != "distance":
        loc = AutoLocator()
        # Get redshift ticks.
        lc_z = lightcone.lightcone_redshifts

        if zticks == "redshift":
            coords = lc_z
        elif zticks == "frequency":
            coords = f_21 / (1 + lc_z) * un.MHz
        else:
            try:
                coords = getattr(lightcone.cosmo_params.cosmo, zticks)(lc_z)
            except AttributeError:
                raise AttributeError(f"zticks '{zticks}' is not a cosmology function.")

        zlabel = " ".join(z.capitalize() for z in zticks.split("_"))
        units = getattr(coords, "unit", None)
        if units:
            zlabel += f" [{str(coords.unit)}]"
            coords = coords.value

        ticks = loc.tick_values(coords.min(), coords.max())

        if ticks.min() < coords.min() / 1.00001:
            ticks = ticks[1:]
        if ticks.max() > coords.max() * 1.00001:
            ticks = ticks[:-1]

        if coords[1] < coords[0]:
            ticks = ticks[::-1]

        if zticks == "redshift":
            z_ticks = ticks
        elif zticks == "frequency":
            z_ticks = f_21 / ticks - 1
        else:
            z_ticks = [
                z_at_value(getattr(lightcone.cosmo_params.cosmo, zticks), z * units)
                for z in ticks
            ]

        d_ticks = (
            lightcone.cosmo_params.cosmo.comoving_distance(z_ticks).value
            - lightcone.lightcone_distances[0]
        )
        getattr(ax, f"set_{z_axis}ticks")(d_ticks)
        getattr(ax, f"set_{z_axis}ticklabels")(ticks)

    else:
        zlabel = "Line-of-Sight Distance [Mpc]"
    return zlabel

# JordanFlitter: I modified this function
def plot_global_history(
    lightcone: [LightCone],
    kind: Optional[str] = None,
    x_kind: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlog: Optional[bool] = None,
    ylog: Optional[bool] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Plot the global history of a given quantity from a lightcone.

    Parameters
    ----------
    lightcone: :class:`~py21cmfast.wrapper.Lightcone`
        The lightcone object that stores the global quantity to be plotted.
    kind: str, optional
        The quantity to be shown. Must be in the `global_quantities` dict in the lightcone.
        By default, will choose the first entry in the dict.
    x_kind : str, optional
        Defines the x-axis of the plot.
        Can be "redshift" (default), "frequency" or "distance" (which starts at zero
        for the lowest redshift).
    xlabel: str, optional
        Label of the x-axis.
    ylabel: str, optional
        Label of the y-axis.
    xlog: bool, optional
        Whether to plot the x-axis in log scale or linear scale.
    ylog: bool, optional
        Whether to plot the y-axis in log scale or linear scale.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.

    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.

    """

    # Let's find first what is kind
    if kind is None:
        kind = list(lightcone.global_quantities.keys())[0]

    assert (
        kind in lightcone.global_quantities
        or hasattr(lightcone, "global_" + kind)
        or (kind.startswith("global_") and hasattr(lightcone, kind))
    )

    # Now, let's find what are the y_values based on kind
    if kind in lightcone.global_quantities:
        y_values = lightcone.global_quantities[kind]
    elif kind.startswith("global)"):
        y_values = getattr(lightcone, kind)
    else:
        y_values = getattr(lightcone, "global_" + kind)

    # Now, let's find what are the x_values based on x_kind
    if x_kind is None:
        x_kind = 'redshift'
    if x_kind == 'redshift':
        x_values = lightcone.node_redshifts
    elif x_kind == 'frequency':
        x_values = f_21/(np.array(lightcone.node_redshifts)-1.)
    elif x_kind == 'distance':
        x_values = interp1d(lightcone.lightcone_redshifts,lightcone.lightcone_coords,
                        kind='cubic',bounds_error=False)(lightcone.node_redshifts)
        # This can happen sometimes...
        if np.isnan(x_values[0]):
            x_values[0] = lightcone.lightcone_coords[-1]
        if np.isnan(x_values[-1]):
            x_values[-1] = lightcone.lightcone_coords[0]
    else:
        raise ValueError("x_kind must be either redshift, distance, or frequency.")

    # Now, set xlabel and ylabel
    if xlabel is None:
        if x_kind == 'redshift':
            xlabel = '$z$'
        elif x_kind == 'distance':
            xlabel = 'Line-of-Sight Distance [Mpc]'
        else:
            xlabel = '$\\nu\,[\mathrm{MHz}]$'
    if ylabel is None:
        if kind == 'brightness_temp':
            ylabel = '$T_{21}\,[\mathrm{mK}]$'
        elif kind == 'Tk_box':
            ylabel = '$T_k\,[\mathrm{K}]$'
        elif kind == 'Ts_box':
            ylabel = '$T_s\,[\mathrm{K}]$'
        elif kind == 'T_chi_box':
            ylabel = '$T_\chi\,[\mathrm{K}]$'
        elif kind == 'V_chi_b_box':
            ylabel = '$V_{\chi b}\,[\mathrm{km/sec}]$'
        elif kind == 'x_e_box':
            ylabel = '$x_e$'
        elif kind == 'xH_box':
            ylabel = '$x_\mathrm{HI}$'
        elif kind == 'J_Lya_box':
            ylabel = '$J_\\alpha\,[\mathrm{cm^{-2}\,Hz^{-1}\,sec^{-1}\,Str^{-1}}]$'

    # Now, set xlog and ylog
    if xlog is None and lightcone.user_params.OUTPUT_AT_DARK_AGES and x_kind == 'redshift':
        xlog = True
    if ylog is None and not kind in ['brightness_temp','J_Lya_box']:
        ylog = True

    # Now, set fig and ax
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    # Plot the global history!
    ax.plot(x_values,y_values,**kwargs)

    # Prettify the plot
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim([min(x_values),max(x_values)])
    ax.set_xlabel(xlabel,fontsize=25)
    ax.set_ylabel(ylabel,fontsize=25)
    if "label" in kwargs:
        ax.legend(fontsize=20)

    # Return output
    return fig, ax

# JordanFlitter: I added this function
def plot_Cl_data(
    lightcone: [LightCone],
    mode: Optional[str] = 'TT',
    Dl_plot: Optional[bool] = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlog: Optional[bool] = None,
    ylog: Optional[bool] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Plot the CMB power spectrum of a given mode from a lightcone.

    Parameters
    ----------
    lightcone: :class:`~py21cmfast.wrapper.Lightcone`
        The lightcone object that stores the CMB power spectrum to be plotted.
    mode: str, optional
        The mode of CMB power spectrum to plot.
        Can be 'TT', 'EE', 'TE', 'BB', 'PP' or 'TP'.
    Dl_plot: bool, optional
        Whether to plot D_ell = (ell(ell+1)/(2*pi))*C_ell in units of (micro-K)^2.
    xlabel: str, optional
        Label of the x-axis.
    ylabel: str, optional
        Label of the y-axis.
    xlog: bool, optional
        Whether to plot the x-axis in log scale or linear scale.
    ylog: bool, optional
        Whether to plot the y-axis in log scale or linear scale.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.

    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """

    # First, let's extract the Cl data from the lightcone object
    mode = mode.lower()
    ell = lightcone.Cl_data['ell'][2:]
    C_ell = lightcone.Cl_data[mode][2:]

    # Now, let's find the x and y values for the plot
    x_values = lightcone.Cl_data['ell'][2:]
    if Dl_plot:
        T_CMB = 2.728e6 # micro-K
        y_values = ell*(ell+1)*C_ell/2/np.pi*T_CMB**2 # (micro-K)^2
        ylabel = '$[\ell(\ell+1)/2\pi]C_\ell^\mathrm{'+mode.upper()+'}\,[\mathrm{\\mu K}^2]$'
    else:
        y_values = C_ell
        ylabel = '$C_\ell^\mathrm{'+mode.upper()+'}$'

    # Now, set fig and ax
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    else:
        fig = ax.figure

    # Plot the Cl data!
    ax.plot(x_values,y_values,**kwargs);

    # Prettify the plot
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim([min(x_values),max(x_values)])
    ax.set_xlabel(xlabel='$\ell$',fontsize=25)
    ax.set_ylabel(ylabel=ylabel,fontsize=25)
    if "label" in kwargs:
        ax.legend(fontsize=20)

    # Return output
    return fig, ax

# JordanFlitter: I added this function
def plot_1d_power_spectrum(
    power_spectrum: [POWER_SPECTRUM],
    k: Optional[float] = None,
    z: Optional[float] = None,
    nu: Optional[float] = None,
    x_kind: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlog: Optional[bool] = None,
    ylog: Optional[bool] = None,
    smooth: Optional[bool] = None,
    noise_data = None,
    error_bars: Optional[bool] = None,
    redshift_axis_on_top: Optional[bool] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Plot the 21cm power spectrum as a function of redshift/frequency (or wavenumber),
    for a fixed wavnumber (or redshift/frequency)

    Parameters
    ----------
    power_spectrum: :class:`~py21cmfast.power_spectrum.POWER_SPECTRUM`
        The power spectrum object that stores the data to be plotted.
    k: float, optional
        The fixed wavenumber (in Mpc). Power spectrum will be plotted as a
        function of redshift/frequency.
        Must be specified if z and nu are not specified.
    z: float, optional
        The fixed redshift. Power spectrum will be plotted as a
        function of wavnumber.
        Must be specified if nu and k are not specified.
    nu: float, optional
        The fixed frequency (in MHz). Power spectrum will be plotted as a
        function of wavnumber.
        Must be specified if k and z are not specified.
    x_kind : str, optional
        Defines the x-axis of the plot.
        Can be "redshift" (default), or "frequency".
        Only works if k is specified (rather than z or nu).
    xlabel: str, optional
        Label of the x-axis.
    ylabel: str, optional
        Label of the y-axis.
    xlog: bool, optional
        Whether to plot the x-axis in log scale or linear scale.
    ylog: bool, optional
        Whether to plot the y-axis in log scale or linear scale.
    smooth: bool, optional
        Whether to smooth the curve or not.
    noise_data: :class:`~py21cmfast.experiment.NOISE_DATA`, optional
        The noise data object that stores the data of the noise.
        If None or not specified, no noise will be plotted.
    error_bars: bool, optional
        If True, noise will be displayed as error bars. Otherwise noise will be
        plotted as shaded regions. Default is False.
    redshift_axis_on_top: bool, optional
        Whether to display a second horizontal axis of redshift at the top of the
        plot. Only works when x_kind is "frequency".
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.plot`.

    Returns
    -------
    fig, ax:
        figure and axis objects from matplotlib.
    """

    # First, let's extract the power specturm values
    k_values = power_spectrum.k_values
    z_values = power_spectrum.z_values
    ps_values = power_spectrum.ps_values

    # Now, let's look for conflict in x_kind argument
    if k is None:
        if x_kind is not None:
            raise TypeError("You have set 'x_kind' but given no k argument!"
                            " You can choose 'x_kind' to be only 'redshift' or 'frequency',"
                            " but k must be specified.")
    else:
        if (x_kind is not None) and (not x_kind in ['redshift', 'frequency']):
            raise TypeError("When k is specified, x_kind must be either 'redshift' or 'frequency'.")

    # Now, let's find what is the x_kind based on the input
    if ((z is None) and (nu is None)) and (k is not None):
        if len(z_values) > 1:
            if x_kind is None:
                x_kind = 'redshift'
        else:
            raise TypeError("You have computed the power spectrum of a coeval box but attempt to plot it as"
                            " a function of redshift/frequency for a specific k!"
                            " Change the input argument to 'z' or don't specify any argument.")
    elif (z is not None) and (k is None):
        if nu is not None:
            warnings.warn("You have specified both z and nu!"
                          " You can only plot the power spectrum for a fixed z or nu, not both.")
        else:
            x_kind = 'wavenumber'
            if len(z_values) == 1 and (not z == z_values[0]):
                warnings.warn("You have specified a redshift which is inconsistent with your ceoval box!"
                              " Plotting the power spectrum for the redshift of your coeval box.")
                z = z_values[0]
            if (noise_data is not None) and (not z in noise_data.z_values):
                warnings.warn("You attempt to plot noise for a given redshift,"
                              " but your noise data doesn't contain that redshift!"
                              " Finding the closest redshift in the noise data")
                z = noise_data.z_values[np.argmin(np.abs(noise_data.z_values-z))]
    elif (nu is not None) and (k is None):
        if z is not None:
            warnings.warn("You have specified both z and nu!"
                          " You can only plot the power spectrum for a fixed z or nu, not both.")
        else:
            z = f_21/nu - 1.
            x_kind = 'wavenumber'
            if len(z_values) == 1 and (not z == z_values[0]):
                warnings.warn("You have specified a frequency which is inconsistent with your ceoval box!"
                              " Plotting the power spectrum for the frequency of your coeval box.")
                z = z_values[0]
            if (noise_data is not None) and (not z in noise_data.z_values):
                warnings.warn("You attempt to plot noise for a given frequency,"
                              " but your noise data doesn't contain that frequency!"
                              " Finding the closest frequency in the noise data")
                z = noise_data.z_values[np.argmin(np.abs(noise_data.z_values-z))]
    # Wrong input...
    elif (z is not None) and (k is not None):
        raise TypeError("In order to plot the 1d power spectrum you need to specify either k or z, not both.")
    elif (nu is not None) and (k is not None):
        raise TypeError("In order to plot the 1d power spectrum you need to specify either k or nu, not both.")
    else:
        if len(z_values) == 1:
            x_kind = 'wavenumber'
            z = z_values[0]
            if (noise_data is not None) and (not z in noise_data.z_values):
                warnings.warn("You attempt to plot noise for a given redshift,"
                              " but your noise data doesn't contain the same redshift "
                              " of your coeval box!"
                              " Finding the closest redshift in the noise data")
                z = noise_data.z_values[np.argmin(np.abs(noise_data.z_values-z))]
        else:
            raise TypeError("In order to plot the 1d power spectrum you need to specify either k or z.")

    # Now, set xlog and ylog
    if xlog is None and ((x_kind == 'redshift' and max(z_values)/min(z_values) > 10) or
                         (x_kind == 'wavnumber' and max(k_values)/min(k_values) > 10)):
        xlog = True
    if ylog is None and (x_kind == 'redshift' or x_kind == 'frequency'):
        ylog = True

    # We plot the power spectrum vs.redshift!
    if x_kind == 'redshift' or x_kind == 'frequency':
        if x_kind == 'frequency':
            z_values = np.sort(f_21/(1.+z_values)) # We convert z_values to frequency wherever x_kind = 'redshift' or 'frequency'
            ps_values = ps_values[::-1,:] # We also need to mirror the power spectrum matrix since we sorted the frequencies
        if smooth:
            if xlog:
                x_values = np.logspace(np.log10(min(z_values)),np.log10(max(z_values)),4*len(z_values))
            else:
                x_values = np.linspace(min(z_values),max(z_values),4*len(z_values))
        else:
            x_values = z_values
        if (k < min(k_values)):
            raise ValueError(f'Too small k! Minimum k in the power spectrum calculation is k_min={min(k_values):.4}'
                             f' but received k={k}.')
        elif (k > max(k_values)):
            raise ValueError(f'Too large k! Maximum k in the power spectrum calculation is k_max={max(k_values):.4}'
                             f' but received k={k}.')
        else:
            if xlog and ylog:
                y_values = np.exp(RectBivariateSpline(k_values, np.log(z_values), np.log(ps_values.T))(k,np.log(x_values)))[0]
            elif xlog:
                y_values = RectBivariateSpline(k_values, np.log(z_values), ps_values.T)(k,np.log(x_values))[0]
            elif ylog:
                y_values = np.exp(RectBivariateSpline(k_values, z_values, np.log(ps_values.T))(k,x_values))[0]
            else:
                y_values = RectBivariateSpline(k_values, z_values, ps_values.T)(k,x_values)[0]

    # We plot the power spectrum vs.wavenumber!
    if x_kind == 'wavenumber':
        if smooth:
            # TalAdi: Increse the number of points for the interpolation to avoid early edges (4->40)
            if xlog:
                x_values = np.logspace(np.log10(min(k_values)),np.log10(max(k_values)),40*len(k_values))
            else:
                x_values = np.linspace(min(k_values),max(k_values),40*len(k_values))
        else:
            x_values = k_values
        if (z < min(z_values) and len(z_values)>1):
            raise ValueError(f'Too small z! Minimum z in the power spectrum calculation is z_min={min(z_values):.4}'
                             f' but received z={z}.')
        elif (z > max(z_values) and len(z_values)>1):
            raise ValueError(f'Too large z! Maximum z in the power spectrum calculation is z_max={max(z_values):.4}'
                             f' but received z={z}.')
        else:
            if len(z_values) == 1:
                if xlog and ylog:
                    y_values = np.exp(interp1d(np.log(k_values), np.log(ps_values), kind='cubic')(np.log(x_values)))
                elif xlog:
                    y_values = interp1d(np.log(k_values), ps_values, kind='cubic')(np.log(x_values))
                elif ylog:
                    y_values = np.exp(interp1d(k_values, np.log(ps_values), kind='cubic')(x_values))
                else:
                    y_values = interp1d(k_values, ps_values, kind='cubic')(x_values)
            else:
                if xlog and ylog:
                    y_values = np.exp(RectBivariateSpline(np.log(k_values), z_values, np.log(ps_values.T))(np.log(x_values),z))
                elif xlog:
                    y_values = RectBivariateSpline(np.log(k_values), z_values, ps_values.T)(np.log(x_values),z)
                elif ylog:
                    y_values = np.exp(RectBivariateSpline(k_values, z_values, np.log(ps_values.T))(x_values,z))
                else:
                    y_values = RectBivariateSpline(k_values, z_values, ps_values.T)(x_values,z)

    # Now, set xlabel and ylabel
    if xlabel is None:
        if x_kind == 'redshift':
            xlabel = '$z$'
        elif x_kind == 'frequency':
            xlabel = '$\\nu\,[\mathrm{MHz}]$'
        else:
            xlabel = '$k\,[\mathrm{Mpc}^{-1}]$'
    if ylabel is None:
        ylabel = '$\Delta^2_'
        if power_spectrum.kind == 'brightness_temp':
            ylabel += '{21}\,[\mathrm{mK}^2]\,'
        elif power_spectrum.kind == 'Tk_box':
            ylabel += '{T_k}\,[\mathrm{K}^2]\,'
        elif power_spectrum.kind == 'Ts_box':
            ylabel += '{T_s}\,[\mathrm{K}^2]\,'
        elif power_spectrum.kind == 'T_chi_box':
            ylabel += '{T_\chi}\,[\mathrm{K}^2]\,'
        elif power_spectrum.kind == 'V_chi_b_box':
            ylabel += '{V_{\chi b}}\,[\mathrm{km/sec}^2]\,'
        elif power_spectrum.kind == 'x_e_box':
            ylabel += '{x_e}\,'
        elif power_spectrum.kind == 'xH_box':
            ylabel += '{x_\mathrm{HI}}\,'
        elif power_spectrum.kind == 'J_Lya_box':
            ylabel += '{J_\\alpha}\,[\mathrm{cm^{-4}\,Hz^{-2}\,sec^{-2}\,Str^{-2}}]\,'
        if x_kind == 'redshift' or x_kind == 'frequency':
            ylabel += f'(k={k}'+'\,\mathrm{Mpc}^{-1})$'
        else:
            if nu is None:
                ylabel += f'(z={z})$'
            else:
                if (noise_data is not None) and (not (f_21/nu-1.) in noise_data.z_values):
                    ylabel += f'(\\nu={f_21/(1.+z)}'+'\,\mathrm{MHz})$'
                else:
                    ylabel += f'(\\nu={nu}'+'\,\mathrm{MHz})$'

    # Now, set fig and ax
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    # Plot the power spectrum!
    ax.plot(x_values,y_values,**kwargs)

    # Prettify the plot
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim([min(x_values),max(x_values)])
    ax.set_xlabel(xlabel,fontsize=25)
    ax.set_ylabel(ylabel,fontsize=22)
    if "label" in kwargs:
        ax.legend(fontsize=20)

    # Plot noise!
    if noise_data is not None:
        # Get error bars
        if x_kind == 'redshift' or x_kind == 'frequency':
            if x_kind == 'redshift':
                x_values_noise = noise_data.z_values
            else:
                x_values_noise = np.sort(f_21/(1.+noise_data.z_values)) # We convert noise redshifts to frequency wherever x_kind = 'redshift' or 'frequency'
            errors = np.zeros_like(x_values_noise)
            for z_ind, z in enumerate(x_values_noise):
                if x_kind == 'redshift':
                    k_noise = noise_data.k_noise[z]
                    noise = noise_data.noise[z]
                else:
                    k_noise = noise_data.k_noise[noise_data.z_values[-1-z_ind]]
                    noise = noise_data.noise[noise_data.z_values[-1-z_ind]]
                good_inds = ~np.isinf(noise)
                errors[z_ind] = interp1d(k_noise[good_inds],noise[good_inds],
                                         kind='cubic',bounds_error=False)(k)
                if errors[z_ind] < 0.:
                    errors[z_ind] = abs(errors[z_ind])
                if np.isnan(errors[z_ind]):
                    errors[z_ind] = 1e16 # infinity (almost)
            # Get centers of the error bars
            if xlog and ylog:
                y_values_noise = np.exp(RectBivariateSpline(k_values, np.log(z_values), np.log(ps_values.T))(k,np.log(x_values_noise)))[0]
            elif xlog:
                y_values_noise = RectBivariateSpline(k_values, np.log(z_values), ps_values.T)(k,np.log(x_values_noise))[0]
            elif ylog:
                y_values_noise = np.exp(RectBivariateSpline(k_values, z_values, np.log(ps_values.T))(k,x_values_noise))[0]
            else:
                y_values_noise = RectBivariateSpline(k_values, z_values, ps_values.T)(k,x_values_noise)[0]
        if x_kind == 'wavenumber':
            x_values_noise = noise_data.k_noise[z]
            errors = copy.deepcopy(noise_data.noise[z])
            errors[np.isinf(errors)] = 1e16 # infinity (almost)
            # Get centers of the error bars
            if len(z_values) == 1:
                if xlog and ylog:
                    y_values_noise = np.exp(interp1d(np.log(k_values), np.log(ps_values), kind='cubic', bounds_error=False)(np.log(x_values_noise)))
                elif xlog:
                    y_values_noise = interp1d(np.log(k_values), ps_values, kind='cubic', bounds_error=False)(np.log(x_values_noise))
                elif ylog:
                    y_values_noise = np.exp(interp1d(k_values, np.log(ps_values), kind='cubic', bounds_error=False)(x_values_noise))
                else:
                    y_values_noise = interp1d(k_values, ps_values, kind='cubic', bounds_error=False)(x_values_noise)
            else:
                if xlog and ylog:
                    y_values_noise = np.exp(RectBivariateSpline(np.log(k_values), z_values, np.log(ps_values.T))(np.log(x_values_noise),z))
                elif xlog:
                    y_values_noise = RectBivariateSpline(np.log(k_values), z_values, ps_values.T)(np.log(x_values_noise),z)
                elif ylog:
                    y_values_noise = np.exp(RectBivariateSpline(k_values, z_values, np.log(ps_values.T))(x_values_noise,z))
                else:
                    y_values_noise = RectBivariateSpline(k_values, z_values, ps_values.T)(x_values_noise,z)
                y_values_noise = y_values_noise.T[0]
        ylim = ax.get_ylim()
        if error_bars:
            # Plot noise as error bars!
            ax.errorbar(x_values_noise,y_values_noise,yerr=errors, ls='none', capsize=3,
                        color=ax.lines[-1].get_color())
        else:
            y_upper = y_values_noise + errors
            y_lower = y_values_noise - errors
            # TalAdi: Set threshold for infinities
            thresh = 1e15
            # Find good inds for interpolation (we don't care for what lies beyond x_values)
            up_inds = (x_values_noise >= min(x_values)) & (x_values_noise <= max(x_values)) & (y_upper < thresh)    # TalAdi: Set threshold for infinities
            # TalAdi: Find the indices of the True values with more than one False in between
            bol_bridge_ind = []
            prev_true_index = None
            for i, val in enumerate(up_inds):
                if val:
                    if prev_true_index is not None and i - prev_true_index > 1:
                        bol_bridge_ind.extend([prev_true_index, i])
                    prev_true_index = i
            # TalAdi: If no bol_bridge isn't found, exclude the values that are above the threshold
            if len(bol_bridge_ind) == 0:
                interp_kind = 'cubic'
                # Keep x_values_noise where y_upper is biggger than the threshold
                x_values_noise_inf_noise = x_values_noise[y_upper >= thresh]
            # TalAdi: If bol_bridge is found, set the interpolation kind to linear and include the values that are above the threshold
            else:
                interp_kind = 'linear'
                up_inds = (x_values_noise >= min(x_values)) & (x_values_noise <= max(x_values))
                x_values_noise_inf_noise = None
            if ylog:
                low_inds = (y_lower > 0.) & (x_values_noise >= min(x_values)) & (x_values_noise <= max(x_values))
                x_values_noise_minf_noise = None
            else:
                low_inds = (x_values_noise >= min(x_values)) & (x_values_noise <= max(x_values)) & (y_lower > -thresh)  # TalAdi: Set threshold for infinities
                # TalAdi: Again, if bol_bridge isn't found, exclude the values that are below the threshold
                if len(bol_bridge_ind) == 0:
                    # Keep x_values_noise where y_lower is smaller than the threshold
                    x_values_noise_minf_noise = x_values_noise[y_upper <= -thresh]
                # TalAdi: If bol_bridge is found, set the interpolation kind to linear and include the values that are below the threshold
                else:
                    low_inds = (x_values_noise >= min(x_values)) & (x_values_noise <= max(x_values))
                    x_values_noise_minf_noise = None
            # In case of shaded region, we need to interpolate the boundaries at x_values (so the shaded region brackets the curve)
            if xlog and ylog:
                y_upper = np.exp(interp1d(np.log(x_values_noise[up_inds]), np.log(y_upper[up_inds]), kind=interp_kind, bounds_error=False)(np.log(x_values)))   # TalAdi: Set interpolation kind
                y_lower = np.exp(interp1d(np.log(x_values_noise[low_inds]), np.log(y_lower[low_inds]), kind=interp_kind, bounds_error=False)(np.log(x_values)))  # TalAdi: Set interpolation kind
            elif xlog:
                y_upper = interp1d(np.log(x_values_noise[up_inds]), y_upper[up_inds], kind=interp_kind, bounds_error=False)(np.log(x_values))   # TalAdi: Set interpolation kind
                y_lower = interp1d(np.log(x_values_noise[low_inds]), y_lower[low_inds], kind=interp_kind, bounds_error=False)(np.log(x_values)) # TalAdi: Set interpolation kind
            elif ylog:
                y_upper = np.exp(interp1d(x_values_noise[up_inds], np.log(y_upper[up_inds]), kind=interp_kind, bounds_error=False)(x_values))   # TalAdi: Set interpolation kind
                y_lower = np.exp(interp1d(x_values_noise[low_inds], np.log(y_lower[low_inds]), kind=interp_kind, bounds_error=False)(x_values)) # TalAdi: Set interpolation kind
            else:
                y_upper = interp1d(x_values_noise[up_inds], y_upper[up_inds], kind=interp_kind, bounds_error=False)(x_values)   # TalAdi: Set interpolation kind
                y_lower = interp1d(x_values_noise[low_inds], y_lower[low_inds], kind=interp_kind, bounds_error=False)(x_values) # TalAdi: Set interpolation kind
            # TalAdi: Add x_values_noise_inf_noise to x_values_noise in their appropriate places, and the corresponding y_upper and y_lower as nans
            if x_values_noise_inf_noise is not None:
                combined_x = np.concatenate((x_values_noise_inf_noise, x_values))
                combined_y_upper = np.concatenate((np.nan*np.ones_like(x_values_noise_inf_noise), y_upper))
                combined_y_lower = np.concatenate((np.nan*np.ones_like(x_values_noise_inf_noise), y_lower))
                sorted_indices = np.argsort(combined_x)
                x_values = combined_x[sorted_indices]
                y_upper = combined_y_upper[sorted_indices]
                y_lower = combined_y_lower[sorted_indices]
            if x_values_noise_minf_noise is not None:
                combined_x = np.concatenate((x_values_noise_minf_noise, x_values))
                combined_y_upper = np.concatenate((np.nan*np.ones_like(x_values_noise_minf_noise), y_upper))
                combined_y_lower = np.concatenate((np.nan*np.ones_like(x_values_noise_minf_noise), y_lower))
                sorted_indices = np.argsort(combined_x)
                x_values = combined_x[sorted_indices]
                y_upper = combined_y_upper[sorted_indices]
                y_lower = combined_y_lower[sorted_indices]
            # ================================================================
            # Fix nans
            y_upper[np.isnan(y_upper)] = 1e16
            if ylog:
                y_lower[np.isnan(y_lower)] = 1e-16
            else:
                y_lower[np.isnan(y_lower)] = -1e16
            # Plot noise as a shaded region!
            ax.fill_between(x_values,y_lower,y_upper,
                            color=ax.lines[-1].get_color(),alpha=0.3)
        ax.set_ylim(ylim)

    # Put redshift axis on top (courtesy of Tal Adi)
    if x_kind == 'frequency' and redshift_axis_on_top:
        # Reset the x-axis to integer redshifts
        # Get the frequency ticks
        freq_lim = ax.get_xlim()    # Get the frequency limits
        # Set frequency ticks at 25 MHz intervals between the nearest 25 round values between freq_lim
        freq_ticks = np.arange(np.ceil(freq_lim[0]/25)*25, np.floor(freq_lim[1]/25)*25+1, 25)
        round_redshifts = np.round(f_21/freq_ticks - 1)   # Get the rounded redshifts corresponding to freq_ticks
        z_freq_ticks = f_21/(round_redshifts + 1)   # Get the corresponding frequencies
        ax.set_xticks(z_freq_ticks)
        # Initialize a redshift axis
        ax2 = ax.twiny()    # Create a twin x-axis
        ax2.xaxis.set_tick_params(labelsize=20)   # Set the tick labels font size
        ax2.set_xticks(ax.get_xticks())   # Set the ticks at the same positions as the frequency ticks
        ax2.set_xticklabels([f'{z:.0f}' for z in round_redshifts])   # Set the tick labels as redshifts
        ax2.set_xlabel("$z$",fontsize=25)
        ax2.set_xlim(ax.get_xlim())
        # Return the x-axis to frequency
        ax.set_xticks(freq_ticks)
    # Return output
    return fig, ax

# JordanFlitter: I added this function
def plot_2d_power_spectrum(
    power_spectrum: [POWER_SPECTRUM],
    cbar: Optional[bool] = True,
    cbar_horizontal: Optional[bool] = False,
    cbar_label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
):
    """
    Show a 2d image of the (log) 21cm power spectrum as a function of (log)
    wavenumber and (log) redshift.

    Parameters
    ----------
    power_spectrum: :class:`~py21cmfast.power_spectrum.POWER_SPECTRUM`
        The power spectrum object that stores the data to be plotted.
    cbar: bool, optional
        Whether or not to show a colorbar.
    cbar_horizontal: bool, optional
        Whether to place the colorbar horizontally or veritcally.
    cbar_label: str, optional
        A label for the colorbar.
    ax: Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    kwargs:
        Optional keywords to pass to :func:`maplotlib.imshow`.

    Returns
    -------
    fig, ax, cb :
        figure, axis and colorbar objects from matplotlib.

    """
    # First, let's extract the power specturm values. We use logarithmic values
    k_values = np.log10(power_spectrum.k_values)
    z_values = np.log10(power_spectrum.z_values)
    ps_values = np.log10(power_spectrum.ps_values)

    # We find the minimum and maximum values of the power specturm
    min_value = np.min(ps_values)
    max_value = np.max(ps_values)

    # Now, set colorbar label
    if cbar_label is None:
        cbar_label = '$\log_{10}(\Delta^2_'
        if power_spectrum.kind == 'brightness_temp':
            cbar_label += '{21}/\mathrm{mK}^2)$'
        elif power_spectrum.kind == 'Tk_box':
            cbar_label += '{T_k}/\mathrm{K}^2)$'
        elif power_spectrum.kind == 'Ts_box':
            cbar_label += '{T_s}/\mathrm{K}^2)$'
        elif power_spectrum.kind == 'T_chi_box':
            cbar_label += '{T_\chi}/\mathrm{K}^2)$'
        elif power_spectrum.kind == 'V_chi_b_box':
            cbar_label += '{V_{\chi b}}/\mathrm{km/sec}^2)$'
        elif power_spectrum.kind == 'x_e_box':
            cbar_label += '{x_e})$'
        elif power_spectrum.kind == 'xH_box':
            cbar_label += '{x_\mathrm{HI}})$'
        elif power_spectrum.kind == 'J_Lya_box':
            cbar_label += '{J_\\alpha}/\mathrm{cm^{-4}\,Hz^{-2}\,sec^{-2}\,Str^{-2}})$'

    # Now, set fig and ax
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
    else:
        fig = ax.figure
    plt.sca(ax)

    # Plot the 2d power specturm!
    plt.imshow(ps_values,aspect='auto',vmin=min_value,vmax=max_value,
               origin='lower',extent=(min(k_values),max(k_values),min(z_values),max(z_values)),**kwargs)

    # Add colorbar
    if cbar:
        cb = plt.colorbar(orientation="horizontal" if cbar_horizontal else "vertical")
        cb.ax.tick_params(labelsize=20)
        cb.ax.set_ylabel(cbar_label,fontsize=20)
    else:
        cb = None

    # Prettify the plot
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(xlabel='$\log_{10}(k/\mathrm{Mpc}^{-1})$',fontsize=25)
    ax.set_ylabel(ylabel='$\log_{10}(z)$',fontsize=25)

    # Return output
    return fig, ax, cb
