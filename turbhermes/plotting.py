import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import animatplot as amp
from matplotlib.animation import PillowWriter
from .functions import get_radial_statistics, get_statistics

def _normalise_time_coord(time_values):
    """
    amp.Timeline() does not do a good job of displaying time values that are very small
    (they get rounded to zero). This function scales the time values by a power of 10 to
    get nice values and modifies the units by the scale factor.
    """
    tmax = time_values.max()
    if tmax < 1.0e-2 or tmax > 1.0e6:
        scale_pow = int(np.floor(np.log10(tmax)))
        scale_factor = 10**scale_pow
        time_values = time_values / scale_factor
        suffix = f"e{scale_pow}"
    else:
        suffix = ""

    return time_values, suffix

def _add_controls(anim, controls, t_label):
    if controls == "both":
        # Add both time slider and play/pause toggle
        anim.controls(timeline_slider_args={"text": t_label})
    elif controls == "timeline":
        # Add time slider
        anim.timeline_slider(text=t_label)
    elif controls == "toggle":
        # Add play/pause toggle
        anim.toggle()
    elif controls is None or controls == "":
        # Add no controls
        pass
    else:
        raise ValueError(f"Unrecognised value for controls={controls}")

def multi_line_raw(bd_list, variable, coordinates, ax=None, **kwargs):
    """
    Plots a time trace of all the datasets in the bd_list, converted to correct units
    bd_list is the list of datasets
    variable is the string of the variable to plot
    coordinates is the tuple of SPATIAL dimensions, x, theta, zeta, 
    """
    if type(coordinates) is not tuple:
        print("Coordinates must be a tuple")
        return
    else:
        if not ax:
            fig, ax = plt.subplots()
            
        x_index = coordinates[0]
        theta_index = coordinates[1]
        zeta_index = coordinates[2]
        for dataset_iterator in bd_list:
            if variable in dataset_iterator:
                normal_time = dataset_iterator['t']
                normal_amp = dataset_iterator[variable][:,x_index,theta_index,zeta_index]
                ax.plot(normal_time, normal_amp, **kwargs)
                ax.axvline(normal_time[-1], ls = ':', alpha = 0.5, **kwargs)
                
                variable_name = bd_list[-1][variable].attrs["long_name"]
                units = bd_list[-1][variable].attrs["units"]
                

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(variable_name + '  ' + units)
    
    return

def multi_line_zeta(bd_list, variable, coordinates, ax=None, **kwargs):
    """
    Plots a time trace of all the datasets in the bd_list, converted to correct units, averaged over zeta. 
    bd_list is the list of datasets
    variable is the string of the variable to plot
    coordinates is the TUPLE of SPATIAL dimensions, x, theta, 
    """
    if type(coordinates) is not tuple:
        print("Coordinates must be a tuple")
        return
    else:
        if not ax:
            fig, ax = plt.subplots()
            
            
        x_index = coordinates[0]
        theta_index = coordinates[1]
        for dataset_iterator in bd_list:
            if variable in dataset_iterator:
                normal_time = dataset_iterator['t']
                normal_amp = dataset_iterator[variable][:,x_index,theta_index].mean('zeta')
                ax.plot(normal_time, normal_amp, **kwargs)
                ax.axvline(normal_time[-1], ls = ':', alpha = 0.5, **kwargs)
                
                variable_name = bd_list[-1][variable].attrs["long_name"]
                units = bd_list[-1][variable].attrs["units"]
                

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(variable_name + '  ' + units)
    
    return


def _parse_coord_option(coord, axis_coords, da):
    if isinstance(axis_coords, dict):
        option_value = axis_coords.get(coord, None)
    else:
        option_value = axis_coords

    if option_value is None:
        c = da[coord]
        if "long_name" in c.attrs:
            label = c.long_name
        else:
            label = coord
        if "units" in c.attrs:
            label = label + f" [{c.units}]"
        return c, label
    elif option_value == "index":
        return np.arange(da.sizes[coord]), f"{coord} index"
    elif isinstance(option_value, str):
        c = da[option_value]
        if "long_name" in c.attrs:
            label = c.long_name
        else:
            label = option_value
        if "units" in c.attrs:
            label = label + f" [{c.units}]"
        return c, label
    else:
        return option_value, None
    
def _get_R_coord_option(coord, da):
    if 'ROMP' in da.coords:
        R = da['ROMP']
        separatrix_index = da.metadata['ixseps1']
        if separatrix_index < da.metadata['nx']:
            # if the input is sliced in the x-direction, this index will not be the same as the ixseps value
            separatrix_index_true = np.where(np.array(da.coords['x']) == separatrix_index)[0][0]
            
        else:
            separatrix_index_true = 0
        physical_grid = []
        sep_R = R[separatrix_index_true]

        physical_grid = R - sep_R
        label = "R-R_sep (m)"

    else:
        Rxy = da['R']
        Zxy = da['Z']
        label = "R-R_sep (m)"
        
        separatrix_index = da.metadata['ixseps1']
        
        
        if separatrix_index < da.metadata['nx']:
            # if the input is sliced in the x-direction, this index will not be the same as the ixseps value
            separatrix_index_true = np.where(np.array(da.coords['x']) == separatrix_index)[0][0]
            
        else:
            separatrix_index_true = 0

        sep_R = Rxy[separatrix_index_true]
        sep_Z = Zxy[separatrix_index_true]
            # calculate the deviation of the coordinate from the separatrix location
        physical_grid = []
        for x_index in range(len(da.coords['x'])):
            particular_R = Rxy[x_index]
            particular_Z = Zxy[x_index]
            if x_index < separatrix_index_true:
                delta_R = particular_R - sep_R 
                delta_Z = particular_Z - sep_Z
                grid_distance = -np.sqrt(delta_R**2 + delta_Z**2)
                physical_grid.append(grid_distance)
            elif x_index >= separatrix_index_true:
                delta_R = particular_R - sep_R 
                delta_Z = particular_Z - sep_Z
                grid_distance = np.sqrt(delta_R**2 + delta_Z**2)
                physical_grid.append(grid_distance)

        physical_grid = np.array(physical_grid)
    
    return physical_grid, label



def diagonal_slice_plotting(
    data,
    animate_over=None,
    animate=True,
    axis_coords=None,
    vmin=None,
    vmax=None,
    logscale=False,
    fps=10,
    save_as=None,
    sep_pos=None,
    ax=None,
    aspect=None,
    controls="both",
    **kwargs,
):
    """
    Plots a line plot which is animated with time.

    Currently only supports 1D+1 data, which it plots with animatplot's Line animation.

    Parameters
    ----------
    data : xarray.DataArray
    animate_over : str, optional
        Dimension over which to animate, defaults to the time dimension
    animate : bool, optional
        If set to false, do not create the animation, just return the block
    axis_coords : None, str, dict
        Coordinates to use for axis labelling.

        - None: Use the dimension coordinate for each axis, if it exists.
        - "index": Use the integer index values.
        - dict: keys are dimension names, values set axis_coords for each axis
          separately. Values can be: None, "index", the name of a 1d variable or
          coordinate (which must have the dimension given by 'key'), or a 1d
          numpy array, dask array or DataArray whose length matches the length of
          the dimension given by 'key'.
    vmin : float, optional
        Minimum value to use for colorbar. Default is to use minimum value of
        data across whole timeseries.
    vmax : float, optional
        Maximum value to use for colorbar. Default is to use maximum value of
        data across whole timeseries.
    logscale : bool or float, optional
        If True, default to a logarithmic color scale instead of a linear one.
        If a non-bool type is passed it is treated as a float used to set the linear
        threshold of a symmetric logarithmic scale as
        linthresh=min(abs(vmin),abs(vmax))*logscale, defaults to 1e-5 if True is
        passed.
    fps : int, optional
        Frames per second of resulting gif
    save_as : True or str, optional
        If str is passed, save the animation as save_as+'.gif'.
        If True is passed, save the animation with a default name,
        '<variable name>_over_<animate_over>.gif'
    sep_pos : int, optional
        Radial position at which to plot the separatrix
    ax : Axes, optional
        A matplotlib axes instance to plot to. If None, create a new
        figure and axes, and plot to that
    aspect : str or None, optional
        Argument to set_aspect(), defaults to "auto"
    controls : string or None, default "both"
        By default, add both the timeline and play/pause toggle to the animation. If
        "timeline" is passed add only the timeline, if "toggle" is passed add only the
        play/pause toggle. If None or an empty string is passed, add neither.
    kwargs : dict, optional
        Additional keyword arguments are passed on to the plotting function
        animatplot.blocks.Line

    Returns
    -------
    animation or block
        If animate==True, returns an animatplot.Animation object, otherwise
        returns an animatplot.blocks.Line instance.
    """

    if animate_over is None:
        animate_over = data.metadata.get("bout_tdim", "t")

    if aspect is None:
        aspect = "auto"

    variable = data.name

    # Check plot is the right orientation
    t_read, x_read = data.dims
    if t_read == animate_over:
        x = x_read
    else:
        data = data.transpose(animate_over, t_read, transpose_coords=True)
        x = t_read

    # Load values eagerly otherwise for some reason the plotting takes
    # 100's of times longer - for some reason animatplot does not deal
    # well with dask arrays!
    image_data = data.values

    # If not specified, determine max and min values across entire data series
    if vmax is None:
        vmax = np.max(image_data)
    if vmin is None:
        vmin = np.min(image_data)

    x_values, x_label = _get_R_coord_option(x, data)

    if not ax:
        fig, ax = plt.subplots()

    ax.set_aspect(aspect)

    # set range of plot
    ax.set_ylim([vmin, vmax])

    line_block = amp.blocks.Line(x_values, image_data, ax=ax, **kwargs)

    if animate:
        t_values, t_label = _parse_coord_option(animate_over, axis_coords, data)
        t_values, t_suffix = _normalise_time_coord(t_values)

        timeline = amp.Timeline(t_values, fps=fps, units=t_suffix)
        anim = amp.Animation([line_block], timeline)

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x_label)
    if "long_name" in data.attrs:
        y_label = data.long_name
    else:
        y_label = variable
    if "units" in data.attrs:
        y_label = y_label + f" [{data.units}]"
    ax.set_ylabel(y_label)

    if logscale:
        if vmin * vmax > 0.0:
            ax.set_yscale("log")
        else:
            if not isinstance(logscale, bool):
                linear_scale = logscale
            else:
                linear_scale = 1.0e-5
            linear_threshold = min(abs(vmin), abs(vmax)) * linear_scale
            ax.set_yscale("symlog", linthresh=linear_threshold)

    # Plot separatrix
    if sep_pos is None:
        sep_pos = 0

    ax.axvline(sep_pos, ls = "--", color= 'black')

    if animate:
        _add_controls(anim, controls, t_label)

        if save_as is not None:
            if save_as is True:
                save_as = "{}_over_{}".format(variable, animate_over)
            anim.save(save_as + ".gif", writer=PillowWriter(fps=fps))

        return anim

    return line_block

def diagonal_slice(
    data,
    vmin=None,
    vmax=None,
    logscale=False,
    save_as=None,
    sep_pos=None,
    ax=None,
    aspect=None,
    **kwargs,
):
    x = data.dims  # the data inputted should be a slice in the x-direction, to get a radial profile
    if aspect is None:
        aspect = "auto"
    # Load values eagerly otherwise for some reason the plotting takes
    # 100's of times longer - for some reason animatplot does not deal
    # well with dask arrays!
    image_data = data.values
    variable = data.name

    # If not specified, determine max and min values across entire data series
    if vmax is None:
        vmax = np.max(image_data)*1.1
    if vmin is None:
        vmin = np.min(image_data)*1.1

    x_values, x_label = _get_R_coord_option(x, data)

    if not ax:
        fig, ax = plt.subplots()

    ax.set_aspect(aspect)

    # set range of plot
    ax.set_ylim([vmin, vmax])

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x_label)
    if "long_name" in data.attrs:
        y_label = data.long_name
    else:
        y_label = variable
    if "units" in data.attrs:
        y_label = y_label + f" [{data.units}]"
    ax.set_ylabel(y_label)

    if sep_pos is None:
        sep_pos = 0 # these are in normalised units, R_sep = 0

    ax.axvline(sep_pos, ls = ':', color = 'black')

    line_element = ax.plot(x_values, image_data)

    return line_element

def plot_heatmap(
    data,
    vmin=None,
    vmax=None,
    logscale=False,
    sep_pos=None,
    ax=None,
    aspect=None,
    **kwargs,
):
    dims = data.dims # assume we should have time and x. so dims = ['t', 'x']
    if aspect is None:
        aspect = "auto"
    if 'x' in dims:
        x_axis_grid, x_label = _get_R_coord_option('x', data)
    else:
        x_axis_grid= data[dims[1]]
        x_label = str(dims[1])

    time_array = data['t'].data
    amplitude_array = data.values # should already have been sliced to the desired coordinates
    variable = data.name

    if not ax:
        fig, ax = plt.subplots()

    pcl = ax.pcolormesh(x_axis_grid, time_array, amplitude_array, vmin=vmin, vmax=vmax)
    ax.set_xlabel(x_label)
    if sep_pos is None:
        sep_pos = 0 # these are in normalised units, R_sep = 0

    ax.axvline(sep_pos, ls = ':', color = 'black')

    ax.set_title(variable)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time [s]")

    return pcl

def mean_diagonal_slice(
    data,
    color,
    label,
    vmin=None,
    vmax=None,
    logscale=False,
    save_as=None,
    sep_pos=None,
    ax=None,
    aspect=None,
    **kwargs,
):
    x = data.dims  # the data inputted should be a slice in the x-direction, to get a radial profile
    if aspect is None:
        aspect = "auto"
 
    image_data = data.values
    x_values, x_label = _get_R_coord_option('x', data)
    variable = data.name

    mean, stdev, skews, kurt = get_radial_statistics(image_data, x_array = data['x'])

        # If not specified, determine max and min values across entire data series
    if vmax is None:
        vmax = (np.max(mean)+np.max(stdev))*1.1
    if vmin is None:
        vmin = (np.min(mean)-np.max(stdev))*1.1
    
    
    if not ax:
        fig, ax = plt.subplots()

    ax.set_aspect(aspect)

    # set range of plot
    ax.set_ylim([vmin, vmax])

    # Add title and axis labels
    ax.set_title(variable)
    ax.set_xlabel(x_label)
    if "long_name" in data.attrs:
        y_label = data.long_name
    else:
        y_label = variable
    if "units" in data.attrs:
        y_label = y_label + f" [{data.units}]"
    ax.set_ylabel(y_label)

    if sep_pos is None:
        sep_pos = 0 # these are in normalised units, R_sep = 0

    ax.axvline(sep_pos, ls = ':', color = 'black')

    f1 = ax.plot(x_values, mean, color =color, label = label)
    f2 = ax.fill_between(x_values, (mean - stdev), (mean + stdev), color =color, alpha=0.2)

    return f1, f2


def fluctuation_cross_plot(
        data_1, 
        data_2,
        ax=None,
        aspect='equal'
):
    # get the standard deviation of these data
    # then histogram it.
    # fluctuation_cross_plot(ds['ne_tilde'][:,10,8,:], ds['v_tilde_x'][:,10,8,:], ax=ax)

    data_1_values = data_1.values
    data_2_values = data_2.values

    data_1_name = data_1.name
    data_2_name = data_2.name

    dummy, stdev_1, dummy, dummy = get_statistics(data_1_values)
    dummy, stdev_2, dummy, dummy = get_statistics(data_2_values)

    print(stdev_1)
    print(stdev_2)

    data_1_flatten = data_1_values.flatten()
    data_2_flatten = data_2_values.flatten()

    if ax is None:
        fig, ax = plt.subplots()


    hist = ax.hist2d(data_1_flatten/stdev_1, data_2_flatten/stdev_2, bins = [50,50], density =False, range = [[-4,4],[-4,4]], cmap = 'plasma')

    ax.set_xlabel(data_1_name)
    ax.set_ylabel(data_2_name)

    ax.set_aspect(aspect)

    """
    cbarEMv = fig.colorbar(hEMv[3], ax=ax[0,0])
    cbarEMnv = fig.colorbar(hEMnv[3], ax=ax[0,1])
    cbarESv = fig.colorbar(hESv[3], ax=ax[1,0])
    cbarESnv = fig.colorbar(hESnv[3], ax=ax[1,1])

    cbarEMv.set_label('Probability')
    cbarEMnv.set_label('Probability')
    cbarESv.set_label('Probability')
    cbarESnv.set_label('Probability')
    """

    return hist