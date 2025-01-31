import matplotlib.pyplot as plt
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
