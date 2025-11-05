import numpy as np
import xarray

def post_processing(dictionary_list, magnetic_flag=False):
    for label in dictionary_list:
        ds = dictionary_list[label]
        ds.utils.calculate_temp
        ds.utils.radial_E_field
        ds.utils.calculate_parallel_velocity
        ds.utils.calculate_perpendicular_velocity

        mean_ZF = ds["V_ExB_y"][:,:,:,:].mean('theta').mean('zeta')
        mean_ZF.attrs['long_name'] = 'zonal flow velocity'
        mean_ZF.attrs['standard_name'] = 'zonal flow velocity'
        mean_ZF.attrs['conversion'] = 1
        mean_ZF.attrs['units'] = 'm / s'
        ds['mean_ZF'] = mean_ZF

        v_tilde_y = ds["V_ExB_y"][:,:,:,:]-ds["V_ExB_y"][:,:,:,:].mean('zeta').mean('theta')
        v_tilde_y.attrs['long_name'] = 'poloidal fluctuation velocity'
        v_tilde_y.attrs['standard_name'] = 'poloidal fluctuation velocity'
        v_tilde_y.attrs['conversion'] = 1
        v_tilde_y.attrs['units'] = 'm / s'
        ds['v_tilde_y'] = v_tilde_y

        v_tilde_x = ds["V_ExB_x"][:,:,:,:]-ds["V_ExB_x"][:,:,:,:].mean('zeta').mean('theta')
        v_tilde_x.attrs['long_name'] = 'radial fluctuation velocity'
        v_tilde_x.attrs['standard_name'] = 'radial fluctuation velocity'
        v_tilde_x.attrs['conversion'] = 1
        v_tilde_x.attrs['units'] = 'm / s'
        ds['v_tilde_x'] = v_tilde_x

        reynold_stress = (v_tilde_y * v_tilde_x).mean('zeta').mean('theta')
        reynold_stress.attrs['long_name'] = 'Reynold Stress'
        reynold_stress.attrs['standard_name'] = 'Reynold Stress'
        reynold_stress.attrs['conversion'] = 1
        reynold_stress.attrs['units'] = '(m / s)^2'
        ds['reynold_stress'] = reynold_stress

        density_fluctuation = ds["Ne"][:,:,:,:] - ds["Ne"][:,:,:,:].mean('zeta').mean('t')
        density_fluctuation.attrs['long_name'] = 'density fluctuation'
        density_fluctuation.attrs['standard_name'] = 'density fluctuation'
        density_fluctuation.attrs['conversion'] = 1
        density_fluctuation.attrs['units'] = 'm^-3'
        ds['ne_tilde'] = density_fluctuation

        density_fluctuation_normalised = (ds["Ne"][:,:,:,:] - ds["Ne"][:,:,:,:].mean('zeta').mean('t'))/ds["Ne"][:,:,:,:].mean('zeta').mean('t')
        density_fluctuation_normalised.attrs['long_name'] = 'normalised density fluctuation'
        density_fluctuation_normalised.attrs['standard_name'] = 'normalised density fluctuation'
        density_fluctuation_normalised.attrs['conversion'] = 1
        density_fluctuation_normalised.attrs['units'] = 'N/A'
        ds['ne_tilde_norm'] = density_fluctuation_normalised

        potential_fluctuation = ds["phi"][:,:,:,:] - ds["phi"][:,:,:,:].mean('zeta').mean('t')
        potential_fluctuation.attrs['long_name'] = 'potential fluctuation'
        potential_fluctuation.attrs['standard_name'] = 'potential fluctuation'
        potential_fluctuation.attrs['conversion'] = 1
        potential_fluctuation.attrs['units'] = 'm^-3'
        ds['phi_tilde'] = potential_fluctuation

        if magnetic_flag == True:
            magnetic_fluctuation = ds["Apar"][:,:,:,:] - ds["Apar"][:,:,:,:].mean('zeta').mean('t')
            magnetic_fluctuation.attrs['long_name'] = 'magnetic fluctuation'
            magnetic_fluctuation.attrs['standard_name'] = 'magnetic fluctuation'
            magnetic_fluctuation.attrs['conversion'] = 1
            magnetic_fluctuation.attrs['units'] = 'T m'
            ds['Apar_tilde'] = magnetic_fluctuation


        dictionary_list[label] = ds

    return dictionary_list

def extract_series_data(ds_dict, variable_name):
    label_list = list(ds_dict.keys())
    ds_0 = ds_dict[label_list[0]]
    da_0 = ds_0[variable_name]

    if len(label_list) > 1:
        for label in label_list[1:]:
            ds = ds_dict[label]
            da = ds[variable_name]
            da_0 = xarray.concat([da_0, da], dim='t')
    
    return da_0

def get_physical_grid(da):
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
