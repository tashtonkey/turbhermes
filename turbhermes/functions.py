import numpy as np
import xarray
from scipy.stats import skew
from scipy.stats import kurtosis
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor

def post_processing(dictionary_list, magnetic_flag=False, time_average_fluctuations=False):
    for label in dictionary_list:
        ds = dictionary_list[label]
        ds.utils.calculate_temp
        ds.utils.radial_E_field
        ds.utils.calculate_parallel_velocity
        ds.utils.calculate_perpendicular_velocity

        # get the core boundaries for the theta averages
        ylower_core = ds.regions['core'].ylower_ind
        yupper_core = ds.regions['core'].yupper_ind

        mean_ZF = ds["V_ExB_theta"][:,:,ylower_core:yupper_core,:].mean('theta').mean('zeta')
        mean_ZF.attrs['long_name'] = 'zonal flow velocity'
        mean_ZF.attrs['standard_name'] = 'zonal flow velocity'
        mean_ZF.attrs['conversion'] = 1
        mean_ZF.attrs['units'] = 'm / s'
        ds['mean_ZF'] = mean_ZF

        v_tilde_y = ds["V_ExB_theta"][:,:,:,:]-ds["V_ExB_theta"][:,:,ylower_core:yupper_core,:].mean('zeta').mean('theta')
        v_tilde_y.attrs['long_name'] = 'poloidal fluctuation velocity'
        v_tilde_y.attrs['standard_name'] = 'poloidal fluctuation velocity'
        v_tilde_y.attrs['conversion'] = 1
        v_tilde_y.attrs['units'] = 'm / s'
        ds['v_tilde_y'] = v_tilde_y

        v_tilde_x = ds["V_ExB_r"][:,:,:,:]-ds["V_ExB_r"][:,:,ylower_core:yupper_core,:].mean('zeta').mean('theta')
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

        # density fluctuations
        if time_average_fluctuations==True:
            density_fluctuation = ds["Ne"][:,:,:,:] - ds["Ne"][:,:,:,:].mean('zeta').mean('t')
        else:
            density_fluctuation = ds["Ne"][:,:,:,:] - ds["Ne"][:,:,:,:].mean('zeta')
        density_fluctuation.attrs['long_name'] = 'density fluctuation'
        density_fluctuation.attrs['standard_name'] = 'density fluctuation'
        density_fluctuation.attrs['conversion'] = 1
        density_fluctuation.attrs['units'] = 'm^-3'
        ds['ne_tilde'] = density_fluctuation

        # normalised density fluctuations
        if time_average_fluctuations==True:
            density_fluctuation_normalised = (ds["Ne"][:,:,:,:] - ds["Ne"][:,:,:,:].mean('zeta').mean('t'))/ds["Ne"][:,:,:,:].mean('zeta').mean('t')
        else:
            density_fluctuation_normalised = (ds["Ne"][:,:,:,:] - ds["Ne"][:,:,:,:].mean('zeta'))/ds["Ne"][:,:,:,:].mean('zeta')
        density_fluctuation_normalised.attrs['long_name'] = 'normalised density fluctuation'
        density_fluctuation_normalised.attrs['standard_name'] = 'normalised density fluctuation'
        density_fluctuation_normalised.attrs['conversion'] = 1
        density_fluctuation_normalised.attrs['units'] = 'N/A'
        ds['ne_tilde_norm'] = density_fluctuation_normalised

        # electric potential fluctuations
        if time_average_fluctuations==True:
            potential_fluctuation = ds["phi"][:,:,:,:] - ds["phi"][:,:,:,:].mean('zeta').mean('t')
        else:
            potential_fluctuation = ds["phi"][:,:,:,:] - ds["phi"][:,:,:,:].mean('zeta')
        potential_fluctuation.attrs['long_name'] = 'potential fluctuation'
        potential_fluctuation.attrs['standard_name'] = 'potential fluctuation'
        potential_fluctuation.attrs['conversion'] = 1
        potential_fluctuation.attrs['units'] = 'V'
        ds['phi_tilde'] = potential_fluctuation

        # normalised potential fluctuations
        if time_average_fluctuations==True:
            potential_fluctuation_normalised = (ds["phi"][:,:,:,:] - ds["phi"][:,:,:,:].mean('zeta').mean('t'))/ds['phi'][:,:,:,:].mean('zeta').mean('t')
        else:
            potential_fluctuation_normalised = (ds["phi"][:,:,:,:] - ds["phi"][:,:,:,:].mean('zeta'))/ds['phi'][:,:,:,:].mean('zeta')
        potential_fluctuation_normalised.attrs['long_name'] = 'normalised potential fluctuation'
        potential_fluctuation_normalised.attrs['standard_name'] = 'normalised potential fluctuation'
        potential_fluctuation_normalised.attrs['conversion'] = 1
        potential_fluctuation_normalised.attrs['units'] = 'N/A'
        ds['phi_tilde_norm'] = potential_fluctuation_normalised        

        # magnetic fluctuations
        if magnetic_flag == True:
            if time_average_fluctuations==True:
                magnetic_fluctuation = ds["Apar"][:,:,:,:] - ds["Apar"][:,:,:,:].mean('zeta').mean('t')
            else:
                magnetic_fluctuation = ds["Apar"][:,:,:,:] - ds["Apar"][:,:,:,:].mean('zeta')
            magnetic_fluctuation.attrs['long_name'] = 'magnetic fluctuation'
            magnetic_fluctuation.attrs['standard_name'] = 'magnetic fluctuation'
            magnetic_fluctuation.attrs['conversion'] = 1
            magnetic_fluctuation.attrs['units'] = 'T m'
            ds['Apar_tilde'] = magnetic_fluctuation

            # normalised potential fluctuations
            if time_average_fluctuations==True:
                magnetic_fluctuation_normalised = (ds["Apar"][:,:,:,:] - ds["Apar"][:,:,:,:].mean('zeta').mean('t'))/ds['Apar'][:,:,:,:].mean('zeta').mean('t')
            else:
                magnetic_fluctuation_normalised = (ds["Apar"][:,:,:,:] - ds["Apar"][:,:,:,:].mean('zeta'))/ds['Apar'][:,:,:,:].mean('zeta')
            magnetic_fluctuation_normalised.attrs['long_name'] = 'normalised magnetic fluctuation'
            magnetic_fluctuation_normalised.attrs['standard_name'] = 'normalised magnetic fluctuation'
            magnetic_fluctuation_normalised.attrs['conversion'] = 1
            magnetic_fluctuation_normalised.attrs['units'] = 'N/A'
            ds['Apar_tilde_norm'] = magnetic_fluctuation_normalised    

            # now getting the delta-br delta-btheta terms 
            jacobian = ds['J']
            g_12 = ds['g_12']
            g_22 = ds['g_22']
            g_23 = ds['g_23']
            g_11 = ds['g_11']
            g_33 = ds['g_33']
            g22 = ds['g22']
            Apar = ds["Apar"][:,:,:,:]

            curl_A_x = (1/jacobian) * ( ((Apar * g_23)/(np.sqrt(g_22))).bout.ddy() - ((Apar * g_23)/(np.sqrt(g_22))).bout.ddz() ) * np.sqrt(g_11)
            curl_A_y = (1/jacobian) * ( ((Apar * g_12)/(np.sqrt(g_22))).bout.ddz() - ((Apar * g_23)/(np.sqrt(g_23))).bout.ddx() ) * np.sqrt(g_22)
            curl_A_z = (1/jacobian) * ( ((Apar * g_22)/(np.sqrt(g_22))).bout.ddx() - ((Apar * g_23)/(np.sqrt(g_12))).bout.ddy() ) * np.sqrt(g_33)

            curl_A_x.attrs['long_name'] = 'delta-B x componenet magnetic fluctuation'
            curl_A_x.attrs['standard_name'] = 'delta-B x componenet magnetic fluctuation'
            curl_A_x.attrs['conversion'] = 1
            curl_A_x.attrs['units'] = 'T'
            ds['delta_B_x'] = curl_A_x

            curl_A_y.attrs['long_name'] = 'delta-B y componenet magnetic fluctuation'
            curl_A_y.attrs['standard_name'] = 'delta-B y componenet magnetic fluctuation'
            curl_A_y.attrs['conversion'] = 1
            curl_A_y.attrs['units'] = 'T'
            ds['delta_B_y'] = curl_A_y

            curl_A_z.attrs['long_name'] = 'delta-B z componenet magnetic fluctuation'
            curl_A_z.attrs['standard_name'] = 'delta-B z componenet magnetic fluctuation'
            curl_A_z.attrs['conversion'] = 1
            curl_A_z.attrs['units'] = 'T'
            ds['delta_B_z'] = curl_A_z

            # calculate maxwell stress. check if it needs to be y or theta, that will need a conversion.

            sigma_B_pol = 1 # this should be the sigma-b-pol, check what the way to get this is.

            delta_B_r = curl_A_x * sigma_B_pol
            delta_B_theta = curl_A_y * np.sqrt(1/(g22 * g_22))
            delta_B_zeta = curl_A_y * (g_23/np.sqrt(g_22 * g_33)) + curl_A_z

            delta_B_r.attrs['long_name'] = 'radial delta-B component'
            delta_B_theta.attrs['long_name'] = 'theta delta-B component'
            delta_B_zeta.attrs['long_name'] = 'zeta delta-B component'
            delta_B_r.attrs['standard_name'] = 'radial delta-B component'
            delta_B_theta.attrs['standard_name'] = 'theta delta-B component'
            delta_B_zeta.attrs['standard_name'] = 'zeta delta-B component'
            delta_B_r.attrs['conversion'] = 1
            delta_B_theta.attrs['conversion'] = 1
            delta_B_zeta.attrs['conversion'] = 1
            delta_B_r.attrs['units'] = 'T'
            delta_B_theta.attrs['units'] = 'T'
            delta_B_zeta.attrs['units'] = 'T'

            ds['delta_B_r'] = delta_B_r
            ds['delta_B_theta'] = delta_B_theta
            ds['delta_B_zeta'] = delta_B_zeta

            # maxwell stress

            maxwell_stress = (delta_B_r * delta_B_theta).mean('zeta').mean('theta')
            maxwell_stress.attrs['long_name'] = 'Maxwell Stress'
            maxwell_stress.attrs['standard_name'] = 'Maxwell Stress'
            maxwell_stress.attrs['conversion'] = 1
            maxwell_stress.attrs['units'] = 'T^2'
            ds['maxwell_stress'] = maxwell_stress

        # turbulent particle flux <\delta v_x \delta n>
        if time_average_fluctuations==True:
            turb_particle_flux = (v_tilde_x * density_fluctuation).mean('zeta')# not taking time average for this .mean('time')
        else:
            turb_particle_flux = (v_tilde_x * density_fluctuation).mean('zeta')
        turb_particle_flux.attrs['long_name'] = 'turbulent particle flux'
        turb_particle_flux.attrs['standard_name'] = 'turbulent particle flux'
        turb_particle_flux.attrs['conversion'] = 1
        turb_particle_flux.attrs['units'] = 'm^2 / s'
        ds['turb_particle_flux'] = turb_particle_flux

        # ion pressure fluctuations
        if time_average_fluctuations==True:
            ion_pressure_fluctuation = ds["Pi"][:,:,:,:] - ds["Pi"][:,:,:,:].mean('zeta').mean('t')
        else:
            ion_pressure_fluctuation = ds["Pi"][:,:,:,:] - ds["Pi"][:,:,:,:].mean('zeta')
        ion_pressure_fluctuation.attrs['long_name'] = 'ion pressure fluctuation'
        ion_pressure_fluctuation.attrs['standard_name'] = 'ion pressure fluctuation'
        ion_pressure_fluctuation.attrs['conversion'] = 1
        ion_pressure_fluctuation.attrs['units'] = 'Pa'
        ds['Pi_tilde'] = ion_pressure_fluctuation

        # electron pressure fluctuations
        if time_average_fluctuations==True:
            electron_pressure_fluctuation = ds["Pe"][:,:,:,:] - ds["Pe"][:,:,:,:].mean('zeta').mean('t')
        else:
            electron_pressure_fluctuation = ds["Pe"][:,:,:,:] - ds["Pe"][:,:,:,:].mean('zeta')
        electron_pressure_fluctuation.attrs['long_name'] = 'electron pressure fluctuation'
        electron_pressure_fluctuation.attrs['standard_name'] = 'electron pressure fluctuation'
        electron_pressure_fluctuation.attrs['conversion'] = 1
        electron_pressure_fluctuation.attrs['units'] = 'Pa'
        ds['Pe_tilde'] = electron_pressure_fluctuation

        dictionary_list[label] = ds

        # turbulent ion energy flux <\delta v_x \delta Pi>
        if time_average_fluctuations==True:
            turb_ion_pressure_flux = (v_tilde_x * ion_pressure_fluctuation).mean('zeta')# not taking time average for this .mean('time')
        else:
            turb_ion_pressure_flux = (v_tilde_x * ion_pressure_fluctuation).mean('zeta')
        turb_ion_pressure_flux.attrs['long_name'] = 'turbulent ion pressure flux'
        turb_ion_pressure_flux.attrs['standard_name'] = 'turbulent ion pressure flux'
        turb_ion_pressure_flux.attrs['conversion'] = 1
        turb_ion_pressure_flux.attrs['units'] = 'W / m^2'
        ds['turb_Pi_flux'] = turb_ion_pressure_flux

        # turbulent electron energy flux <\delta v_x \delta Pi>
        if time_average_fluctuations==True:
            turb_electron_pressure_flux = (v_tilde_x * electron_pressure_fluctuation).mean('zeta')# not taking time average for this .mean('time')
        else:
            turb_electron_pressure_flux = (v_tilde_x * electron_pressure_fluctuation).mean('zeta')
        turb_electron_pressure_flux.attrs['long_name'] = 'turbulent electron pressure flux'
        turb_electron_pressure_flux.attrs['standard_name'] = 'turbulent electron pressure flux'
        turb_electron_pressure_flux.attrs['conversion'] = 1
        turb_electron_pressure_flux.attrs['units'] = 'W / m^2'
        ds['turb_Pe_flux'] = turb_electron_pressure_flux

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

def get_statistics(da_slice):
    mean = np.sqrt((da_slice*da_slice).mean())
    stdev = (np.std(da_slice))
    skews = (skew(da_slice, axis = None))
    kurt = (kurtosis(da_slice, axis = None))
    return mean, stdev, skews, kurt

def get_radial_statistics(da, x_array):
    mean_list=[]
    stdev_list=[]
    skews_list=[]
    kurt_list=[]

    for xcoord in x_array:
        da_slice = da[:,xcoord,:]
        mean, stdev, skews, kurt = get_statistics(da_slice)

        mean_list.append(mean)
        stdev_list.append(stdev)
        skews_list.append(skews)
        kurt_list.append(kurt)

    mean_list=np.array(mean_list)
    stdev_list=np.array(stdev_list)
    skews_list=np.array(skews_list)
    kurt_list=np.array(kurt_list)

    return mean_list, stdev_list, skews_list, kurt_list