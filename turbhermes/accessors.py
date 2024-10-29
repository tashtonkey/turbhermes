from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
import numpy as np


@register_dataset_accessor("utils")
class UtilityDatasetAccessor(BoutDatasetAccessor):
    """
    Class specifically for calculating ExB velocities of BOUT++ data.
    
    Requires that the BOUT++ data has a 'phi' field and 'x' and 'z' coordinates,
    in addition to the default 'Bxy' magnetic field.
    """

    def __init__(self, ds):
        super().__init__(ds)
        self.data = ds

    # This is where module-specific methods would go
    # For example maybe elm-pb would have a .elm_growth_rate() method?
    @property
    def radial_E_field(self):
        """Calculates local radial electric field"""
        #THIS NEEDS NORMALISING BECAUSE OF THE GRADIENT WITH rho_s0, maybe?
        if "radial_E" not in self.data:
            E_x = self.data["phi"].bout.ddx()
            E_x.attrs["standard_name"] = "radial E field"
            E_x.attrs["long_name"] = "radial electric field"
            E_x.attrs["units"] = "V m^-1"
            self.data["radial_E"] = E_x
        return self.data["radial_E"]
    
    @property
    def calculate_temp(self):
        """Calculates local radial electric field
        species = string of the species type, ie 'd', 'e', to match formatting
        """
        species_list = []
        for variable in list(self.data):
            if variable[0] == 'N':
                if variable[1].isupper() == False:
                    index = variable.find('_')
                    if index == -1:
                        species_variable = variable[1:]
                        species_list.append(species_variable)
        new_species = species_list[0]      
        for species in species_list:
            if "T" + species not in self.data:
                pressure = self.data["P" + species]
                density = self.data["N" + species]
                temperature = pressure/density
                temperature.attrs["standard_name"] = "temperature"
                temperature.attrs["long_name"] = species + " temperature"
                temperature.attrs["units"] = "eV"
                temperature.attrs["conversion"] = self.data.metadata['Tnorm']
                self.data["T" + species] = temperature
                new_species = species
        return self.data["T" + new_species]
    
    @property
    def calculate_parallel_velocity(self):
        """
        Calcualtes the parallel velocity of the particular species
        species_list is list of species, ie 'd','e',etc.
        """
        from fractions import Fraction
        
        species_list = []
        for variable in list(self.data):
            if variable[0] == 'N':
                if variable[1].isupper() == False:
                    index = variable.find('_')
                    if index == -1:
                        species_variable = variable[1:]
                        species_list.append(species_variable)
        new_species = species_list[0] 
        
        for species in species_list:
            if "V" + species not in self.data:
                density = self.data["N" + species]
                momentum = self.data["NV" + species]
                particle_mass = Fraction(str(self.data.options[species]['AA']).replace(" ", "")) # mass in terms of atomic units, converted to string to remove spaces, then into a fraction (to account for e)
                AA = 1.66054e-27
                mass_in_kg = AA * particle_mass

                NVe_conv = momentum.attrs['conversion']
                Ne_conv = density.attrs['conversion']
                
                new_conversion = NVe_conv/(mass_in_kg*Ne_conv)

                parallel_velocity = self.data['NV' + species]/self.data['N' + species] # If this used by xhermes, these will be normalised already
                parallel_velocity.attrs["standard_name"] = "parallel velocity"
                parallel_velocity.attrs["long_name"] = species + " parallel velocity"
                parallel_velocity.attrs["units"] = "m / s"
                parallel_velocity.attrs["conversion"] = new_conversion
                self.data["V" + species] = parallel_velocity
                new_species = species
        return self.data["V" + new_species]
    
    @property
    def calculate_perpendicular_velocity(self):
        """
        Calcualtes the perpendicular velocity of the particular species, with a diamagnetic component and an ExB component
        species_list is list of species, ie 'd','e',etc.
        """
        import math
        
        species_list = []
        for variable in list(self.data):
            if variable[0] == 'N':
                if variable[1].isupper() == False:
                    index = variable.find('_')
                    if index == -1:
                        species_variable = variable[1:]
                        species_list.append(species_variable)
        
        species_list = self.data.metadata['species_list']
        new_species = species_list[0] 

        # Add something to check that the metric tensors have been normalised? Or assume that the normalisation has already happened?
        
        jacobian = self.data['J']
        g_12 = self.data['g_12']
        g_22 = self.data['g_22']
        g_23 = self.data['g_23']
        g_11 = self.data['g_11']
        g_33 = self.data['g_33']
        
        g11 = self.data['g11']
        g22 = self.data['g22']
        g33 = self.data['g33']

        rho_s0 = self.data.metadata['rho_s0']

        poloidal_flow_bool_string = self.data.options['vorticity']['poloidal_flows']
        if poloidal_flow_bool_string == 'true':
            poloidal_flow_bool = True
        elif poloidal_flow_bool_string == 'false':
            poloidal_flow_bool = False
        else:
            print("Poloidal flow bool in incorrect format, assuming false")
            poloidal_flow_bool = False
    
        for species in species_list:
            if "V_dia_x_" + species not in self.data:
                if "P" + species in self.data:
                    charge_sign = math.copysign(1, self.data.options[species]['charge']) # getting the sign of the charge for the diamagnetic direction
                    pressure = self.data['P' + species]
                    density = self.data['N' + species]

                    p_ddx = pressure.bout.ddx()
                    p_ddy = pressure.bout.ddy()
                    p_ddz = pressure.bout.ddz()

                    # this is -ve because ExB and diamagnetic need to oppose, and one document says this way, another says the other way, idk what's correct
                    V_dia_x = -(charge_sign * (p_ddy * g_23 - p_ddz * g_22) / (density *  1.602e-19 * g_22 )) * np.sqrt(g_11)
                    V_dia_y = -(charge_sign * (p_ddz * g_12 - p_ddx * g_23) / (density *  1.602e-19 * g_22 )) * np.sqrt(g_22)
                    V_dia_z = -(charge_sign * (p_ddx * g_22 - p_ddy * g_12) / (density *  1.602e-19 * g_22 )) * np.sqrt(g_33)

                    V_dia_x.attrs['long_name'] = 'radial diamagnetic velocity'
                    V_dia_y.attrs['long_name'] = 'poloidal diamagnetic velocity'
                    V_dia_z.attrs['long_name'] = 'toroidal diamagnetic velocity'
                    V_dia_x.attrs['standard_name'] = 'radial diamagnetic velocity'
                    V_dia_y.attrs['standard_name'] = 'poloidal diamagnetic velocity'
                    V_dia_z.attrs['standard_name'] = 'toroidal diamagnetic velocity'
                    V_dia_x.attrs['conversion'] = 1
                    V_dia_y.attrs['conversion'] = 1
                    V_dia_z.attrs['conversion'] = 1
                    V_dia_x.attrs['units'] = 'm / s'
                    V_dia_y.attrs['units'] = 'm / s'
                    V_dia_z.attrs['units'] = 'm / s'

                    self.data['V_dia_x_' + species] = V_dia_x
                    self.data['V_dia_y_' + species] = V_dia_y
                    self.data['V_dia_z_' + species] = V_dia_z

                    new_species = species
                else:
                    print("No pressure in dataset")

        if "V_ExB_x" not in self.data:
            potential = self.data['phi']
            potential_conversion = potential.attrs['conversion'] # converting the psi computational units into real units

            phi_ddx = -potential.bout.ddx() # this is Ex
            phi_ddy = -potential.bout.ddy() # this is Ey
            phi_ddz = -potential.bout.ddz() # this is Ez

            # g_12 = 0

            V_ExB_x = potential_conversion * (phi_ddy * g_23 - phi_ddz * g_22) / (rho_s0 * g_22) * np.sqrt(g11)
            
            if poloidal_flow_bool == True:
                V_ExB_y = potential_conversion * (phi_ddz * g_12 - phi_ddx * g_23) / (rho_s0 * g_22) * np.sqrt(g22)
            elif poloidal_flow_bool == False:
                V_ExB_y = 0 * (phi_ddz * g_12 - phi_ddx * g_23) / (rho_s0 * g_22) * np.sqrt(g_22)
            
            V_ExB_z = potential_conversion * (phi_ddx * g_22 - phi_ddy * g_12) / (rho_s0 * g_22) * np.sqrt(g33)

            V_ExB_x.attrs['long_name'] = 'radial ExB velocity'
            V_ExB_y.attrs['long_name'] = 'poloidal ExB velocity'
            V_ExB_z.attrs['long_name'] = 'toroidal ExB velocity'
            V_ExB_x.attrs['standard_name'] = 'radial ExB velocity'
            V_ExB_y.attrs['standard_name'] = 'poloidal ExB velocity'
            V_ExB_z.attrs['standard_name'] = 'toroidal ExB velocity'
            V_ExB_x.attrs['conversion'] = 1
            V_ExB_y.attrs['conversion'] = 1
            V_ExB_z.attrs['conversion'] = 1
            V_ExB_x.attrs['units'] = 'm / s'
            V_ExB_y.attrs['units'] = 'm / s'
            V_ExB_z.attrs['units'] = 'm / s'

            self.data['V_ExB_x'] = V_ExB_x
            self.data['V_ExB_y'] = V_ExB_y
            self.data['V_ExB_z'] = V_ExB_z

            #### New Version Below

            rho_s0 = ds.metadata['rho_s0']
            Bnorm = ds.metadata['Bnorm']

            metric_component_list = ['g11', 'g12', 'g13', 'g22', 'g23', 'g33', 'g_11', 'g_12', 'g_13', 'g_22', 'g_23', 'g_33']

            metric_unnormalise_dict = {'g11': 1, 'g22':1/(rho_s0)**2, 'g33':1/(rho_s0)**2, 'g12':Bnorm, 'g13':Bnorm, 'g23': 1/(rho_s0)**2,
                                        'g_11': 1/(Bnorm*rho_s0)**2, 'g_22': 1, 'g_33':(rho_s0)**2, 'g_12':1/Bnorm, 'g_13':1/Bnorm, 'g_23': (rho_s0)**2}

            for metric in metric_component_list:
                ds[metric] = ds[metric] * metric_unnormalise_dict[metric]
                
            potential = ds['phi']

            phi_ddx = -potential.bout.ddx() # this is Ex. No need for any conversion
            phi_ddy = -potential.bout.ddy() # this is Ey
            phi_ddz = -potential.bout.ddz() # this is Ez

            # Don't need to convert units because xHermes does it for me
            # -ve in front to be consistent with documentation from the BOUT++ manual : v_ExB = ExB/B**2
            V_ExB_x = ((phi_ddy * ds['g_23'] - phi_ddz * ds['g_22']) / (ds['g_22'])) * np.sqrt(ds['g_11'])
            V_ExB_y = ((phi_ddz * ds['g_12'] - phi_ddx * ds['g_23']) / (ds['g_22'])) * np.sqrt(ds['g_22'])
            V_ExB_z = ((phi_ddx * ds['g_22'] - phi_ddy * ds['g_12']) / (ds['g_22'])) * np.sqrt(ds['g_33'])



        return "Calculated"

"""
Examples from John tutorial

# We can un-normalise our data, so it's in physical units
n0 = ds.options["model"]["n0"]
ds["n"] = ds["n"] * n0
ds["n"].attrs["units"] = "m^-3"
"""

@register_dataarray_accessor("turbo")
class TurbulenceDataArrayAccessor(BoutDataArrayAccessor):
    """
    Class for calculating things/plotting things that I think will be useful for the 3d turbulence analysis.
    
    Will likely try to incorporate the information geometry

    """
    def __init__(self, ds):
        super().__init__(ds)
        self.data = ds
        self.metadata = ds.attrs.get("metadata")  
        self.coords = ds.coords
        self.attrs = ds.attrs
        
    
    @property
    def plot_diagonal_profile(self):
        """
        This take a radial profile at an arbitrary theta location, measured against real space distance from the separatrix
        
        It requires the 'self' to be selected at a particular time, with a theta index already selected. 
        ### The show_location option shows a poloidal cross section of the plasma with a line over the location of the 
        No to the above.
        
        This function will do one thing and one thing only, and that is plot a poloidal profile. It requires the input to
        only be 1D, with a poloidal slice selected already.
        
        If the data input is zeta-averaged, it will plot the average. If not, it won't.
        """
        #poloidal_slice = bd.isel(theta = theta_index) # should already be pre-selected
        R_values = np.array(self.coords['R'])
        Z_values = np.array(self.coords['Z'])
        # get the grid cell of the separatrix
        
        separatrix_index = self.metadata['ixseps1']
        
        if separatrix_index < self.metadata['nx']:
            sep_R = R_values[separatrix_index]
            sep_Z = Z_values[separatrix_index]
        else:
            sep_R = R_values[-1]
            sep_Z = Z_values[-1]

        # calculate the deviation of the coordinate from the separatrix location
        physical_grid = []
        for x_index in self.coords['x']:
            particular_R = R_values[x_index]
            particular_Z = Z_values[x_index]
            if x_index < separatrix_index:
                delta_R = particular_R - sep_R 
                delta_Z = particular_Z - sep_Z
                grid_distance = -np.sqrt(delta_R**2 + delta_Z**2)
                physical_grid.append(grid_distance)
            elif x_index >= separatrix_index:
                delta_R = particular_R - sep_R 
                delta_Z = particular_Z - sep_Z
                grid_distance = np.sqrt(delta_R**2 + delta_Z**2)
                physical_grid.append(grid_distance)

        physical_grid = np.array(physical_grid)
        
        self_name = str(self.attrs['long_name'])
        self_conversion = self.attrs['conversion']
        self_units = str(self.attrs['units'])

        slice_amplitudes = np.array(self.data)*self_conversion
        
        fig, ax = plt.subplots()
        ax.set_xlabel("R-R_sep (m)")
        ax.set_ylabel(self_name + ' (' + self_units + ')')
        ax.plot(physical_grid, slice_amplitudes)
        ax.axvline(0, ls = ':', color = 'black')
        plt.show()
        
        return 'Plotting'
    
    
    @property
    def fluctuations(self):
        """
        This gets delta-n compared to the mean profiles of the given variable. You cannot log10 this, because it goes +ve and -ve.
        This will return a data array that can be put straight back into a bd dataset.
        """
        mean_values = self.data.mean('zeta')
        expanded_means = mean_values.expand_dims(dim={'zeta':self.coords['zeta']}, axis=3)

        # before this will work, I need to turn mean_values into a repeat along the zeta axis, the correct number of times
        delta_values = self.data - expanded_means # +ve delta values means there is more fluctuation there than expected from background, ie overdense. -ve is underdense
        
        short_name = self.data.name
        
        attributes = self.attrs
        attributes['long_name'] = str(attributes['long_name'] + ' fluctuations')
        attributes['standard_name'] = str(attributes['standard_name'] + ' fluctuations')
        
        delta_value_array = xarray.DataArray(delta_values, dims = self.data.dims, coords = self.coords, attrs = attributes)
        
        return delta_value_array
    
    @property
    def plot_location(self):
        """
        This will show the cell centres where the diagnostic is taken
        """
        self_R = np.array(self.coords['R'])
        self_Z = np.array(self.coords['Z'])
        
        location = (self_R, self_Z)
        full_grid_plotting(grid_dataset, location)
        
        return "plotted"
        
    
@register_dataset_accessor("turbo")
class TurbulenceDatasetAccessor(BoutDatasetAccessor):
    """
    Class for calculating things/plotting things that I think will be useful for the 3d turbulence analysis.
    
    Will likely try to incorporate the information geometry

    """
    def __init__(self, ds):

        super().__init__(ds)
        self.data = ds
        self.metadata = ds.attrs.get("metadata")  
        self.coords = ds.coords
        self.attrs = ds.attrs
        
    
