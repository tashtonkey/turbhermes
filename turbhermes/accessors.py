from xarray import register_dataset_accessor, register_dataarray_accessor
from xbout import BoutDatasetAccessor, BoutDataArrayAccessor
import numpy as np
import xarray
import matplotlib.pyplot as plt
from .plotting import diagonal_slice_plotting, diagonal_slice

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
    def normalise_metric(self):
    
        rho_s0 = self.data.metadata['rho_s0']
        Bnorm = self.data.metadata['Bnorm']

        for varname in list(self.data):
            da = self.data[varname]
            #elif varname == "g11":
            #    # Metric tensor term
            #    da.attrs.update(
            #        {
            #            "units_type": "hermes",
            #            "units": "T2 m2",
            #            "conversion": (Bnorm * rho_s0)**2,
            #            "standard_name": "g11",
            #            "long_name": "g11 term in the metric tensor",
            #        }
            #    )
            if varname == "g22":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "m-2",
                        "conversion": 1/(rho_s0)**2,
                        "standard_name": "g22",
                        "long_name": "g22 term in the metric tensor",
                    }
                )
                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"

            elif varname == "g33":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "m-2",
                        "conversion": 1/(rho_s0)**2,
                        "standard_name": "g33",
                        "long_name": "g33 term in the metric tensor",
                    }
                )
                
                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"
            elif varname == "g12":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "T",
                        "conversion": Bnorm,
                        "standard_name": "g12",
                        "long_name": "g12 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"
            elif varname == "g13":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "T",
                        "conversion": Bnorm,
                        "standard_name": "g13",
                        "long_name": "g13 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"

            elif varname == "g23":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "m-2",
                        "conversion": 1/(rho_s0)**2,
                        "standard_name": "g23",
                        "long_name": "g23 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"

            elif varname == "g_11":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "T-2m-2",
                        "conversion": 1/(Bnorm * rho_s0)**2,
                        "standard_name": "g_11",
                        "long_name": "g_11 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"

            #elif varname == "g_22":
            #    # Metric tensor term
            #    da.attrs.update(
            #        {
            #            "units_type": "hermes",
            #            "units": "m2",
            #            "conversion": (rho_s0)**2,
            #            "standard_name": "g_22",
            #            "long_name": "g_22 term in the metric tensor",
            #        }
            #    )
            elif varname == "g_33":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "m2",
                        "conversion": (rho_s0)**2,
                        "standard_name": "g_33",
                        "long_name": "g_33 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"

            elif varname == "g_12":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "T-1",
                        "conversion": 1/Bnorm,
                        "standard_name": "g_12",
                        "long_name": "g_12 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"

            elif varname == "g_13":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "T-1",
                        "conversion": 1/Bnorm,
                        "standard_name": "g_13",
                        "long_name": "g_13 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"

            elif varname == "g_23":
                # Metric tensor term
                da.attrs.update(
                    {
                        "units_type": "hermes",
                        "units": "m2",
                        "conversion": (rho_s0)**2,
                        "standard_name": "g_23",
                        "long_name": "g_23 term in the metric tensor",
                    }
                )

                da *= da.attrs["conversion"]
                da.attrs["units_type"] = "SI"


    @property
    def radial_E_field(self):
        """Calculates local radial electric field"""
        
        if "radial_E" not in self.data:
            E_x = -self.data["phi"].bout.ddx()
            E_x.attrs["standard_name"] = "radial E field"
            E_x.attrs["long_name"] = "radial electric field"
            E_x.attrs["units"] = "V m^-1"
            self.data["radial_E"] = E_x
        return self.data["radial_E"]
    
    @property
    def calculate_temp(self):
        """Calculates local temperature
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
        
        species_list = self.data.metadata['species']
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

            phi_ddx = -potential.bout.ddx() # this is Ex. No need for any conversion
            phi_ddy = -potential.bout.ddy() # this is Ey
            phi_ddz = -potential.bout.ddz() # this is Ez

            # Don't need to convert units because xHermes does it for me
            # -ve in front to be consistent with documentation from the BOUT++ manual : v_ExB = ExB/B**2
            V_ExB_x = ((phi_ddy * g_23 - phi_ddz * g_22) / (g_22)) * np.sqrt(g_11)
            V_ExB_y = ((phi_ddz * g_12 - phi_ddx * g_23) / (g_22)) * np.sqrt(g_22)
            V_ExB_z = ((phi_ddx * g_22 - phi_ddy * g_12) / (g_22)) * np.sqrt(g_33)

            V_ExB_x.attrs['long_name'] = 'x ExB velocity component'
            V_ExB_y.attrs['long_name'] = 'y ExB velocity component'
            V_ExB_z.attrs['long_name'] = 'z ExB velocity component'
            V_ExB_x.attrs['standard_name'] = 'x ExB velocity component'
            V_ExB_y.attrs['standard_name'] = 'y ExB velocity component'
            V_ExB_z.attrs['standard_name'] = 'z ExB velocity component'
            V_ExB_x.attrs['conversion'] = 1
            V_ExB_y.attrs['conversion'] = 1
            V_ExB_z.attrs['conversion'] = 1
            V_ExB_x.attrs['units'] = 'm / s'
            V_ExB_y.attrs['units'] = 'm / s'
            V_ExB_z.attrs['units'] = 'm / s'

            self.data['V_ExB_x'] = V_ExB_x
            self.data['V_ExB_y'] = V_ExB_y
            self.data['V_ExB_z'] = V_ExB_z

            sigma_B_pol = 1 # this should be the sigma-b-pol, check what the way to get this is.
            
            V_ExB_r = V_ExB_x * sigma_B_pol
            V_ExB_theta = V_ExB_y * np.sqrt(1/(g22 * g_22))
            V_ExB_zeta = V_ExB_y * (g_23/np.sqrt(g_22 * g_33)) + V_ExB_z

            V_ExB_r.attrs['long_name'] = 'radial ExB velocity component'
            V_ExB_theta.attrs['long_name'] = 'theta ExB velocity component'
            V_ExB_zeta.attrs['long_name'] = 'zeta ExB velocity component'
            V_ExB_r.attrs['standard_name'] = 'radial ExB velocity component'
            V_ExB_theta.attrs['standard_name'] = 'theta ExB velocity component'
            V_ExB_zeta.attrs['standard_name'] = 'zeta ExB velocity component'
            V_ExB_r.attrs['conversion'] = 1
            V_ExB_theta.attrs['conversion'] = 1
            V_ExB_zeta.attrs['conversion'] = 1
            V_ExB_r.attrs['units'] = 'm / s'
            V_ExB_theta.attrs['units'] = 'm / s'
            V_ExB_zeta.attrs['units'] = 'm / s'

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
        
    
    def plot_diagonal_slice(
        self,
        save_as=None,
        sep_pos=None,
        ax=None,
        **kwargs,
    ):
        data = self.data
        variable = data.name
        n_dims = len(data.dims)

        if n_dims == 1:
            print(
                "{} data passed has {} dimensions - plotting 1D profile".format(variable, str(n_dims))
            )
            line_block = diagonal_slice(
                data=data,
                sep_pos=sep_pos,
                save_as=save_as,
                ax=ax,
                **kwargs,
            )
            return line_block
        else:
            print(
                "{} data passed has {} dimensions - incorrect dimensions".format(variable, str(n_dims))
            )
            return
    
    def animate_diagonal_profile(
        self,
        animate_over=None,
        animate=True,
        axis_coords=None,
        fps=10,
        save_as=None,
        sep_pos=None,
        ax=None,
        **kwargs,
    ):
        """
        Plots a line plot which is animated over time over the specified coordinate.

        Currently only supports 1D+1 data, which it plots with animatplot's wrapping of
        matplotlib's plot.

        Parameters
        ----------
        animate_over : str, optional
            Dimension over which to animate, defaults to the time dimension
        axis_coords : None, str, dict
            Coordinates to use for axis labelling.

            - None: Use the dimension coordinate for each axis, if it exists.
            - "index": Use the integer index values.
            - dict: keys are dimension names, values set axis_coords for each axis
              separately. Values can be: None, "index", the name of a 1d variable or
              coordinate (which must have the dimension given by 'key'), or a 1d
              numpy array, dask array or DataArray whose length matches the length of
              the dimension given by 'key'.
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
            Argument to ``ax.set_aspect()``, defaults to "auto"
        kwargs : dict, optional
            Additional keyword arguments are passed on to the plotting function
            (animatplot.blocks.Line).

        Returns
        -------
        animation or block
            If ``animate==True``, returns an animatplot.Animation object, otherwise
            returns an animatplot.blocks.Line instance.
        """

        data = self.data
        variable = data.name
        n_dims = len(data.dims)

        if n_dims == 2:
            print(
                "{} data passed has {} dimensions - will use "
                "animatplot.blocks.Line()".format(variable, str(n_dims))
            )
            line_block = diagonal_slice_plotting(
                data=data,
                animate_over=animate_over,
                axis_coords=axis_coords,
                sep_pos=sep_pos,
                animate=animate,
                fps=fps,
                save_as=save_as,
                ax=ax,
                **kwargs,
            )
            return line_block
    


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
        
    
