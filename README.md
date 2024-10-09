# turbhermes

The github for my python methods to analyse Hermes-3 turbulence simulations. Built on the xhermes and boutdata/boututils frameworks, which in turn are build on xarray.

turbhermes is the folder that contains the __init__.py, and will be the actual python module to be loaded.

The setup.py should be able to allow a local install of the module, it has been mostly copied from the xhermes setup.py version.

The accessors and functions will need cleaning and editing, some do not work effectively, others are based on xbout instead of xhermes, so need factors of normalisation to be removed. 
