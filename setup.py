import os
from setuptools import setup, find_packages

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "turbhermes",
	version="0.1",
	description="Analyse Hermes-3 simulations, extending from xhermes",
	license="Apache",
	long_description = read("README.md"),
	classifiers=[
		"Development Status :: 2 - Pre-Alpha",
		"Intended Audience :: Science/Research",
		"Intended Audience :: Education",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: Apache License",
		"Natural Language :: English",
		"Operating System :: POSIX :: Linux",
		"Programming Language :: Python :: 3.6",
		"Topic :: Scientific/Engineering :: Physics",
	],
	install_requires=["xarray", "xbout", "xhermes"],
	packages=find_packages(),
	include_package_data=True,
)
