# -*- coding: utf-8 -*-
"""
Specify all dependecies for installing the wavefront reconstruction package.

@author: sklykov

@license: GPLv3, general terms on: https://www.gnu.org/licenses/gpl-3.0.en.html

"""
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as readme_info:
    long_description = readme_info.read()

setup(name="wavefront_reconstruction",
      version="0.0.3", license="GPLv3",
      license_files=['LICENSE'],
      url="...",
      download_url="...",
      long_description=long_description,
      author="S. K.", author_email="contact at GitHub",
      packages=["wfs_reconstruction"],
      package_data={"wfs_reconstruction": ['calibrations/*', 'pics/*', 'README.md']},
      # data_files=[('.', ['README.md'])],  # adding some info into data folder only
      include_package_data=True,
      description="GUI and functions for reconstruction of wavefronts acquired by a Shack-Hartmann sensor",
      install_requires=['matplotlib', 'numpy', 'scikit-image', 'scipy'])
