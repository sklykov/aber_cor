# -*- coding: utf-8 -*-
"""
Specify all dependecies for installing the wavefront reconstruction package.

@author: Sergei Klykov (GitHub: @ssklykov)
@license: GPLv3 (check https://www.gnu.org/licenses/ )
"""
from setuptools import setup, find_packages

setup(name="wavefront_reconstruction",
      version="0.0.1.dev1", license="GPLv3",
      url="https://github.com/ssklykov/aber_cor",
      author="Sergei Klykov", author_email="contact at GitHub",
      packages=["wavefront_reconstruction"], package_dir={"devices_ctrl"},
      # packages=find_packages(),
      description="GUI and functions for reconstruction of wavefronts acquired by a Shack-Hartmann sensor",
      install_requires=['matplotlib', 'numpy', 'scikit-image', 'scipy'])
