# aber_cor
Collection of Python scripts for reconstruction and estimation of optical aberrations. 

- It consists of two folders up to now, in one there are files for calculation of Zernike polynomials sums for representing 
a phase front, reconstruction of wafefronts (see details in the folder "ZernikePolynomials"). Another one - collection of 
scripts for controlling sCMOS camera, providing interface for specification of Zernike's polynomials 
amplitudes (for further sending to a device under development) and their sum representation for the user. 
- All scripts have been developed in the Spyder IDE with common dependencies for numerical calculations: scipy(numpy),
matplotlib, scikit-image (skimage), pyqt5, pyqtgraph. I don't see any necessity to specify it now in any "requirements"
file for installation from a scratch. In the case of any interest, please, refer to import sections of the scripts.
- I would be happy if some parts of this repository will be useful for anyone.
