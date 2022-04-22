# -*- coding: utf-8 -*-
"""
Compose tests for implemented modal phase wavefront reconstruction using decomposition to Zernike polynomials.

@author: ssklykov
"""
# %% Imports
from calc_zernikes_sh_wfs import calc_integral_matrix_zernike
from calc_zernikes_sh_wfs import get_polynomials_coefficients
from calc_zernikes_sh_wfs import get_integral_limits_nonaberrated_centers, get_coms_shifts
from zernike_pol_calc import plot_zps_polar, get_classical_polynomial_name
import os
import time
import numpy as np

# %% Type of calibration
shwfs = False; repo_pics = True; n_zernikes = 14

# %% Making calibration (integration of Zernike polynomials over sub-apertures) once and reading the integral matrix later
# zernikes_set1 = [(-1, 1), (1, 1)]
# zernikes_set2 = [(-2, 2), (0, 2), (2, 2)]
# zernikes_set3 = [(-3, 3), (-1, 3), (1, 3), (3, 3)]
# zernikes_set4 = [(-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]
# zernikes_set5 = [(-5, 5), (-3, 5), (-1, 5), (1, 5), (3, 5), (5, 5)]
zernikes_set14 = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]
zernikes_set20 = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4),
                  (-5, 5), (-3, 5), (-1, 5), (1, 5), (3, 5), (5, 5)]
if n_zernikes == 20:
    zernikes_set = zernikes_set20
else:
    zernikes_set = zernikes_set14

# %% Testing the another calibration mechanism - calculate only once the integral matrix for the non-aberrated image
if repo_pics:
    # Parameters for integration
    plot = True; aperture_radius = 14.0; threshold = 58.0; region_size = 16; n_integration_steps = 80; min_dist_peaks = 18
    # Manual specification of relative paths to the files
    current_path = os.path.dirname(__file__)  # get path to the folder containing the script
    calibrations = os.path.join(current_path, "calibrations")  # the "calibrations" folder with all saved calculations data
    precalculated_zernikes = os.path.join(calibrations, "IntegralMatrix20TabularZernike_RepoPics.npy")
    # precalculated_zernikes2 = os.path.join(calibrations, "integral_calibration_matrix.npy")
    precalculated_nonaberration = os.path.join(calibrations, "CoMsNonaberrated_RepoPics.npy")
    # precalculated_nonaberration2 = os.path.join(calibrations, "detected_focal_spots.npy")
    if not(os.path.exists(precalculated_zernikes)):
        t1 = time.time()  # get the current time measurement
        (coms_nonaberrated, pic_integral_limits,
         theta0, rho0, integration_limits) = get_integral_limits_nonaberrated_centers(plot_results=plot, threshold_abs=threshold,
                                                                                      region_size=region_size,
                                                                                      aperture_radius=aperture_radius,
                                                                                      min_dist_peaks=min_dist_peaks)
        integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0, aperture_radius=aperture_radius,
                                                       n_steps=n_integration_steps, use_tabular_functions=True)
        np.save(precalculated_zernikes, integral_matrix)
        if not(os.path.exists(precalculated_nonaberration)):  # if CoMs from non-aberrated image not saved, save them
            np.save(precalculated_nonaberration, coms_nonaberrated)
        t2 = time.time()  # get the current time measurement
        if np.round(t2-t1, 3) > 60:
            print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round((t2-t1)/60, 1), "minutes")
        else:
            print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
    else:
        integral_matrix = np.load(precalculated_zernikes); coms_nonaberrated = np.load(precalculated_nonaberration)
        # coms_nonaberrated2 = np.load(precalculated_nonaberration2); diff_coms = coms_nonaberrated2 - coms_nonaberrated
        # integral_matrix2 = np.load(precalculated_zernikes2); diff_int_matricies = integral_matrix2 - integral_matrix
        (coms_shifts, integral_matrix_aberrated) = get_coms_shifts(coms_nonaberrated, integral_matrix, plot_results=plot,
                                                                   threshold_abs=threshold, region_size=region_size)
        alpha_coefficients = list(get_polynomials_coefficients(integral_matrix_aberrated, coms_shifts)*np.pi)  # !!! * by pi
        # Plotting the found reconstructed wavefront
        plot_zps_polar(zernikes_set, title="reconstruction of pics from the open repository",
                       alpha_coefficients=alpha_coefficients)
        # Plotting the 14 Zernikes, assuming that the graph from the calculation gives the coefficients for Zernike polynomials
        repo_coefficients = [-0.987, 0.367, 0.341, -1.434, 0.843, -0.306, 0.192, 0.375, -0.277, 0.343, -0.159, 0.171, -0.037, -0.016,
                             0.0, -0.01, 0.25, 0.0, 0.004, 0.0]
        for i in range(len(zernikes_set)):
            print(get_classical_polynomial_name(zernikes_set[i]), ":", np.round(alpha_coefficients[i], 3), "(SK) ",
                  repo_coefficients[i], "(Jacopo's)")
        plot_zps_polar(zernikes_set, title="profile reconstructed by Jacopo", alpha_coefficients=repo_coefficients)

# %% Tests on the recorded pictures from the Shack-Hartmann sensor
if shwfs:
    # Parameters for integration
    plot = False; aperture_radius = 10.0; threshold = 54.0; region_size = 18; n_integration_steps = 80; min_dist_peaks = 16
    # Manual change the working directory to the folder with stored pictures, outside the repository
    current_path = os.path.dirname(__file__)  # get path to the folder containing the script
    calibrations = os.path.join(current_path, "calibrations")  # the "calibrations" folder with all saved calculations data
    aberrated_pic_name = "AstigmatismPic2.png"  # Name of prerecorded picture with aberrations
    precalculated_zernikes = os.path.join(calibrations, "IntegralMatrix20TabZernike_RecordedAberrations.npy")
    precalculated_nonaberration = os.path.join(calibrations, "CoMsNonaberrated_RecordedAberrations.npy")
    os.chdir(".."); os.chdir(".."); os.chdir("sh_wfs")  # Navigation to the local storage with recorded aberrations
    if not(os.path.exists(precalculated_zernikes)):
        t1 = time.time()  # get the current time measurement
        (coms_nonaberrated, pic_integral_limits,
         theta0, rho0, integration_limits) = get_integral_limits_nonaberrated_centers(pics_folder=os.getcwd(),
                                                                                      background_pic_name="backgroundPic2.png",
                                                                                      nonaberrated_pic_name="nonAberrationPic2.png",
                                                                                      plot_results=plot, threshold_abs=threshold,
                                                                                      region_size=region_size,
                                                                                      aperture_radius=aperture_radius,
                                                                                      min_dist_peaks=min_dist_peaks)
        integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0, aperture_radius=aperture_radius,
                                                       n_steps=n_integration_steps, use_tabular_functions=True)
        np.save(precalculated_zernikes, integral_matrix)
        if not(os.path.exists(precalculated_nonaberration)):  # if CoMs from non-aberrated image not saved, save them
            np.save(precalculated_nonaberration, coms_nonaberrated)
        t2 = time.time()  # get the current time measurement
        if np.round(t2-t1, 3) > 60:
            print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round((t2-t1)/60, 1), "minutes")
        else:
            print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
    else:
        integral_matrix = np.load(precalculated_zernikes); coms_nonaberrated = np.load(precalculated_nonaberration)
        plot = False; threshold = 54.0
        (coms_shifts, integral_matrix_aberrated) = get_coms_shifts(coms_nonaberrated, integral_matrix,
                                                                   pics_folder=os.getcwd(), aberrated_pic_name=aberrated_pic_name,
                                                                   plot_results=plot, threshold_abs=threshold, region_size=region_size)
        alpha_coefficients = list(get_polynomials_coefficients(integral_matrix_aberrated, coms_shifts)*np.pi)   # !!! * by pi
        # Plot the sum profile of all
        if aberrated_pic_name[0:7] == "Defocus":  # because naming was inverted in the recorded aberrations
            title_name = "astigmatism"
        else:
            title_name = "defocus"
        plot_zps_polar(zernikes_set, title="reconstruction of actual " + title_name, alpha_coefficients=alpha_coefficients)
        for i in range(len(zernikes_set)):
            print(get_classical_polynomial_name(zernikes_set[i]), ":", np.round(alpha_coefficients[i], 3))
