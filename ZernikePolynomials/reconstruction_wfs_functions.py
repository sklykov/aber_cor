# -*- coding: utf-8 -*-
"""
Specification of calculation functions for reconstruction using non- and aberrated wavefronts recorded by a Shack-Hartmann sensor.

These functions are called in 'wfs_reconstruction_ui.py' for using it in the GUI controlling program.
According to the doctoral thesis by Antonello, J. (2014): https://doi.org/10.4233/uuid:f98b3b8f-bdb8-41bb-8766-d0a15dae0e27

@author: ssklykov
"""

# %% Imports and globals
import numpy as np
from skimage.feature import peak_local_max
# import time
from matplotlib.patches import Circle
# from matplotlib.patches import Rectangle  # uncomment if need for visualization
from scipy import ndimage
from calc_zernikes_sh_wfs import check_img_coordinate
from threading import Thread
from queue import Empty, Queue
from zernike_pol_calc import normalization_factor
from calc_zernikes_sh_wfs import (rho_ab, rho_integral_funcX, rho_integral_funcY,
                                  r_integral_tabular_funcX, r_integral_tabular_funcY)


# %% Function definitions
def get_localCoM_matrix(image: np.ndarray, axes_fig, min_dist_peaks: int = 15, threshold_abs: float = 55.0,
                        region_size: int = 16) -> np.array:
    """
    Calculate local center of masses in the region around of found peaks (maximums).

    Parameters
    ----------
    image : np.ndarray
        Shack-Hartmann image with focal spots of focused wavefront.
    axes_fig : matplotlib.Axes
        Axes class of the figure shown in the GUI widget (window).
    min_dist_peaks : int, optional
        Minimal distance between two local peaks. The default is 15.
    threshold_abs : float, optional
        Absolute minimal intensity value for start searching of a local peak. The default is 55.0.
    region_size : int, optional
        Size of local rectangle, there the center of mass is calculated. The default is 16.

    Returns
    -------
    coms : np.array
        Calculated center of masses (CoMs) coordinates.

    """
    detected_centers = peak_local_max(image, min_distance=min_dist_peaks, threshold_abs=threshold_abs)
    (rows, cols) = image.shape
    # axes_fig.plot(detected_centers[:, 1], detected_centers[:, 0], '.', color="red")  # plot local peaks
    half_size = region_size // 2  # Half of rectangle area for calculation of CoM
    size = np.size(detected_centers, 0)  # Number of found local peaks
    coms = np.zeros((size, 2), dtype='float')  # Center of masses coordinates initialization
    for i in range(size):
        x_left_upper = check_img_coordinate(cols, detected_centers[i, 1] - half_size)
        y_left_upper = check_img_coordinate(rows, detected_centers[i, 0] - half_size)
        # Plot found regions for CoM calculations
        # axes_fig.add_patch(Rectangle((x_left_upper, y_left_upper), 2*half_size, 2*half_size,
        #                              linewidth=1, edgecolor='yellow', facecolor='none'))
        # CoMs calculation
        subregion = image[y_left_upper:y_left_upper+2*half_size, x_left_upper:x_left_upper+2*half_size]
        (coms[i, 0], coms[i, 1]) = ndimage.center_of_mass(subregion)
        coms[i, 0] += y_left_upper; coms[i, 1] += x_left_upper
    # Plot found CoMs
    # axes_fig.plot(coms[:, 1], coms[:, 0], '.', color="green")
    return coms


def get_integral_limits_nonaberrated_centers(axes_fig, picture_as_array: np.ndarray, threshold_abs: float = 55.0,
                                             aperture_radius: float = 15.0) -> tuple:
    """
    Calculate the center of masses of localized focal spots and also the integration limits for further modal wavefront reconstruction.

    Focal spots are recorded within the non-aberrated image (flat wavefront) from Shack-Hartmann sensor.

    Parameters
    ----------
    axes_fig : matplotlib.Axes
        Axes class of the figure shown in the GUI widget (window).
    picture_as_array : np.ndarray
        Shack-Hartmann image with focal spots of focused wavefront.
    threshold_abs : float
        Absolute minimal intensity value for start searching of a local peak. The default is 55.0.
    aperture_radius : float
        Radius of sub-aperture radius in pixels. The default is 15.0.

    Returns
    -------
    tuple
        (Centers of masses for non-aberrated image without the central sub-aperture, Theta (polar) coordinates of sub-apertures,
         Rho (polar) coordinates of sub-apertures, Integration limits for sub-apertures).

    """
    # Use the specified radius of sub-apertures for calculation of parameters for CoMs defining
    min_dist_peaks = int(np.round(1.5*aperture_radius, 0))
    region_size = int(np.round(1.4*aperture_radius, 0))
    # CoMs = center of masses, calculated in the areas around found local peaks
    coms_nonaberrated = get_localCoM_matrix(picture_as_array, axes_fig, min_dist_peaks=min_dist_peaks,
                                            threshold_abs=threshold_abs, region_size=region_size)
    # If the specified parameters provide no result for searching for focal spots, then skip all calculations
    if len(coms_nonaberrated) == 0:
        print("No detected focal spots")
        subapertures_wt_central = theta0 = rho0 = integration_limits = -1
    else:
        # Searching for the central sub-aperture that should be close to the center of the image
        (rows, cols) = picture_as_array.shape; x_img_center = cols//2; y_img_center = rows//2
        x_min_dist = cols; y_min_dist = rows; i_center_subaperture = -1
        min_distance = np.sqrt(np.power(x_min_dist, 2) + np.power(y_min_dist, 2))
        # Looking for minimal distance between sub-apertures and the center of the frame and saving its index
        for i in range(np.size(coms_nonaberrated, 0)):
            distance = np.sqrt(np.power((coms_nonaberrated[i, 1] - x_img_center), 2)
                               + np.power((coms_nonaberrated[i, 0] - y_img_center), 2))
            if min_distance > distance:
                i_center_subaperture = i; min_distance = distance
        x_central_subaperture = coms_nonaberrated[i_center_subaperture, 1]
        y_central_subaperture = coms_nonaberrated[i_center_subaperture, 0]
        # Plotting the found center of image and central sub-aperture
        axes_fig.plot(x_central_subaperture, y_central_subaperture, '+', color="red")
        # axes_fig.plot(x_img_center, y_img_center, '+', color="blue")
        # Detected centers below are searched for on the aberrated image, so the shifts in CoMs should be calculated relatively to it
        rho0 = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # polar coordinate r of a sub-aperture (lens)
        theta0 = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # polar coordinate theta of a sub-aperture (lens)
        theta_a = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # integration limits (lower) on theta (1)
        theta_b = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # integration limits (higher) on theta (2)
        subapertures_wt_central = np.zeros((np.size(coms_nonaberrated, 0)-1, 2), dtype='float')
        integration_limits = [(0.0, 0.0) for i in range(np.size(coms_nonaberrated, 0)-1)]
        # Calculation of integration limits - below
        j = 0
        for i in range(np.size(coms_nonaberrated, 0)):
            # Plot found regions for CoM calculations
            if not(i == i_center_subaperture):
                axes_fig.add_patch(Circle((coms_nonaberrated[i, 1], coms_nonaberrated[i, 0]), aperture_radius,
                                          edgecolor='green', facecolor='none'))
            if i == i_center_subaperture:  # Deleting the central subaperture from the calculations
                continue
            else:
                # Calculation of the boundaries of the integration intervals for equations from the thesis
                x_relative = coms_nonaberrated[i, 1] - x_central_subaperture  # relative to the central sub-aperture
                y_relative = -coms_nonaberrated[i, 0] + y_central_subaperture   # relative to the central sub-aperture
                rho0[j] = np.sqrt(np.power(x_relative, 2) + np.power(y_relative, 2))  # radial coordinate of lens pupil
                # Calculation of integration limits for the angles theta
                theta0[j] = np.arctan2(y_relative, x_relative)*(180/np.pi)  # Calculation arctan with right quadrant selection in grads!
                # ??? Maybe redundant but seems better to make conversion to all positive values angles:
                if theta0[j] < 0.0:
                    theta0[j] += 360.0   # all negative angles become positive
                theta_a[j] = np.round(theta0[j] - np.arctan(aperture_radius/rho0[j])*(180/np.pi), 3)  # calculation in grads!
                theta_b[j] = np.round(theta0[j] + np.arctan(aperture_radius/rho0[j])*(180/np.pi), 3)  # calculation in grads!
                integration_limits[j] = (theta_a[j], theta_b[j])
                subapertures_wt_central[j, 0] = coms_nonaberrated[i, 0]; subapertures_wt_central[j, 1] = coms_nonaberrated[i, 1]
                j += 1
        # Plot the found and used CoMs for calculations
        axes_fig.plot(subapertures_wt_central[:, 1], subapertures_wt_central[:, 0], '.', color="red")
        # Before the calculation for angles made in grads, below the transfer to radians for passing it further to trigonometric functions
        theta0 = np.radians(theta0); integration_limits = np.radians(integration_limits)
    return subapertures_wt_central, theta0, rho0, integration_limits


def get_zernike_coefficients_list(selected_order: int) -> list:
    """
    Return list with tuples containing azimuthal and radial orders (m, n).

    Parameters
    ----------
    selected_order : int
        Selected Zernike order, now in range from 1 to 5.

    Returns
    -------
    list
        List with sequential azimuthal and radial orders stored in tuples as (m, n).

    """
    zernike_coefficients_dict = {1: [(-1, 1), (1, 1)], 2: [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2)],
                                 3: [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3)],
                                 4: [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3),
                                     (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)],
                                 5: [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3),
                                     (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4), (-5, 5), (-3, 5), (-1, 5), (1, 5),
                                     (3, 5), (5, 5)]}
    if selected_order >= 1 and selected_order <= 5:
        return zernike_coefficients_dict[selected_order]
    else:
        return []


def get_zernike_order_from_coefficients_number(coefficients_number: int) -> int:
    """
    Return maximum Zernike polynomial order.

    Parameters
    ----------
    coefficients_number : int
        Number of coeffficients with orders (m ,n).

    Returns
    -------
    int
        Order of maximum Zernike polynomial.

    """
    coefficients_number_dict = {2: 1, 5: 2, 9: 3, 14: 4, 20: 5, 27: 6, 35: 7}
    if coefficients_number >= 2 and coefficients_number <= 35:
        return coefficients_number_dict[coefficients_number]
    else:
        return -1


def calc_integrals_on_apertures_unit_circle(integration_limits: np.ndarray, theta0: np.ndarray, rho0: np.ndarray, m: int, n: int,
                                            messages_queue: Queue, n_polynomial: int = 1, aperture_radius: float = 15.0,
                                            n_steps: int = 50, swapXY: bool = True) -> np.ndarray:
    """
    Calculate integrals using trapezodial rule inside the sub-apertures, which lies inside the unit circle.

    Parameters
    ----------
    integration_limits : np.ndarray
        Calculated previously on theta polar coordinate.
    theta0 : np.ndarray
        Polar coordinate theta of sub-aperture centers.
    rho0 : np.ndarray
        Polar coordinates r of sub-aperture centers.
    m : int
        Azimuthal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.
    messages_queue : Queue
        Queue pipe for checking of Stop event and sending back the messages about the calculation progress.
    n_polynomial : int, optional
        Number of polynomial for which the integration calculated. The default is 1.
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    n_steps : int, optional
        Number of integration steps for both integrals. The default is 50.
    swapXY : bool, optional
        For swapping the direction of Y axis pointing up on the picture, instead down as for pixel coordinates (y, x),
        for conforming with the thesis calculations. The default is True.

    Returns
    -------
    integral_values : ndarray with sizes (Number of sub-apertures, 2)
        Resulting integration values for each sub-aperture, they are listed for Y and X axes (relatively to image coordinate system).

    """
    integral_values = np.zeros((len(integration_limits), 2), dtype='float')  # Doubled values - for X,Y axes
    # calibration = 1.0  # TODO: Calibration taking into account the wavelength, focal length should be implemented later
    # Introduction of Zernike's polynomials normalization coefficients => tune of each polynomial contribution
    calibration = normalization_factor(m, n)  # use it for recalculate integral values for testing
    rho_unit_calibration = np.max(rho0) + aperture_radius  # For making integration on rho on unit circle
    # For each sub-aperture below integration on (r, theta) of the Zernike polynomials
    calculation_flag = True  # for stopping the calculation if appropriate messages received
    for i_subaperture in range(len(integration_limits)):
        if not messages_queue.empty():
            try:
                message = messages_queue.get_nowait()
                if message == "Stop integration":
                    calculation_flag = False
                    messages_queue.put_nowait(message)
            except Empty:
                pass
        if calculation_flag:
            # print(f"Integration for #{n_polynomial} pol. started on {i_subaperture} subaperture out of {len(integration_limits)}")
            # Integration limits and steps on theta
            (theta_a, theta_b) = integration_limits[i_subaperture]  # Calculated previously integration steps on theta (radians)
            delta_theta = (theta_b - theta_a)/n_steps  # Step for integration on theta
            theta = theta_a  # initial value of theta
            # Getting limits for integration on rho
            (rho_a, rho_b) = rho_ab(rho0[i_subaperture], theta, theta0[i_subaperture], aperture_radius)
            # All rho values should be normalized to the maximum rho0 coordinate + radius_subaperture
            rho_a /= rho_unit_calibration; rho_b /= rho_unit_calibration
            # Integration over theta (trapezoidal rule)
            integral_sumX = 0.0; integral_sumY = 0.0
            for j_theta in range(n_steps+1):
                # Integration on rho for X and Y axis (trapezoidal formula)
                rho = rho_a  # Lower integration boundary
                delta_rho = (rho_b - rho_a)/n_steps
                integral_sum_rhoX1 = 0.0; integral_sum_rhoX2 = 0.0; integral_sum_rhoY1 = 0.0; integral_sum_rhoY2 = 0.0
                for j_rho in range(n_steps+1):
                    # get 2 parts of functions according the thesis
                    if n <= 7:  # tabular functions specified up to this order
                        (X1, X2) = r_integral_tabular_funcX(rho, theta, m, n)
                        (Y1, Y2) = r_integral_tabular_funcY(rho, theta, m, n)
                    else:
                        (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                        (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    # multiplication depends on starting / finishing point according to the formula
                    if (j_rho == 0) or (j_rho == n_steps):
                        integral_sum_rhoX1 += 0.5*X1; integral_sum_rhoX2 += 0.5*X2  # on X axis
                        integral_sum_rhoY1 += 0.5*Y1; integral_sum_rhoY2 += 0.5*Y2  # on Y axis
                    else:
                        integral_sum_rhoX1 += X1; integral_sum_rhoX2 += X2
                        integral_sum_rhoY1 += Y1; integral_sum_rhoY2 += Y2
                    rho += delta_rho
                integral_sum_rhoX = (integral_sum_rhoX1 - integral_sum_rhoX2)  # Equations from thesis
                integral_sum_rhoY = (integral_sum_rhoY1 + integral_sum_rhoY2)  # Equations from thesis
                # End of integration on rho (i.e. r from polar coordinates)
                integral_sum_rhoX *= delta_rho; integral_sum_rhoY *= delta_rho
                if (j_theta == 0) and (j_theta == n_steps):
                    integral_sumX += 0.5*integral_sum_rhoX; integral_sumY += 0.5*integral_sum_rhoY
                else:
                    integral_sumX += integral_sum_rhoX; integral_sumY += integral_sum_rhoY
                theta += delta_theta
            integral_sumX *= delta_theta; integral_sumY *= delta_theta  # End of integration on theta
            # actually, the integral values should be calibrated to each sub-aperture area - depending on the integration limits
            integral_sumX /= 0.5*(theta_b - theta_a)*((rho_b*rho_b)-(rho_a*rho_a))  # 0.5 - due to integration from (rdr)dtheta
            integral_sumY /= 0.5*(theta_b - theta_a)*((rho_b*rho_b)-(rho_a*rho_a))
            # The final integral values should be also calibrated to focal and wavelengths, but it's not yet implemented
            if swapXY:  # Choosing the relation between X and Y axis calculation (swap them on demand)
                integral_values[i_subaperture, 1] = calibration*integral_sumX  # Not yet implemented calibration, not necessary now
                integral_values[i_subaperture, 0] = calibration*integral_sumY
            else:
                integral_values[i_subaperture, 0] = calibration*integral_sumX
                integral_values[i_subaperture, 1] = calibration*integral_sumY
            integral_values = np.round(integral_values, 8)  # rounding up to ... digits after coma
        else:
            break  # stop the integration on each sub-aperture
    return integral_values


def calc_integral_matrix_zernike(progress_bar, zernike_polynomials_list: list, integration_limits: np.ndarray, theta0: np.ndarray,
                                 rho0: np.ndarray, messages_queue: Queue, aperture_radius: float = 15.0,
                                 n_steps: int = 10, swapXY: bool = True) -> np.ndarray:
    """
    Wrap calculation of integral values on sub-apertures performing on several Zernike polynomials.

    For shortening time of calculation, use the increasing order of listing of polynomials coefficients (like [(-1, 1), (1, 1)])

    Parameters
    ----------
    zernike_polynomials_list: list
        All polynomial specification as 2 orders (m, n) in list.
    integration_limits : np.ndarray
        Calculated previously on theta polar coordinate.
    theta0 : np.ndarray
        Polar coordinate theta of sub-aperture centers.
    rho0 : np.ndarray
        Polar coordinates r of sub-aperture centers.
    messages_queue : Queue
        Queue pipe for checking of Stop event and sending back the messages about the calculation progress.
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    n_steps : int, optional
        Number of integration steps for both integrals. The default is 100.
    swapXY: bool, optional
        For swapping the direction of Y axis pointing up on the picture, instead down as for pixel coordinates (y, x),
        for conforming with the thesis calculations. The default is True.

    Returns
    -------
    integral_matrix : ndarray with sizes (Number of sub-apertures, Number of Zernikes)
        Resulting integration values for each sub-aperture and for both X and Y axes and specified Zernike values.

    """
    progress_bar['value'] = 0  # refresh progress bar
    n_rows = np.size(rho0, 0)
    n_cols = 2*len(zernike_polynomials_list)  # Because calculation needed for both X and Y axes
    integral_matrix = np.zeros((n_rows, n_cols), dtype='float')
    # Shortening the time of calculation because of symmetrical integrals over sub-apertures for (-1, 1) and (1, 1),
    # (-2, 2) and (2, 2), (-3, 3) and (-4, 4) - applying below reassignment
    symmetrical_substitution = False; i_symmetry = -1
    calculation_flag = True  # flag for stopping calculation
    length_add = (100 // len(zernike_polynomials_list))  # portion for progress bar
    s = 0  # for calculation of increasing progress bar value
    progress_bar['value'] = 5  # some visually initial progress bar value
    # Integration for each polynomial:
    for i in range(len(zernike_polynomials_list)):
        # Check the messages for stopping integration
        if not messages_queue.empty():
            try:
                message = messages_queue.get_nowait()
                if message == "Stop integration":
                    calculation_flag = False
            except Empty:
                pass
        if calculation_flag:
            (m, n) = zernike_polynomials_list[i]
            if (((m == -1) and (n == 1)) or ((m == -2) and (n == 2)) or ((m == -3) and (n == 3)) or ((m == -4) and (n == 4))
                    or ((m == -5) and (n == 5))):
                i_symmetry = i; symmetrical_substitution = True  # Regulates if shortening of calculation happen below
            if symmetrical_substitution:
                if (((m == 1) and (n == 1)) or ((m == 2) and (n == 2)) or ((m == 3) and (n == 3)) or ((m == 4) and (n == 4))
                        or ((m == 5) and (n == 5))):
                    # Shortening the time of calculation because of symmetrical integrals over sub-apertures for (-1, 1) and (1, 1),
                    # (-2, 2) and (2, 2), (-3, 3) and (-4, 4) - applying below reassignment
                    integral_matrix[:, 2*i] = -integral_matrix[:, 2*i_symmetry+1]
                    integral_matrix[:, 2*i+1] = integral_matrix[:, 2*i_symmetry]
                else:
                    # Normal integrals calculation
                    integral_values = calc_integrals_on_apertures_unit_circle(integration_limits, theta0, rho0, m, n,
                                                                              messages_queue=messages_queue, n_polynomial=i+1,
                                                                              aperture_radius=aperture_radius, n_steps=n_steps,
                                                                              swapXY=swapXY)
                    integral_matrix[:, 2*i] = integral_values[:, 0]; integral_matrix[:, 2*i+1] = integral_values[:, 1]
            else:
                # Normal integrals calculation
                integral_values = calc_integrals_on_apertures_unit_circle(integration_limits, theta0, rho0, m, n,
                                                                          messages_queue=messages_queue, n_polynomial=i+1,
                                                                          aperture_radius=aperture_radius, n_steps=n_steps,
                                                                          swapXY=swapXY)
                integral_matrix[:, 2*i] = integral_values[:, 0]; integral_matrix[:, 2*i+1] = integral_values[:, 1]
            print(f"Calculated {i+1} polynomial out of {len(zernike_polynomials_list)}")
            s += length_add; progress_bar['value'] = s
        else:
            # Integration was aborted
            progress_bar['value'] = 0; integral_matrix = []; break  # stop for loop above
    if calculation_flag:
        messages_queue.put_nowait("Integration finished")
        if s < 100:
            progress_bar['value'] = 100
    else:
        messages_queue.put_nowait("Integration aborted")
    return integral_matrix


def get_coms_shifts(coms_nonaberrated: np.ndarray, integral_matrix: np.ndarray, coms_aberrated: np.ndarray,
                    aperture_radius: float = 15.0) -> tuple:
    """
    Calculate shifts between center of masses around local focal spots for aberrated and non-aberrated images.

    Parameters
    ----------
    coms_nonaberrated : np.ndarray
        Calculated center of masses around local focal spots on the non-aberrated image.
    coms_aberrated : np.ndarray
        Calculated center of masses around local focal spots on the aberrated image.
    integral_matrix : np.ndarray
        Calculated the integral matrix for Zernike polynomials.
    aperture_radius : float
        Radius of sub-aperture radius in pixels. The default is 15.0.

    Returns
    -------
    tuple
        (shifts of CoMs, integral matrix according to the detected CoMs on the aberrated image).

    """
    min_dist_peaks = int(np.round(1.5*aperture_radius, 0))  # estimation based on specified aperture radius
    # Calculate the shifts between CoMs in aberrated and non-aberrated images
    coms_shifts = np.zeros((np.size(coms_aberrated, 0), 2), dtype='float')  # Shifts between CoMs
    # Recalculate the integration values that will be used further for calculation of alpha coefficient
    integral_matrix_aberrated = np.zeros((np.size(coms_aberrated, 0), np.size(integral_matrix, 1)), dtype='float')
    i_central_aperture = -1  # for defining the central aperture in the aberrated list of CoMs
    # Below - calculation of shifts between CoMs
    for i in range(np.size(coms_aberrated, 0)):
        central_aperture_found = True  # flag for defining the index of central sub-aperture in the aberrated list
        for j in range(0, np.size(coms_nonaberrated, 0)):
            diffY = abs(coms_aberrated[i, 0] - coms_nonaberrated[j, 0]); diffX = abs(coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
            diffY_sign = (coms_aberrated[i, 0] - coms_nonaberrated[j, 0]); diffX_sign = (coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
            if (diffX < float((min_dist_peaks/2))) and (diffY < float((min_dist_peaks/2))):
                central_aperture_found = False  # the matching is defined, shift found => it's not the central sub-aperture
                coms_shifts[i, 0] = -diffY_sign  # Direction of Y axis swapped (not as on the picture, from top to bottom)
                coms_shifts[i, 1] = diffX_sign  # Direction of X axis is the same as on the picture (from left to right)
                for k in range(np.size(integral_matrix, 1)):
                    integral_matrix_aberrated[i, k] = integral_matrix[j, k]
                break  # Stop searching the candidates within CoMs from nonaberrated image if the actual shift found
        if central_aperture_found:
            # print("Central subaperture found:", i)
            i_central_aperture = i  # record the found index for further removing of the central aperture integral values and coms
    # Below: removing belonging to central subaperture values
    coms_shifts = np.delete(coms_shifts, i_central_aperture, axis=0)
    coms_aberrated = np.delete(coms_aberrated, i_central_aperture, axis=0)
    integral_matrix_aberrated = np.delete(integral_matrix_aberrated, i_central_aperture, axis=0)

    return coms_shifts, integral_matrix_aberrated, coms_aberrated


# %% Threaded class for integral matrix calculation
class IntegralMatrixThreaded(Thread):
    """Calculate integral matrix in the threaded manner."""

    messages_queue: Queue
    order: int
    theta0: np.ndarray
    rho0: np.ndarray
    integration_limits: np.ndarray
    radius_subaperture: float
    integral_matrix: np.ndarray

    def __init__(self, messages_queue: Queue, order: int, theta0: np.ndarray, rho0: np.ndarray,
                 integration_limits: np.ndarray, radius_subaperture: float, progress_bar,
                 integral_matrix: np.ndarray):
        self.messages_queue = messages_queue; self.order = order; self.theta0 = theta0
        self.rho0 = rho0; self.integration_limits = integration_limits; self.radius_subaperture = radius_subaperture
        self.progress_bar = progress_bar; self.integral_matrix = integral_matrix
        super().__init__()  # initialization of a new thread

    def run(self):
        """
        Calculate integral matrix calling specified functions above (look also for defaul values).

        Returns
        -------
        None.

        """
        print("Integral matrix calculation started")
        self.integral_matrix = calc_integral_matrix_zernike(self.progress_bar, get_zernike_coefficients_list(self.order),
                                                            self.integration_limits, self.theta0, self.rho0,
                                                            self.messages_queue, self.radius_subaperture)
        print("Integral matrix calculation finished")


# %% Tests
if __name__ == "__main__":
    print(get_zernike_coefficients_list(1))
