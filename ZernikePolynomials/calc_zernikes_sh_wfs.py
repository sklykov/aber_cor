# -*- coding: utf-8 -*-
"""
Specification of calculation functions for reconstruction using non- and aberrated wavefronts recorded by a Shack-Hartmann sensor.

These functions are called further in the 'reconstruction_test.py' file for testing reconstructions.
According to the doctoral thesis by Antonello, J. (2014): https://doi.org/10.4233/uuid:f98b3b8f-bdb8-41bb-8766-d0a15dae0e27

@author: ssklykov
"""

# %% Imports and globals
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from skimage.util import img_as_ubyte
from skimage.feature import peak_local_max
import time
from matplotlib.patches import Rectangle, Circle
from scipy import ndimage
from zernike_pol_calc import radial_polynomial, radial_polynomial_derivative_dr, triangular_function
from zernike_pol_calc import triangular_derivative_dtheta, normalization_factor
from zernike_pol_calc import tabular_radial_polynomial, tabular_radial_derivative_dr  # for speeding up calculations
from numpy.linalg import lstsq
plt.close('all')


# %% Function definitions
def check_img_coordinate(max_coordinate, coordinate):
    """
    Check that specified coordinate lays in the image height or width - maximum coordinate value.

    Parameters
    ----------
    max_coordinate : float or int
        Height or width.
    coordinate : float or int
        Coordinate under revision.

    Returns
    -------
    Corrected coordinate.

    """
    if type(max_coordinate) is int:
        min_coordinate = 0
    else:
        min_coordinate = 0.0
    # Comparing to the return laying in [0, max_coordinate] value
    if coordinate > max_coordinate:
        return max_coordinate
    elif coordinate < min_coordinate:
        return min_coordinate
    else:
        return coordinate


def get_localCoM_matrix(image: np.ndarray, min_dist_peaks: int = 15, threshold_abs: float = 55.0,
                        region_size: int = 16, plot: bool = False) -> np.array:
    """
    Calculate local center of masses in the region around of found peaks (maximums).

    Parameters
    ----------
    image : np.ndarray
        Shack-Hartmann image with recorded wavefront.
    min_dist_peaks : int, optional
        Minimal distance between two local peaks. The default is 15.
    threshold_abs : float, optional
        Absolute minimal intensity value of the local peak. The default is 2.
    region_size : int, optional
        Size of local rectangle, there the center of mass is calculated. The default is 16.
    plot : bool, optional
        If it's true, it allows plotting of found local peaks, boxes and CoMs. The default is True.

    Returns
    -------
    coms : np.array
        Calculated center of masses coordinates.

    """
    detected_centers = peak_local_max(image, min_distance=min_dist_peaks, threshold_abs=threshold_abs)
    (rows, cols) = image.shape
    if plot:
        plt.figure(); plt.imshow(image)  # Plot found local peaks
        plt.plot(detected_centers[:, 1], detected_centers[:, 0], '.', color="red")
        plt.title("Found local peaks")
    half_size = region_size // 2  # Half of rectangle area for calculation of CoM
    size = np.size(detected_centers, 0)  # Number of found local peaks
    coms = np.zeros((size, 2), dtype='float')  # Center of masses coordinates initialization
    if plot:
        plt.figure(); plt.imshow(image)  # Plot found regions for CoM calculations
        plt.title("Local regions for center of mass calculations")
    for i in range(size):
        x_left_upper = check_img_coordinate(cols, detected_centers[i, 1] - half_size)
        y_left_upper = check_img_coordinate(rows, detected_centers[i, 0] - half_size)
        # left_upper_corner = [x_left_upper, y_left_upper]
        if plot:
            # Plot found regions for CoM calculations
            plt.gca().add_patch(Rectangle((x_left_upper, y_left_upper),
                                          2*half_size, 2*half_size, linewidth=1,
                                          edgecolor='yellow', facecolor='none'))
        # CoMs calculation
        subregion = image[y_left_upper:y_left_upper+2*half_size, x_left_upper:x_left_upper+2*half_size]
        (coms[i, 0], coms[i, 1]) = ndimage.center_of_mass(subregion)
        coms[i, 0] += y_left_upper; coms[i, 1] += x_left_upper
    # Plot found CoMs
    if plot:
        plt.figure(); plt.imshow(image)
        plt.plot(coms[:, 1], coms[:, 0], '.', color="green")
        plt.title("Found center of masses")
    return coms

# !!! def get_integr_limits_centralized_subapertures(...) - deleted as not used anymore


def rho_ab(rho0: float, theta: float, theta0: float, aperture_radius: float) -> tuple:
    """
    Calculate of integration limits on sub-aperture (crossed by global radius from the polar coordinates).

    For the drawing - check mentioned dissertation.

    Parameters
    ----------
    rho0 : float
        R coordinate of the sub-aperture center (O).
    theta : float
        Current angle from the global polar coordinate (relative to the frame's center).
    theta0 : float
        Theta coordinate of the sub-aperture center (O).
    aperture_radius : float
        Radius of a sub-aperture.

    Returns
    -------
    tuple
        As (rho_a, rho_b) - the integration limits for further integration of Zernike polynomials.

    """
    cosin = np.cos(theta-theta0)
    cosinsq = cosin*cosin
    rho0sq = rho0*rho0
    apertureRsq = aperture_radius*aperture_radius
    # ???: for some reason, for big polar coordinates relative to radius aperture, the equation below becomes
    # complex (negative under np.sqrt). As the workaround - assume that under sqrt operation the actual value is positive
    if (rho0sq*(cosinsq - 1) + apertureRsq) < 0:
        # print("angle diff:", np.round((theta-theta0)*(180/np.pi), 1),
        #       "np.sqrt from", np.round((rho0sq*(cosinsq - 1) + apertureRsq), 2))  # Debugging
        rho_a = rho0*cosin - np.sqrt(abs(rho0sq*(cosinsq - 1) + apertureRsq))
        rho_b = rho0*cosin + np.sqrt(abs(rho0sq*(cosinsq - 1) + apertureRsq))
    else:
        rho_a = rho0*cosin - np.sqrt(rho0sq*(cosinsq - 1) + apertureRsq)
        rho_b = rho0*cosin + np.sqrt(rho0sq*(cosinsq - 1) + apertureRsq)
    return rho_a, rho_b


def rho_integral_funcX(r: float, theta: float, m: int, n: int) -> tuple:
    """
    Return 2 functions under the integral equation on X axis specified in the thesis.

    Parameters
    ----------
    r : float
        Polar coordinate (rho).
    theta : float
        Polar coordinate (theta).
    m : int
        Azimuthal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.

    Returns
    -------
    tuple
        2 values composing sub-integral parts on axis X.

    """
    derivRmn = radial_polynomial_derivative_dr(m, n, r)
    Rmn = radial_polynomial(m, n, r)
    Angularmn = triangular_function(m, theta)
    derivAngularmn = triangular_derivative_dtheta(m, theta)
    return (derivRmn*r*Angularmn*np.cos(theta)), (Rmn*derivAngularmn*np.sin(theta))


def r_integral_tabular_funcX(r: float, theta: float, m: int, n: int) -> tuple:
    """
    Return 2 functions under the integral equation on X axis specified in the thesis, using tabular functions as called ones.

    Parameters
    ----------
    r : float
        Polar coordinate (rho).
    theta : float
        Polar coordinate (theta).
    m : int
        Azimuthal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.

    Returns
    -------
    tuple
        2 values composing sub-integral parts on axis X.

    """
    derivRmn = tabular_radial_derivative_dr(m, n, r)
    Rmn = tabular_radial_polynomial(m, n, r)
    Angularmn = triangular_function(m, theta)
    derivAngularmn = triangular_derivative_dtheta(m, theta)
    return (derivRmn*r*Angularmn*np.cos(theta)), (Rmn*derivAngularmn*np.sin(theta))


def rho_integral_funcY(r: float, theta: float, m: int, n: int) -> tuple:
    """
    Return 2 functions under the integral equation on Y axis specified in the thesis.

    Parameters
    ----------
    r : float
        Polar coordinate (rho).
    theta : float
        Polar coordinate (theta).
    m : int
        Azimuthal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.

    Returns
    -------
    tuple
        2 values composing sub-integral parts on axis Y.

    """
    derivRmn = radial_polynomial_derivative_dr(m, n, r)
    Rmn = radial_polynomial(m, n, r)
    Angularmn = triangular_function(m, theta)
    derivAngularmn = triangular_derivative_dtheta(m, theta)
    return (derivRmn*r*Angularmn*np.sin(theta)), (Rmn*derivAngularmn*np.cos(theta))


def r_integral_tabular_funcY(r: float, theta: float, m: int, n: int) -> tuple:
    """
    Return 2 functions under the integral equation on Y axis specified in the thesis, using tabular functions as called ones.

    Parameters
    ----------
    r : float
        Polar coordinate (rho).
    theta : float
        Polar coordinate (theta).
    m : int
        Azimuthal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.

    Returns
    -------
    tuple
        2 values composing sub-integral parts on axis Y.

    """
    derivRmn = tabular_radial_derivative_dr(m, n, r)
    Rmn = tabular_radial_polynomial(m, n, r)
    Angularmn = triangular_function(m, theta)
    derivAngularmn = triangular_derivative_dtheta(m, theta)
    return (derivRmn*r*Angularmn*np.sin(theta)), (Rmn*derivAngularmn*np.cos(theta))


# !!! def calc_integrals_on_apertures - was similar to the function below, but it calculates the integrals on the
# radial polar coordinates without calibration it to the distance from the center to the most distant sub-aperture,
# that makes below integration on the unit circular aperture on the radial coordinate rho, that seems helps to
# reproduce associated tests of wavefront reconstruction


def calc_integrals_on_apertures_unit_circle(integration_limits: np.ndarray, theta0: np.ndarray, rho0: np.ndarray, m: int, n: int,
                                            n_polynomial: int = 1, aperture_radius: float = 15.0, n_steps: int = 50,
                                            swapXY: bool = True, use_tabular_functions: bool = False) -> np.ndarray:
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
    n_polynomial: int
        Number of polynomial for which the integration calculated. The default is 1.
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    n_steps : int, optional
        Number of integration steps for both integrals. The default is 50.
    swapXY: bool, optional
        For swapping the direction of Y axis pointing up on the picture, instead down as for pixel coordinates (y, x),
        for conforming with the thesis calculations. The default is True.
    use_tabular_functions: bool, optional
        If True, then the tabular functions up to 7th order will be used for integrals calculation. The default is False.

    Returns
    -------
    integral_values : ndarray with sizes (Number of sub-apertures, 2)
        Resulting integration values for each sub-aperture, they are listed for Y and X axes (relatively to image coordinate system).

    """
    integral_values = np.zeros((len(integration_limits), 2), dtype='float')  # Doubled values - for X,Y axes
    calibration = 1.0  # TODO: Calibration taking into account the wavelength, focal length should be implemented later
    # Introduction of Zernike's polynomials normalization coefficients => possible tune of each polynomial contribution
    if use_tabular_functions:  # For faster testing and acquiring the integral matrices
        calibration = normalization_factor(m, n)  # use it for recalculate integral values for testing
    rho_unit_calibration = np.max(rho0) + aperture_radius  # For making integration on rho on unit circle
    # For each sub-aperture below integration on (r, theta) of the Zernike polynomials
    for i_subaperture in range(len(integration_limits)):
        print(f"Integration for #{n_polynomial} polynomial started on {i_subaperture} subaperture out of {len(integration_limits)}")
        # Integration limits and steps on theta
        (theta_a, theta_b) = integration_limits[i_subaperture]  # Calculated previously integration steps on theta (radians)
        delta_theta = (theta_b - theta_a)/n_steps  # Step for integration on theta
        theta = theta_a  # initial value of theta
        # Getting limits for integration on rho
        (rho_a, rho_b) = rho_ab(rho0[i_subaperture], theta, theta0[i_subaperture], aperture_radius)
        # !!!: Another approach to integration on rho: the entire picture with sub-apertures will be accounted as unit-radius circle
        # For that, all rho values should be normalized to the maximum rho0 coordinate + radius_subaperture
        rho_a /= rho_unit_calibration; rho_b /= rho_unit_calibration
        # Integration over theta (trapezoidal rule)
        integral_sumX = 0.0; integral_sumY = 0.0
        for j_theta in range(n_steps+1):
            # Integration on rho for X and Y axis (trapezoidal formula)
            rho = rho_a  # Lower integration boundary
            delta_rho = (rho_b - rho_a)/n_steps
            integral_sum_rhoX1 = 0.0; integral_sum_rhoX2 = 0.0; integral_sum_rhoY1 = 0.0; integral_sum_rhoY2 = 0.0
            for j_rho in range(n_steps+1):
                if (j_rho == 0) or (j_rho == n_steps):
                    if use_tabular_functions:
                        (X1, X2) = r_integral_tabular_funcX(rho, theta, m, n)
                    else:
                        (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += 0.5*X1; integral_sum_rhoX2 += 0.5*X2  # on X axis
                    if use_tabular_functions:
                        (Y1, Y2) = r_integral_tabular_funcY(rho, theta, m, n)
                    else:
                        (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += 0.5*Y1; integral_sum_rhoY2 += 0.5*Y2  # on Y axis
                else:
                    if use_tabular_functions:
                        (X1, X2) = r_integral_tabular_funcX(rho, theta, m, n)
                    else:
                        (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += X1; integral_sum_rhoX2 += X2
                    if use_tabular_functions:
                        (Y1, Y2) = r_integral_tabular_funcY(rho, theta, m, n)
                    else:
                        (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += Y1; integral_sum_rhoY2 += Y2
                rho += delta_rho
            integral_sum_rhoX = (integral_sum_rhoX1 - integral_sum_rhoX2)  # From thesis equations
            integral_sum_rhoY = (integral_sum_rhoY1 + integral_sum_rhoY2)
            # End of integration on rho (i.e. r from polar coordinates)
            integral_sum_rhoX *= delta_rho; integral_sum_rhoY *= delta_rho
            if (j_theta == 0) and (j_theta == n_steps):
                integral_sumX += 0.5*integral_sum_rhoX; integral_sumY += 0.5*integral_sum_rhoY
            else:
                integral_sumX += integral_sum_rhoX; integral_sumY += integral_sum_rhoY
            theta += delta_theta
        integral_sumX *= delta_theta; integral_sumY *= delta_theta  # End of integration on theta
        # integral values should be also calibrated to the areas of sub-apertures - but in which form?
        # integral_sumX /= (np.pi*np.power((aperture_radius/rho_unit_calibration), 2))
        # integral_sumY /= (np.pi*np.power((aperture_radius/rho_unit_calibration), 2))
        # actually, the integral values should be calibrated to each sub-aperture area - depending on the integration limits
        integral_sumX /= 0.5*(theta_b - theta_a)*((rho_b*rho_b)-(rho_a*rho_a))  # 0.5 - due to integration from (rdr)dtheta
        integral_sumY /= 0.5*(theta_b - theta_a)*((rho_b*rho_b)-(rho_a*rho_a))
        # The final integral values should be also calibrated to focal and wavelengths, but it's not yet implemented
        if swapXY:  # Choosing the relation between X and Y axis calculation, meaning - see the documentation
            integral_values[i_subaperture, 1] = calibration*integral_sumX  # Not yet implemented calibration, not necessary now
            integral_values[i_subaperture, 0] = calibration*integral_sumY
        else:
            integral_values[i_subaperture, 0] = calibration*integral_sumX
            integral_values[i_subaperture, 1] = calibration*integral_sumY
        integral_values = np.round(integral_values, 8)
    return integral_values


def calc_integral_matrix_zernike(zernike_polynomials_list: list, integration_limits: np.ndarray, theta0: np.ndarray,
                                 rho0: np.ndarray, aperture_radius: float = 15.0, n_steps: int = 50,
                                 swapXY: bool = True, use_tabular_functions: bool = False) -> np.ndarray:
    """
    Wrap for calculation of integral values on sub-apertures performing on several Zernike polynomials.

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
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    n_steps : int, optional
        Number of integration steps for both integrals. The default is 50
    swapXY: bool, optional
        For swapping the direction of Y axis pointing up on the picture, instead down as for pixel coordinates (y, x),
        for conforming with the thesis calculations. The default is True.
    use_tabular_functions: bool, optional
        If True, then the tabular functions up to 7th order will be used for integrals calculation. The default is False.

    Returns
    -------
    integral_matrix : ndarray with sizes (Number of sub-apertures, Number of Zernikes)
        Resulting integration values for each sub-aperture and for both X and Y axes and specified Zernike values.

    """
    n_rows = np.size(rho0, 0)
    n_cols = 2*len(zernike_polynomials_list)  # Because calculation needed for both X and Y axes
    integral_matrix = np.zeros((n_rows, n_cols), dtype='float')
    # Shortening the time of calculation because of symmetrical integrals over sub-apertures for (-1, 1) and (1, 1),
    # (-2, 2) and (2, 2), (-3, 3) and (-4, 4) - applying below reassignment
    symmetrical_substitution = False; i_symmetry = -1
    for i in range(len(zernike_polynomials_list)):
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
                print("Shortening of calculation used")
            else:
                # Normal integrals calculation
                integral_values = calc_integrals_on_apertures_unit_circle(integration_limits, theta0, rho0, m, n, n_polynomial=i+1,
                                                                          aperture_radius=aperture_radius, n_steps=n_steps,
                                                                          swapXY=swapXY, use_tabular_functions=use_tabular_functions)
                integral_matrix[:, 2*i] = integral_values[:, 0]
                integral_matrix[:, 2*i+1] = integral_values[:, 1]
        else:
            # Normal integrals calculation
            integral_values = calc_integrals_on_apertures_unit_circle(integration_limits, theta0, rho0, m, n, n_polynomial=i+1,
                                                                      aperture_radius=aperture_radius, n_steps=n_steps, swapXY=swapXY,
                                                                      use_tabular_functions=use_tabular_functions)
            integral_matrix[:, 2*i] = integral_values[:, 0]
            integral_matrix[:, 2*i+1] = integral_values[:, 1]
        print(f"Calculated {i+1} polynomial out of {len(zernike_polynomials_list)}")
    return integral_matrix


def get_polynomials_coefficients(integral_matrix: np.ndarray, coms_shifts: np.ndarray) -> np.ndarray:
    """
    Get the solution to the equation S = E*Alpha.

    There S - CoMs shifts, E - integral matrix, Alpha - coefficients for decomposition of the wavefront on sum of Zernike polynomials.

    Parameters
    ----------
    integral_matrix : np.ndarray
        Integrals of Zernike polynomials on sub-apertures.
    coms_shifts : np.ndarray
        Shifts of center of masses coordinates for each focal spot of each sub-aperture.

    Returns
    -------
    alpha_coefficients : np.ndarray
        Coefficients for decomposition.

    """
    # alpha_coefficientsXY  = lstsq(integral_matrix, coms_shifts, rcond=1E-6)[0]  # Provides with solutions more than 1E-6, that is "0"
    # The matrices should be changed in sizes for calculation the alpha coefficients for each Zernike polynomial.
    # This made according to suggestion in the paper Dai G.-M., 1994
    n_subapertures = np.size(coms_shifts, 0); n_polynomials = np.size(integral_matrix, 1) // 2
    integral_matrix_swapped = np.zeros((2*n_subapertures, n_polynomials), dtype='float')
    coms_shifts_swapped = np.zeros(2*n_subapertures, dtype='float')
    for i in range(n_subapertures):
        coms_shifts_swapped[2*i] = coms_shifts[i, 0]  # X values of all polynomials
        coms_shifts_swapped[2*i+1] = coms_shifts[i, 1]  # Y values of all polynomials
        for j in range(n_polynomials):
            integral_matrix_swapped[2*i, j] = integral_matrix[i, 2*j]  # X values of all polynomials
            integral_matrix_swapped[2*i+1, j] = integral_matrix[i, 2*j+1]   # Y values of all polynomials
    alpha_coefficients = lstsq(integral_matrix_swapped, coms_shifts_swapped, rcond=1E-6)[0]  # Provides with solutions more than 1E-6

    return alpha_coefficients


# !!! def get_overall_coms_shifts() - deleted, it was for calculation of CoMs shifts for each pair for further calibration


def get_integral_limits_nonaberrated_centers(pics_folder: str = "pics", background_pic_name: str = "picBackground.png",
                                             nonaberrated_pic_name: str = "nonAberrationPic.png", min_dist_peaks: int = 18,
                                             threshold_abs: float = 60.0, region_size: int = 20, subtract_background: bool = False,
                                             aperture_radius: float = 15.0, plot_results: bool = False) -> tuple:
    """
    Calculate the center of masses of localized focal spots and also the integration limits for further modal wavefront reconstruction.

    Focal spots are recorded within the non-aberrated image (flat wavefront) from Shack-Hartmann sensor.
    Integrations limits are calculated according to the mentioned above in this file thesis.

    Parameters
    ----------
    pics_folder : str, optional
        Path to the folder with pictures for processing. The default is "pics".
    background_pic_name : str, optional
        Picture name containing background. The default is "picBackground.png".
    nonaberrated_pic_name : str, optional
        Picture name containing non-aberrated (flat) wavefront. The default is "nonAberrationPic.png".
    min_dist_peaks : int, optional
        Minimal distance between two local peaks. The default is 18.
    threshold_abs : float, optional
        Absolute minimal intensity value of the local peak, seeding value for the iterative searching of them. The default is 60.0.
    region_size : int, optional
        Size of local rectangle, there the center of mass is calculated. The default is 20.
    subtract_background : bool, optional
        If False, from the opened images the stored background picture won't be subtracted. The default is False.
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    plot_results : bool, optional
        If True, the images with found local peaks and sub-apertures will be plotted. The default is False.

    Raises
    ------
    Exception
        If the specified path to the folder with pictures doesn't exist or not a directory and if images there aren't actual files.

    Returns
    -------
    tuple
        (Centers of masses for non-aberrated image without the central sub-aperture, non-aberrated image,
         Theta (polar) coordinates of sub-apertures, Rho (polar) coordinates of sub-apertures, Integration limits for sub-apertures).

    """
    # Open images on some specified folder
    if pics_folder == "pics":
        # Default folder in the repository
        absolute_path = os.path.join(os.getcwd(), pics_folder)
        backgroundPath = os.path.join(absolute_path, background_pic_name)
        nonaberratedPath = os.path.join(absolute_path, nonaberrated_pic_name)
    else:
        if not(os.path.exists(pics_folder)):
            raise Exception("Specified Path doesn't exist")
        else:
            if not(os.path.isdir(pics_folder)):
                raise Exception("Specified object isn't a directory")
            else:
                # Looking for images on the specified path
                if subtract_background:
                    backgroundPath = os.path.join(pics_folder, background_pic_name)
                    if not(os.path.isfile(backgroundPath)):
                        raise Exception("The background image doesn't exist or not a file, check root path and its name")
                nonaberratedPath = os.path.join(pics_folder, nonaberrated_pic_name)
                if not(os.path.isfile(nonaberratedPath)):
                    raise Exception("Some of specified images are not actual files, check path and their names")
    # Open the stored files and extracting the recorded background from the wavefronts
    nonaberrated = (io.imread(nonaberratedPath, as_gray=True))
    if subtract_background:
        # Subtracting from the recorded pictures (non- and aberrated) the recorded background
        background = (io.imread(backgroundPath, as_gray=True))  # reads background image only if it's requested by input parameters
        diff_nonaberrated = (nonaberrated - background); diff_nonaberrated = img_as_ubyte(diff_nonaberrated)
        threshold_abs = np.round(np.mean(diff_nonaberrated), 0) + 1
    else:
        # Subtracting background as the minimal value on the picture and manually transferring the pictures into the U8 (ubyte) type
        # Additionally, stretch a bit the contrast
        diff_nonaberrated = nonaberrated - abs(np.min(nonaberrated)); diff_nonaberrated *= (255/np.max(diff_nonaberrated))
        diff_nonaberrated = np.uint8(diff_nonaberrated)
    # If the recorded background helps to enhance the contrast, then preserve the operation of background substraction
    if plot_results and subtract_background:
        plt.figure(); plt.imshow(background); plt.tight_layout(); plt.title("Background")
        plt.figure(); plt.imshow(nonaberrated); plt.tight_layout(); plt.title("Non-aberrated")
        plt.figure(); plt.imshow(diff_nonaberrated); plt.tight_layout(); plt.title("Non-aberrated - Background")
    elif plot_results:
        plt.figure(); plt.imshow(nonaberrated); plt.tight_layout(); plt.title("Non-aberrated")
    # Plotting of results on an image for debugging (for coms_nonaberrated)
    # CoMs = center of masses, calculated in the areas around found local peaks
    coms_nonaberrated = get_localCoM_matrix(diff_nonaberrated, min_dist_peaks=min_dist_peaks,
                                            threshold_abs=threshold_abs,
                                            region_size=region_size, plot=plot_results)
    # Searching for the central sub-aperture that should be close to the center of the image
    (rows, cols) = diff_nonaberrated.shape; x_img_center = cols//2; y_img_center = rows//2
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
    if plot_results:
        plt.figure(); plt.imshow(diff_nonaberrated)  # for plotting the found central sub-aperture
        plt.plot(x_central_subaperture, y_central_subaperture, '+', color="red")
        plt.plot(x_img_center, y_img_center, '+', color="blue")
        plt.title("Estimated circular apertures of micro-lenses array")
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
        if plot_results and not(i == i_center_subaperture):
            plt.gca().add_patch(Circle((coms_nonaberrated[i, 1], coms_nonaberrated[i, 0]), aperture_radius,
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
    # Plotting the calculated coordinates in polar projection for checking that all their centers defined correctly
    if plot_results:
        plt.figure()
        plt.axes(projection='polar')
        plt.plot(np.radians(theta0), rho0, '.')
        plt.tight_layout()
    # Before the calculation for angles made in grads, below the transfer to radians for passing it further to trigonometric functions
    theta0 = np.radians(theta0)
    integration_limits = np.radians(integration_limits)
    return subapertures_wt_central, diff_nonaberrated, theta0, rho0, integration_limits


def get_coms_shifts(coms_nonaberrated: np.ndarray, integral_matrix: np.ndarray, pics_folder: str = "pics",
                    background_pic_name: str = "picBackground.png", aberrated_pic_name: str = "aberrationPic.png",
                    min_dist_peaks: int = 18, threshold_abs: float = 60.0, region_size: int = 20,
                    subtract_background: bool = False, plot_results: bool = False) -> tuple:
    """
    Calculate shifts between center of masses around local focal spots for aberrated and non-aberrated images.

    Parameters
    ----------
    coms_nonaberrated : np.ndarray
        Calculated center of masses around local focal spots on the non-aberrated image.
    integral_matrix : np.ndarray
        Calculated the integral matrix for Zernike polynomials.
    pics_folder : str, optional
        Path to the folder with pictures for processing. The default is "pics".
    background_pic_name : str, optional
        Picture name containing background. The default is "picBackground.png"
    aberrated_pic_name : str, optional
        Picture name containing aberrated wavefront. The default is "aberrationPic.png".
    min_dist_peaks : int, optional
        Minimal distance between two local peaks. The default is 18.
    threshold_abs : float, optional
        Absolute minimal intensity value of the local peak, seeding value for the iterative searching of them. The default is 60.0.
    region_size : int, optional
        Size of local rectangle, there the center of mass is calculated. The default is 20.
    subtract_background : bool, optional
        If False, from the opened images the stored background picture won't be subtracted. The default is False.
    plot_results : bool, optional
        If True, the images with found local peaks and sub-apertures will be plotted. The default is False.

    Raises
    ------
    Exception
        If the specified path to the folder with pictures doesn't exist or not a directory and if images there aren't actual files.

    Returns
    -------
    tuple
        (shifts of CoMs, integral matrix according to the detected CoMs on the aberrated image).

    """
    # Open images on some specified folder
    if pics_folder == "pics":
        # Default folder in the repository
        absolute_path = os.path.join(os.getcwd(), pics_folder)
        backgroundPath = os.path.join(absolute_path, background_pic_name)
        aberratedPath = os.path.join(absolute_path, aberrated_pic_name)
    else:
        if not(os.path.exists(pics_folder)):
            raise Exception("Specified Path doesn't exist")
        else:
            if not(os.path.isdir(pics_folder)):
                raise Exception("Specified object isn't a directory")
            else:
                # Looking for images on the specified path
                if subtract_background:
                    backgroundPath = os.path.join(pics_folder, background_pic_name)
                    if not(os.path.isfile(backgroundPath)):
                        raise Exception("The background image doesn't exist or not a file, check root path and its name")
                aberratedPath = os.path.join(pics_folder, aberrated_pic_name)
                if not(os.path.isfile(aberratedPath)):
                    raise Exception("The aberrated image doesn't exist or not a file, check root path and its name")
    # Open the stored files and extracting the recorded background from the wavefronts
    aberrated = (io.imread(aberratedPath, as_gray=True))
    if subtract_background:
        # Subtracting from the recorded pictures (non- and aberrated) the recorded background
        background = (io.imread(backgroundPath, as_gray=True))
        diff_aberrated = (aberrated - background); diff_aberrated = img_as_ubyte(diff_aberrated)
        threshold_abs = np.round(np.mean(diff_aberrated), 0) + 1
    else:
        # Subtracting background as the minimal value on the picture and manually transferring the pictures into the U8 (ubyte) type
        # Additionally, stretch a bit the contrast
        diff_aberrated = aberrated - abs(np.min(aberrated)); diff_aberrated *= (255/np.max(diff_aberrated))
        diff_aberrated = np.uint8(diff_aberrated)
        if plot_results and not subtract_background:
            plt.figure(); plt.imshow(diff_aberrated); plt.tight_layout(); plt.title("Aberrated")
    # If the recorded background helps to enhance the contrast, then preserve the operation of background subtraction
    if plot_results and subtract_background:
        plt.figure(); plt.imshow(background); plt.tight_layout(); plt.title("Background")
        plt.figure(); plt.imshow(diff_aberrated); plt.tight_layout(); plt.title("Aberrated - Background")
    # CoMs = center of masses, calculated in the areas around found local peaks, below - for specified aberrated picture
    coms_aberrated = get_localCoM_matrix(diff_aberrated, min_dist_peaks=min_dist_peaks, threshold_abs=threshold_abs,
                                         region_size=region_size, plot=plot_results)
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
    integral_matrix_aberrated = np.delete(integral_matrix_aberrated, i_central_aperture, axis=0)

    return coms_shifts, integral_matrix_aberrated


# %% Tests of functionality
# Calculation of shifts between non- and aberrated images, integrals of Zernike polynomials on sub-apertures
if __name__ == '__main__':
    t1 = time.time(); zernikes_set = [(-3, 3), (3, 3)]
    show_plots = False; min_dist_peaks = 18; threshold = 60.0; region_size = 20; aperture_radius = 14.0  # Parameters
    # Below - calculating CoMs of focal spots in the non-aberrated picture, get integral limits and the picture
    (coms_nonaberrated, pic_integral_limits,
     theta0, rho0, integration_limits) = get_integral_limits_nonaberrated_centers(plot_results=show_plots,
                                                                                  threshold_abs=threshold,
                                                                                  region_size=region_size,
                                                                                  aperture_radius=aperture_radius,
                                                                                  min_dist_peaks=min_dist_peaks)
    # Below - integration of polynomials in each sub-aperture area using calculated before limits
    integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0,
                                                   aperture_radius=aperture_radius, n_steps=20)
    # Below - calculation of CoM shifts between aberrated and non-aberrated pictures
    (coms_shifts, integral_matrix_aberrated) = get_coms_shifts(coms_nonaberrated, integral_matrix, plot_results=show_plots,
                                                               threshold_abs=threshold, region_size=region_size,
                                                               min_dist_peaks=min_dist_peaks)
    # Below - checking the timing of integration and get the alpha coefficients
    t2 = time.time(); print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
    alpha_coefficients = get_polynomials_coefficients(integral_matrix_aberrated, coms_shifts)
    # Below - testing performance of the integration using tabular functions instead of recursive formulas
    t3 = time.time(); use_tabular_functions = True
    integral_matrix2 = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0,
                                                    aperture_radius=aperture_radius, n_steps=20,
                                                    use_tabular_functions=use_tabular_functions)
    diff_integral_matrices = np.absolute(integral_matrix2 - integral_matrix)
    t4 = time.time(); print(f"Integration using tabular polynomials ({zernikes_set}) takes:", np.round(t4-t3, 3), "s")
