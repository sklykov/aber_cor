# -*- coding: utf-8 -*-
"""
Calculation of Zernike polynomials, plot their surface maps on the unit apertures.

@author: ssklykov
"""
# %% Imports and globals
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close('all')  # closing all opened and pending figures


# %% Functions definitions
def normalization_factor(m: int, n: int) -> float:
    """
    Calculate according to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m : int
        Azimuthal order.
    n : int
        Radial order.

    Returns
    -------
    float
        Calculated value.

    """
    n = abs(n)  # Guarantees that input parameter is always > 0
    if m == 0:
        return np.sqrt(n+1)
    else:
        return np.sqrt(2*(n+1))


def radial_polynomial(m: int, n: int, r: float) -> float:
    """
    According to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m: int
        Azimuthal order.
    n: int
        Radial order.
    r: float
        Radius from the polar coordinate.

    Returns
    -------
    float:
        Recursively calculated radial polynomial.
    """
    if (n == 0) and (m == 0):
        return 1.0
    elif m > n:
        return 0.0
    else:
        # Recursion formula that should be more effective than direct calculation (only for high radial orders)
        return r*(radial_polynomial(abs(m-1), n-1, r) + radial_polynomial(m+1, n-1, r)) - radial_polynomial(m, n-2, r)


def triangular_function(m: int, theta: float) -> float:
    """
    Calculate triangular function according to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m : int
        Azimutal order.
    theta : float
        Angle from polar coordinates associated with the unit circle .

    Returns
    -------
    float
        Calculated value.

    """
    if m > 0:
        return np.cos(m*theta)
    elif m < 0:
        return -np.sin(m*theta)
    else:
        return 1.0


def zernike_polynomial(m: int, n: int, r: float, theta: float) -> float:
    """
    Calculate according to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m: int
        Azimuthal order.
    n: int
        Radial order.
    r: float
        Radius from a polar coordinate system.
    theta: float
        Angle from a polar coordinate system.

    Returns
    -------
    float:
        Recursively calculated Zernike polynomial.
    """
    return normalization_factor(m, n)*radial_polynomial(m, n, r)*triangular_function(m, theta)


def zernike_polynomials_sum_tuned(orders: list, alpha_coefficients: list, step_r: float = 0.01,
                                  step_theta: float = 1.0) -> tuple:
    """
    Calculate sum of Zernike's polynomials using specified amplitudes (alpha coefficients).

    NOTE: this sum calculation using the tabular values up to 7th order and vectorization of triangular function calculation.
    It's speed up, but not universal. Rewrite in the need of customizing to the arbitrary order calculation.

    Parameters
    ----------
    orders : list
        List of Zernike polynomials orders recorded in tuples (m, n) inside the list like [(m, n), ...].
    alpha_coefficients: list
        List with tunning coefficients (amplitudes) of each polynomial for their sum calculation.
    step_r : float, optional
        Step for calculation of radius for a summing map (colormap). The default is 0.01.
    step_theta : float, optional
        Step (in grades) for calculation of angle for a summing map (colormap). The default is 1.0.

    Raises
    ------
    TypeError
        If the input list doesn't contain tuples formatted as (m,n) AND if length of orders != alpha coefficients.

    Returns
    -------
    tuple
        In the form (R, Theta, S) there R - radial coordinates vector, Theta - angular coordinates vector,
        S - polynomials sum.

    """
    R = np.arange(0.0, 1.0+step_r, step_r)  # steps on r (polar coordinate)
    Theta = np.arange(0.0, (2.0*np.pi + np.radians(step_theta)), np.radians(step_theta))  # steps on theta (polar coordinates)
    (i_size, j_size) = (np.size(R, 0), np.size(Theta, 0))
    # Calculation of sum of Zernike's polynomials (on all points sequentially => slow, deleted)
    # Speed up the calculation by using vectorization of calculation on angles theta and using tabulated function values
    Z = np.zeros((i_size, j_size), dtype='float')  # for storing Zernike's polynomial values
    S = np.zeros((i_size, j_size), dtype='float')  # initial sum
    for k in range(len(orders)):
        if abs(alpha_coefficients[k]) > 1.0E-6:  # the alpha or amplitude coefficient is actually non-zero
            tuple_orders = orders[k]
            if (not(isinstance(tuple_orders, tuple)) and not(len(orders) == len(alpha_coefficients))
                    and len(tuple_orders != 2)):  # checking for conformity with specification of orders
                raise TypeError
            else:
                (m, n) = tuple_orders
                norm = normalization_factor(m, n)
                for i in range(i_size):
                    Z[i, :] = (alpha_coefficients[k]*norm*tabular_radial_polynomial(m, n, R[i])
                               * (vectorized_triangular_function(m, Theta)[:]))  # get the pol. value
                S += Z  # adding to the final sum of all contributed Zernike's polynomials
        else:
            continue  # goes further on the loop for the next polynomial with non-zero amplitude
    return R, Theta, S    # tuple can be defined by coma separation


def plot_zps_polar(orders: list, step_r: float = 0.005, step_theta: float = 0.5, title: str = "Sum of Zernike polynomials",
                   alpha_coefficients: list = [], show_amplitudes: bool = False):
    """
    Plot Zernike's polynomials sum ("zps") in polar projection for the unit radius circle.

    Parameters
    ----------
    orders : list
        List of Zernike polynomials orders recorded in tuples (m, n) inside the list like [(m, n), ...].
    step_r : float, optional
        Step for calculation of radius for a summing map (colormap). The default is 0.01.
    step_theta : float, optional
        Step (in grades) for calculation of angle for a summing map (colormap). The default is 1.0.
    title : str, optional
        Title for placing on the plot, e.g. for specific single polynomial. The default is "Sum of Zernike polynomials".
    alpha_coefficients: list, optional
        List with tunning coefficients (amplitudes) of each polynomial for their sum calculation.
    show_amplitudes : bool, optional
        Shows the colour-bar on the plot with amplitudes. The default is False.

    Returns
    -------
    None. The plot is shown in the separate window.

    """
    if len(alpha_coefficients) == 0:
        alpha_coefficients = [1.0]*len(orders)
    R, Theta, Z = zernike_polynomials_sum_tuned(orders, alpha_coefficients, step_r=step_r, step_theta=step_theta)
    # Plotting and formatting - Polar projection + plotting the colormap
    plt.figure(); axes = plt.axes(projection='polar'); axes.set_theta_direction(-1)  # set the clockwise counting of theta
    # plt.grid(False); plt.pcolormesh(Theta, R, Z, cmap=cm.coolwarm)  # the windows with plots are not responsive!
    plt.contourf(Theta, R, Z, 100, cmap=cm.coolwarm)  # produces more responsive plots!
    plt.title(title); plt.axis('off')
    if show_amplitudes:
        plt.colorbar()  # shows the colour bar with shown on image amplitudes
    plt.tight_layout()


def get_plot_zps_polar(figure, orders: list, alpha_coefficients: list, step_r: float = 0.01,
                       step_theta: float = 1.0, show_amplitudes: bool = False):
    """
    Plot Zernike's polynomials sum ("zps") in polar projection for the unit radius circle on the provided matplotlib.figure instance.

    Parameters
    ----------
    figure : matplotlib.figure.Figure()
        The Figure() class instance from matplotlib.figure module for plotting the Zernkike's polynomials sum on that.
    orders : list
        List of Zernike polynomials orders recorded in tuples (m, n) inside the list like [(m, n), ...].
    alpha_coefficients: list
        List with tuning coefficients (amplitudes) of each polynomial for their sum calculation.
    step_r : float, optional
        Step for calculation of radius for a summing map (colormap). The default is 0.01.
    step_theta : float, optional
        Step (in grades) for calculation of angle for a summing map (colormap). The default is 1.0.
    show_amplitudes : bool, optional
        Shows the colourbar on the plot with amplitudes. The default is False.

    Raises
    ------
    TypeError
        If the input list doesn't contain tuples formatted as (m,n) AND if length of orders != alpha coefficients.

    Returns
    -------
    figure : matplotlib.figure.Figure()
        The Figure() instance with plotted colormesh graph.

    """
    R, Theta, S = zernike_polynomials_sum_tuned(orders, alpha_coefficients, step_r=step_r, step_theta=step_theta)
    # Plotting and formatting - Polar projection + plotting the colormap as colour mesh
    # Below - clearing of picture axes, for preventing adding many axes to the single figure
    figure.clear(); axes = figure.add_subplot(projection='polar')  # axes - the handle for drawing functions
    # Below - manual deletion and reinitializing of axes
    # if figure.get_axes() == []:
    #     axes = figure.add_subplot(projection='polar')  # axes - the handle for drawing functions
    # else:
    #     figure.delaxes(figure.get_axes()[0])
    #     axes = figure.add_subplot(projection='polar')  # axes - the handle for drawing functions
    # !!!: using contourf function is too slow for providing refreshing upon calling by the button
    axes.grid(False)  # demanded by pcolormesh function, if not called - deprecation warning
    im = axes.pcolormesh(Theta, R, S, cmap=cm.coolwarm)  # plot the colour map by using the Z map according to Theta, R coord-s
    axes.axis('off')  # off polar coordinate axes
    axes.set_theta_direction(-1)  # the counterclockwise counting of angle switched to clockwise!
    if show_amplitudes:
        figure.colorbar(im, ax=axes)  # shows the colour bar with shown on image amplitudes
    figure.tight_layout()
    return figure


def vectorized_triangular_function(m: int, Theta: np.ndarray) -> np.ndarray:
    """
    Calculate triangular Zernike function on the input array of angles theta (vectorization).

    Parameters
    ----------
    m : int
        Angular order of Zernike's polynomial.
    Theta : np.ndarray
        Angle in radians.

    Returns
    -------
    np.ndarray
        Calculated angular function, the size the same as input size of the angles array.

    """
    if m >= 0:
        return np.cos(m*Theta)
    else:
        return np.sin(m*Theta)


def tabular_radial_polynomial(m: int, n: int, r: float) -> float:
    """
    Return calculated by the explicit equation (from Lakshminarayanan V., Fleck A. (2011)) radial Zernike's polynomial value.

    Return values only up to 7th order! (-3, 7) and (3, 7) in the sited above paper have the typos (2 instead of 21).
    For the orders higher than 7, return 0.0 value.

    Parameters
    ----------
    m : int
        Angular order of Zernike's polynomial.
    n : int
        Radial order of Zernike's polynomial.
    r : float
        Polar radial coordinate r (rho).

    Returns
    -------
    float
        Radial Zernike's polynomial value.

    """
    if ((m == -1) and (n == 1)) or ((m == 1) and (n == 1)):
        return r
    elif ((m == -2) and (n == 2)) or ((m == 2) and (n == 2)):
        return r*r  # r^2
    elif (m == 0) and (n == 2):
        return 2.0*r*r - 1.0  # 2r^2 - 1
    elif ((m == -3) and (n == 3)) or ((m == 3) and (n == 3)):
        return r*r*r  # r^3
    elif ((m == -1) and (n == 3)) or ((m == 1) and (n == 3)):
        return 3.0*r*r*r - 2.0*r  # 3r^3 - 2r
    elif ((m == -4) and (n == 4)) or ((m == 4) and (n == 4)):
        return r*r*r*r  # r^4
    elif ((m == -2) and (n == 4)) or ((m == 2) and (n == 4)):
        return 4.0*r*r*r*r - 3.0*r*r  # 4r^4 - 3r^2
    elif (m == 0) and (n == 4):
        return 6.0*r*r*r*r - 6.0*r*r + 1.0  # 6r^4 - 6r^2 + 1
    elif ((m == -5) and (n == 5)) or ((m == 5) and (n == 5)):
        return r*r*r*r*r  # r^5
    elif ((m == -3) and (n == 5)) or ((m == 3) and (n == 5)):
        return 5.0*r*r*r*r*r - 4.0*r*r*r  # 5r^5 - 4r^3
    elif ((m == -1) and (n == 5)) or ((m == 1) and (n == 5)):
        return 10.0*r*r*r*r*r - 12.0*r*r*r + 3.0*r  # 10r^5 - 12r^3 + 3r
    elif ((m == -6) and (n == 6)) or ((m == 6) and (n == 6)):
        return r*r*r*r*r*r  # r^6
    elif ((m == -4) and (n == 6)) or ((m == 4) and (n == 6)):
        return 6.0*r*r*r*r*r*r - 5.0*r*r*r*r  # 6r^6 - 5r^4
    elif ((m == -2) and (n == 6)) or ((m == 2) and (n == 6)):
        return 15.0*r*r*r*r*r*r - 20.0*r*r*r*r + 6.0*r*r  # 15r^6 - 20r^4 + 6r^2
    elif (m == 0) and (n == 6):
        return 20.0*r*r*r*r*r*r - 30.0*r*r*r*r + 12.0*r*r - 1.0  # 20r^6 - 30r^4 + 12r^2 - 1
    elif ((m == -7) and (n == 7)) or ((m == 7) and (n == 7)):
        return r*r*r*r*r*r*r  # r^7
    elif ((m == -5) and (n == 7)) or ((m == 5) and (n == 7)):
        return 7.0*r*r*r*r*r*r*r - 6.0*r*r*r*r*r  # 7r^7 - 6r^5
    elif ((m == -3) and (n == 7)) or ((m == 3) and (n == 7)):
        return 21.0*r*r*r*r*r*r*r - 30.0*r*r*r*r*r + 10.0*r*r*r  # 21r^7 - 30r^5 + 10r^3
    elif ((m == -1) and (n == 7)) or ((m == 1) and (n == 7)):
        return 35.0*r*r*r*r*r*r*r - 60.0*r*r*r*r*r + 30.0*r*r*r - 4.0*r  # 35r^7 - 60r^5 + 30r^3 - 4r
    else:
        return 0.0  # default return value for the orders more than 7.


def tabular_radial_derivative_dr(m: int, n: int, r: float) -> float:
    """
    Return calculated by the explicit equation derivative on r of radial polynomial specified above.

    Return values only up to 7th order! For the orders higher than 7, return 0.0 value.

    Parameters
    ----------
    m : int
        Angular order of Zernike's polynomial.
    n : int
        Radial order of Zernike's polynomial.
    r : float
        Polar radial coordinate r (rho).

    Returns
    -------
    float
        Derivative of radial Zernike's polynomial on r value.
    """
    if ((m == -1) and (n == 1)) or ((m == 1) and (n == 1)):
        return 1.0  # dr/dr = 1.0
    elif ((m == -2) and (n == 2)) or ((m == 2) and (n == 2)):
        return 2.0*r  # dr^2/dr = 2r
    elif (m == 0) and (n == 2):
        return 4.0*r  # d(2r^2 - 1)/dr = 4r
    elif ((m == -3) and (n == 3)) or ((m == 3) and (n == 3)):
        return 3.0*r*r  # d(r^3)/dr = 3*(r^2)
    elif ((m == -1) and (n == 3)) or ((m == 1) and (n == 3)):
        return 9.0*r*r - 2.0  # d(3r^3 - 2r)/dr = 9*(r^2) - 2
    elif ((m == -4) and (n == 4)) or ((m == 4) and (n == 4)):
        return 4.0*r*r*r  # d(r^4)/dr = 4*(r^3)
    elif ((m == -2) and (n == 4)) or ((m == 2) and (n == 4)):
        return 16.0*r*r*r - 6.0*r  # d(4r^4 - 3r^2)/dr = 16*(r^3) - 6r
    elif (m == 0) and (n == 4):
        return 24.0*r*r*r - 12.0*r  # d(6r^4 - 6r^2 + 1)/dr = 24*(r^3) - 12r
    elif ((m == -5) and (n == 5)) or ((m == 5) and (n == 5)):
        return 5.0*r*r*r*r  # d(r^5)/dr = 5*(r^4)
    elif ((m == -3) and (n == 5)) or ((m == 3) and (n == 5)):
        return 25.0*r*r*r*r - 12.0*r*r  # d(5r^5 - 4r^3)/dr = 25*(r^4) - 12*(r^2)
    elif ((m == -1) and (n == 5)) or ((m == 1) and (n == 5)):
        return 50.0*r*r*r*r - 36.0*r*r + 3.0  # d(10r^5 - 12r^3 + 3r)/dr = 50*(r^4) - 36*(r^2) + 3
    elif ((m == -6) and (n == 6)) or ((m == 6) and (n == 6)):
        return 6.0*r*r*r*r*r  # d(r^6)/dr = 6*(r^5)
    elif ((m == -4) and (n == 6)) or ((m == 4) and (n == 6)):
        return 36.0*r*r*r*r*r - 20.0*r*r*r  # d(6r^6 - 5r^4)/dr = 36*(r^5) - 20*(r^3)
    elif ((m == -2) and (n == 6)) or ((m == 2) and (n == 6)):
        return 90.0*r*r*r*r*r - 80.0*r*r*r + 12.0*r  # d(15r^6 - 20r^4 + 6r^2)/dr = 90*(r^5) - 80*(r^3) + 12r
    elif (m == 0) and (n == 6):
        return 120.0*r*r*r*r*r - 120.0*r*r*r + 24.0*r  # d(20r^6 - 30r^4 + 12r^2 - 1)/dr = 120*(r^5) - 120*(r^3) + 24r
    elif ((m == -7) and (n == 7)) or ((m == 7) and (n == 7)):
        return 7.0*r*r*r*r*r*r  # d(r^7)/dr = 7*(r^6)
    elif ((m == -5) and (n == 7)) or ((m == 5) and (n == 7)):
        return 49.0*r*r*r*r*r*r - 30.0*r*r*r*r  # d(7r^7 - 6r^5)/dr = 49*(r^6) - 30*(r^4)
    elif ((m == -3) and (n == 7)) or ((m == 3) and (n == 7)):
        return 147.0*r*r*r*r*r*r - 150.0*r*r*r*r + 30.0*r*r  # d(21r^7 - 30r^5 + 10r^3)/dr = 147*(r^6) - 150*(r^4) + 30*(r^2)
    elif ((m == -1) and (n == 7)) or ((m == 1) and (n == 7)):  # d(35r^7 - 60r^5 + 30r^3 - 4r) = 245*(r^6) - 300*(r^4) + 90*(r^2)-4
        return 245.0*r*r*r*r*r*r - 300.0*r*r*r*r + 90.0*r*r - 4.0
    else:
        return 0.0  # default return value for the orders more than 7.


def radial_polynomial_derivative_dr(m: int, n: int, r: float) -> float:
    """
    Calculate derivative on radius of radial polynomial (like dR(m,n)/dr), that is calculated by the recurrence equation.

    Parameters
    ----------
    m: int
        Azimuthal order.
    n: int
        Radial order.
    r: float
        Radius from the polar coordinate.

    Returns
    -------
    float:
        Recursively calculated derivative on radius from the radial polynomial.

    """
    if (n == 0) and (m == 0):
        return 0.0
    elif m > n:
        return 0.0
    else:
        # Recursion formula that should be more effective than direct calculation
        return ((radial_polynomial(abs(m-1), n-1, r) + radial_polynomial(m+1, n-1, r)) +
                + r*(radial_polynomial_derivative_dr(abs(m-1), n-1, r) + radial_polynomial_derivative_dr(m+1, n-1, r)) +
                -radial_polynomial_derivative_dr(m, n-2, r))


def triangular_derivative_dtheta(m: int, theta: float) -> float:
    """
    Calculate derivative of triangular function on the angle theta (like dTriangular/dtheta).

    Parameters
    ----------
    m : int
        Azimuthal order.
    theta : float
        Angle from polar coordinates associated with the unit circle .

    Returns
    -------
    float
        Calculated value.

    """
    if m > 0:
        return -m*math.sin(m*theta)
    elif m < 0:
        return -m*math.cos(m*theta)
    else:
        return 0.0


def get_classical_polynomial_name(mode: tuple, short_names: bool = False) -> str:
    """
    Return the classical name of Zernike polynomial.

    Till the 4th order (including) - the names taken from the Wikipedia artuicle https://en.wikipedia.org/wiki/Zernike_polynomials
    5th order names - from the website https://www.telescope-optics.net/monochromatic_eye_aberrations.htm.
    6th order and 7th names - my guess about the naming.

    Parameters
    ----------
    mode : tuple
        Zernike polynomial specification as (m, n).
    short_names: bool, optional
        Return shortened names for the polynomials. The default is false.

    Returns
    -------
    str
        Classical name.

    """
    name = ""
    dictionary_names = {(-1, 1): "Vertical (Y) tilt", (1, 1): "Horizontal (X) tilt", (-2, 2): "Oblique astigmatism",
                        (0, 2): "Defocus", (2, 2): "Vertical astigmatism", (-3, 3): "Vertical trefoil",
                        (-1, 3): "Vertical coma", (1, 3): "Horizontal coma", (3, 3): "Oblique trefoil",
                        (-4, 4): "Oblique quadrafoil", (-2, 4): "Oblique secondary astigmatism",
                        (0, 4): "Primary spherical", (2, 4): "Vertical secondary astigmatism",
                        (4, 4): "Vertical quadrafoil", (-5, 5): "Vertical pentafoil",
                        (-3, 5): "Vertical secondary trefoil", (-1, 5): "Vertical secondary coma",
                        (1, 5): "Horizontal secondary coma", (3, 5): "Oblique secondary trefoil",
                        (5, 5): "Oblique pentafoil", (-6, 6): "Oblique sexfoil",
                        (-4, 6): "Oblique secondary quadrafoil", (-2, 6): "Oblique thirdly astigmatism",
                        (0, 6): "Secondary spherical", (2, 6): "Vertical thirdly astigmatism",
                        (4, 6): "Vertical secondary quadrafoil", (6, 6): "Vertical sexfoil",
                        (-7, 7): "Vertical septfoil", (-5, 7): "Vertical secondary pentafoil",
                        (-3, 7): "Vertical thirdly trefoil", (-1, 7): "Vertical thirdly coma",
                        (1, 7): "Horizontal thirdly coma", (3, 7): "Oblique thirdly trefoil",
                        (5, 7): "Oblique secondary pentafoil", (7, 7): "Oblique septfoil"}
    dictionary_short_names = {(-1, 1): "Vert. tilt", (1, 1): "Hor. tilt", (-2, 2): "Obliq. astigm.",
                              (0, 2): "Defocus", (2, 2): "Vert. astigm.", (-3, 3): "Vert. 3foil",
                              (-1, 3): "Vert. coma", (1, 3): "Hor. coma", (3, 3): "Obliq. 3foil",
                              (-4, 4): "Obliq. 4foil", (-2, 4): "Obliq. 2d ast.",
                              (0, 4): "Spherical", (2, 4): "Vert. 2d ast.", (4, 4): "Vert. 4foil",
                              (-5, 5): "Vert. 5foil", (-3, 5): "Vert. 2d 3foil", (-1, 5): "Vert. 2d coma",
                              (1, 5): "Hor. 2d coma", (3, 5): "Obliq. 2d 3foil",
                              (5, 5): "Obliq. 5foil", (-6, 6): "Obliq. 6foil", (-4, 6): "Obliq.2d 4foil",
                              (-2, 6): "Obliq. 3d ast.", (0, 6): "2d spherical", (2, 6): "Vert. 3d ast.",
                              (4, 6): "Vert. 2d 4foil", (6, 6): "Vert. 6foil", (-7, 7): "Vert. 7foil",
                              (-5, 7): "Vert. 2d 5foil", (-3, 7): "Vert. 3d 3foil", (-1, 7): "Vert. 3d coma",
                              (1, 7): "Hor. 3d coma", (3, 7): "Obliq.3d 3foil",
                              (5, 7): "Obliq.2d 5foil", (7, 7): "Obliq. 7foil"}
    if short_names:
        if mode in dictionary_short_names.keys():
            name = dictionary_short_names[mode]
    else:
        if mode in dictionary_names.keys():
            name = dictionary_names[mode]
    return name


def get_osa_standard_index(m: int, n: int) -> int:
    """
    Calculate OSA/ANSI standard single index for Zernike polynomial (m, n).

    Parameters
    ----------
    m : int
        Azimuthal order.
    n : int
        Radial order.

    Returns
    -------
    int
        OSA/ANSI standard single index.

    """
    return (m + n*(n+2))//2


def test():
    """
    Perform tests of implemented recurrence equations.

    Raise
    -----
    AssertionError

    Returns
    -------
    None if all tests passed.

    """
    r = 0.841; theta = np.radians(30.0); theta = float(theta)  # some predefined values for polar coordinates
    # Tests of implemented recursive values using the equations from Wiki as tabulated
    m = 1; n = 1; zrp = zernike_polynomial(m, n, r, theta)
    assert abs(zrp - 2.0*r*np.cos(theta)) < 1E-4, f'Implemented Z{m, n} not equal to tabulated one'
    m = -1; n = 1; zrp = zernike_polynomial(m, n, r, theta)
    assert abs(zrp - 2.0*r*np.sin(theta)) < 1E-4, f'Implemented Z{m, n} not equal to tabulated one'
    m = 0; n = 2; zrp = zernike_polynomial(m, n, r, theta)
    assert abs(zrp - np.sqrt(3.0)*(2.0*r*r - 1.0)) < 1E-4, f'Implemented Z{m, n} != to tabulated one'
    m = -2; n = 2; zrp = zernike_polynomial(m, n, r, theta)
    assert abs(zrp - np.sqrt(6.0)*r*r*np.sin(2.0*theta)) < 1E-4, f'Implemented Z{m, n} != to tabulated one'
    m = 1; n = 3; zrp = zernike_polynomial(m, n, r, theta)
    assert abs(zrp - np.sqrt(8.0)*(3.0*r*r*r - 2.0*r)*np.cos(theta)) < 1E-4, f'Implemented Z{m, n} != to tabulated one'
    m = -4; n = 4; zrp = zernike_polynomial(m, n, r, theta)
    assert abs(zrp - np.sqrt(10.0)*r*r*r*r*np.sin(4.0*theta)) < 1E-4, f'Implemented Z{m, n} != to tabulated one'
    m = 0; n = 4; zrp = zernike_polynomial(m, n, r, theta)
    assert abs(zrp - np.sqrt(5.0)*(6.0*r*r*r*r - 6.0*r*r + 1.0)) < 1E-4, f'Implemented Z{m, n} != to tabulated one'
    # Test the tabulated values, assuming that the recursive implemented correctly
    test_orders = [(-7, 7), (-5, 7), (-3, 7), (-1, 7), (-6, 6), (-4, 6), (-2, 6), (-5, 5), (-3, 5),
                   (-1, 5), (-4, 4), (-2, 4), (0, 4), (-3, 3), (-1, 3), (-2, 2), (0, 2), (-1, 1)]
    for test_order in test_orders:
        (m, n) = test_order
        assert abs(radial_polynomial(m, n, r)-tabular_radial_polynomial(m, n, r)) < 1.0E-6, f'Check tabulated R{m, n}'
        assert abs(radial_polynomial_derivative_dr(m, n, r)-tabular_radial_derivative_dr(m, n, r)) < 1.0E-6, f'Tab. dR{m, n}!'
        # print(radial_polynomial_derivative_dr(m, n, r), tabular_radial_derivative_dr(m, n, r))
        # print(radial_polynomial(m, n, r), tabular_radial_polynomial(m, n, r))
    print("All tests passed")


# %% Tests
if __name__ == '__main__':
    test()  # Testing implemented recurrence equations and tabular values

    # %% Plotting results over the surface
    step_r = 0.01; step_theta = 1.0  # in grads
    orders = [(-1, 1)]  # Y tilt
    plot_zps_polar(orders, step_r, step_theta, "Y tilt")
    zernikes_set = [(-1, 1), (1, 1)]
    plot_zps_polar(zernikes_set, step_r, step_theta, "Sum tilts with equal amplitudes", show_amplitudes=True)
