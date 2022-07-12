import numpy as np
import scipy.interpolate as interp


def adjust_contours(contours):
    """TODO: add docstring"""

    new_contours = []
    for stack in contours:
        fixed = []
        for contour in stack:
            fixed.append(contour[:, 0, :])
        new_contours.append(fixed)

    return new_contours


def to_splines(stack, degree=3, smoothing=6, stdev=1):
    """Takes a Stack object, and for each of its tracked contours returns a list of splines in the form of (t, c, k),
    a tuple containing the vector of knots, the B-spline coefficients, and the degree of the coefficients.

    Parameters
    ----------
    stack : image.Stack
        The stack object that needs to be processed.
    degree : int, default 3
        The degree of the splines, the default is cubic, which is recommanded.
    smoothing : float, default 6
        The degree of smoothing of the fit. There is a tradeoff between the closeness and smootheness of fit, however,
        with larger values resulting in a smoother but less accurate fit.
    stdev : float or list, default 1
        If it is a list, the uncertainty for each pixel measurement, otherwise the uncertainty for all pixels.

    Returns
    -------
    splines_list : list
        List containing lists of (t, c, k) spline representations for each contour at a given instance in time.
    """

    # Get all tracked contours as list(list(ndarray[x, y])) where each ndarray[x, y] forms a closed (cyclic) contour
    tracked = stack.tracked
    assert tracked, 'No tracked contours in the current stack'

    # Interpolate
    splines_list = []
    for contour_object in tracked:
        contour_splines = []
        for contour in contour_object:
            # Adjust so that first row is all x-positions and second row is all y-positions
            adjusted_contour = contour[:, 0, :].T
            x = adjusted_contour[0]
            y = adjusted_contour[1]

            # Append the starting coordinates, since we apply a periodic fit
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

            # Get weights from standard deviation vector
            weights = 1 / np.array(stdev) if isinstance(stdev, list) else np.ones(len(x)) / stdev

            # Find spline
            tck, _ = interp.splprep([x, y], w=weights, s=smoothing, k=degree, per=True)
            contour_splines.append(tck)
        splines_list.append(contour_splines)

    return splines_list


def evaluate_splines(splines_list, increment=0.001):
    """TODO: add docstring"""

    # Parameter of spline
    u = np.linspace(0, 1 - increment, int(1 / increment))

    # Evaluate each spline at each frame for all tracked contours
    coords_list = []
    for splines in splines_list:
        coords = []
        for spline in splines:
            x, y = interp.splev(u, spline)
            coords.append(np.array([x, y]))
        coords_list.append(coords)

    return coords_list


def remove_border(stack):
    pass


def calculate_curvature(stack):
    pass
