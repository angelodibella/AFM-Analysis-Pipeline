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


def to_splines(stack, degree=3):
    """TODO: add docstring"""

    # Get all tracked contours as list(list(ndarray[x, y])) where each ndarray[x, y] forms a closed (cyclic) contour
    tracked = stack.tracked

    # Interpolate
    splines = []
    for contour_object in tracked:
        contour_splines = []
        for contour in contour_object:
            # Adjust so that first row is all x-positions and second row is all y-positions
            adjusted_contour = contour[:, 0, :].T

            # Find spline
            tck, _ = interp.splprep([adjusted_contour[0], adjusted_contour[1]], s=0, k=degree, per=True)
            contour_splines.append(tck)
        splines.append(contour_splines)

    return splines




def remove_border(stack):
    pass


def calculate_curvature(stack):
    pass
