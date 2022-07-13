import cv2 as cv
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

import process as ps
import image as im

# Constants to be used
RED = [1.0, 0.0, 0.0]
NOBLUE_MORERED = [0.644, 0.356, 0]


def thin_defects(stack):
    """TODO: finish docstring
    Probably for thin defects
    """

    stack.gaussian_blur((3, 3), 1, coeffs=RED)
    stack.binary_threshold(0, otsu=True)
    contours, _, _ = stack.get_contours(hierarchy=cv.RETR_TREE, append=True)

    stack.intensity_band(0.90, 1, which=2, otsu=True)

    stack.add_contours(contours, which=5, color=(255, 0, 0))

    stack.get_contour_centers(append=True)

    # Print information
    stack.print_info('Thin Defects')


def transmembrane_defects(stack):
    """TODO: finish docstring
    Probably for transmembrane defects
    """

    stack.gaussian_blur((13, 13), 1, coeffs=NOBLUE_MORERED)
    stack.intensity_band(0, 60, binary=True)

    stack.get_contours(hierarchy=cv.RETR_EXTERNAL, append=True)

    stack.get_contour_centers(append=True)

    # Track some contours
    stack.track_contour((13, 5))

    # Evaluate spline for tracked contour
    splines_list = ps.to_splines(stack)
    coords_list = ps.evaluate_splines(splines_list)

    # Plot spline
    plt.figure(figsize=(5, 5))
    spline = coords_list[0][-1]
    plt.plot(spline[0], spline[1])
    plt.gca().invert_yaxis()
    plt.show()

    # Print information
    stack.print_info('Transmemebrane Defects')


def canny_1(stack):
    stack.gaussian_blur((3, 3), 1)
    stack.canny(0, 0)

    contours, _, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()
