import cv2 as cv
import numpy as np
import scipy.interpolate as interp

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import process as ps
import image as im

# Set font parameters for plots
plt.rcParams["font.family"] = "Serif"
plt.rcParams.update({'font.size': 10})

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
    stack.track_contour((13, 31))

    # Evaluate spline for tracked contour
    splines_list = ps.to_splines(stack)

    # Animate the spline with curvature
    ps.animate_contour_spline(splines_list, 0, 'trans_13_5', sigma=1.2, px_xlen=stack.px_xlen, px_ylen=stack.px_ylen)
    ps.animate_contour_spline(splines_list, 1, 'trans_13_31', sigma=1.3, px_xlen=stack.px_xlen, px_ylen=stack.px_ylen)

    # Print information
    stack.print_info('Transmemebrane Defects')


def canny_1(stack):
    stack.gaussian_blur((3, 3), 1)
    stack.canny(0, 0)

    contours, _, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()
