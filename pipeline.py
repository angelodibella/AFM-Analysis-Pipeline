import cv2 as cv

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
    stack.track_contour((13, 31))

    # Create spline object
    spline = ps.Spline(stack)

    # Animate the spline with curvature
    # spline.animate_spline(0, 'trans', sigma=1.2)
    # spline.animate_spline(1, 'trans', sigma=1.3)

    # Compare two splines
    spline.compare_splines(0, (10, 11), 'trans')
    spline.compare_splines(0, (10, 14), 'trans')

    # Print information
    stack.print_info('Transmemebrane Defects')


def canny_1(stack):
    stack.gaussian_blur((3, 3), 1)
    stack.canny(0, 0)

    contours, _, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()
