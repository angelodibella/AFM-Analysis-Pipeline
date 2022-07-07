import cv2 as cv

# Constants to be used
COEFFS_1 = [0.34, 0.66, 0.0]


def thin_defects(stack):
    """TODO: finish docstring
    Probably for thin defects
    """

    stack.gaussian_blur((3, 3), 1)
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

    stack.gaussian_blur((13, 13), 1)
    stack.intensity_band(0, 35, binary=True)

    contours, _, _ = stack.get_contours(hierarchy=cv.RETR_EXTERNAL, append=True)

    # stack.intensity_band(30, 74, which=2)

    stack.get_contour_centers(append=True)

    stack.track_contour((40, 80))

    # Print information
    stack.print_info('Transmemebrane Defects')


def canny_1(stack):
    stack.gaussian_blur((3, 3), 1, coeffs=COEFFS_1)
    stack.canny(0, 0)

    contours, _, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()
