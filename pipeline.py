import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import process as ps
import image as im

# Constants to be used
RED = [1.0, 0.0, 0.0]
NOBLUE_MORERED = [0.644, 0.356, 0]

# Set font parameters for plots
plt.rcParams['font.family'] = 'Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 10})

# Create a save auxiliary figures
auxiliary_figures = False


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

    thresh_range = (0, 63)  # 0-60 is optimal

    stack.gaussian_blur((13, 13), 1, coeffs=NOBLUE_MORERED)
    stack.intensity_band(*thresh_range, binary=True)

    # # Open
    # kernel = np.ones((3, 3), dtype='uint8')
    # stack.dilate(kernel, iterations=1)
    # stack.erode(kernel, iterations=1)

    stack.get_contours(hierarchy=cv.RETR_EXTERNAL, append=True)

    stack.get_contour_centers(append=True)

    # Track some contours
    stack.track_contour((13, 5))
    stack.track_contour((13, 30))
    stack.track_contour((57, 120))
    # stack.track_contour((57, 1)) # FIND THE BIG ONE!!!

    # Create spline object
    spline = ps.Spline(stack, increment=0.0005)

    # Create curvature-velocity diagram
    spline.create_curvature_velocity_plot(0, 0, 'trans', averages=True)
    spline.create_curvature_velocity_plot(1, 0, 'trans', averages=True)
    spline.create_curvature_velocity_plot(2, 0, 'trans', averages=True)

    # # Animate the spline with curvature
    # spline.animate_spline(0, 'trans', sigma=1.2, start=0)
    # spline.animate_spline(1, 'trans', sigma=1.3)
    # spline.animate_spline(2, 'trans', sigma=1.3)

    # # Compare splines
    # spline.compare_splines(0, (10, 11), 'trans')
    # spline.compare_splines(0, (11, 12), 'trans')
    # spline.compare_splines(0, (12, 13), 'trans')
    # spline.compare_splines(0, (13, 14), 'trans')
    # spline.compare_splines(0, (14, 15), 'trans')
    # spline.compare_splines(0, (15, 16), 'trans')
    # spline.compare_splines(0, (17, 18), 'trans')
    #
    # spline.compare_splines(1, (10, 11), 'trans')
    # spline.compare_splines(1, (11, 12), 'trans')
    # spline.compare_splines(1, (12, 13), 'trans')

    # for i in range(len(spline.tracked[0]) - 1):
    #     spline.compare_splines(0, (i, i + 1), 'trans')
    #     spline.compare_splines(0, (i, i + 1), 'trans', secondary_map=True)
    #
    # for i in range(len(spline.tracked[1]) - 1):
    #     spline.compare_splines(1, (i, i + 1), 'trans')
    #     spline.compare_splines(1, (i, i + 1), 'trans', secondary_map=True)

    # # Create kymographs
    # spline.create_curvature_kymograph(0, 'trans', hline=1910)
    # spline.create_curvature_kymograph(1, 'trans')
    # spline.create_curvature_kymograph(2, 'trans')
    #
    # spline.create_velocity_kymograph(0, 'trans', hline=1910)
    # spline.create_velocity_kymograph(1, 'trans')
    # spline.create_velocity_kymograph(2, 'trans')

    if auxiliary_figures:
        print('Saving auxiliary figures...', end=' ')

        # Get the last image from the grayscale stack
        hist_data = stack.stacks[1][-1].flatten()

        # Create figure
        plt.figure()
        plt.xlim(-5, 255)
        plt.vlines([0, 60], 0, 4000, color='blue', linestyles='dashed')
        plt.hist(hist_data, bins=256, color='#000000')
        plt.savefig('Output Images/Auxiliary Figures/histogram.png', dpi=300)

        print('(Done)')

    # Print information
    stack.print_info('Transmemebrane Defects')


def canny_1(stack):
    stack.gaussian_blur((3, 3), 1)
    stack.canny(0, 0)

    contours, _, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()
