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
    curvatures_list = ps.splines_curvature(splines_list)

    # Plot spline
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots(1, 1)

    splines = coords_list[0]
    curvatures = curvatures_list[0]

    all_x = []
    all_y = []
    for spline in splines:
        all_x.append(spline[0])
        all_y.append(spline[1])
    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Create color map for curvature
    minima, maxima = (np.min(curvatures), np.max(curvatures))
    norm = colors.Normalize(vmin=-1, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='jet')
    plt.colorbar(mapper)

    def animate(n):
        ax.cla()
        ax.set_xlim(np.min(all_x) - 10, np.max(all_x) + 10)
        ax.set_ylim(np.min(all_y) - 10, np.max(all_y) + 10)
        ax.invert_yaxis()
        sc = ax.scatter(splines[n][0], splines[n][1], color=mapper.to_rgba(curvatures[n]), marker='.', s=20)

        return sc,

    # Create animation
    ani = animation.FuncAnimation(fig, animate, len(splines), interval=10, blit=True)

    # Write animation to video
    writervideo = animation.FFMpegWriter(fps=5)
    ani.save('Output Images/trans_13_5.mp4', writer=writervideo, dpi=300)

    # Print information
    stack.print_info('Transmemebrane Defects')


def canny_1(stack):
    stack.gaussian_blur((3, 3), 1)
    stack.canny(0, 0)

    contours, _, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()
