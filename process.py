import numpy as np
import scipy.interpolate as interp
import skimage.segmentation as seg

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set font parameters for plots
plt.rcParams["font.family"] = "Serif"
plt.rcParams.update({'font.size': 10})


def adjust_contours(contours):
    """TODO: add docstring"""

    new_contours = []
    for stack in contours:
        fixed = []
        for contour in stack:
            fixed.append(contour[:, 0, :])
        new_contours.append(fixed)

    return new_contours


class Spline:
    """TODO: add class docstring"""

    def __init__(self, stack, degree=3, smoothing=7, stdev=1, increment=0.0005):

        # Retrieve the tracked contours
        self.tracked = stack.tracked
        self.tracked_positions = stack.tracked_positions

        self.splines_list = []
        self.coords_list = []
        self.curvatures_list = []

        self.px_xlen = stack.px_xlen
        self.px_ylen = stack.px_ylen

        self.to_splines(degree, smoothing, stdev, append=True)
        self.evaluate_splines(increment=increment, append=True)
        self.splines_curvature(increment=increment)

    def to_splines(self, degree, smoothing, stdev, append=False):
        """
        Takes a Stack object, and for each of its tracked contours returns a list of splines in the form of (t, c, k),
        a tuple containing the vector of knots, the B-spline coefficients, and the degree of the coefficients.

        Parameters
        ----------
        stack : image.Stack
            The stack object that needs to be processed.
        degree : int, default 3
            The degree of the splines, the default is cubic, which is recommanded.
        smoothing : float, default 7
            The degree of smoothing of the fit. There is a tradeoff between the closeness and smootheness of fit, however,
            with larger values resulting in a smoother but less accurate fit.
        stdev : float or list, default 1
            If it is a list, the uncertainty for each pixel measurement, otherwise the uncertainty for all pixels.
        append : bool, default False
            Append values to the object attribute instances.

        Returns
        -------
        splines_list : list
            List containing lists of (t, c, k) spline representations for each contour at a given instance in time.
        """

        assert self.tracked, 'No tracked contours in the current stack'

        # Interpolate
        splines_list = []
        for contour_object in self.tracked:
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

        if append:
            self.splines_list = splines_list

        return splines_list

    def evaluate_splines(self, increment=0.0005, der=0, append=False):
        """TODO: add docstring"""

        # Parameter of spline
        u = np.linspace(0, 1 - increment, int(1 / increment))

        # Evaluate each spline at each frame for all tracked contours
        coords_list = []
        for splines in self.splines_list:
            coords = []
            for spline in splines:
                x, y = interp.splev(u, spline, der=der)
                coords.append(np.array([x, y]))
            coords_list.append(coords)

        if append:
            self.coords_list = coords_list

        return coords_list

    def splines_curvature(self, increment=0.0005, append=True):
        """TODO: add docstring, check that splines must be at least parabolic (degree > 1); this is signed curvature"""

        # TODO: FIX! Curvatures are set to zero for some reason...

        # Calculate derivatives of spline, which have the same length
        coords_list_d1 = self.evaluate_splines(increment=increment, der=1)
        coords_list_d2 = self.evaluate_splines(increment=increment, der=2)

        curvatures_list = []
        for i, coords_d1 in enumerate(coords_list_d1):
            curvatures = []
            for j, coord_d1 in enumerate(coords_d1):
                coord_d1 = np.array(coord_d1)
                coord_d2 = np.array(coords_list_d2[i][j])

                # Calculate curvature for current list of coordinates at each point
                curvature_numerator = coord_d1[1] * coord_d2[0] - coord_d1[0] * coord_d2[1]
                curvature_denominator = (coord_d1[0] ** 2 + coord_d1[1] ** 2) ** (3 / 2)
                curvatures.append(curvature_numerator / curvature_denominator)
            curvatures_list.append(curvatures)

        if append:
            self.curvatures_list = curvatures_list

        return curvatures_list

    def animate_contour_spline(self, which, file_name, figsize=(5, 5), sigma=1, cmap='jet', start=100):
        """TODO: add docstring"""

        # Treat file name
        if file_name:
            file_name += f'_{self.tracked_positions[which][0]}_{self.tracked_positions[which][1]}'
        else:
            file_name += f'{self.tracked_positions[which][0]}_{self.tracked_positions[which][1]}'

        # Convert pixel scale to micrometers
        self.px_xlen /= 1000  # micrometers
        self.px_ylen /= 1000  # micrometers

        # Initialize the figure
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots(1, 1)

        # Get the desired contour out of the tracked ones
        splines = self.coords_list[which]
        curvatures = self.curvatures_list[which]

        # Get all the x- and y-values
        all_x = []
        all_y = []
        for spline in splines:
            all_x.append(spline[0])
            all_y.append(spline[1])
        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Since there may be outliers of high leverage, consider only the curvature values that are in the
        # range +/- (sigma * standard deviation), where the (sample) standard deviation is that of the curvature values,
        # and sigma is a multiplier chosen by the user for the specific contour
        curvature_range = sigma * np.std(curvatures, ddof=1)

        # Create color map for curvature
        norm = colors.Normalize(vmin=-curvature_range, vmax=curvature_range, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(mapper)

        def animate(n):
            """Update the scatter plot.

            Parameters
            ----------
            n : int
                Current frame in the series.

            Returns
            -------
            matplotlib.collections.PathCollection
                The artist object used for blitting.
            """
            ax.cla()
            ax.set_xlabel(r'x ($\mu$m)')
            ax.set_ylabel(r'y ($\mu$m)')
            ax.set_xlim((np.min(all_x) - 10) * self.px_xlen, (np.max(all_x) + 10) * self.px_xlen)
            ax.set_ylim((np.min(all_y) - 10) * self.px_ylen, (np.max(all_y) + 10) * self.px_ylen)
            ax.invert_yaxis()
            sc = ax.scatter(splines[n][0] * self.px_xlen, splines[n][1] * self.px_ylen,
                            color=mapper.to_rgba(curvatures[n]), marker='.', s=20)
            point = ax.scatter(splines[n][0][start] * self.px_xlen, splines[n][1][start] * self.px_ylen, c='#DE88EA',
                               marker='X', s=15, linewidths=0.2)

            return sc, point,

        # Create animation
        ani = animation.FuncAnimation(fig, animate, len(splines), interval=10, blit=True)

        # Write animation to video (uses FFMpeg)
        print(f'\nWriting animation... (Output Images/{file_name}.mp4)')
        writervideo = animation.FFMpegWriter(fps=5)
        ani.save(f'Output Images/{file_name}.mp4', writer=writervideo, dpi=300)


def remove_border(stack):
    pass


def calculate_curvature(stack):
    pass
