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

        self.degree = degree
        self.smoothing = smoothing
        self.stdev = stdev
        self.increment = increment

        # Retrieve the tracked contours
        self.tracked = stack.tracked
        self.tracked_positions = stack.tracked_positions
        self.centers = stack.tracked_centers

        self.splines_list = []
        self.coords_list = []
        self.curvatures_list = []

        self.px_xlen = stack.px_xlen
        self.px_ylen = stack.px_ylen
        self.timings = stack.timings

        # Store the map between values of the spline (coordinates or derivatives) across frames
        self.map = []

        print(f'\nCreating splines from {len(self.tracked)} contours...', end=' ')
        self.to_splines(degree, smoothing, stdev, append=True)
        print(f'(Done)\nEvaluating splines at {int(1 / increment)} points -> ', end=' ')
        self.evaluate_splines(increment=increment, append=True)
        print(f'Calculating curvature of splines...', end=' ')
        self.splines_curvature(increment=increment)
        print('(Done)')

    # -------------------------------- Numerical Methods --------------------------------

    def to_splines(self, degree, smoothing, stdev, append=False):
        """
        Takes a Stack object, and for each of its tracked contours returns a list of splines in the form of (t, c, k),
        a tuple containing the vector of knots, the B-spline coefficients, and the degree of the coefficients.

        Parameters
        ----------
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

        # Generate map if necessary, then apply it
        if der == 0:
            self.generate_LSSD_map(coords_list)
        coords_list = self.apply_map(coords_list)

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

    # -------------------------------- Graphical Methods --------------------------------

    def animate_spline(self, which, file_name, figsize=None, sigma=1, cmap='jet', start=1000, frame_time=True):
        """TODO: add docstring"""

        # Treat file name
        if file_name:
            file_name += f'_{self.tracked_positions[which][0]}_{self.tracked_positions[which][1]}'
        else:
            file_name += f'{self.tracked_positions[which][0]}_{self.tracked_positions[which][1]}'

        # Convert pixel scale to micrometers
        px_xlen_um = self.px_xlen / 1000  # micrometers
        px_ylen_um = self.px_ylen / 1000  # micrometers

        # Initialize the figure
        fig = plt.figure() if figsize is None else plt.figure(figsize=figsize)
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
            if frame_time:
                ax.set_title(f'Membrane Defect at Frame {n}')
            else:
                ax.set_title(f'Membrane Defect at Time {n * self.timings} s')
            ax.set_xlabel(r'x ($\mu$m)')
            ax.set_ylabel(r'y ($\mu$m)')
            ax.set_xlim((np.min(all_x) - 10) * px_xlen_um, (np.max(all_x) + 10) * px_xlen_um)
            ax.set_ylim((np.min(all_y) - 10) * px_ylen_um, (np.max(all_y) + 10) * px_ylen_um)
            ax.invert_yaxis()
            sc = ax.scatter(splines[n][0] * px_xlen_um, splines[n][1] * px_ylen_um,
                            color=mapper.to_rgba(curvatures[n]), marker='.', s=20)
            point = ax.scatter(splines[n][0][start] * px_xlen_um, splines[n][1][start] * px_ylen_um, c='#DE88EA',
                               marker='X', s=15, linewidths=0.2)

            return sc, point,

        # Create animation
        ani = animation.FuncAnimation(fig, animate, len(splines), interval=10, blit=True)

        # Write animation to video (uses FFMpeg)
        print(f'\nWriting animation... (Output Images/Animations/{file_name}.mp4)', end=' ')
        writervideo = animation.FFMpegWriter(fps=5)
        ani.save(f'Output Images/Animations/{file_name}.mp4', writer=writervideo, dpi=300)
        print('(Done)')

        return ani

    def compare_splines(self, which, positions, file_name, figsize=None, interval=50, cmap='jet', frame_time=True):
        """Compare two splines with the current map.

        Parameters
        ----------
        which : int
            Which tracked spline to compare.
        positions : tuple of int
            The two time positions of the spline.
        file_name : str
            Prefix of the file name of the saved figure.
        figsize : tuple of int, optional
            The size of the figure.
        interval : int, default 50
            Interval between points where to draw the connecting lines.
        cmap : str
            Color map for displacement vectors
        frame_time : bool, default True
            Whether the time is displayed as frames in the stack or the true time in seconds.
        """

        # Select the splines
        coord_1 = self.coords_list[which][np.min(positions)]
        coord_2 = self.coords_list[which][np.max(positions)]

        # Get pixel scale in micrometers
        px_xlen_um = self.px_xlen / 1000  # micrometers
        px_ylen_um = self.px_ylen / 1000  # micrometers

        # Get the displacements
        displacements = []
        for i, pair_1 in enumerate(coord_1.T):
            pair_2 = coord_2.T[i]
            displacements.append(np.sqrt(np.sum((pair_2 - pair_1) ** 2)))

            # TODO: implement method to verify the sign of propagation (+ if expanding, - if retracting)
            # # Check which is closer to the center
            # d_2 = np.sum((pair_2 - self.centers[which][positions[0]]) ** 2)
            # d_1 = np.sum((pair_1 - self.centers[which][positions[0]]) ** 2)
            # print(d_1, d_2)
            # if d_2 < d_1:
            #     displacements[-1] = -displacements[-1]
        displacements = displacements[::interval]

        # Initialize figure
        fig = plt.figure() if figsize is None else plt.figure(figsize=figsize)
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        plt.gca().invert_yaxis()

        # Create color map (WARNING: does not support px_xlen != px_ylen)
        # TODO: display color bar in nm instead of um, and add title to it
        displacement_range = np.max(np.abs(displacements)) * px_xlen_um
        norm = colors.Normalize(vmin=-displacement_range, vmax=displacement_range, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(mapper)

        # Plot displacement lines
        for i, pair_1 in enumerate(coord_1.T[::interval]):
            pair_2 = coord_2.T[::interval][i]
            plt.plot(np.array([pair_1[0], pair_2[0]]) * px_xlen_um, np.array([pair_1[1], pair_2[1]]) * px_ylen_um,
                     color=mapper.to_rgba(displacements[i] * px_xlen_um))

        # Plot the two splines
        label_1 = f'Frame {np.min(positions)}' if frame_time else f'Time {np.min(positions) * self.timings} s'
        plt.scatter(coord_1[0] * px_xlen_um, coord_1[1] * px_ylen_um, c='#000000',
                    label=label_1, s=10)
        label_2 = f'Frame {np.max(positions)}' if frame_time else f'Time {np.max(positions) * self.timings} s'
        plt.scatter(coord_2[0] * px_xlen_um, coord_2[1] * px_ylen_um, c='#808080',
                    label=label_2, s=10)
        plt.legend()

        print(f'\nSaving comparison... (Output Images/Comparisons/'
              f'{file_name}_{which}_({positions[0]}_{positions[1]}).png)')
        plt.savefig(f'Output Images/Comparisons/{file_name}_{which}_({positions[0]}_{positions[1]}).png', dpi=300)

    # -------------------------------- Mapping Methods --------------------------------

    def generate_LSSD_map(self, coords_list):
        """Generated a least sum square distance map between points of successive splines. TODO: add docstring"""

        # Iterate through the coordinates, finding the minimum squared distance
        print('Generating spline map [LSSD method]...', end=' ')
        min_diffs_list = []
        for i, coords in enumerate(coords_list):
            min_diffs = []
            for frame, curr_coord in enumerate(coords[:-1]):
                # Treat current coordinate list if it is not the first in the succession for the current tracked spline
                if min_diffs:
                    curr_coord = np.roll(curr_coord, min_diffs[frame - 1], axis=1)

                next_coord = coords[frame + 1]
                sum_squared_distances = []
                for d in range(len(curr_coord[0])):
                    # Shift the spline by i
                    next_candidate_x = np.roll(next_coord[0], d)
                    next_candidate_y = np.roll(next_coord[1], d)
                    diff_x = next_candidate_x - curr_coord[0]
                    diff_y = next_candidate_y - curr_coord[1]

                    # Append the current squared distance
                    sum_squared_distances.append(np.sum(diff_x ** 2 + diff_y ** 2))
                min_diffs.append(np.argmin(sum_squared_distances))
            min_diffs_list.append(min_diffs)
        print('(Done)')

        self.map = min_diffs_list

    def apply_map(self, coords_list):
        """TODO: add docstring"""

        # Apply the map for each pair (x, y) where x and y are lists respectively
        new_coords_list = []
        for i, coords in enumerate(coords_list):
            new_coords = [coords[0]]
            for frame, coord in enumerate(coords[1:]):
                new_coord = np.roll(coord, self.map[i][frame], axis=1)  # apply the map
                new_coords.append(new_coord)
            new_coords_list.append(new_coords)

        return new_coords_list

    def perpendicular_gradient_map(self):
        """TODO: add docstring"""

        # TODO: implement `perpendicular_gradient_map`; this should calculate, for every point in every spline, the
        #       next (in the successive frame) best point, preferring the points that are closest to the perpendicular
        #       to the gradient of the spline at the current point

        pass



