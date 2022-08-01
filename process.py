import numpy as np
from numpy import ndarray
import scipy.interpolate as interp

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from sklearn.cluster import KMeans

# Set font parameters for plots
plt.rcParams['font.family'] = 'Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 10})


def adjust_contours(contours):
    """Convert the standard contour output of OpenCV to lists: list(tuple(ndarray[N, 1, 2]))
    -> list(list(ndaray[N, 2])).

    Parameters
    ----------
    contours : list of (tuple of ndarray)
        List of raw contour outputs of OpenCV.

    Returns
    -------
    new_contours : list of (list of ndarray)
        Adjusted list of contours.
    """

    new_contours = []
    for stack in contours:
        fixed = []
        for contour in stack:
            fixed.append(contour[:, 0, :])
        new_contours.append(fixed)

    return new_contours


def is_in_spline(coord, coord_spline):
    """Check if an array of coordinates is inside a spline given its coordinates.

    Parameters
    ----------
    coord : ndarray of float
        Two-dimensional array containing the coordinates to be verified.
    coord_spline : ndarray of float
        Coordinates that comprise the spline

    Returns
    -------
    mask : ndarray of bool
        One-dimensional array indicating whether each point is inside the spline.
    """

    # Convert to list of tuples
    coord_spline_tuples = [(c[0], c[1]) for c in coord_spline.T]

    # Approximate spline as polygon
    polygon = Polygon(coord_spline_tuples)

    # Create list of points to check
    coord_points = [Point(c[0], c[1]) for c in coord.T]

    # Check if each points is in the spline
    mask = np.array([polygon.contains(point) for point in coord_points])

    return mask


def rotate_coord(coord, theta):
    """Rotates the axes of a two-dimensional array of coordinates of shape (2, N) from its original orthonormal basis.

    Parameters
    ----------
    coord : ndarray of float
        Array of coordinates with shape (2, N) where N is the total number of coordinate points.
    theta : float
        Angle the coordinate system must be rotated by

    Returns
    -------
    new_coord: ndarray of float
        Coordinates represented in the new, rotated coordinate system.
    """

    rotating_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    new_coord = rotating_matrix @ coord

    return new_coord


class Spline:
    """TODO: add class docstring"""

    def __init__(self, stack, degree=3, smoothing=7, stdev=1, increment=0.0005, secondary_map=None):

        self.degree = degree
        self.smoothing = smoothing
        self.stdev = stdev
        self.increment = increment
        self.use_secondary_map = secondary_map

        self.frames = stack.frames

        # Retrieve the tracked contours
        self.tracked = stack.tracked
        self.tracked_positions = stack.tracked_positions
        self.centers = stack.tracked_centers

        self.splines_list = []
        self.coords_list = []
        self.curvatures_list = []
        self.displacements_list = []

        self.secondary_coords_list = []
        self.secondary_displacements_list = []  # TODO: implement!

        self.px_xlen = stack.px_xlen
        self.px_ylen = stack.px_ylen
        self.timings = stack.timings

        # Store the map between values of the spline (coordinates or derivatives) across frames
        self.map = []
        self.secondary_maps_list = []

        print(f'\nCreating splines from {len(self.tracked)} contours...', end=' ')
        self.to_splines(degree, smoothing, stdev, append=True)
        print(f'(Done)\nEvaluating splines at {int(1 / increment)} points -> ', end=' ')
        self.evaluate_splines(increment=increment, append=True)
        print(f'Calculating curvature of splines...', end=' ')
        self.splines_curvature(increment=increment)
        print('(Done)\nEvaluating displacements...', end=' ')
        self.splines_displacements()
        print('(Done)')

        if self.use_secondary_map == 'normal':
            print('Generating secondary map [normal vector method]...', end=' ')
            self.generate_NV_secondary_map()
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

    def splines_displacements(self):
        """TODO: add docstring"""

        for coords in self.coords_list:
            displacements = []
            for i, curr_coord in enumerate(coords[:-1]):
                next_coord = coords[i + 1]
                abs_d = np.sqrt(np.sum((next_coord - curr_coord) ** 2, axis=0))
                d = np.where(is_in_spline(next_coord, curr_coord), -abs_d, abs_d)

                displacements.append(d)
            self.displacements_list.append(np.array(displacements))

    # -------------------------------- Graphical Methods --------------------------------

    def animate_spline(self, which, file_name, figsize=None, sigma=1, cmap='jet', start=1000, frame_time=True):
        """TODO: add docstring"""

        # Treat file name TODO: simplify this since `file_name` is not optional
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

    def compare_splines(self, which, positions, file_name, figsize=None, interval=50, cmap='jet', frame_time=True,
                        secondary_map=False):
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
        secondary_map : bool, default False
            Use the secondary map as point mapping between spline coordinates.
        """

        # Get pixel scale in micrometers
        px_xlen_um = self.px_xlen / 1000  # micrometers
        px_ylen_um = self.px_ylen / 1000  # micrometers

        if secondary_map:
            coords_list = self.apply_secondary_map(self.coords_list)
            file_name += f'_{self.use_secondary_map}'
        else:
            coords_list = self.coords_list

        # Select the splines
        coord_1 = coords_list[which][np.min(positions)]
        coord_2 = coords_list[which][np.max(positions)]

        # Get the displacements
        abs_displacements = np.sqrt(np.sum((coord_2 - coord_1) ** 2, axis=0))
        displacements = np.where(is_in_spline(coord_2, coord_1), -abs_displacements, abs_displacements)[::interval]

        # Initialize figure
        fig = plt.figure() if figsize is None else plt.figure(figsize=figsize)
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        plt.gca().invert_yaxis()
        plt.gca().axis('equal')

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
                    label=label_1, s=6)
        label_2 = f'Frame {np.max(positions)}' if frame_time else f'Time {np.max(positions) * self.timings} s'
        plt.scatter(coord_2[0] * px_xlen_um, coord_2[1] * px_ylen_um, c='#808080',
                    label=label_2, s=6)
        plt.legend()

        print(f'\nSaving comparison... (Output Images/Comparisons/'
              f'{file_name}_{which}_({positions[0]}_{positions[1]}).png)')
        plt.savefig(f'Output Images/Comparisons/{file_name}_{which}_({positions[0]}_{positions[1]}).png', dpi=300)

    def create_curvature_kymograph(self, which, file_name, figsize=None, sigma=1, cmap='jet', start=100,
                                   frame_time=True, sampling_interval=1, hline=None):
        """TODO: add docstring"""

        # Convert pixel scale to micrometers
        px_xlen_um = self.px_xlen / 1000  # micrometers
        px_ylen_um = self.px_ylen / 1000  # micrometers

        # Get the desired curvatures
        curvatures = np.array(self.curvatures_list[which]).T[::sampling_interval].T

        # Since there may be outliers of high leverage, consider only the curvature values that are in the
        # range +/- (sigma * standard deviation), where the (sample) standard deviation is that of the curvature values,
        # and sigma is a multiplier chosen by the user for the specific contour
        curvature_range = sigma * np.std(curvatures, ddof=1)

        # Create color map for curvature
        norm = colors.Normalize(vmin=-curvature_range, vmax=curvature_range, clip=True)

        fig = plt.figure() if figsize is None else plt.figure(figsize=figsize)
        im = plt.imshow(np.roll(curvatures.T, -start, axis=0), aspect='auto', origin='lower', cmap='jet', norm=norm,
                        extent=(0, curvatures.shape[0] * self.timings, 0, curvatures.shape[1] * self.px_xlen))
        plt.xlabel('Time (s)')
        plt.ylabel('Arc Length Position on Spline (nm)')
        plt.colorbar(im)

        # Add horizontal line
        if hline is not None:
            y_line = ((hline - start) * self.px_xlen) % (curvatures.shape[1] * self.px_xlen)
            plt.hlines(y_line, 0, curvatures.shape[0] * self.timings, linestyles='dashed')

        print(f'\nWriting kymograph... (Output Images/Kymographs/{file_name}_{which}_curv.png)', end=' ')
        plt.savefig(f'Output Images/Kymographs/{file_name}_{which}_curv.png')
        print('(Done)')

    def create_velocity_kymograph(self, which, file_name, sampling_interval=1, start=100, hline=None, sigma=1.5):
        """TODO: add docstring"""

        # Get the desired velocities
        velocities = (np.array(self.displacements_list[which]) / self.timings).T[::sampling_interval].T

        # Since there may be outliers of high leverage, consider only the velocity values that are in the
        # range +/- (sigma * standard deviation), where the (sample) standard deviation is that of the velocity values,
        # and sigma is a multiplier chosen by the user for the specific contour
        velocity_range = sigma * np.std(velocities, ddof=1)

        # Create color map for velocity
        norm = colors.Normalize(vmin=-velocity_range, vmax=velocity_range, clip=True)

        fig = plt.figure()
        im = plt.imshow(np.roll(velocities.T, -start, axis=0), aspect='auto', origin='lower', cmap='jet', norm=norm,
                        extent=(0, velocities.shape[0] * self.timings, 0, velocities.shape[1] * self.px_xlen))

        plt.xlabel('Time (s)')
        plt.ylabel('Arc Length Position on Spline (nm)')
        plt.colorbar(im)

        # Add horizontal line
        if hline is not None:
            y_line = ((hline - start) * self.px_xlen) % (velocities.shape[1] * self.px_xlen)
            plt.hlines(y_line, 0, velocities.shape[0] * self.timings, linestyles='dashed')

        print(f'\nWriting kymograph... (Output Images/Kymographs/{file_name}_{which}_vel.png)', end=' ')
        plt.savefig(f'Output Images/Kymographs/{file_name}_{which}_vel.png')
        print('(Done)')

    def create_curvature_velocity_plot(self, which, point, file_name, d_velocity_approx=0, averages=False,
                                       secondary_map=False):
        """TODO: add docstring"""

        plt.figure()

        # Calculate velocities and curvatures
        velocities = self.displacements_list[which] / self.timings * self.px_xlen  # nm / s
        if d_velocity_approx:
            vel_inst = velocities[:-d_velocity_approx]
            vel_next = np.roll(velocities, d_velocity_approx)[d_velocity_approx:]
            velocities = (vel_inst + vel_next) / (d_velocity_approx + 1)
        curvatures = np.array(self.curvatures_list[which])[:-(1 + d_velocity_approx)]

        if averages:
            avg_curvature = np.mean(curvatures, axis=0)
            avg_velocity = np.mean(velocities, axis=0)
            plt.xlim(np.min(avg_curvature), np.max(avg_curvature))

            plt.title(f'Points on Contour {which} across Time')
            plt.xlabel('Average Curvature')
            plt.ylabel(r'Average Velocity ($\mathrm{nm}\;\mathrm{s}^{-1}$)')
            plt.scatter(avg_curvature, avg_velocity, marker='.', color='#000000', label='Spline data', s=2)

            # Linear fit
            degree = 1
            fit, cvm = np.polyfit(avg_curvature, avg_velocity, degree, cov=True)
            dfit = [np.sqrt(cvm[i, i]) for i in range(2)]
            fitted = fit[0] * avg_curvature + fit[1]

            endpoint_curvatures = np.r_[np.min(avg_curvature),
                                        avg_curvature, np.max(avg_curvature)]
            fitted_max = (fit[0] + dfit[0]) * endpoint_curvatures + (fit[1] + dfit[1])
            fitted_min = (fit[0] - dfit[0]) * endpoint_curvatures + (fit[1] - dfit[1])

            # Calculate coefficient of determination
            p = np.poly1d(fit)
            y_hat = p(avg_curvature)
            y_bar = np.mean(avg_velocity)
            ss_reg = np.sum((y_hat - y_bar) ** 2)
            ss_tot = np.sum((avg_velocity - y_bar) ** 2)
            r_sq = ss_reg / ss_tot

            # Plot linear fit
            plt.plot(avg_curvature, fitted, color='orange', label=f'Linear fit: $R^2 = {r_sq:0.5f}$')

            print(f'\nSaving average curvature/velocity plot of contour {which}...'
                  f' (Output Images/Curvature with Velocity/{file_name}_{which}_avg.png)...', end=' ')
        else:
            plt.xlim(np.min(curvatures[:, point]), np.max(curvatures[:, point]))

            plt.title(f'Point {point} of Contour {which} across Time')
            plt.xlabel('Curvature')
            plt.ylabel(r'Velocity ($\mathrm{nm}\;\mathrm{s}^{-1}$)')
            plt.scatter(curvatures[:, point], velocities[:, point], marker='.', color='#000000', label='Spline data')

            # Linear fit
            degree = 1
            fit, cvm = np.polyfit(curvatures[:, point], velocities[:, point], degree, cov=True)
            dfit = [np.sqrt(cvm[i, i]) for i in range(2)]
            fitted = fit[0] * curvatures[:, point] + fit[1]

            endpoint_curvatures = np.r_[np.min(curvatures[:, point]), curvatures[:, point],
                                        np.max(curvatures[:, point])]
            fitted_max = (fit[0] + dfit[0]) * endpoint_curvatures + (fit[1] + dfit[1])
            fitted_min = (fit[0] - dfit[0]) * endpoint_curvatures + (fit[1] - dfit[1])

            # Calculate coefficient of determination
            p = np.poly1d(fit)
            y_hat = p(curvatures[:, point])
            y_bar = np.mean(velocities[:, point])
            ss_reg = np.sum((y_hat - y_bar) ** 2)
            ss_tot = np.sum((velocities[:, point] - y_bar) ** 2)
            r_sq = ss_reg / ss_tot

            # Plot linear fit
            plt.plot(curvatures[:, point], fitted, color='orange', label=f'Linear fit: $R^2 = {r_sq:0.5f}$')

            print(f'\nSaving curvature/velocity plot of contour {which} at point {point}/{int(1 / self.increment)}...'
                  f' (Output Images/Curvature with Velocity/{file_name}_{which}_{point}.png)...', end=' ')

        plt.fill_between(endpoint_curvatures, fitted_min, fitted_max, alpha=0.2, color='orange',
                         label='Confidence interval at 1-sigma')
        plt.legend()

        file_path = f'Output Images/Curvature with Velocity/{file_name}_{which}_avg.png' if averages \
            else f'Output Images/Curvature with Velocity/{file_name}_{which}_{point}.png'
        plt.savefig(file_path, dpi=300)
        print('(Done)')

    # -------------------------------- Mapping Methods --------------------------------

    def generate_LSSD_map(self, coords_list):
        """Generated a least sum square distance map between points of successive splines. TODO: finish docstring"""

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

    def load_map(self):
        """TODO: add docstring"""

        # TODO: save map in a cache file from which to load

        pass

    def generate_NV_secondary_map(self):
        """TODO: add docstring"""

        # Calculate derivatives of spline, which have the same length
        coords_list_d1 = self.evaluate_splines(increment=self.increment, der=1)

        # Iterate through the splines, changing their coordinate system for every point
        for which, coords in enumerate(self.coords_list):
            secondary_maps = []
            for frame, curr_coord in enumerate(coords[:-1]):
                secondary_map = []

                # Calculate next coordinates
                next_coord = coords[frame + 1]

                # Find the closes points to the normal vector
                tangent_vectors = coords_list_d1[which][frame]  # normalization is trivial since arc tangent is computed
                for i, tangent_vector in enumerate(tangent_vectors.T):  # remove [::200]
                    # Calculate rotation angle
                    theta = np.arctan2(tangent_vector[1], tangent_vector[0])

                    # Rotate coordinates
                    rotated_next_coord = rotate_coord(next_coord, theta)  # list of rotated next coordinates
                    rotated_curr_coord_pair = rotate_coord(curr_coord[:, i], theta)  # rotated current coordinate

                    # rotated_curr_coord = rotate_coord(curr_coord, theta)  # remove

                    # test_rotation(rotated_next_coord, rotated_curr_coord_pair, tangent_vector, theta)  # remove

                    # Calculate square x-differences of candidate points from current point
                    dx_sq = (rotated_next_coord[0] - rotated_curr_coord_pair[0]) ** 2
                    dy_sq = (rotated_next_coord[1] - rotated_curr_coord_pair[1]) ** 2

                    # Select the kth smallest square x-differences
                    f = 100  # factor of sample points to be selected
                    k = int(1 / (self.increment * f))
                    idx = np.argpartition(dx_sq, k)[:k]  # indices of the first k smallest points

                    kmeans = False
                    if kmeans:
                        rotated_next_coord_candidates = rotated_next_coord.T[idx]

                        dy_sq_candidates = dy_sq[idx]
                        dy_sq_candidates_dict = {dy_sq[i]: i for i in idx}

                        # Partition the smallest square x-difference points into two clusters
                        # TODO: add algorithm to choose whether to use 1 or 2 clusters
                        clustering_labels = KMeans(n_clusters=2).fit_predict(rotated_next_coord_candidates)
                        index_cluster_1 = index_cluster_2 = []
                        for j, l in enumerate(clustering_labels):
                            if l:
                                index_cluster_1.append(dy_sq_candidates_dict[dy_sq_candidates[j]])
                            else:
                                index_cluster_2.append(dy_sq_candidates_dict[dy_sq_candidates[j]])

                        # Calculate mean square y-difference for each cluster, selecting the one with minimum difference
                        mean_dy_sq_cluster_1 = np.mean(dy_sq[index_cluster_1])
                        mean_dy_sq_cluster_2 = np.mean(dy_sq[index_cluster_2])
                        index_cluster = index_cluster_1 if mean_dy_sq_cluster_1 < mean_dy_sq_cluster_2 \
                            else index_cluster_2

                        index_cluster_dict = {dx_sq[i]: i for i in index_cluster}

                        # Finally, retrieve minimum index
                        min_index = index_cluster_dict[np.min(dx_sq[index_cluster])]
                        secondary_map.append(min_index)
                    else:
                        # Partition these points into two classes: ones with high y-differences, and with low
                        # y-differences
                        dy_sq_candidates = dy_sq[idx]
                        idy = np.argwhere(dy_sq < np.mean(dy_sq_candidates)).flatten()

                        # Remove the indices that correspond to the highest (above mean) squared y-differences, keeping
                        # the order of the indices such that they correspond to the
                        idxy = np.intersect1d(idx, idy)
                        dict_dx_sq_candidates = {dx_sq[i]: i for i in idxy}
                        idxy = [dict_dx_sq_candidates[i] for i in np.sort(dx_sq[idxy])]  # sort indices

                        secondary_map.append(idxy[0])
                secondary_maps.append(secondary_map)
                print("#")
            self.secondary_maps_list.append(np.array(secondary_maps))

    def apply_secondary_map(self, coords_list):
        """TODO: add docstring"""

        # WARNING: secondary maps are non-bijective

        # Apply the map for each pair (x, y) where x and y are lists respectively
        # TODO: figure out, maybe, if the next contour is mapping onto the current instead of the opposite
        new_coords_list = []
        for i, coords in enumerate(coords_list):
            new_coords = [coords[0]]
            for frame, coord in enumerate(coords[1:]):
                new_coord = [coord[:, new_index] for new_index in self.secondary_maps_list[i][frame]]  # apply map
                new_coords.append(np.transpose(new_coord))
            new_coords_list.append(np.array(new_coords))

        return new_coords_list

    def load_secondary_map(self):
        """TODO: add docstring"""
        pass
