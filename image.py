import copy
import numpy as np
import tifffile as tiff
import cv2 as cv

import track


def icc(img):
    """Invert color channels.

	Parameters
	----------
	img : array_like
	    Single image (grayscale or otherwise).

	Returns
	-------
	np.ndarray
	    Image with inverted color channels.
	"""

    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


def play(frames, fps=20, title=''):
    """TODO: add docstring"""

    # From frames per second to milliseconds per frame
    mspf = int(1000 / fps)

    # Show consecutive images in stack
    keep = True
    while keep:
        for frame in frames:
            cv.imshow(title, icc(frame))
            cv.waitKey(mspf - 25)

            # If keyboard input `q` then stop playing the video, if `w` restart the video
            key = cv.waitKey(25)
            if key & 0xFF == ord('q'):
                keep = False
                break
            elif key & 0xFF == ord('w'):
                break
            elif key & 0xFF == ord('d'):
                new_mspf = int(mspf - 5)
                mspf = new_mspf if new_mspf > 25 else mspf
            elif key & 0xFF == ord('a'):
                mspf = int(mspf + 5)


def to_RGB(stack):
    """TODO: add docstring"""

    if len(stack.shape) == 3:
        return np.array([cv.cvtColor(frame.astype('uint8'), cv.COLOR_GRAY2RGB) for frame in stack])
    else:
        return np.array(stack)


def adjust_contours(contours):
    """TODO: add docstring"""

    new_contours = []
    for stack in contours:
        fixed = []
        for contour in stack:
            fixed.append(contour[:, 0, :])
        new_contours.append(fixed)

    return new_contours


class Stack:
    """Allows for the implementation of an image processing pipeline from an initial stack (TIFF) of images. Each
    method has an optional boolean parameter `append`: if false the initial stack is copied, processed, and returned,
    if true (default) the latest stack in the pipeline is copied, processed, returned, and added as a new member of the
    pipeline history. The pipeline history is stored in the `stacks` attribute, not to be confused with the initial
    stack `stack`.

	Parameters
	----------
	path : str
	    The path to the initial stack of images.
	timings : one-dimensional array_like (same size as the number of frames in the initial stack) or float, default 1
	    Relative time assigned to each frame of the stacks if array-like, otherwise the time interval between frames,
	    all assumed to be in seconds.
	px_xlen : float, default 1
	    Scale length of a pixel in the x-dimension.
	px_ylen : float, optional
	    Scale length of a pixel in the y-dimension. If not specified, takes the same value as `px_xlen`.

	Attributes
	----------
	stack : np.ndarray
	    Initial stack of images.
	frames : int
	    Number of frames in the initial stack.
	xsize : int
	    Number of pixels in the x-direction of the images in the stack.
	ysize : int
	    Number of pixels in the y-direction of the images in the stack.
	stacks : list of np.ndarray
	    Contains the image stacks at each stage in the pipeline.
	contours : list of int
	    Contains the contours retrieved when attempting to find them.
	info : list of str
	    Description of each step in the pipeline.
	tracked : list
	    Individually tracked contours.
	"""

    RETR_MODES = {0: 'external retrieval', 1: 'list retrieval', 2: 'two-level retrieval', 3: 'tree retrieval',
                  4: 'floodfill retrieval'}

    APPROX_METHODS = {1: 'no approximation', 2: 'simple approximation', 3: 'Teh-Chin approximation (L1)',
                      4: 'Teh-Chin approximation (KCOS)'}

    def __init__(self, path, timings=1, px_xlen=1, px_ylen=0):
        self.path = path
        self.timings = timings
        self.px_xlen = px_xlen
        self.px_ylen = px_xlen if px_ylen == 0 else px_ylen

        # Read image
        self.stack = tiff.imread(path)
        self.frames, self.xsize, self.ysize, _ = self.stack.shape

        # Record progress in the pipeline with an array of modified stacks
        # WARNING: list of ndarray objects, shape of stacks might vary throughout it
        self.stacks = [np.copy(self.stack)]

        # Record contours
        self.contours = []

        # Keep track of processes
        self.info = []

        # Keep track of individual contours
        self.tracked = []

    # ------------------- Getter-ish Methods -------------------

    def stack_select(self, append, which=-1):
        """ TODO: add docstring
        """
        return np.copy(self.stacks[which]) if append else np.copy(self.stack)

    def last(self):
        """TODO: add docstring"""
        return self.stacks[-1]

    def length(self):
        """TODO: add docstring"""
        return len(self.stacks)

    def print_info(self, sub_fixture=''):
        """TODO: add docstring"""

        if sub_fixture != '':
            sub_fixture = f"'{sub_fixture}' "
        fixture = f"----------------- Pipeline {sub_fixture}for {self.path.split('/')[-1]} -----------------"
        print('\n' + fixture)
        for i, s in enumerate(self.info):
            print(f'{i + 1}.', s)
        print('-' * len(fixture))

    def get_contours(self, hierarchy=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE, inv=True,
                     coeffs=[0.299, 0.587, 0.114], append=False, color=(255, 255, 255)):
        """TODO: add docstring"""

        # If the image is not greyscale, then make it
        to_process = np.copy(self.stacks[-1])
        if len(self.stacks[-1].shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=False)

        # Invert image
        if inv:
            to_process = 255 - to_process

        # Find contours in each frame
        contours = []
        hiers = []
        out_stack = np.zeros_like(to_process)
        for i, frame in enumerate(to_process):
            contour, hier = cv.findContours(frame, hierarchy, method)
            cv.drawContours(out_stack[i], contour, -1, color, 1)
            contours.append(contour)
            hiers.append(hier)

        if append:
            self.info.append(f'Find contours using {self.RETR_MODES[hierarchy]} with {self.APPROX_METHODS[method]}, '
                             f'and draw them')
            self.stacks.append(out_stack)
            self.contours.append(contours)

        return contours, hiers, out_stack

    def get_contour_data(self, which=-1):
        """TODO: add docstring"""

        assert self.contours, 'Contours have to be detected at least once'

        # TODO: implement `get_contour_data`

    def get_histogram(self, which='FIX', contours=None):
        """TODO: add docstring, should get this histogram inside the contours"""
        pass

    def get_contour_centers(self, which=-1, mask=-1, append=False):
        """TODO: add docstring"""

        # Select which contours to analyze
        contours_list = self.contours[which]

        # If `append`, choose to which stack to draw
        to_draw = to_RGB(self.stacks[mask])

        # Find centers using image moments
        centers_list = []
        for i, contours in enumerate(contours_list):
            centers = []
            for contour in contours:
                moment = cv.moments(contour)
                if moment['m00'] != 0:
                    cx = int(moment['m10'] / moment['m00'])
                    cy = int(moment['m01'] / moment['m00'])

                    # Draw a circle at the center
                    if append:
                        cv.circle(to_draw[i], (cx, cy), 3, (0, 0, 255), -1)

                # FOR DEBUG PURPOSES
                else:
                    cx = cy = -1
                centers.append([cx, cy])
            centers_list.append(np.array(centers))

        # Draw centers
        if append:
            self.stacks.append(to_draw)
            self.info.append(f'Drew centers of contours {which} of stack {mask}')

        return centers_list

    def copy(self):
        """Returns a deep copy of the current stack object.

        Returns
        -------
        Stack
            A deep copy of the current stack object.
        """

        return copy.deepcopy(self)

    def save(self, directory, name, frame_num=0):
        """TODO: add docstring"""

        if frame_num:
            print(f"\nSaving stack at position {frame_num}... ({{directory+name+'.tiff'}})")
            tiff.imwrite(directory + name + '.tiff', self.stacks[frame_num], imagej=True)
        else:
            print(f"\nSaving all stacks... ({directory+name+'.tiff'})")

            # Convert all images to RGB to ensure homogeneity across frames in all stacks
            rgb_stacks = np.array([to_RGB(stack) for stack in self.stacks])

            # Write the image
            tiff.imwrite(directory + name + '.tiff', rgb_stacks, imagej=True, metadata={'axes': 'TCYXS'})

    # ------------------- Setter-ish Methods -------------------

    def add_contours(self, contours, which=-1, append=True, color=(255, 255, 255)):
        """TODO: add docstring"""

        # Select the image to process
        to_process = self.stack_select(append=append, which=which)

        # To RGB to highlight contours
        out_stack = to_RGB(to_process)

        # Draw contours in each frame
        for i, frame in enumerate(out_stack):
            cv.drawContours(out_stack[i], contours[i], -1, color, 1)

        if append:
            self.info.append(f'Drew contours from external source')
            self.stacks.append(out_stack)
            self.contours.append(contours)

        return out_stack

    def add(self, stack, info='Add external stack'):
        """TODO: add docstring"""

        self.stacks.append(stack)
        self.info.append(info)

    # ---------------- Image Processing Methods ----------------

    def grayscale(self, coeffs=[0.299, 0.587, 0.114], append=True):
        """Returns the last stack in grayscale using the linear NTSC method, and adds the stack to the pipeline history.
        If `append` is false, returns the first stack in grayscale using the linear NTSC method.

        Parameters
        ----------
        coeffs : array_like, default [0.299, 0.587, 0.114]
            Three-vector of floats (must sum to 1) with which respectively the red, green, and blue color channels are
            to be weighted when converting to grayscale. The default is the NTSC color formula for best human
            perception.
        append : bool, default True
            If true processes the last stack in the stack pipeline and appends to it, if false processes the initial
            stack.

		Returns
		-------
		gray : np.ndarray
		    Grayscale copy of the selected stack.
		"""

        # Select which stack to process
        to_process = self.stack_select(append)

        # Vectorize color coefficients
        coeffs = np.array(coeffs).reshape((1, 3))

        # Assume the images are RGB
        err_msg = f'Last stack in pipeline at position {len(self.stacks) - 1} must be composed of RGB images' \
            if append else 'Input stack must be composed of RGB images'
        assert len(to_process.shape) == 4, err_msg

        # Image in grayscale using the linear NTSC method
        gray = np.copy(to_process)[:, :, :, 0]
        for i, frame in enumerate(to_process):
            gray[i] = cv.transform(frame, coeffs)

        # Add grayscale stack to the pipeline
        if append:
            self.info.append(f'Convert to 8bit grayscale using RGB weights {coeffs.reshape(3)}')
            self.stacks.append(gray)

        return gray

    def binary_threshold(self, max_val, coeffs=[0.299, 0.587, 0.114], otsu=False, append=True):
        """Thresholds (binary) the last stack in the pipeline and appends to it. If `append` is false, returns the
        first stack treated with a binary threshold.

        Parameters
        ----------
        max_val : int
            Grey value (0-255) below which all pixels in the stack are set to 0 and above which to 255.
        coeffs : array_like, default [0.299, 0.587, 0.114]
            Three-vector of floats (must sum to 1) with which respectively the red, green, and blue color channels are
            to be weighted when converting to grayscale. The default is the NTSC color formula for best human
            perception.
        otsu : bool, default False
            TODO: add parameter description for `otsu`
        append : bool, default True
            If true processes the last stack in the stack pipeline and appends to it, if false processes the initial
            stack.

        Returns
        -------
        thresh : np.ndarray
            Thresholded copy of the selected stack.
        """

        # Select which stack to process
        to_process = self.stack_select(append)

        # If the image is not greyscale, then make it
        if len(to_process.shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=append)

        # Apply the binary threshold (set any pixel exceeding `max_val` to white, and below it to black)
        thresh = np.copy(to_process)
        for i, frame in enumerate(to_process):
            if otsu:
                _, thresh[i] = cv.threshold(frame, max_val, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            else:
                _, thresh[i] = cv.threshold(frame, max_val, 255, cv.THRESH_BINARY)

        # Add thresholded stack to the pipeline
        if append:
            self.info.append(f'Apply a binary threshold')
            self.info[-1] += ' using the Otsu method' if otsu else f' with maximum value {max_val}'
            self.stacks.append(thresh)

        return thresh

    def adaptive_gaussian_threshold(self, block_size, c, coeffs=[0.299, 0.587, 0.114], append=True):
        """Thresholds (adaptive Gaussian) the last stack in the pipeline and appends to it. If `append` is false,
        returns the first stack treated with a binary threshold.

        Parameters
        ----------
        block_size : int
            Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        c : float
            Constant subtracted from the mean or weighted mean.
        coeffs : array_like, default [0.299, 0.587, 0.114]
            Three-vector of floats (must sum to 1) with which respectively the red, green, and blue color channels are
            to be weighted when converting to grayscale. The default is the NTSC color formula for best human
            perception.
        append : bool, default=True
            If true processes the last stack in the stack pipeline and appends to it, if false processes the initial
            stack.

        Returns
        -------
        thresh : np.ndarray
            Thresholded copy of the selected stack.
        """

        # Select which stack to process
        to_process = self.stack_select(append)

        # If the image is not greyscale, then make it
        if len(to_process.shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=append)

        # Apply the binary threshold (set any pixel exceeding `max_val` to white, and below it to black)
        thresh = np.copy(to_process)
        for i, frame in enumerate(to_process):
            thresh[i] = cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, c)

        # Add thresholded stack to the pipeline
        if append:
            self.info.append(f'Apply a binary adaptive Gaussian threshold with block size {block_size} and mean '
                             f'offset {c}')
            self.stacks.append(thresh)

        return thresh

    def gaussian_blur(self, ker, sigma, coeffs=[0.299, 0.587, 0.114], append=True):
        """TODO: add docstring"""

        # Select which stack to process
        to_process = self.stack_select(append)

        # If the image is not greyscale, then make it
        if len(to_process.shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=append)

        # Apply the binary threshold (set any pixel exceeding `max_val` to white, and below it to black)
        gauss = np.copy(to_process)
        for i, frame in enumerate(to_process):
            gauss[i] = cv.GaussianBlur(frame, ker, sigma)

        # Add thresholded stack to the pipeline
        if append:
            self.info.append(f'Apply Gaussian blur with kernel shape {ker} and sigma {sigma}')
            self.stacks.append(gauss)

        return gauss

    def median_blur(self, ksize, coeffs=[0.299, 0.587, 0.114], append=True):
        """TODO: add docstring"""

        # Select which stack to process
        to_process = self.stack_select(append)

        # If the image is not greyscale, then make it
        if len(to_process.shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=append)

        # Apply the binary threshold (set any pixel exceeding `max_val` to white, and below it to black)
        gauss = np.copy(to_process)
        for i, frame in enumerate(to_process):
            gauss[i] = cv.medianBlur(frame, ksize)

        # Add thresholded stack to the pipeline
        if append:
            self.info.append(f'Apply median blur with kernel size {ksize}')
            self.stacks.append(gauss)

        return gauss

    def canny(self, low, high, coeffs=[0.299, 0.587, 0.114], otsu=True, append=True):
        """TODO: add docstring"""

        # Select which stack to process
        to_process = self.stack_select(append)

        # If the image is not greyscale, then make it
        if len(to_process.shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=append)

        # Apply canny edge detector
        canny = np.copy(to_process)
        for i, frame in enumerate(to_process):
            # Find Otsu threshold values
            if otsu:
                high, _ = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                low = 0.5 * high
            canny[i] = cv.Canny(frame, low, high)

        # Add edge-detected stack to the pipeline
        if append:
            self.info.append(f'Apply Canny edge detector using')
            if otsu:
                self.info[-1] += ' Otsu threshold values (different for each frame)'
            else:
                self.info[-1] += f' threshold values from {low} to {high}'
            self.stacks.append(canny)

        return canny

    def intensity_band(self, low, high, which=-1, coeffs=[0.299, 0.587, 0.114], binary=False, otsu=False, append=True):
        """TODO: add docstring"""

        # Select which stack to process
        to_process = self.stack_select(append, which=which)

        # If the image is not greyscale, then make it
        if len(to_process.shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=append)

        if otsu:
            # If Otsu is chosen, dynamic updating of the threshold values will occur; the `low` and `high` variables
            # now serve as factors that be multiplied respectively to these updating values
            band = np.copy(to_process)
            for i, frame in enumerate(to_process):
                otsu_high, _ = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                otsu_low = 0.5 * otsu_high
                band[i] = np.where((otsu_low * low <= frame) & (frame <= otsu_high * high), frame, 255)
        else:
            # Filter all pixels which are outside the range [low, high] by setting them to white
            band = np.where((low <= to_process) & (to_process <= high), to_process, 255)

        # Turn the band to binary
        if binary:
            band = np.where(band < 255, 0, 255).astype('uint8')

        if append:
            self.stacks.append(band)
            if which == -1:
                self.info.append(f'Filtered intensity band in range [{low}, {high}]')
            else:
                self.info.append(f'Filtered intensity band in range [{low}, {high}] of stack {which + 1}')

            if binary:
                self.info[-1] += ' and converted it to binary'

        return band

    def track_contour(self, contour_loc, which=-1, append=True):
        """TODO: add docstring"""

        # Get index path
        pointers = track.track_contour(self, contour_loc, which=which)

        # Get contours to be used
        all_contours = self.contours[which][contour_loc[0]:]
        similar_contours = []
        for frame, contours in enumerate(all_contours):
            similar_contours.append(contours[pointers[frame][1]])

        if append:
            # Draw contour over time
            images = np.zeros_like(self.stack)
            for frame, image in enumerate(images[contour_loc[0]:]):
                cv.drawContours(images[frame + contour_loc[0]], similar_contours[frame], -1, (255, 255, 255), 1)

            self.stacks.append(images)
            self.info.append(f'Tracked contours {which} from {contour_loc[0]}')
            self.tracked.append(similar_contours)

        return similar_contours

    def apply(self, func, info='', append=True, **kwargs):
        """Apply external processing function to the last stack in the stack pipeline and add it to the pipeline
        history. If `append` is false, apply external processing function to the first stack in the stack pipeline.

        Parameters
        ----------
        func : callable
            The function that is to be applied to each frame of the last stack in the pipeline.
        info : str, optional
            Information of the function to be stored in the pipeline description.
        append : bool, default True
            If true processes the last stack in the stack pipeline and appends to it, if false processes the initial
             stack.
        **kwargs : optional
            Additional parameters to be passed to `func`.

        Returns
        -------
        np.ndarray
            Last image in the stack pipeline, which has been now treated with `func`.
        """

        # Process each frame of the original stack
        to_process = self.stack_select(append)
        processed = np.copy(to_process)
        for i, frame in enumerate(to_process):
            processed[i] = func(frame, **kwargs)

        # Add it to the pipeline history
        if append:
            if info:
                self.info.append(info)
            else:
                self.info.append(f'Apply function {func.__name__} to each frame')
            self.stacks.append(processed)

        return processed

    # ----------------- Image Analysis Methods -----------------




