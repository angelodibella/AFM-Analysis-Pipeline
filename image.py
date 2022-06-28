import numpy as np
import tifffile as tiff
import cv2 as cv


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
	info : list of str
	    Description of each step in the pipeline.
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

        # Keep track of processes
        self.info = []

    # ---------------- Extraction Methods ----------------

    def stack_select(self, append):
        """ TODO: add docstring
        """
        return self.stacks[-1] if append else self.stack

    def last(self):
        """TODO: add docstring"""
        return self.stacks[-1]

    def length(self):
        """TODO: add docstring"""
        return len(self.stacks)

    def print_info(self, sub_fixture=''):
        """TODO: add docstring"""

        if sub_fixture != '':
            sub_fixture += ' '
        fixture = f"----------------- Pipeline {sub_fixture}for {self.path.split('/')[-1]} -----------------"
        print('\n' + fixture)
        for i, s in enumerate(self.info):
            print(f'{i + 1}.', s)
        print('-' * len(fixture))

    def get_contours(self, hierarchy=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE, coeffs=[0.114, 0.587, 0.299],
                     append=False):
        """TODO: add docstring"""

        # If the image is not greyscale, then make it
        to_process = np.copy(self.stacks[-1])
        if len(self.stacks[-1].shape) == 4:
            to_process = self.grayscale(coeffs=coeffs, append=False)

        # Find contours in each frame
        contours = []
        out_stack = np.zeros_like(to_process)
        for i, frame in enumerate(to_process):
            contour, _ = cv.findContours(frame, hierarchy, method)
            cv.drawContours(out_stack[i], contour, -1, (255, 255, 255), 1)
            contours.append(contour)

        if append:
            self.info.append(f'Find contours using {self.RETR_MODES[hierarchy]} with {self.APPROX_METHODS[method]}, '
                             f'and draw them')
            self.stacks.append(out_stack)

        return contours, out_stack

    # ----------------------------------------------------

    def grayscale(self, coeffs=[0.114, 0.587, 0.299], append=True):
        """Returns the last stack in grayscale using the linear NTSC method, and adds the stack to the pipeline history.
        If `append` is false, returns the first stack in grayscale using the linear NTSC method.

        Parameters
        ----------
        coeffs : array_like, default [0.114, 0.587, 0.299]
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

    def binary_threshold(self, max_val, coeffs=[0.114, 0.587, 0.299], otsu=False, append=True):
        """Thresholds (binary) the last stack in the pipeline and appends to it. If `append` is false, returns the
        first stack treated with a binary threshold.

        Parameters
        ----------
        max_val : int
            Grey value (0-255) below which all pixels in the stack are set to 0 and above which to 255.
        coeffs : array_like, default [0.114, 0.587, 0.299]
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

    def adaptive_gaussian_threshold(self, block_size, c, coeffs=[0.114, 0.587, 0.299], append=True):
        """Thresholds (adaptive Gaussian) the last stack in the pipeline and appends to it. If `append` is false,
        returns the first stack treated with a binary threshold.

        Parameters
        ----------
        block_size : int
            Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        c : float
            Constant subtracted from the mean or weighted mean.
        coeffs : array_like, default [0.114, 0.587, 0.299]
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

    def gaussian_blur(self, ker, sigma, coeffs=[0.114, 0.587, 0.299], append=True):
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

    def median_blur(self, ksize, coeffs=[0.114, 0.587, 0.299], append=True):
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

    def canny(self, low, high, coeffs=[0.114, 0.587, 0.299], otsu=True, append=True):
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

    def apply(self, func, append=True):
        """Apply external processing function to the last stack in the stack pipeline and add it to the pipeline
        history. If `append` is false, apply external processing function to the first stack in the stack pipeline.

        Parameters
        ----------
        func : callable
            The function that is to be applied to each frame of the last stack in the pipeline.
        append : bool, default True
            If true processes the last stack in the stack pipeline and appends to it, if false processes the initial
             stack.

        Returns
        -------
        np.ndarray
            Last image in the stack pipeline, which has been now treated with `func`.
        """

        # Process each frame of the original stack
        to_process = self.stack_select(append)
        processed = np.copy(to_process)
        for i, frame in enumerate(to_process):
            processed[i] = func(frame)

        # Add it to the pipeline history
        if append:
            self.info.append(f'Apply function {func.__name__} to each frame')
            self.stacks.append(processed)

        return processed
