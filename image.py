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
	ndarray
	    Image with inverted color channels.
	"""

    return cv.cvtColor(img, cv.COLOR_RGB2BGR)


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
	stack : ndarray
	    Initial stack of images.
	frames : int
	    Number of frames in the initial stack.
	xsize : int
	    Number of pixels in the x-direction of the images in the stack.
	ysize : int
	    Number of pixels in the y-direction of the images in the stack.
	stacks : list of ndarray
	    Contains the image stacks at each stage in the pipeline.
	"""

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

    def grayscale(self, append=True):
        """Returns the last stack in grayscale using the linear NTSC method, and adds the stack to the pipeline history.
        If `append` is false, returns the first stack in grayscale using the linear NTSC method.

        Parameters
        ----------
        append : bool, default True
            If true processes the last stack in the stack pipeline and appends to it, if false processes the initial
             stack.

		Returns
		-------
		gray : ndarray
		    Grayscale copy of the initial stack, now also the last element in the `stacks` list.
		"""

        if append:
            # Assume the images are RGB
            assert len(self.stacks[-1].shape) == 4, f'Last stack in pipeline at position {len(self.stacks)} should ' \
                                                    f'be composed of RGB images'

            # Image in grayscale using the linear NTSC method
            gray = np.copy(self.stacks[-1])[:, :, :, 0]
            for i, frame in enumerate(self.stacks[-1]):
                gray[i] = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            # Add grayscale image to the stack pipeline
            self.stacks.append(gray)

            return gray
        else:
            # Assume the images are RGB
            assert len(self.stack.shape) == 4, 'Input stack should be composed of RGB images'

            # Image in grayscale using the linear NTSC method
            gray = np.copy(self.stack)[:, :, :, 0]
            for i, frame in enumerate(self.stack):
                gray[i] = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            return gray

    def threshold(self, min_, max_, append=True):
        pass

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
        ndarray
            Last image in the stack pipeline, which has been now treated with `func`.
        """

        if append:
            # Process each frame in the last stack of the pipeline history
            processed = np.copy(self.stacks[-1])
            for i, frame in enumerate(self.stacks[-1]):
                processed[i] = func(frame)

            # Add it to the pipeline history
            self.stacks.append(processed)

            return processed
        else:
            # Process each frame of the original stack
            processed = np.copy(self.stack)
            for i, frame in enumerate(self.stack):
                processed[i] = func(frame)

            return processed
