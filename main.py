import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import image as im
import pipeline as pl

TOT_AREA = 4e3  # nanometers squared

# Define data for 375 nM epidermicin topography
PATH_375nM = 'Images/POPCPG8_375nM_epi.tiff'
TIME_375nM = 25.85507  # seconds

# Define data for 750 nM epidermicin topography
PATH_750nM = 'Images/POPCPG6_750nM_epi.tiff'
TIME_750nM = 23.81818  # seconds

# Create initial stacks
stack_375nM = im.Stack(PATH_375nM, timings=TIME_375nM)
stack_750nM = im.Stack(PATH_750nM, timings=TIME_750nM)

contours = pl.test(stack_375nM)
stack_375nM.print_info()
im.play(stack_375nM.last())

plt.figure()
contour_numbers = [len(frame_cont) for frame_cont in contours]
frames = np.arange(stack_375nM.frames)
plt.plot(frames, contour_numbers)
plt.show()
