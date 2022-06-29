import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import image as im
import pipeline as pl

# Define directories
OUT_DIR = 'Output Images/'
STACK_OUT_DIR = 'Output Images/Stack Pipelines/'

# -------------------- Data --------------------

TOT_AREA = 4e6  # nanometers squared

# Define data for 375 nM epidermicin topography
PATH_375nM = 'Images/POPCPG8_375nM_epi.tiff'
TIME_375nM = 25.85507  # seconds
INJ_FRAMES_375nM = [7, 29, 45, 60]

# Define data for 750 nM epidermicin topography
PATH_750nM = 'Images/POPCPG6_750nM_epi.tiff'
TIME_750nM = 23.81818  # seconds
INJ_FRAMES_750nM = [2]

# ----------------------------------------------

# Assuming pixels represent square areas
px_len = np.sqrt(TOT_AREA) / 512

# Create initial stacks
stack_375nM = im.Stack(PATH_375nM, timings=TIME_375nM, px_xlen=px_len)
stack_750nM = im.Stack(PATH_750nM, timings=TIME_750nM, px_xlen=px_len)

thresh_375nM = stack_375nM.copy()

contours_thresh = pl.otsu_1(thresh_375nM)
thresh_375nM.print_info()

# im.play(copy_375nM.last())
# im.play(stack_375nM.last())

thresh_375nM.save(STACK_OUT_DIR, 'otsu_1_375nM')

# plt.figure()
# contour_numbers = [len(frame_cont) for frame_cont in contours]
# frames = np.arange(stack_375nM.frames)
# plt.plot(frames, contour_numbers)
# plt.show()
