import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import image as im
import pipeline as pl
import process as ps

# Save output?
save = input('Save output files? [y/n] ').lower() == 'y'

# Define directories
OUT_DIR = 'Output Images/'
STACK_OUT_DIR = 'Output Images/Stack Pipelines/'

# -------------------- Data --------------------

TOT_AREA = 4e6  # nanometers squared

# Define data for 375 nM epidermicin topography
PATH_375nM = 'Images/POPCPG8_375nM_epi.tiff'
TIME_375nM = 25.85507  # seconds
INJ_FRAMES_375nM = [8, 29, 45, 60]  # From timings data

# Define data for 750 nM epidermicin topography
PATH_750nM = 'Images/POPCPG6_750nM_epi.tiff'
TIME_750nM = 23.81818  # seconds
INJ_FRAMES_750nM = [3]

# ----------------------------------------------

# Assuming pixels represent square areas
px_len = np.sqrt(TOT_AREA) / 512

# Create initial stacks, models for further processing
stack_375nM = im.Stack(PATH_375nM, timings=TIME_375nM, px_xlen=px_len)
stack_750nM = im.Stack(PATH_750nM, timings=TIME_750nM, px_xlen=px_len)

# Use `thresh_1` pipeline
thin_375nM = stack_375nM.copy()
pl.thin_defects(thin_375nM)

# Use `thresh_2` pipeline
transmembrane_375nM = stack_375nM.copy()
pl.transmembrane_defects(transmembrane_375nM)

# Save pipeline info
if save:
    thin_375nM.save(STACK_OUT_DIR, 'thin_defects_375nM')
    transmembrane_375nM.save(STACK_OUT_DIR, 'transmembrane_defects_375nM')

# Calculate average membrane height from grayscale stack, using the image data before the first injection
mem_375nM = stack_375nM.grayscale(append=False)[:INJ_FRAMES_375nM[0]]
mean_mem_375nM = np.mean(mem_375nM)
stdev_mem_375nM = np.std(mem_375nM, ddof=1)

mem_750nM = stack_750nM.grayscale(append=False)[:INJ_FRAMES_750nM[0]]
mean_mem_750nM = np.mean(mem_750nM)
stdev_mem_750nM = np.std(mem_750nM, ddof=1)

print(f'\n------------- Membrane Gray Value -------------')
print(f'\t\tMean\t\t\t\tStandard Deviation')
print(f'375 nM\t{mean_mem_375nM}\t{stdev_mem_375nM}')
print(f'750 nM\t{mean_mem_750nM}\t{stdev_mem_750nM}')

#################################
adj_contours_thin_375nM = ps.adjust_contours(thin_375nM.contours[-1])[10][11].T


# plt.figure(figsize=(6, 6))
# plt.scatter(adj_contours_thin_375nM[0], adj_contours_thin_375nM[1], marker='s', s=200)
# plt.gca().invert_yaxis()  # coordinates in original stacks are +x (right) +y (down), i.e., origin is upper left
# plt.show()
#################################




