import cv2 as cv

import image as im
import pipeline as pl

# Define data
path_375nm = 'Images/POPCPG8_375nm_epi.tiff'
time_375nm = 25.85507
path_750nm = 'Images/POPCPG6_750nm_epi.tiff'
time_750nm = 23.81818

# Create initial stacks
stack_375nm = im.Stack(path_375nm, timings=time_375nm)
stack_750nm = im.Stack(path_750nm, timings=time_750nm)

pl.test(stack_375nm)
stack_375nm.print_info()
im.play(stack_375nm.last())

