import cv2 as cv

import image as im
import pipeline as pl

# Define data
PATH_375nm = 'Images/POPCPG8_375nm_epi.tiff'
TIME_375nm = 25.85507
PATH_750nm = 'Images/POPCPG6_750nm_epi.tiff'
TIME_750nm = 23.81818

# Create initial stacks
stack_375nm = im.Stack(PATH_375nm, timings=TIME_375nm)
stack_750nm = im.Stack(PATH_750nm, timings=TIME_750nm)

pl.test(stack_375nm)
stack_375nm.print_info()
im.play(stack_375nm.last())

