import cv2 as cv

import image as im
import pipeline as pl

stack_375nm = im.Stack('Images/POPCPG8_375nm_epi.tiff', timings=25.85507)
stack_750nm = im.Stack('Images/POPCPG6_750nm_epi.tiff', timings=23.81818)

pl.test(stack_375nm)
# im.play(stack_375nm.last())
stack_375nm.print_info()
