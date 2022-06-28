import cv2 as cv

import image as im
import pipeline as pl

stack_375nm = im.Stack('Images/POPCPG8_375nm_epi.tiff', timings=25.85507)
stack_750nm = im.Stack('Images/POPCPG6_750nm_epi.tiff', timings=23.81818)

coeffs = [0.34, 0.66, 0.0]
block_size = 11
c = 2

im.play(stack_375nm.stack)

# cv.imshow('', stack_375nm.median_blur(9, coeffs=coeffs)[-5])
# print(len(stack_375nm.stacks))
# cv.waitKey(0)
#
# cv.imshow('', stack_375nm.binary_threshold(0, otsu=True)[-5])
# print(len(stack_375nm.stacks))
# cv.waitKey(0)

