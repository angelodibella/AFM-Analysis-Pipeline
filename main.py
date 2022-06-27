import cv2 as cv

import image as im
import pipeline as pl

stack_375nm = im.Stack('Images/POPCPG8_375nm_epi.tiff', timings=25.85507)
stack_750nm = im.Stack('Images/POPCPG6_750nm_epi.tiff', timings=23.81818)

coeffs = [0.34, 0.66, 0.0]
cv.imshow('', im.icc(stack_375nm.grayscale(coeffs=coeffs, append=False)[-5]))
cv.waitKey(0)

