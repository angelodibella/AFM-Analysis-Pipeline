import cv2 as cv

import image as im
import pipeline as pl

stack_375nm = im.Stack('Images/POPCPG8_375nm_epi.tiff')
stack_750nm = im.Stack('Images/POPCPG6_750nm_epi.tiff')


def test_pipeline():
    cv.imshow('', im.icc(stack_375nm.stack[-1]))
    cv.waitKey(0)

    cv.imshow('', im.icc(stack_375nm.stacks[-1][-1]))
    cv.waitKey(0)

    cv.imshow('', im.icc(stack_375nm.grayscale(append=False)[-1]))
    cv.waitKey(0)

    cv.imshow('', im.icc(stack_375nm.stacks[-1][-1]))
    cv.waitKey(0)

    cv.imshow('', im.icc(stack_375nm.grayscale()[-1]))
    cv.waitKey(0)

    cv.imshow('', im.icc(stack_375nm.stacks[-1][-1]))
    cv.waitKey(0)

    cv.imshow('', im.icc(stack_375nm.stacks[0][-1]))
    cv.waitKey(0)

    print('Pipeline length:', len(stack_375nm.stacks))
