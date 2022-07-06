import cv2 as cv
import numpy as np

import image as im


def BNCS_alg(matches, distances):
    """TODO: add docstring"""
    # TODO: implement best next contour selection algorithm; MUST be reproducible and stable under edge cases
    pass


def track_contour(stack: im.Stack, contour_loc: tuple, which=-1, end_frame=-1) -> list:
    """TODO: add docstring, explain how this uses image (Hu) moments
    contour_loc must be (frame, contour)
    """

    # Get preliminary contours properties
    contours_list = stack.contours[which]
    adj_contours_list = im.adjust_contours(contours_list)
    centers_list = stack.get_contour_centers()

    # Iterate through each frame, starting from the frame succeeding the initial contour
    pointer_indeces = [contour_loc]
    for next_frame, next_contours in enumerate(contours_list[contour_loc[0] + 1:end_frame + 1]):
        # Set up current contour
        curr_frame, curr_index = pointer_indeces[-1]
        curr_contour = contours_list[curr_frame][curr_index]
        adj_curr_contour = adj_contours_list[curr_frame][curr_index]
        center_curr_contour = centers_list[curr_frame][curr_index]

        # Calculate region of interest (ROI)
        # TODO: restrict the search area to a definite region of interest using the extreme points of the current
        #       contour, such that ROI(next_contours) <- next_contours in the following `for` loop

        # Iterate through each contour in the next frame
        matches = []
        distances = []
        for next_index, next_contour in enumerate(next_contours):
            # Set up next contour
            adj_next_contour = adj_contours_list[next_frame][next_index]
            center_next_contour = centers_list[next_frame][next_index]

            # Calculate the matching parameter using image moments and an overlap function (cf. cv::ShapeMatchModes)
            matches.append(cv.matchShapes(curr_contour, next_contour, cv.CONTOURS_MATCH_I1, 0.0))

            # Calculate distance from current to next candidate contour
            distances.append(np.abs(np.array(center_curr_contour) - np.array(center_next_contour)))

        # Best next contour selection (BNCS) algorithm
        pointer_indeces.append((next_frame, BNCS_alg(matches, distances)))




