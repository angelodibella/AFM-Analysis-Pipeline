import cv2 as cv
import numpy as np


def BNCS_alg(matches, distances, weight_powers=(2, 1.6)):
    """TODO: add docstring"""

    # To numpy arrays
    matches = np.copy(matches)
    distances = np.copy(distances)

    # Retrieve the indices of the best match (minimum value, in this case) and the least distance between centers
    min_match_index = np.argmin(matches)
    min_distance_index = np.argmin(distances)

    # Find the best, closest match
    if min_match_index == min_distance_index:
        return min_match_index
    else:
        return np.argmin(matches ** weight_powers[0] + distances ** weight_powers[1])


def track_contour(stack, contour_loc: tuple, which=-1) -> list:
    """TODO: add docstring, explain how this uses image (Hu) moments
    contour_loc must be of shape (frame, contour)
    """

    # Get preliminary contours properties
    contours_list = stack.contours[which]
    centers_list = stack.get_contour_centers()

    # Iterate through each frame, starting from the frame succeeding the initial contour
    pointer_indeces = [contour_loc]
    for next_frame, next_contours in enumerate(contours_list[contour_loc[0] + 1:], start=contour_loc[0] + 1):
        # Set up current contour
        curr_frame, curr_index = pointer_indeces[-1]
        curr_contour = contours_list[curr_frame][curr_index]
        center_curr_contour = centers_list[curr_frame][curr_index]

        # Calculate region of interest (ROI)
        # TODO: restrict the search area to a definite region of interest using the extreme points of the current
        #       contour, such that ROI(next_contours) <- next_contours in the following `for` loop

        # Iterate through each contour in the next frame
        matches = []
        distances = []
        for next_index, next_contour in enumerate(next_contours):
            # Set up next contour
            center_next_contour = centers_list[next_frame][next_index]

            # Calculate the matching parameter using image moments and an overlap function (cf. cv::ShapeMatchModes)
            matches.append(cv.matchShapes(curr_contour, next_contour, cv.CONTOURS_MATCH_I1, 0.0))

            # Calculate distance from current to next candidate contour
            distances.append(np.sqrt(np.sum((np.array(center_curr_contour) - np.array(center_next_contour)) ** 2)))

        # Best next contour selection (BNCS) algorithm
        pointer_indeces.append((next_frame, BNCS_alg(matches, distances)))

    return pointer_indeces
