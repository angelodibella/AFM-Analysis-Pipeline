# Constants to be used
COEFFS_1 = [0.34, 0.66, 0.0]


def otsu_1(stack):
    stack.gaussian_blur((3, 3), 1, coeffs=COEFFS_1)
    stack.binary_threshold(32, otsu=True)
    contours, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()

    return contours


def canny_1(stack):
    stack.gaussian_blur((3, 3), 1, coeffs=COEFFS_1)
    stack.canny(0, 0)
    contours, _ = stack.get_contours(append=True)

    # Print information
    stack.print_info()

    return contours
