import image as im

coeffs = [0.34, 0.66, 0.0]
block_size = 11
c = 2


def otsu_1(stack):
    stack.gaussian_blur((3, 3), 0, coeffs=coeffs)
    stack.binary_threshold(32, otsu=True)
    contours, _ = stack.get_contours(append=True)

    return contours


def canny_1(stack):
    stack.gaussian_blur((3, 3), 0, coeffs=coeffs)
    stack.canny(0, 0)
    contours, _ = stack.get_contours(append=True)

    return contours
