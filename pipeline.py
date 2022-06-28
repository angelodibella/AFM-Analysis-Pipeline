import image as im

coeffs = [0.34, 0.66, 0.0]
block_size = 11
c = 2


def test(stack):
    stack.gaussian_blur((3, 3), 0, coeffs=coeffs)
    stack.binary_threshold(32, otsu=True)
    stack.get_contours(append=True)
