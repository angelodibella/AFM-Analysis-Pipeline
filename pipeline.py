import image as im

coeffs = [0.34, 0.66, 0.0]
block_size = 11
c = 2

ID = {}


def test(stack):
    stack.gaussian_blur((5, 5), 0)
    stack.binary_threshold(0, otsu=True)


    print(f'Pipeline {test.__name__} has length {stack.length()}')
