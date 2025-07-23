import cv2

def apply_erosion(binary_image, iterations=1, invert=True):
    if invert:
        binary_image = cv2.bitwise_not(binary_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    eroded = cv2.erode(binary_image, kernel, iterations=iterations)

    if invert:
        eroded = cv2.bitwise_not(eroded)

    return eroded
