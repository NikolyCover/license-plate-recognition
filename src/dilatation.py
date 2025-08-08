import cv2

def apply_dilation(binary_image, iterations=1):
    binary_image = cv2.bitwise_not(binary_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    dilated = cv2.dilate(binary_image, kernel, iterations=iterations)

    dilated = cv2.bitwise_not(dilated)

    return dilated