import cv2


def apply_dilation(binary_image, iterations=1):
    # Invert the image
    binary_image = cv2.bitwise_not(binary_image)

    # Define a cross-shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Apply dilation
    dilated = cv2.dilate(binary_image, kernel, iterations=iterations)

    # Invert the result back
    dilated = cv2.bitwise_not(dilated)

    return dilated