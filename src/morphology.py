import numpy as np
import cv2
from moments import calculate_invariant_moments

def apply_erosion(binary_image, iterations=1):
    binary_image = cv2.bitwise_not(binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(binary_image, kernel, iterations=iterations)
    return cv2.bitwise_not(eroded)

def apply_dilation(binary_image, iterations=1):
    binary_image = cv2.bitwise_not(binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(binary_image, kernel, iterations=iterations)
    return cv2.bitwise_not(dilated)

def apply_closing(binary_image, kernel_size=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def apply_opening(binary_image, kernel_size=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
