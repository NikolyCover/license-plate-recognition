import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_erosion(binary_image, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=iterations)

    return eroded
