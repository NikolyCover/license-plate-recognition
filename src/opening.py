import cv2
import numpy as np

def apply_opening(binary_image, kernel_size=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Aplica a operação morfológica de abertura: erosão seguida de dilatação
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return opened_image
