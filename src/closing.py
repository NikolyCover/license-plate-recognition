import cv2
import numpy as np

def apply_closing(binary_image, kernel_size=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Aplica a operação morfológica de fechamento: dilatação seguida de erosão
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return closed_image
