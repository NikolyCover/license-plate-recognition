import cv2
import numpy as np
import os
import pickle

def calculate_invariant_moments(character_img):
    character_img = cv2.threshold(character_img, 127, 1, cv2.THRESH_BINARY_INV)[1].astype(np.uint8)

    moments = cv2.moments(character_img)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Escala logarítmica para normalização
    for i in range(len(hu_moments)):
        hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-30)

    return hu_moments[:5]  # conforme o artigo, usar apenas os 5 primeiros momentos
