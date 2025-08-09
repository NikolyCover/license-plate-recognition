# -*- coding: utf-8 -*-
from find_plate_area import find_plate_area
from threshold import apply_threshold
from erosion import apply_erosion
from dilatation import apply_dilation 
from utils import load_image, show_image
from segment_characters import segment_characters, extract_characters
from recognition import recognize_character
from closing import apply_closing
from opening import apply_opening

import cv2

if __name__ == "__main__":
    image_path = "mock/PLATE_3.png"

    plate = load_image(image_path)
    #plate = find_plate_area(plate)  

    if plate is not None:
        show_image("Placa Recortada", plate, cmap=None)

        plate = cv2.GaussianBlur(plate, (5, 5), 0)

        plate = apply_threshold(plate)
        show_image("Placa Binarizada (Otsu)", plate)

        plate = apply_dilation(plate, iterations=3)
        show_image("Placa Após Dilatar", plate)

        plate = apply_erosion(plate, iterations=1)
        show_image("Placa Após Erosão", plate)

        segmented_image = segment_characters(plate)
        show_image("Imagem Segmentada", segmented_image, cmap='gray')

        characters = extract_characters(plate, segmented_image)

        for i, char in enumerate(characters):
            label = recognize_character(char)
            show_image(f"Caractere {i + 1}: {label}", char)

    else:
        print("Nenhuma placa detectada.")
