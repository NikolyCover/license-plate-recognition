import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    return edges

# Extrai a região da placa com base nas bordas
def extract_plate_region(original_image, edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)

            if 2 < ratio < 6:
                return original_image[y:y+h, x:x+w]
    
    print("Placa não encontrada.")
    return None

def find_plate_area(image):
    edges = preprocess_image(image)
    plate = extract_plate_region(image, edges)
    return plate
