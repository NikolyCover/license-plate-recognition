import cv2
import numpy as np

def calculate_invariant_moments(character_img):
    character_img = cv2.threshold(character_img, 127, 1, cv2.THRESH_BINARY_INV)[1].astype(np.uint8)

    moments = cv2.moments(character_img)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Escala logarítmica para normalização
    for i in range(len(hu_moments)):
        hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-30)

    return hu_moments[:5]  

def normalize_moments(moments):
    # Normaliza os momentos invariantes utilizando a norma L2
    return moments / np.linalg.norm(moments)

def calculate_geometric_features(character_img):
    # Encontra os contornos na imagem binária
    contours, _ = cv2.findContours(character_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {}
    
    for contour in contours:
        area = cv2.contourArea(contour)  # Área
        perimeter = cv2.arcLength(contour, True)  # Perímetro
        x, y, w, h = cv2.boundingRect(contour)  # Retângulo delimitador
        aspect_ratio = float(w) / float(h)  # Razão de aspecto (largura/altura)
        convex_hull = cv2.convexHull(contour)  # Convexidade
        convexity = float(area) / cv2.contourArea(convex_hull)  # Convexidade

        features['area'] = area
        features['perimeter'] = perimeter
        features['aspect_ratio'] = aspect_ratio
        features['convexity'] = convexity
    
    return features
