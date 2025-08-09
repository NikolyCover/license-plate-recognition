import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_characters(binary_image):
    segmented_image = np.zeros_like(binary_image, dtype=int)
    current_label = 1 
    
    height, width = binary_image.shape
    
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 0 and segmented_image[i, j] == 0:
                segmented_image[i, j] = current_label
                stack = [(i, j)]
                
                while stack:
                    x, y = stack.pop()

                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width:
                            if binary_image[nx, ny] == 0 and segmented_image[nx, ny] == 0:
                                segmented_image[nx, ny] = current_label
                                stack.append((nx, ny))
                
                current_label += 1
    
    return segmented_image

def extract_characters(binary_image, labeled_image):
    """
    Retorna a lista de caracteres recortados ORDENADOS da ESQUERDA para a DIREITA.
    Também filtra ruídos muito pequenos.
    """
    characters = []
    boxes = []
    num_labels = int(np.max(labeled_image))

    H, W = binary_image.shape
    min_area = 0.002 * (H * W)   # filtro de área mínima (~0.2% da placa) – ajuste se precisar
    max_area = 0.30  * (H * W)   # filtro de área máxima (evitar pegar a placa toda/borda)

    for label in range(1, num_labels + 1):
        mask = np.zeros_like(binary_image, dtype=np.uint8)
        mask[labeled_image == label] = 255

        # um componente pode ter múltiplos contornos “internos”; pegue o maior externo
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue  # ignora ruído / componentes gigantes (borda, logo etc.)

        # recorte do caractere (mantém como você tinha)
        char = mask[y:y+h, x:x+w]
        char = cv2.bitwise_not(char)  # preto em fundo branco (ok pro seu fluxo)

        boxes.append((x, y, w, h))
        characters.append(char)

    # ORDENAR: esquerda→direita (x), com desempate por y (linhas)
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0], boxes[i][1]))
    characters_sorted = [characters[i] for i in order]
    boxes_sorted = [boxes[i] for i in order]

    return characters_sorted, boxes_sorted

