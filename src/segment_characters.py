import cv2
import numpy as np
from typing import List, Tuple

# Retorna a lista de vizinhos 8-conectados (dx, dy).
def neighbors_8() -> List[Tuple[int, int]]:
    return [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Faz flood fill iterativo para rotular um componente a partir de (i, j).
def flood_fill_label(
    binary_image: np.ndarray,
    labeled_image: np.ndarray,
    start_i: int,
    start_j: int,
    label: int,
    foreground_value: int = 0
) -> None:
    h, w = binary_image.shape
    stack = [(start_i, start_j)]
    labeled_image[start_i, start_j] = label
    for dx, dy in neighbors_8():
        pass  # apenas força o carregamento da função antes do loop real

    neigh = neighbors_8()
    while stack:
        x, y = stack.pop()
        for dx, dy in neigh:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                if binary_image[nx, ny] == foreground_value and labeled_image[nx, ny] == 0:
                    labeled_image[nx, ny] = label
                    stack.append((nx, ny))


# Rotula componentes conectados considerando pixels pretos (0) como “caracteres”.
def segment_characters(binary_image: np.ndarray) -> np.ndarray:
    h, w = binary_image.shape
    labeled = np.zeros((h, w), dtype=int)
    current_label = 1
    for i in range(h):
        for j in range(w):
            if binary_image[i, j] == 0 and labeled[i, j] == 0:
                flood_fill_label(binary_image, labeled, i, j, current_label, foreground_value=0)
                current_label += 1
    return labeled


# Calcula limites de área mínimos e máximos em pixels com base em razões do tamanho da placa.
def area_thresholds(h: int, w: int, min_ratio: float = 0.002, max_ratio: float = 0.30) -> Tuple[float, float]:
    total = h * w
    return min_ratio * total, max_ratio * total


# Cria uma máscara (0/255) para um rótulo específico.
def component_mask(labeled_image: np.ndarray, label: int) -> np.ndarray:
    mask = np.zeros_like(labeled_image, dtype=np.uint8)
    mask[labeled_image == label] = 255
    return mask


# Encontra o maior contorno externo na máscara (quando existir).
def largest_external_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# Recorta o caractere pela bounding box e inverte para “preto no branco”.
def crop_invert_char(mask: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    char = mask[y:y+h, x:x+w]
    return cv2.bitwise_not(char)  # retorna caractere preto em fundo branco


# Aplica os filtros de área para descartar ruídos e componentes gigantes.
def pass_area_filter(w: int, h: int, min_area: float, max_area: float) -> bool:
    area = w * h
    return (area >= min_area) and (area <= max_area)


# Ordena caixas da esquerda para a direita, desempate por y (linhas).
def sort_boxes_left_to_right(boxes: List[Tuple[int, int, int, int]]) -> List[int]:
    return sorted(range(len(boxes)), key=lambda i: (boxes[i][0], boxes[i][1]))


# Extrai caracteres e seus bounding boxes ordenados; filtra ruídos por área.
def extract_characters(
    binary_image: np.ndarray,
    labeled_image: np.ndarray,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.30
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    h, w = binary_image.shape
    min_area, max_area = area_thresholds(h, w, min_area_ratio, max_area_ratio)

    characters: List[np.ndarray] = []
    boxes: List[Tuple[int, int, int, int]] = []

    num_labels = int(np.max(labeled_image))
    for label in range(1, num_labels + 1):
        mask = component_mask(labeled_image, label)
        cnt = largest_external_contour(mask)
        if cnt is None:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if not pass_area_filter(bw, bh, min_area, max_area):
            continue

        char = crop_invert_char(mask, x, y, bw, bh)
        characters.append(char)
        boxes.append((x, y, bw, bh))

    order = sort_boxes_left_to_right(boxes)
    characters_sorted = [characters[i] for i in order]
    boxes_sorted = [boxes[i] for i in order]
    return characters_sorted, boxes_sorted
