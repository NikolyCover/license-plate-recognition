import cv2
import numpy as np
from typing import List, Tuple, Optional

# =========================
# Configs padrão
# =========================
DEFAULT_CONNECTIVITY = 8          # 4 ou 8
DEFAULT_FOREGROUND_VALUE = 0      # pixels do caractere (0 = preto)
MIN_AREA_RATIO = 0.002            # ~0.2% da área da placa
MAX_AREA_RATIO = 0.30             # 30% da área da placa

# =========================
# Utilidades
# =========================
def _neighbors(connectivity: int = DEFAULT_CONNECTIVITY) -> List[Tuple[int, int]]:
    """Retorna offsets de vizinhos para 4- ou 8-conectividade."""
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8-conectividade
    return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

def _in_bounds(x: int, y: int, h: int, w: int) -> bool:
    return 0 <= x < h and 0 <= y < w

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    return img if img.dtype == np.uint8 else img.astype(np.uint8)

# =========================
# Rotulagem (componentes conexos)
# =========================
def connected_components(
    binary_image: np.ndarray,
    foreground_value: int = DEFAULT_FOREGROUND_VALUE,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> np.ndarray:
    """
    Rótula componentes onde 'foreground_value' é considerado '1' (pixels do caractere).
    Retorna uma imagem de rótulos (int) com 0 = fundo e 1..N = componentes.
    """
    img = _ensure_uint8(binary_image)
    h, w = img.shape
    labeled = np.zeros((h, w), dtype=int)
    current_label = 1
    neigh = _neighbors(connectivity)

    for i in range(h):
        for j in range(w):
            if img[i, j] == foreground_value and labeled[i, j] == 0:
                # DFS iterativo (pilha)
                labeled[i, j] = current_label
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    for dx, dy in neigh:
                        nx, ny = x + dx, y + dy
                        if _in_bounds(nx, ny, h, w):
                            if img[nx, ny] == foreground_value and labeled[nx, ny] == 0:
                                labeled[nx, ny] = current_label
                                stack.append((nx, ny))
                current_label += 1

    return labeled

def segment_characters(
    binary_image: np.ndarray,
    foreground_value: int = DEFAULT_FOREGROUND_VALUE,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> np.ndarray:
    """
    Wrapper para manter compatibilidade com seu código antigo.
    """
    return connected_components(binary_image, foreground_value, connectivity)

# =========================
# Extração de caracteres a partir dos rótulos
# =========================
def _component_mask(labeled_image: np.ndarray, label: int) -> np.ndarray:
    """Retorna máscara uint8 (255 dentro do componente, 0 fora)."""
    mask = (labeled_image == label).astype(np.uint8) * 255
    return mask

def _largest_external_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Retorna o maior contorno externo do componente (ou None)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def _area_filters(H: int, W: int, min_ratio: float, max_ratio: float) -> Tuple[float, float]:
    img_area = float(H * W)
    return min_ratio * img_area, max_ratio * img_area

def _extract_char_from_mask(mask: np.ndarray, cnt: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Recorta o char usando o bounding box do maior contorno e inverte (caractere preto em fundo branco),
    mantendo a compatibilidade com o seu fluxo.
    """
    x, y, w, h = cv2.boundingRect(cnt)
    crop = mask[y:y+h, x:x+w]                 # dentro do recorte, char = 255
    char = cv2.bitwise_not(crop)              # inverte → char preto (0), fundo branco (255)
    return char, (x, y, w, h)

def _sort_boxes_left_to_right(boxes: List[Tuple[int, int, int, int]]) -> List[int]:
    """Ordena por X e desempata por Y, retornando índices ordenados."""
    return sorted(range(len(boxes)), key=lambda i: (boxes[i][0], boxes[i][1]))

def extract_characters(
    binary_image: np.ndarray,
    labeled_image: np.ndarray,
    min_area_ratio: float = MIN_AREA_RATIO,
    max_area_ratio: float = MAX_AREA_RATIO,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Retorna (characters, boxes) ordenados da esquerda para a direita.
      - characters: lista de recortes (uint8) com caractere preto em fundo branco.
      - boxes: lista de (x, y, w, h) no sistema de coordenadas da placa.
    Filtra ruídos pequenos e componentes gigantes (borda, logos) via área.
    """
    H, W = binary_image.shape
    min_area, max_area = _area_filters(H, W, min_area_ratio, max_area_ratio)

    characters: List[np.ndarray] = []
    boxes: List[Tuple[int, int, int, int]] = []

    num_labels = int(np.max(labeled_image))
    if num_labels <= 0:
        return characters, boxes

    for label in range(1, num_labels + 1):
        mask = _component_mask(labeled_image, label)
        cnt = _largest_external_contour(mask)
        if cnt is None:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        area = float(w * h)
        if area < min_area or area > max_area:
            continue  # ignora ruído / componentes gigantes

        char, box = _extract_char_from_mask(mask, cnt)
        characters.append(char)
        boxes.append(box)

    # Ordenar esquerda→direita (com desempate vertical):
    order = _sort_boxes_left_to_right(boxes)
    characters_sorted = [characters[i] for i in order]
    boxes_sorted = [boxes[i] for i in order]
    return characters_sorted, boxes_sorted
