# moments.py (refatorado)
from typing import Dict, Tuple
import cv2
import numpy as np

# =========================
# Configuração
# =========================
DEFAULT_CANVAS_W = 208
DEFAULT_CANVAS_H = 130
DEFAULT_TARGET_H = 100
CLOSE_KERNEL = np.ones((3, 3), np.uint8)  # para selar microfrestas em buracos

# =========================
# Utilidades básicas
# =========================
def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Garante dtype uint8."""
    return img if img.dtype == np.uint8 else img.astype(np.uint8)

def _to_binary(img: np.ndarray) -> np.ndarray:
    """
    Binariza por Otsu. Garante caractere BRANCO (255) e fundo PRETO (0).
    Se o branco estiver dominando a imagem, inverte.
    """
    img = _ensure_uint8(img)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.count_nonzero(bw == 255) > np.count_nonzero(bw == 0):
        bw = cv2.bitwise_not(bw)
    return bw

def _largest_external_contour(bw: np.ndarray):
    """Retorna o maior contorno externo (ou None se não houver)."""
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def _center_by_moments(canvas: np.ndarray) -> np.ndarray:
    """
    Centraliza o caractere no canvas usando o centróide dos momentos.
    """
    m = cv2.moments(canvas, binaryImage=True)
    if m['m00'] <= 0:
        return canvas
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    H, W = canvas.shape
    M = np.float32([[1, 0, (W // 2) - cx],
                    [0, 1, (H // 2) - cy]])
    return cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

# =========================
# Normalização em canvas
# =========================
def normalize_char_canvas(
    img: np.ndarray,
    canvas_w: int = DEFAULT_CANVAS_W,
    canvas_h: int = DEFAULT_CANVAS_H,
    target_h: int = DEFAULT_TARGET_H
) -> np.ndarray:
    """
    Recorta o maior componente (caractere), redimensiona para 'target_h' e
    posiciona centralizado em um canvas (caractere BRANCO em fundo PRETO).
    """
    bw = _to_binary(img)
    cnt = _largest_external_contour(bw)
    if cnt is None:
        return np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(cnt)
    crop = bw[y:y + h, x:x + w]

    scale = (target_h / h) if h > 0 else 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y0 = max(0, (canvas_h - new_h) // 2)
    x0 = max(0, (canvas_w - new_w) // 2)
    canvas[y0:y0 + min(new_h, canvas_h), x0:x0 + min(new_w, canvas_w)] = \
        resized[:canvas_h - y0, :canvas_w - x0]

    canvas = _center_by_moments(canvas)
    return canvas

def normalized_binary(
    img: np.ndarray,
    cw: int = DEFAULT_CANVAS_W,
    ch: int = DEFAULT_CANVAS_H,
    th: int = DEFAULT_TARGET_H
) -> np.ndarray:
    """
    Retorna binária a partir do canvas normalizado:
    caractere (255) em fundo (0).
    """
    canvas = normalize_char_canvas(img, cw, ch, th)
    return (canvas > 0).astype(np.uint8) * 255

# =========================
# Features (Hu, geom, buracos, tiny)
# =========================
def calculate_invariant_moments(character_img: np.ndarray) -> np.ndarray:
    """
    Calcula Hu moments (log transform, sinal preservado), usando a imagem normalizada.
    Retorna apenas os 5 primeiros (suficientes para este problema).
    """
    bw = normalized_binary(character_img)
    moments = cv2.moments((bw > 0).astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    for i in range(len(hu)):
        hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30)
    return hu[:5]

def _hole_features_from_binary(bw: np.ndarray) -> Dict[str, float]:
    """
    Conta buracos (contornos filhos) e calcula razão da área dos buracos
    sobre a área em branco. Usa borda de 1px e fechamento leve para robustez.
    """
    h, w = bw.shape
    bordered = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bordered[1:-1, 1:-1] = bw
    bordered = cv2.morphologyEx(bordered, cv2.MORPH_CLOSE, CLOSE_KERNEL, iterations=1)

    contours, hierarchy = cv2.findContours(bordered, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes, hole_area = 0, 0.0
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, hrc in enumerate(hierarchy):
            if hrc[3] != -1:  # tem pai => buraco
                holes += 1
                hole_area += cv2.contourArea(contours[i])

    total_area = float(np.count_nonzero(bordered > 0)) or 1.0
    return {'holes': float(holes), 'hole_area_ratio': hole_area / total_area}

def _geom_from_contour(cnt: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calcula extent, solidity, aspect_ratio e perimeter_norm a partir do maior contorno.
    """
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))

    x, y, w, h = cv2.boundingRect(cnt)
    box_area = float(w * h) if w * h > 0 else 1.0
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    hull = cv2.convexHull(cnt)
    convex_area = float(cv2.contourArea(hull)) or 1.0

    extent = area / box_area
    solidity = area / convex_area
    perimeter_norm = perim / (2.0 * (w + h)) if (w + h) > 0 else 0.0
    return extent, solidity, aspect_ratio, perimeter_norm

def calculate_geometric_features(character_img: np.ndarray) -> Dict[str, float]:
    """
    Extrai features geométricas (extent, solidity, aspect_ratio, perimeter_norm)
    + topologia de buracos (holes, hole_area_ratio) a partir da imagem normalizada.
    """
    bw = normalized_binary(character_img)
    cnt = _largest_external_contour(bw)
    if cnt is None:
        return {
            'extent': 0.0, 'solidity': 0.0, 'aspect_ratio': 1.0,
            'perimeter_norm': 0.0, 'holes': 0.0, 'hole_area_ratio': 0.0
        }

    extent, solidity, aspect_ratio, perimeter_norm = _geom_from_contour(cnt)
    hf = _hole_features_from_binary(bw)
    return {
        'extent': extent,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'perimeter_norm': perimeter_norm,
        'holes': hf['holes'],
        'hole_area_ratio': hf['hole_area_ratio'],
    }

def tiny_signature(character_img: np.ndarray, size: int = 32) -> np.ndarray:
    """
    Assinatura binária (0/1) reduzida do canvas normalizado para template matching leve.
    """
    bw = normalized_binary(character_img)
    tiny = cv2.resize(bw, (size, size), interpolation=cv2.INTER_NEAREST)
    return (tiny > 0).astype(np.uint8)
