import cv2
import numpy as np

def _to_binary(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Queremos o CARACTERE como branco (255) e fundo preto (0).
    # Se branco domina, provavelmente é fundo → inverte.
    if np.count_nonzero(bw == 255) > np.count_nonzero(bw == 0):
        bw = cv2.bitwise_not(bw)

    return bw

def normalize_char_canvas(img, canvas_w=208, canvas_h=130, target_h=100):
    bw = _to_binary(img)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = bw[y:y+h, x:x+w]

    scale = (target_h / h) if h > 0 else 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y0 = max(0, (canvas_h - new_h) // 2)
    x0 = max(0, (canvas_w - new_w) // 2)
    canvas[y0:y0+min(new_h, canvas_h), x0:x0+min(new_w, canvas_w)] = \
        resized[:canvas_h - y0, :canvas_w - x0]

    m = cv2.moments(canvas, binaryImage=True)
    if m['m00'] > 0:
        cx = int(m['m10'] / m['m00']); cy = int(m['m01'] / m['m00'])
        M = np.float32([[1, 0, (canvas_w // 2) - cx],
                        [0, 1, (canvas_h // 2) - cy]])
        canvas = cv2.warpAffine(canvas, M, (canvas_w, canvas_h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    return canvas

def calculate_invariant_moments(character_img):
    img = normalize_char_canvas(character_img, canvas_w=208, canvas_h=130, target_h=100)
    img01 = (img > 0).astype(np.uint8)
    moments = cv2.moments(img01)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(len(hu)):
        hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30)
    return hu[:5]

def calculate_geometric_features(character_img):
    bw = _to_binary(character_img)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'extent':0.0,'solidity':0.0,'aspect_ratio':1.0,'perimeter_norm':0.0}

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    box_area = float(w*h) if w*h>0 else 1.0
    aspect_ratio = float(w)/float(h) if h>0 else 1.0

    hull = cv2.convexHull(cnt)
    convex_area = float(cv2.contourArea(hull)) or 1.0

    extent = area / box_area
    solidity = area / convex_area
    perimeter_norm = perimeter / (2.0*(w+h)) if (w+h)>0 else 0.0

    return {
        'extent': extent,
        'solidity': solidity,
        'aspect_ratio': aspect_ratio,
        'perimeter_norm': perimeter_norm
    }
