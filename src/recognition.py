# recognition.py
from typing import Dict, List, Tuple, Optional
import pickle
import numpy as np
from morphology import apply_opening, apply_closing
from moments import (
    calculate_invariant_moments,
    calculate_geometric_features,
    tiny_signature,
)

# =========================
# Configuração / limites
# =========================
AR_Z_MAX: float = 3.5     # gate por aspect ratio (z-score)
TAU: float = 4.0          # limiar final do voto médio
TINY_SIZE: int = 32       # tamanho da assinatura binária

# Pesos do score (Hu, geom, tiny)
W_HU: float   = 0.55
W_GEOM: float = 0.25
W_TINY: float = 0.20
TINY_SCALE: float = 10.0  # escala do Hamming [0..1] -> [0..10]

# Penalidades de buracos
P_HOLES_DIFF: float = 2.5     # por diferença no número de buracos
P_HOLE_AREA: float  = 3.0     # |Δ hole_area_ratio|
P_BS_HARD: float    = 5.0     # regra dura B vs S


# =========================
# Helpers de vetor/estatística
# =========================
def _geom_vec(g: Dict) -> np.ndarray:
    return np.array([
        g['extent'], g['solidity'], g['aspect_ratio'],
        g['perimeter_norm'], float(g.get('holes', 0)),
        float(g.get('hole_area_ratio', 0.0))
    ], dtype=float)

def _zscore(val: float, mean: float, std: float) -> float:
    std = (std if std and std != 0 else 1.0)
    return abs((val - mean) / std)

def _compare_geom(g1: Dict, g2: Dict, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> float:
    v1, v2 = _geom_vec(g1), _geom_vec(g2)
    if mean is not None and std is not None:
        std = np.where(std == 0, 1.0, std)
        v1 = (v1 - mean) / std
        v2 = (v2 - mean) / std
    return float(np.linalg.norm(v1 - v2))

def _sig_dist32(q_sig: np.ndarray, db_sig: np.ndarray) -> float:
    """Distância de Hamming normalizada [0..1] entre assinaturas 32x32 binárias (0/1)."""
    return float(np.mean(q_sig.astype(np.uint8) != db_sig.astype(np.uint8)))


# =========================
# Gates e restrições
# =========================
def _label_allowed(label: str, pos: Optional[int]) -> bool:
    import string
    LETTERS = set(list(string.ascii_uppercase))
    DIGITS  = set(list("0123456789"))
    if pos in (0, 1, 2, 4):
        allow = LETTERS
    elif pos in (3, 5, 6):
        allow = DIGITS
    else:
        allow = LETTERS | DIGITS
    return (label in allow)

def _pass_ar_gate(ar_q: float, label: str, per_label: Dict[str, Dict[str, float]]) -> bool:
    st = per_label.get(label)
    if st is None:
        return True
    return _zscore(ar_q, st['ar_mean'], st['ar_std']) <= AR_Z_MAX


# =========================
# Penalidades topológicas (buracos)
# =========================
def _hole_penalty(geo_q: Dict, geo_db: Dict, label: str) -> float:
    holes_q  = int(geo_q.get('holes', 0))
    holes_db = int(geo_db.get('holes', 0))
    hole_area_q  = float(geo_q.get('hole_area_ratio', 0.0))
    hole_area_db = float(geo_db.get('hole_area_ratio', 0.0))

    pen = 0.0
    if holes_q != holes_db:
        pen += P_HOLES_DIFF * abs(holes_q - holes_db)
    pen += P_HOLE_AREA * abs(hole_area_q - hole_area_db)

    # Regra dura B vs S
    if label in ('B', 'S'):
        if holes_q >= 1 and label == 'S':
            pen += P_BS_HARD
        if holes_q == 0 and label == 'B':
            pen += P_BS_HARD
    return pen


# =========================
# Scoring
# =========================
def _score_candidate(
    im_q: np.ndarray,
    geo_q: Dict,
    sig_q: Optional[np.ndarray],
    db_entry: Dict,
    geom_mean: np.ndarray,
    geom_std: np.ndarray,
    label: str
) -> float:
    db_mom = np.array(db_entry['moments'], dtype=float)
    db_geo = db_entry['geometry']
    db_sig = np.array(db_entry.get('tiny', []), dtype=np.uint8) if 'tiny' in db_entry else None

    hu_dist = float(np.linalg.norm(im_q - db_mom))
    geom_dist = _compare_geom(geo_q, db_geo, geom_mean, geom_std)

    total = W_HU * hu_dist + W_GEOM * geom_dist

    if db_sig is not None and db_sig.size > 0 and sig_q is not None:
        h = _sig_dist32(sig_q, db_sig)    # 0..1
        total += W_TINY * (h * TINY_SCALE)

    total += _hole_penalty(geo_q, db_geo, label)
    return total


# =========================
# Matching (por variante)
# =========================
def _compute_query_features(var_img: np.ndarray) -> Tuple[np.ndarray, Dict, np.ndarray, float]:
    """Extrai (Hu, geom, tiny, aspect_ratio) para a variante do query."""
    im_q  = calculate_invariant_moments(var_img)
    geo_q = calculate_geometric_features(var_img)
    sig_q = tiny_signature(var_img, size=TINY_SIZE)
    ar_q  = geo_q['aspect_ratio']
    return im_q, geo_q, sig_q, ar_q

def _match_variant(
    var_img: np.ndarray,
    db: Dict,
    geom_mean: np.ndarray,
    geom_std: np.ndarray,
    per_label: Dict[str, Dict[str, float]],
    pos: Optional[int]
) -> Tuple[Optional[str], float]:
    im_q, geo_q, sig_q, ar_q = _compute_query_features(var_img)

    best_label, min_dist = None, float('inf')
    for label, entries in db.items():
        if label in ('_stats', '_per_label'):
            continue
        if pos is not None and not _label_allowed(label, pos):
            continue
        if not _pass_ar_gate(ar_q, label, per_label):
            continue

        for entry in entries:
            total = _score_candidate(im_q, geo_q, sig_q, entry, geom_mean, geom_std, label)
            if total < min_dist:
                min_dist, best_label = total, label
    return best_label, min_dist


# =========================
# Fallback (sem AR gate)
# =========================
def _match_variant_no_gate(
    var_img: np.ndarray,
    db: Dict,
    geom_mean: np.ndarray,
    geom_std: np.ndarray,
    pos: Optional[int]
) -> Tuple[Optional[str], float]:
    im_q, geo_q, sig_q, _ = _compute_query_features(var_img)

    best, bestd = None, float('inf')
    for label, entries in db.items():
        if label in ('_stats', '_per_label'):
            continue
        if pos is not None and not _label_allowed(label, pos):
            continue
        for entry in entries:
            total = _score_candidate(im_q, geo_q, sig_q, entry, geom_mean, geom_std, label)
            if total < bestd:
                bestd, best = total, label
    return best, bestd


# =========================
# Carregamento de DB / Stats
# =========================
def _load_db(db_path: str) -> Dict:
    with open(db_path, "rb") as f:
        return pickle.load(f)

def _load_stats(db: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    stats = db.get('_stats', {})
    geom_mean = np.array(stats.get('geom_mean', [0, 0, 1, 0, 0, 0]), dtype=float)
    geom_std  = np.array(stats.get('geom_std',  [1, 1, 1, 1, 1, 1]), dtype=float)
    per_label = db.get('_per_label', {})
    return geom_mean, geom_std, per_label


# =========================
# API principal
# =========================
def recognize_character(character_img: np.ndarray, db_path: str = "char_database.pkl", pos: Optional[int] = None) -> str:
    """
    Reconhece um caractere binário/tons de cinza.
    - Usa Hu moments, features geométricas + buracos e assinatura binária 32×32.
    - Aplica gate por aspect ratio por label e fallback sem gate quando necessário.
    """
    db = _load_db(db_path)
    geom_mean, geom_std, per_label = _load_stats(db)

    # variantes simples (robustez morfológica)
    variants = [
        character_img,
        apply_opening(character_img, (3, 3), 1),
        apply_closing(character_img, (3, 3), 1),
    ]

    # 1ª passada: com AR gate
    votes = [_match_variant(v, db, geom_mean, geom_std, per_label, pos) for v in variants]
    labels = [l for (l, _) in votes if l is not None]

    # fallback: tenta sem AR gate se nada passou
    if not labels:
        votes = [_match_variant_no_gate(v, db, geom_mean, geom_std, pos) for v in variants]
        labels = [l for (l, _) in votes if l is not None]
        if not labels:
            return '?'

    # votação por maioria; desempate por menor distância média
    best = max(set(labels), key=labels.count)
    avg_dist = float(np.mean([d for (l, d) in votes if l == best]))
    return best if avg_dist < TAU else '?'
