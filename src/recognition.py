from moments import calculate_invariant_moments, calculate_geometric_features
from morphology import apply_opening, apply_closing
import pickle, numpy as np
import string

TAU = 4.0
LETTERS = set(list(string.ascii_uppercase))
DIGITS  = set(list("0123456789"))

def allowed_labels_for_pos(pos):
    # ABC1D23  -> 0,1,2=Letras; 3=Dígito; 4=Letra; 5,6=Dígitos
    if pos in (0,1,2,4):
        return LETTERS
    if pos in (3,5,6):
        return DIGITS
    return LETTERS | DIGITS


def _geom_vec(g):
    return np.array([g['extent'], g['solidity'], g['aspect_ratio'], g['perimeter_norm']], dtype=float)

def compare_geometric_features(g1, g2, mean=None, std=None):
    v1, v2 = _geom_vec(g1), _geom_vec(g2)
    if mean is not None and std is not None:
        std = np.where(std==0, 1.0, std)
        v1 = (v1 - mean)/std
        v2 = (v2 - mean)/std
    return np.linalg.norm(v1 - v2)

def recognize_character(character_img, db_path="char_database.pkl", pos=None):
    allow = allowed_labels_for_pos(pos) if pos is not None else None

    with open(db_path, "rb") as f:
        db = pickle.load(f)

    stats = db.get('_stats', {})
    geom_mean = np.array(stats.get('geom_mean', [0,0,1,0]), dtype=float)
    geom_std  = np.array(stats.get('geom_std',  [1,1,1,1]), dtype=float)

    # variantes MORFOLÓGICAS LEVES (ensemble)
    variants = [
        character_img,
        apply_opening(character_img, (3,3), 1),
        apply_closing(character_img, (3,3), 1),
    ]

    votes = []
    for var in variants:
        im = calculate_invariant_moments(var)       # já normaliza
        geo = calculate_geometric_features(var)     # idem

        best_label = None
        min_dist = float('inf')

        for label, entries in db.items():
            if label == '_stats': 
                continue
            if allow is not None and label not in allow:
                continue
            for entry in entries:
                db_m = np.array(entry['moments'], dtype=float)
                m_dist = np.linalg.norm(im - db_m)

                g_dist = compare_geometric_features(geo, entry['geometry'], geom_mean, geom_std)
                total = 0.8*m_dist + 0.2*g_dist

                if total < min_dist:
                    min_dist = total
                    best_label = label

        votes.append((best_label, min_dist))

    # majority vote + distância média do rótulo vencedor
    labels = [l for (l, _) in votes]
    best = max(set(labels), key=labels.count)
    avg_dist = np.mean([d for (l, d) in votes if l == best])

    return best if avg_dist < TAU else '?'