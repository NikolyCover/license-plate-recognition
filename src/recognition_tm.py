import pickle, cv2, numpy as np
from typing import Optional
from morphology import apply_opening, apply_closing
from moments import normalize_char_canvas
from recognition import allowed_labels_for_pos  

def _tm_score(img: np.ndarray, tpl: np.ndarray) -> float:
    """
    Retorna score de correlação normalizada (TM_CCOEFF_NORMED) no range [-1, 1].
    Como img e tpl têm o mesmo tamanho, o resultado é 1x1.
    """
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    return float(res[0, 0])

def _normalize01(bw: np.ndarray) -> np.ndarray:
    return (bw > 0).astype(np.float32)

def recognize_character_tm(
    character_img: np.ndarray,
    db_path: str = "template_db.pkl",
    pos: Optional[int] = None,
    use_mean_template: bool = True,
    vote_morph_variants: bool = True,
    tau: float = 0.2  # limiar de aceitação; ajuste com validação
) -> str:
    """ 
    Reconhece um caractere comparando com templates usando Template Matching.
    - Restringe o espaço de busca pelo 'pos' (ABC1D23).
    - Normaliza o caractere no mesmo canvas dos templates.
    - Usa TM_CCOEFF_NORMED; 1.0 é match perfeito.
    """
    with open(db_path, "rb") as f:
        data = pickle.load(f)
    db = data["templates"]
    meta = data["meta"]

    allow = allowed_labels_for_pos(pos) if pos is not None else None

    # variantes opcionalmente (ensemble leve)
    imgs = [character_img]
    if vote_morph_variants:
        imgs += [
            apply_opening(character_img, (3,3), 1),
            apply_closing(character_img, (3,3), 1),
        ]

    label_votes = []
    for var in imgs:
        norm = normalize_char_canvas(var, meta["canvas_w"], meta["canvas_h"], meta["target_h"])
        q = _normalize01(norm)

        best_label, best_score = None, -2.0  # menor que o mínimo possível (-1)
        for label, entry in db.items():
            if allow is not None and label not in allow:
                continue

            # avalia contra o template médio e/ou amostras
            candidates = []
            if use_mean_template:
                candidates.append(entry["mean"])
            else:
                candidates += entry["samples"]

            # testa todos os candidatos e pega o melhor score
            for tpl in candidates:
                s = _tm_score(q, tpl)
                if s > best_score:
                    best_score, best_label = s, label

        label_votes.append((best_label, best_score))

    # majority vote pelo rótulo; desempate pelo maior score médio
    labels = [l for l, _ in label_votes]
    winner = max(set(labels), key=labels.count)
    avg_score = float(np.mean([s for (l, s) in label_votes if l == winner]))

    return winner if avg_score >= tau else "?"
