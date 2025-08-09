import os, cv2, pickle
import numpy as np
from moments import normalize_char_canvas  # tua função

def build_template_db(image_folder, output="template_db.pkl",
                      canvas_w=208, canvas_h=130, target_h=100):
    """
    Lê imagens rotuladas (ex: 'A_1.png', 'B_2.png'), normaliza no mesmo canvas
    e guarda o template médio por label, além de também guardar cada variante.
    """
    per_label = {}

    for fname in os.listdir(image_folder):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')): 
            continue
        label = fname.split('_')[0]  # "A_3.png" -> "A"
        path = os.path.join(image_folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue

        norm = normalize_char_canvas(img, canvas_w=canvas_w, canvas_h=canvas_h, target_h=target_h)
        # normalizar para float [0,1] ajuda no TM_CCOEFF_NORMED
        tpl = (norm > 0).astype(np.float32)

        per_label.setdefault(label, []).append(tpl)

    # opcional: template médio por label (suaviza pequenas variações)
    db = {}
    for label, arrs in per_label.items():
        stack = np.stack(arrs, axis=0)  # (n, H, W)
        mean_tpl = stack.mean(axis=0)
        db[label] = {
            "mean": mean_tpl,           # template médio
            "samples": arrs,            # amostras individuais
        }

    meta = {"canvas_w": canvas_w, "canvas_h": canvas_h, "target_h": target_h}
    with open(output, "wb") as f:
        pickle.dump({"templates": db, "meta": meta}, f)
    print(f"[OK] Templates salvos em {output} (labels: {len(db)})")

if __name__ == "__main__":
    build_template_db("src/characters")
