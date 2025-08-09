import os
import pickle
import cv2
import numpy as np
from moments import calculate_invariant_moments, calculate_geometric_features, tiny_signature

# =======================
# Configuração (fácil de testar/tunar)
# =======================
AUG_KERNEL = np.ones((3, 3), np.uint8)
AUG_ITERS  = 1
USE_OPENING  = True
USE_CLOSING  = True
USE_EROSION  = True
USE_DILATION = True
TINY_SIZE    = 32

IMG_EXTS = (".png", ".jpg", ".jpeg")

# =======================
# Auxiliares de vetor/estatística
# =======================
def geom_vec(g: dict) -> np.ndarray:
    return np.array([
        g['extent'], g['solidity'], g['aspect_ratio'],
        g['perimeter_norm'], float(g.get('holes', 0)),
        float(g.get('hole_area_ratio', 0.0))
    ], dtype=float)

def init_stats_defaults() -> tuple[np.ndarray, np.ndarray]:
    # 6 dimensões (inclui holes e hole_area_ratio)
    return (
        np.array([0, 0, 1, 0, 0, 0], dtype=float),
        np.array([1, 1, 1, 1, 1, 1], dtype=float),
    )

# =======================
# Pipeline de arquivos/imagens
# =======================
def list_image_files(folder: str) -> list[str]:
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]

def parse_label_from_filename(filename: str) -> str | None:
    # Espera algo como "A_001.png"
    parts = filename.split("_")
    if len(parts) != 2:
        return None
    return parts[0]

def load_gray_image(path: str) -> np.ndarray | None:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# =======================
# Augmentação leve
# =======================
def make_variants(gray_img: np.ndarray) -> list[np.ndarray]:
    """Retorna a imagem + variantes morfológicas leves."""
    variants = [gray_img]

    if USE_OPENING:
        variants.append(cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, AUG_KERNEL, iterations=AUG_ITERS))
    if USE_CLOSING:
        variants.append(cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, AUG_KERNEL, iterations=AUG_ITERS))
    if USE_EROSION:
        variants.append(cv2.erode(gray_img, AUG_KERNEL, iterations=AUG_ITERS))
    if USE_DILATION:
        variants.append(cv2.dilate(gray_img, AUG_KERNEL, iterations=AUG_ITERS))

    return variants

# =======================
# Extração de features (1 entrada -> 1 registro)
# =======================
def extract_entry(img: np.ndarray) -> dict:
    """Extrai todas as features necessárias de uma imagem (moments, geom e tiny)."""
    m  = calculate_invariant_moments(img)     # já normaliza internamente
    g  = calculate_geometric_features(img)    # idem
    tg = tiny_signature(img, size=TINY_SIZE)  # binário 0/1
    return {'moments': m.tolist(), 'geometry': g, 'tiny': tg.tolist()}

# =======================
# Acúmulo em memória (DB + estatísticas)
# =======================
def add_entry(database: dict, label: str, entry: dict):
    database.setdefault(label, []).append(entry)

def accumulate_stats(geom_all: list[np.ndarray], label_geom: dict, label: str, g: dict):
    geom_all.append(geom_vec(g))
    label_geom.setdefault(label, []).append(float(g['aspect_ratio']))

def compute_global_stats(geom_all: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not geom_all:
        return init_stats_defaults()
    M = np.vstack(geom_all)
    mean = M.mean(axis=0)
    std  = M.std(axis=0)
    std  = np.where(std == 0, 1.0, std)  # evita divisão por zero
    return mean, std

def compute_per_label_ar_stats(label_geom: dict) -> dict:
    per_label = {}
    for lab, arr in label_geom.items():
        arr = np.array(arr, dtype=float)
        m = float(arr.mean()) if arr.size else 1.0
        s = float(arr.std())  if arr.size else 0.0
        if s == 0.0:
            s = 1.0
        per_label[lab] = {'ar_mean': m, 'ar_std': s}
    return per_label

# =======================
# Persistência
# =======================
def save_database(database: dict, geom_mean: np.ndarray, geom_std: np.ndarray, per_label: dict, output_file: str):
    database['_stats'] = {
        'geom_mean': geom_mean.tolist(),
        'geom_std':  geom_std.tolist(),
        'config': {'canvas_w': 208, 'canvas_h': 130, 'target_h': 100}
    }
    database['_per_label'] = per_label

    with open(output_file, "wb") as f:
        pickle.dump(database, f)

# =======================
# Processo principal (build)
# =======================
def process_single_file(folder: str, filename: str, database: dict,
                        geom_all: list[np.ndarray], label_geom: dict) -> int:
    """Processa um arquivo (com augmentações) e devolve quantas variantes foram salvas."""
    label = parse_label_from_filename(filename)
    if label is None:
        print(f"Nome não esperado: {filename}")
        return 0

    path = os.path.join(folder, filename)
    img  = load_gray_image(path)
    if img is None:
        print(f"Falha ao carregar: {path}")
        return 0

    variants = make_variants(img)
    print(f"Processando: {filename} (+{len(variants)-1} aug)")

    saved = 0
    for v in variants:
        entry = extract_entry(v)
        add_entry(database, label, entry)
        accumulate_stats(geom_all, label_geom, label, entry['geometry'])
        saved += 1
    return saved

def build_database(image_folder: str) -> tuple[dict, np.ndarray, np.ndarray, dict, int]:
    database: dict = {}
    geom_all: list[np.ndarray] = []
    label_geom: dict = {}
    total = 0

    for filename in list_image_files(image_folder):
        total += process_single_file(image_folder, filename, database, geom_all, label_geom)

    geom_mean, geom_std = compute_global_stats(geom_all)
    per_label = compute_per_label_ar_stats(label_geom)
    return database, geom_mean, geom_std, per_label, total

# =======================
# API pública
# =======================
def generate_character_database(image_folder: str, output_file: str = "char_database.pkl"):
    database, geom_mean, geom_std, per_label, total = build_database(image_folder)
    save_database(database, geom_mean, geom_std, per_label, output_file)

    n_labels = len([k for k in database.keys() if k not in ('_stats', '_per_label')])
    print(f"Base criada com {n_labels} labels.")
    print(f"Total de variantes salvas: {total}")

# =======================
# Execução direta
# =======================
if __name__ == "__main__":
    generate_character_database("src/characters")
