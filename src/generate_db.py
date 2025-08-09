import os, pickle, cv2, numpy as np
from moments import calculate_invariant_moments, calculate_geometric_features, normalize_char_canvas

def _geom_vec(g):
    return np.array([g['extent'], g['solidity'], g['aspect_ratio'], g['perimeter_norm']], dtype=float)

def generate_character_database(image_folder, output_file="char_database.pkl"):
    database = {}
    geom_all = []

    total_variants = 0
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        print(f"Processando: {filename}")
        parts = filename.split("_")
        if len(parts) != 2:
            print(f"Nome não esperado: {filename}")
            continue

        label = parts[0]
        path = os.path.join(image_folder, filename)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Falha ao carregar: {path}")
            continue

        # normalizar na MESMA forma do reconhecimento
        norm = normalize_char_canvas(image, canvas_w=208, canvas_h=130, target_h=100)

        moments = calculate_invariant_moments(norm)  # já usa normalização
        geometry = calculate_geometric_features(norm)

        if label not in database:
            database[label] = []
        database[label].append({'moments': moments.tolist(), 'geometry': geometry})
        geom_all.append(_geom_vec(geometry))
        total_variants += 1

    # stats geométricas para z-score
    if geom_all:
        geom_all = np.vstack(geom_all)
        geom_mean = geom_all.mean(axis=0)
        geom_std  = geom_all.std(axis=0)
    else:
        geom_mean = np.array([0,0,1,0], dtype=float)
        geom_std  = np.array([1,1,1,1], dtype=float)

    database['_stats'] = {
        'geom_mean': geom_mean.tolist(),
        'geom_std':  (np.where(geom_std==0, 1.0, geom_std)).tolist(),
        'config': {'canvas_w':208, 'canvas_h':130, 'target_h':100}
    }

    with open(output_file, "wb") as f:
        pickle.dump(database, f)

    print(f"Base criada com {len(database)-1} labels (excluindo _stats).")
    print(f"Total de variantes: {total_variants}")

if __name__ == "__main__":
    generate_character_database("src/characters")
