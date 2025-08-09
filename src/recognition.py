from moments import calculate_invariant_moments, normalize_moments, calculate_geometric_features
import pickle
import numpy as np

def recognize_character(character_img, db_path="char_database.pkl"):
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    # Cálculo dos momentos invariantes do caractere a ser reconhecido
    input_moments = calculate_invariant_moments(character_img)
    input_moments = normalize_moments(input_moments)
    
    # Cálculo das características geométricas
    input_geometry = calculate_geometric_features(character_img)
    
    best_label = None
    min_distance = float("inf")

    for label, entries in db.items():
        for entry in entries:
            # Comparando os momentos invariantes
            db_moments = normalize_moments(np.array(entry['moments']))
            moments_distance = np.linalg.norm(input_moments - db_moments)

            # Comparando as características geométricas
            geometry_distance = compare_geometric_features(input_geometry, entry['geometry'])

            total_distance = 0.8 * moments_distance + 0.2 * geometry_distance

            if total_distance < min_distance:
                min_distance = total_distance
                best_label = label

    return best_label

def compare_geometric_features(geometry1, geometry2):
    # Calculando a diferença nas características geométricas
    area_diff = np.abs(geometry1['area'] - geometry2['area'])
    perimeter_diff = np.abs(geometry1['perimeter'] - geometry2['perimeter'])
    aspect_ratio_diff = np.abs(geometry1['aspect_ratio'] - geometry2['aspect_ratio'])
    convexity_diff = np.abs(geometry1['convexity'] - geometry2['convexity'])
    
    # Ajustando a importância de cada característica (pesos podem ser alterados)
    total_geometry_diff = 0.4 * area_diff + 0.3 * perimeter_diff + 0.2 * aspect_ratio_diff + 0.1 * convexity_diff
    
    return total_geometry_diff