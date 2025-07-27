
from moments import calculate_invariant_moments
import pickle
import numpy as np

def recognize_character(character_img, db_path="char_database.pkl"):
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    input_moments = calculate_invariant_moments(character_img)
    
    best_label = None
    min_distance = float("inf")

    for label, entries in db.items():
        for db_moments in entries:
            distance = np.linalg.norm(np.array(input_moments) - np.array(db_moments))
            if distance < min_distance:
                min_distance = distance
                best_label = label

    return best_label