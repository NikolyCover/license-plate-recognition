import os
import pickle
import cv2
from moments import calculate_invariant_moments

def generate_character_database(image_folder, output_file="char_database.pkl"):
    database = {}

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            name_without_ext = os.path.splitext(filename)[0]  # "A_1"
            if name_without_ext.endswith(("_2")):
                print(f"> Imagem pulada: {filename}")
                continue

            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Não foi possível carregar: {image_path}")
                continue

            label = filename.split("_")[0]  # "A_3.png" -> "A"

            moments = calculate_invariant_moments(image)

            if label not in database:
                database[label] = []

            database[label].append(moments.tolist())

    with open(output_file, "wb") as f:
        pickle.dump(database, f)

    print(f"\Base de caracteres criadas com {len(database)} labels.")

if __name__ == "__main__":
    generate_character_database("src/characters")
