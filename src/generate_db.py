import os
import pickle
import cv2
from moments import calculate_invariant_moments, calculate_geometric_features

def generate_character_database(image_folder, output_file="char_database.pkl"):
    database = {}

    # Variável para contar o total de variantes
    total_variants = 0

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Processando: {filename}")

            # Extração do rótulo (exemplo: "A" de "A_01")
            label = filename.split("_")[0]  # Ex: "A_01" -> "A"
            if len(filename.split("_")) != 2:
                print(f"Nome de arquivo não esperado: {filename}")
                continue

            # Caminho da imagem
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Não foi possível carregar: {image_path}")
                continue

            # Cálculo dos momentos invariantes para a imagem
            moments = calculate_invariant_moments(image)

            # Cálculo das características geométricas
            geometric_features = calculate_geometric_features(image)

            # Armazenando as variantes de caracteres no banco
            if label not in database:
                database[label] = []

            database[label].append({'moments': moments.tolist(), 'geometry': geometric_features})
            total_variants += 1  # Incrementando o total de variantes

    # Salvando o banco de dados em um arquivo pickle
    with open(output_file, "wb") as f:
        pickle.dump(database, f)

    print(f"Base de caracteres criada com {len(database)} labels.")
    print(f"Total de variantes salvas: {total_variants}")

if __name__ == "__main__":
    generate_character_database("src/characters")
