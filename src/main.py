from find_plate_area import find_plate_area
from threshold import apply_threshold
from erosion import apply_erosion
from utils import load_image, show_image
from segment_characters import segment_characters, extract_characters

if __name__ == "__main__":
    image_path = "mock/carro.png"

    plate = load_image(image_path)
    #plate = find_plate_area(plate)  

    if plate is not None:
        show_image("Placa Recortada", plate, cmap=None)

        binary = apply_threshold(plate)
        show_image("Placa Binarizada (Otsu)", binary)

        eroded = apply_erosion(binary, iterations=5)
        show_image("Placa Após Erosão", eroded)

        segmented_image = segment_characters(eroded)
        show_image("Imagem Segmentada", segmented_image, cmap='gray')

        characters = extract_characters(eroded, segmented_image)

        for i, char in enumerate(characters):
            show_image(f"Caractere {i + 1}", char)
    else:
        print("Nenhuma placa detectada.")
