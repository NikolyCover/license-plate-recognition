from find_plate_area import find_plate_area
from threshold import apply_threshold
from erosion import apply_erosion
from utils import load_image, show_image

if __name__ == "__main__":
    image_path = "mock/carro.png"

    plate = load_image(image_path)
    #plate = find_plate_area(plate)  

    if plate is not None:
        show_image("Placa Recortada", plate, cmap=None)

        binary = apply_threshold(plate)
        show_image("Placa Binarizada (Otsu)", binary)

        eroded = apply_erosion(binary, iterations=1)
        show_image("Placa Após Erosão", eroded)
    else:
        print("Nenhuma placa detectada.")
