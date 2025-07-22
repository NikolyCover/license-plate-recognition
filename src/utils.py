import cv2

import matplotlib.pyplot as plt
import cv2

def show_image(title, image, cmap="gray"):
    plt.imshow(image if cmap is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image
