import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_characters(binary_image):
    segmented_image = np.zeros_like(binary_image, dtype=int)
    current_label = 1 
    
    height, width = binary_image.shape
    
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 0 and segmented_image[i, j] == 0:
                segmented_image[i, j] = current_label
                stack = [(i, j)]
                
                while stack:
                    x, y = stack.pop()

                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width:
                            if binary_image[nx, ny] == 0 and segmented_image[nx, ny] == 0:
                                segmented_image[nx, ny] = current_label
                                stack.append((nx, ny))
                
                current_label += 1
    
    return segmented_image

def extract_characters(binary_image, labeled_image):
    characters = []
    num_labels = np.max(labeled_image)
    
    for label in range(1, num_labels + 1):
        mask = np.zeros_like(binary_image, dtype=np.uint8)
        mask[labeled_image == label] = 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            character = mask[y:y+h, x:x+w]
            character = cv2.bitwise_not(character)
            characters.append(character)
    
    return characters
