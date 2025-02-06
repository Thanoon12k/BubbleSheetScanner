
import cv2
import numpy as np


class UserInterface:

    def display_image(self, image,scale_percent=50):
        
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_bubbles(self, image, bubbles, color='gr', thickness=2):
        if len(image.shape) == 2:  # Grayscale frames have 2 dimensions
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        color_dict = {
            'gr': (0, 255, 0),   # Green
            're': (0, 0, 255),   # Red
            'ye': (0, 255, 255), # Yellow
            'bl': (255, 0, 0),   # Blue
            'cy': (255, 255, 0), # Cyan
            'br': (42, 42, 165)  # Brown
        }
        color = color_dict.get(color, (0, 255, 0))  # Default to green if color not found

        for bubble in bubbles:
            cv2.drawContours(image, [bubble], -1, color, thickness)
        return image
    def rectangle(self, image, x, y, w, h):
        if len(image.shape) == 2:  # Grayscale frames have 2 dimensions
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return image
    def put_text(self, image, text, x=30, y=60, text_size=3):
        if len(image.shape) == 2:  # Grayscale frames have 2 dimensions
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,255), 4)
        return image
    
