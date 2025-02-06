import cv2
import os

class ImageManager:

    def __init__(self, path):
        self.path = path
        self.image_name='image_manager_.png'
        self.image = cv2.imread(os.path.join(path))
        self.adaptive_image = self.get_adaptive_image()
    

    def save_image(self, image, image_name=None):
        if image_name is None:
            image_name = self.image_name
        image.save(os.path.join(self.path, image_name))
        print(f'image saved as -> {self.path}/{image_name}')
    
    def get_adaptive_image(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        self.adaptive_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 99, 9)
        return self.adaptive_image
    
    def get_canny_image(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return self.canny_image