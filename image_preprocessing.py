import fitz  # PyMuPDF
import os
from PIL import Image
import cv2
import numpy as np


def pdf_to_images(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_images")
    os.makedirs(output_folder, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
   
    page_list = list(range(len(pdf_document)))

    images = []
    # Convert specified pages
    for page_num in page_list:
        page = pdf_document.load_page(page_num)  # Load page
        pix = page.get_pixmap(dpi=200)  # Render page to an image with 300 DPI
        output_image_path = os.path.join(output_folder, f'page_{page_num + 1}.png')
        pix.save(output_image_path)  # Save image
        
        with Image.open(output_image_path) as img:
            images.append(np.array(img))
            img.save(output_image_path)
        
    print(f"Images saved in: {output_folder}")  

    return images
def resize_images(images,w=800,h=1000):
    for img in images:
        img = img.resize((w, h))
    return images

def display_images(images, title, scale=100):
    for i, img in enumerate(images):
        if scale != 100:
            width = int(img.shape[1] * scale / 100)
            height = int(img.shape[0] * scale / 100)
            img = cv2.resize(img, (width, height))
        
        cv2.imshow(f'{title}_{i}', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_for_squared_in_image_corners(image):
      
        # Define the size of the square to be added in the corners
        square_size = 50
        color = (0, 255, 0)  # Green color in BGR

        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Define the coordinates for the squares in the corners
        corners = [
            (0, 0),  # Top-left corner
            (0, width - square_size),  # Top-right corner
            (height - square_size, 0),  # Bottom-left corner
            (height - square_size, width - square_size)  # Bottom-right corner
        ]

        # Draw the squares in the corners
        for (y, x) in corners:
            image[y:y + square_size, x:x + square_size] = color

        return image

def getCannyFrame( frame, s1=127, s2=255):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        return cv2.Canny(gray, s1, s2)

def getAdaptiveThresh(frame,maxx=99,minn=9):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, maxx, minn)
def save_images(images, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for i, img in enumerate(images):
        filename = os.path.join(folder_name, f"{i}.jpg")
        cv2.imwrite(filename, img)
    print(f"Saved images to folder: {folder_name}")

def getCircularContours(adaptiveFrame):
        contours, _ = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularContours = []
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])  # Sort by y-axis
        total_width = 0
        total_height = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if 0.8 <= aspect_ratio <= 1.4 and w > 20 and h > 20:
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)

                if defects is not None and len(defects) > 5:
                    area = cv2.contourArea(contour)
                    hull_area = cv2.contourArea(cv2.convexHull(contour))
                    solidity = float(area) / hull_area

                    if solidity > 0.9:
                        circularContours.append(contour)
                        total_width += w
                        total_height += h

        if circularContours:
            bubbleWidthAvr = total_width / len(circularContours)
            bubbleHeightAvr = total_height / len(circularContours)

        return circularContours

def draw_contours_on_frame(frame, contours,red=0,green=0,blue=0):
    
    cv2.drawContours(frame, contours, -1, (blue, green, red), 2)
    return frame
def get_sub_images(image):
        image = np.array(image)
        height, width = image.shape[:2]
        coordinates = [
            {'top_left': (8, 22), 'bottom_right': (27, 41)},  # student number image
            {'top_left': (10, 40), 'bottom_right': (30, 90)},  # questions 1 to 25
            {'top_left': (33, 40), 'bottom_right': (52, 90)},  # questions 26 to 50
            {'top_left': (56, 40), 'bottom_right': (74, 90)},  # questions 51 to 75
            {'top_left': (79, 40), 'bottom_right': (96, 90)},  # questions 76 to 100
        ]

        sub_images = []
        for coord in coordinates:
            x1 = int(coord['top_left'][0] * width / 100)
            y1 = int(coord['top_left'][1] * height / 100)
            x2 = int(coord['bottom_right'][0] * width / 100)
            y2 = int(coord['bottom_right'][1] * height / 100)
            sub_image = image[y1:y2, x1:x2]
            sub_images.append(sub_image)

        return sub_images