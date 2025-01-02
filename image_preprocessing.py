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

def resize_images(images, width=1200):
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]
        height = int(width / aspect_ratio)
        resized_img = cv2.resize(img, (width, height))
        resized_images.append(resized_img)
    return resized_images

def display_images(images, title="I", scale=100):
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
def save_images(images, folder_name,image_name_context=''):
    os.makedirs(folder_name, exist_ok=True)
    for i, img in enumerate(images):
        filename = os.path.join(folder_name, f"__{i}__{image_name_context}.jpg")
        cv2.imwrite(filename, img)
    print(f"Saved images to folder: {folder_name}")

def getCircularContours(adaptiveFrame,img=None,analyses=False):
        contours, _ = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if analyses:
            dd=draw_contours_on_frame(img.copy(),contours,red=255)
            display_images([dd])
        circularContours = []
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])  # Sort by y-axis
        total_width = 0
        total_height = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if analyses:
                dd=draw_contours_on_frame(img.copy(),contour,red=255)
                display_images([dd])
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)
                area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = float(area) / hull_area
                
                ll=-1
                if defects is not None:
                    ll=len(defects)
                print(f"ratio: ({w},{h}, {aspect_ratio})  defects: {ll}    solidity: {solidity}  ")
            if 0.7 <= aspect_ratio <= 1.5 and w > 18 and h > 18:
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)

                if defects is not None and len(defects) >= 5:
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
        shift=0
        image = np.array(image)
        height, width = image.shape[:2]
        number_y_factor=0.212
        answers_y_factor=0.39
        number_box_height_factor=0.6
        answers_box_height_factor=0.22

        ## block one x0_y0_start= 0,09    0,23       
        x0_start=int(0.09*width)
        y0_start=int(number_y_factor*height)
        x0_end=int(0.25*width)    # 0.09 is the first x 0.25*width is the last x of the student_number_block
        
        y0_end=int(y0_start+        ((x0_end-x0_start)//number_box_height_factor)         )
        ## block2 x,y start == 
        
       
        
        x1_start=0.1042*width+shift
        y1_start=answers_y_factor *height
        x1_end=0.2875*width+shift
        y1_end=int(y1_start+((x1_end-x1_start)//answers_box_height_factor))

        shift=shift+(0.225*width)
        x2_start=0.1042*width+shift
        y2_start=answers_y_factor *height
        x2_end=0.2875*width+shift
        y2_end=int(y2_start+((x2_end-x2_start)//answers_box_height_factor))
        
        shift=shift+(0.225*width)
        x3_start=0.1042*width+shift
        y3_start=answers_y_factor *height
        x3_end=0.2875*width+shift
        y3_end=int(y3_start+((x3_end-x3_start)//answers_box_height_factor))


        shift=shift+(0.225*width)
        x4_start=0.1042*width+shift
        y4_start=answers_y_factor *height
        x4_end=0.2875*width+shift
        y4_end=int(y4_start+((x4_end-x4_start)//answers_box_height_factor))


        coordinates = {
            'b1': [x0_start, y0_start, x0_end, y0_end],
            'b2': [int(x1_start), int(y1_start), int(x1_end), int(y1_end)],
            'b3': [int(x2_start), int(y2_start), int(x2_end), int(y2_end)],
            'b4': [int(x3_start), int(y3_start), int(x3_end), int(y3_end)],
            'b5': [int(x4_start), int(y4_start), int(x4_end), int(y4_end)],
        }
        
        block1 = image[coordinates['b1'][1]:coordinates['b1'][3], coordinates['b1'][0]:coordinates['b1'][2]]
        block2 = image[coordinates['b2'][1]:coordinates['b2'][3], coordinates['b2'][0]:coordinates['b2'][2]]
        block3 = image[coordinates['b3'][1]:coordinates['b3'][3], coordinates['b3'][0]:coordinates['b3'][2]]
        block4 = image[coordinates['b4'][1]:coordinates['b4'][3], coordinates['b4'][0]:coordinates['b4'][2]]
        block5 = image[coordinates['b5'][1]:coordinates['b5'][3], coordinates['b5'][0]:coordinates['b5'][2]]
        
        sub_images = []

        sub_images.append(block1)
        sub_images.append(block2)
        sub_images.append(block3)
        sub_images.append(block4)
        sub_images.append(block5)
        return sub_images
