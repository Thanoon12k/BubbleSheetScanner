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
        
    # print(f"Images saved in: {output_folder}")  

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

def getCircularContours(adaptiveFrame,canny_frame=None,analyses=False):
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

def draw_contours_on_frame(frame, contours,color='r',display=False,add_colors=False):
    if add_colors:
        frame=cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)

    blue,green,red=0,0,0
    if color=='r':red=255
    elif color=='b':blue=255
    elif color=='g':green=255
    cv2.drawContours(frame, contours, -1, (blue, green, red), 2)
    if display:
        display_images([frame])
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

def analyse_sub_images(sub_images,i):
    for sub_img in sub_images:
        ad_image=getAdaptiveThresh(sub_img)

        bubbles=getCircularContours(ad_image)
        bubbles_count=len(bubbles)
        if bubbles_count != 40 and bubbles_count !=125:
                bubbles=getCircularContours(ad_image,sub_img,analyses=True)
                bubbles_count=len(bubbles)
        drawed_bubbles=draw_contours_on_frame(sub_img.copy(),bubbles,blue=255)
        
        display_images([drawed_bubbles],"img",100)
        print(f'sub_img{i}_    {bubbles_count}')
        # save_images([ad_image,drawed_bubbles],f'page_{i}_',f'_({bubbles_count})_')  

def find_four_squares(frame, analyses=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.9 <= aspect_ratio <= 1.1 and w < frame.shape[1] / 10:  # Check if the contour is approximately square and width is less than frame width/10
                squares.append(approx)
    
    # Sort squares by area and select the largest 4
    squares = sorted(squares, key=cv2.contourArea, reverse=True)[:4]
    
    if analyses:
        analyzed_frame = frame.copy()
        cv2.drawContours(analyzed_frame, squares, -1, (0, 255, 0), 2)
        display_images([analyzed_frame], "Analyzed Squares", 30)
    
    squares = sorted(squares, key=lambda c: cv2.boundingRect(c)[1])  # Sort by y-axis
    squares[-2:] = sorted(squares[-2:], key=lambda c: cv2.boundingRect(c)[0])  # Sort by x-axis

    return squares

def get_five_sub_images(image, four_points):
        image = np.array(image)
        sub_images = []
        
        b1_x, b1_y = map(int, four_points[0][0][0]) #point 1
        b2_x, b2_y = map(int, four_points[1][0][0]) #point 2
        
        b4_x, b4_y = map(int, four_points[3][0][0])#point 3
        square_width = cv2.boundingRect(four_points[0])[2]
        lower_blocks_x =b2_x 
        lower_blocks_y = int(b2_y + (square_width//1.5))
        lower_blocks_x2 = b4_x + square_width
        lower_blocks_y2 = b4_y 
        lower_block_width =int( (lower_blocks_x2 - lower_blocks_x) // 4  )
        upper_block_x1=lower_blocks_x
        upper_block_x2=int(lower_block_width +(square_width//2))
        upper_block_y1=b1_y+square_width
        upper_block_y2=b2_y
        cv2.rectangle(image, (upper_block_x1, upper_block_y1), (upper_block_x2, upper_block_y2), (0, 255, 0), 2)
        sub_images.append(image[upper_block_y1:upper_block_y2, upper_block_x1:upper_block_x2])

        for i in range(0,lower_block_width*4,lower_block_width):
            x_start = lower_blocks_x + i
            x_end = x_start + lower_block_width
            cv2.rectangle(image, (x_start, lower_blocks_y), (x_end, lower_blocks_y2), (255, 0, ), 2)
            sub_image = image[lower_blocks_y:lower_blocks_y2, x_start:x_end]
            sub_images.append(sub_image)
        # display_images([image], "Lower Blocks", 30)
        
        return sub_images


def OrigingetFourPoints( canny):
        """
        Find four corner points of the bubble sheet in the image.
        """
        squareContours = []
        contours, hie = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            fourPoints = []
            i = 0
            for cnt in contours:
                (x, y), (MA, ma), angle = cv2.minAreaRect(cnt)
                epsilon = 0.04 * cv2.arcLength(cnt, False)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                approx_Length=len(approx)
                if approx_Length == 4 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    fourPoints.append((cx, cy))
                    squareContours.append(cnt)
                    i += 1
            return fourPoints, squareContours

def OrigingetCannyFrame( frame):
        """
        Apply Canny edge detection to the input frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        frame = cv2.Canny(gray, 127, 255)
        return frame

def OrigingetAdaptiveThresh(frame):
    """
    Apply adaptive thresholding to the input frame with enhanced effect.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)  # Increase blur effect
    adaptiveFrame = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
    return adaptiveFrame

def OrigingetOvalContours( adaptiveFrame):
        """
        Detect and return contours of oval-shaped bubbles in the image.
        """
        contours, hierarchy = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ovalContours = []

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0, True)
            ret = 0
            x, y, w, h = cv2.boundingRect(contour)

            # Eliminating non-oval shapes by approximation length and aspect ratio
            if (len(approx) > 15 and w / h <= 1.2 and w / h >= 0.8):
                mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)
                ret = cv2.matchShapes(mask, contour, 1, 0.0)

                if (ret < 1):
                    ovalContours.append(contour)
                    

        return ovalContours