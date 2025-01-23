import cv2
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import shutil

import fitz  
from PIL import Image
import numpy as np

def pdf_to_images(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_students")
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

    return images,output_folder

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
def draw_rect_on_frame(frame, axis,color='r',display=False,scale=100):
        x1,y1,x2,y2=axis[0],axis[1],axis[2],axis[3]
        
        if len(frame.shape) == 2:  # Grayscale frames have 2 dimensions
                frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        if scale != 100:
                width = int(frame.shape[1] * scale / 100)
                height = int(frame.shape[0] * scale / 100)
                frame = cv2.resize(frame, (width, height))


        blue,green,red=0,0,0
        if color=='r':red=255
        elif color=='b':blue=255
        elif color=='g':green=255

        cv2.rectangle(frame, (x1, y1), (x2, y2), (blue, green, red), 4)
        if display:
            display_images([frame])
        return frame

def draw_contours_on_frame(frame, contours,color='r',display=False,scale=100):
    if len(frame.shape) == 2:  # Grayscale frames have 2 dimensions
            frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
    if scale != 100:
            width = int(frame.shape[1] * scale / 100)
            height = int(frame.shape[0] * scale / 100)
            frame = cv2.resize(frame, (width, height))


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


def add_missing_bubbles(cnts, expected_count,sorted_slices):
        cnts_length = len(cnts)
        bubble_radius = cv2.boundingRect(cnts[0])[2]

        if expected_count == 125:
            for slice in sorted_slices:
                y_first = cv2.boundingRect(slice[0])[1]
                for c in slice:
                    y = cv2.boundingRect(c)[1]
                    diff = abs(y - y_first)
                    if diff > bubble_radius:
                        missing_bubble = np.copy(c)
                        missing_bubble[:, :, 1] -= diff
                        cnts.append(missing_bubble)
                        return cnts
        elif expected_count == 40:
            for slice in sorted_slices:
                x_first = cv2.boundingRect(slice[0])[0]
                for c in slice:
                    x = cv2.boundingRect(c)[0]
                    diff = abs(x - x_first)
                    if diff > bubble_radius:
                        missing_bubble = np.copy(c)
                        missing_bubble[:, :, 0] -= diff
                        cnts.append(missing_bubble)
                        return cnts
        return "error"
        
def find_student_answers(img,adaptiveFrame,sub_image_counters,i):
        bubbles_rows = 25
        bubbles_columns = 5
        total_bubbles =125
        ad=getAdaptiveThresh(img)
        ad2=OrigingetAdaptiveThresh(img)

        cnts=[]
        cnts1 = getCircularContours(ad)
        cnts2 = getCircularContours(ad2)

        cnts = cnts1 if len(cnts1) == 40 else cnts2
            
        cnts_lenth=len(cnts)
        # display_images([draw_contours_on_frame(img,cnts)],scale=70)
        
        
        while len(cnts) < total_bubbles:
            print(f'invalid answers of bubbles {len(cnts)}__ : img_{i} does not match the expected count. 40')
            # cnts=fix_missing_counters(cnts,5)
            cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
            sliced_contours = [cnts[i:i+bubbles_columns] for i in range(0, cnts_lenth, bubbles_columns)]
            sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
            cnts = [contour for slice in sorted_slices for contour in slice]

            cnts=add_missing_bubbles(cnts, 125,sorted_slices )
        # Convert the adaptive frame to color to display colored contours
        
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
        sliced_contours = [cnts[i:i+bubbles_columns] for i in range(0, cnts_lenth, bubbles_columns)]
        sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
        cnts = [contour for slice in sorted_slices for contour in slice]

        student_number = ''
        answers = []
        answers_bubbles=[]
        for row in range(0, total_bubbles, bubbles_columns):
            row_bubbles = cnts[row:row+bubbles_columns]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in row_bubbles]
            chosen_bubble_index = areas.index(max(areas))
            answers_bubbles.append(row_bubbles[chosen_bubble_index])
            student_number += str(chosen_bubble_index)
            answers.append(chosen_bubble_index)

        return [answers,answers_bubbles]


def find_student_number(img,adaptiveFrame, image_contours, i,analyse=False):
    
    fm = cv2.cvtColor(adaptiveFrame.copy(), cv2.COLOR_GRAY2BGR)

    bubbles_rows = 10
    bubbles_columns = 4
    total_bubbles = 40
    ad=getAdaptiveThresh(img)
    ad2=OrigingetAdaptiveThresh(img)

    cnts=[]
    cnts1 = getCircularContours(ad)
    cnts2 = getCircularContours(ad2)

    cnts = cnts1 if len(cnts1) == 40 else cnts2
        
    cnts_lenth=len(cnts)
    # draw_contours_on_frame(img,cnts,display=True)
    while len(cnts) < total_bubbles:
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        sliced_cnts = [cnts[i:i + bubbles_rows] for i in range(0, len(cnts), bubbles_rows)]
        sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_cnts]
        cnts = [contour for slice in sorted_slices for contour in slice]

        print(f'invalid number of bubbles {len(cnts)}__ : img_{i} does not match the expected count. 40')
        cnts = add_missing_bubbles(cnts, 40,sorted_slices )

    contours = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
    sliced_contours = [contours[i:i + bubbles_rows] for i in range(0, len(contours), bubbles_rows)]
    sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_contours]
    contours = [contour for slice in sorted_slices for contour in slice]

    student_number = ''
   
    # for i,c in enumerate(contours):
    #         print(f"COLUMN ({i}) ")
    #         x, y, w, h = cv2.boundingRect(c)
    #         ff=cv2.circle(fm.copy(),(int(x+w//2),int(y+h//2)),int(w//2),(0,0,255))
    #         cv2.imshow(f'_{i}', ff)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    chosen_bubbles=[]
    for col in range(0, total_bubbles, bubbles_rows):
        column_bubbles = contours[col:col + bubbles_rows]
        areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in column_bubbles]
        chosen_bubble_index = areas.index(max(areas))
        chosen_bubbles.append(column_bubbles[chosen_bubble_index])
        if analyse:
            
            print(f"COLUMN ({col}) index={chosen_bubble_index}")
            cv2.drawContours(fm, column_bubbles[chosen_bubble_index], -1, (0,0,255), 2)
            cv2.imshow(f'_{chosen_bubble_index}', fm)
            cv2.waitKey(0)
        student_number += str(chosen_bubble_index)

    return student_number,chosen_bubbles

def calculate_student_score(student_answers, answer_key):

    score = sum(1 for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans)
    correct_indices = [q_num for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans]
    return score, correct_indices

# def get_student_num_student_score(sub_images):
#     student_answers = []
#     student_number = ''
#     for i, img in enumerate(sub_images):
#         ad_image=getAdaptiveThresh(img)
#         counters=getCircularContours(ad_image)

#         if i == 0:
#             student_number = find_student_number(img,ad_image,counters, i)
#             # print(f"student number : {student_number}")
#         else:
#             student_answers += find_student_answers(ad_image,counters, i)
#             # print(f"block answers {(i-1)*25}-{i*25}: {student_answers[(i-1)*25:i*25]}")
            
#     return student_number,student_answers

def get_pdf_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path



def display_student_results(student_number, score, root):
    result_window = tk.Toplevel(root)
    result_window.title("Student Results")

    # Center the window on the screen
    window_width = 300
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    result_window.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    # Set background color
    result_window.configure(bg='lightblue')

    # Create and pack the labels and button with colors
    tk.Label(result_window, text=f"Student Number: {student_number}", font=("Helvetica", 16), bg='lightblue', fg='darkblue').pack(pady=10)
    tk.Label(result_window, text=f"Score: {score}", font=("Helvetica", 16), bg='lightblue', fg='darkblue').pack(pady=10)
    tk.Button(result_window, text="OK", font=("Helvetica", 14), bg='darkblue', fg='white', command=result_window.destroy).pack(pady=10)


def get_answers_from_xlsx(path):
    data = pd.read_excel(path)
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    answer_key = {}
    for i, row in data.iterrows():
                q=row[0]
                answer_key[i] = answer_mapping.get(row[1], None)  # Handle invalid answers gracefully
    return answer_key
def write_results_to_csv(student_number, score, correct_indices,student_answers):
        # Define file path
        file_path = 'student_results.csv'
        answer_mapping_reverse = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        student_answers_mapped = [answer_mapping_reverse.get(ans, 'N/A') for ans in student_answers]
        # Load existing DataFrame or create a new one
        if os.path.exists(file_path):
            results_df = pd.read_csv(file_path)
        else:
            results_df = pd.DataFrame(columns=['Student Number'] + ['Score from 100'] + [f'Q{i+1}' for i in range(100)])

        # Create a new row for the student
        new_row = pd.DataFrame([[student_number] + [score] + student_answers_mapped], columns=results_df.columns)
        student_number = str(student_number)
        all_students_numbers=results_df['Student Number'].astype(str).values
        
        if student_number in all_students_numbers:
            # Update the existing row
            results_df.loc[results_df['Student Number'] == student_number] = new_row.values[0]
        else:
            # Append the new row to the DataFrame
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Save the DataFrame to CSV
        results_df.to_csv(file_path, index=False)


if __name__ == "__main__":

            ANSWER_KEYS=get_answers_from_xlsx('answers.xlsx')
            student_number=0000
            score=0

            pdf_path = get_pdf_path()
            pdf_images,output_folder = pdf_to_images(pdf_path)
            pdf_images=resize_images(pdf_images,1200)
            # pdf_images=[cv2.imread("Scan1.jpg")]
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            # four_points=find_four_squares(pdf_images[0],analyses=True)
            
            root = tk.Tk()
            root.withdraw()  # Hide the root window

            for pp, image in enumerate(pdf_images):
                student_answers = []
                student_number = ''
                canny=OrigingetCannyFrame(pdf_images[0])
                adaptive=OrigingetCannyFrame(pdf_images[0])
                # display_images([canny],'',50)
                # display_images([adaptive],'',50)
                four_points=find_four_squares(image,analyses=False)
                sub_images =get_five_sub_images(image,four_points)

                for i, img in enumerate(sub_images):
                    # ad_image=getAdaptiveThresh(img)
                    adaptive=getAdaptiveThresh(img)
                    # display_images([adaptive],'ORiginal',100)
                    # adaptive=getAdaptiveThresh(img)
                    # display_images([adaptive],'mine',100)
                    counters=getCircularContours(adaptive,img)
                    n_contours=len(counters)
                    # draw_contours_on_frame(img.copy(),counters,color='b',display=False)
                    # save_images([img],'results',f"img_{pp}")

                    if i == 0:
                        student_number,chosen_bubbles = find_student_number(img,adaptive,counters, i,analyse=False)
                        print(f"student number : {student_number}")
                        # draw_contours_on_frame(img,chosen_bubbles,color='r',display=True)
                    
                        
                    else:
                        
                        block_answers,chosen_bubbles_ans=find_student_answers(img,adaptive,counters, i)
                        student_answers += block_answers
                        print(f"block answers {(i-1)*25}-{i*25}: {student_answers[(i-1)*25:i*25]}")
                        # display_images([draw_contours_on_frame(img,chosen_bubbles_ans)],scale=75)

                        
                
                score, correct_indices = calculate_student_score(student_answers, ANSWER_KEYS)
                write_results_to_csv(student_number, score, correct_indices,student_answers)
                print(f"Student ( {student_number} ) score : {score}, correct_answers: {correct_indices}")

                display_student_results(student_number, score, root)


            root.mainloop()
