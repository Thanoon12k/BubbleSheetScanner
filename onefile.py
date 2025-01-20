import cv2
import pandas as pd
import os
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog

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
 


def fix_missing_contours(ovalContours, expected_count, axis='x'):
    # Determine sorting axis
    axis_index = 0 if axis == 'x' else 1
    # Sort contours by the specified axis
    contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[axis_index])
    
    # Separate contours into groups based on axis distance
    groups = []
    current_group = [contours[0]]
    for contour in contours[1:]:
        if cv2.boundingRect(contour)[axis_index] - cv2.boundingRect(current_group[-1])[axis_index] <= 18:
            current_group.append(contour)
        else:
            groups.append(current_group)
            current_group = [contour]
    groups.append(current_group)
    
    # Add missing contours to groups with less than expected_count
    for group in groups:
        while len(group) < expected_count:
            # Duplicate the last contour in the group
            last_contour = group[-1]
            x, y, w, h = cv2.boundingRect(last_contour)
            if axis == 'x':
                new_contour = np.array([[[x, y + h + 1]]])  # Slightly offset the new contour
            else:
                new_contour = np.array([[[x + w + 1, y]]])  # Slightly offset the new contour
            group.append(new_contour)
    
    # Flatten the list of groups back into a single list of contours
    fixed_contours = [contour for group in groups for contour in group]
    return fixed_contours
    
def find_student_answers(adaptiveFrame,frame_contours,i):
        bubbles_rows = 25
        bubbles_columns = 5
        total_bubbles =125

        ovalContours = frame_contours
        if len(ovalContours) < total_bubbles:
            print(f'invalid answers of bubbles {len(ovalContours)}__ : img_{i} does not match the expected count. 40')
            # ovalContours=fix_missing_counters(ovalContours,5)
            ovalContours=fix_missing_contours(ovalContours, bubbles_columns, axis='Y')
        # Convert the adaptive frame to color to display colored contours
        # color_frame = cv2.cvtColor(adaptiveFrame.copy(), cv2.COLOR_GRAY2BGR)
        
        # # Draw contours on the frame for visualization
        # for contour in ovalContours:
        #     cv2.drawContours(color_frame, [contour], -1, (0, 255, 0), 2)
        
        # # Display the frame with contours
        # cv2.imshow(f'Contours_{i}', color_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[0])
        sliced_contours =   [contours[i:i+bubbles_columns] for i in range(0, len(contours), bubbles_columns)]
        sorted_slices =     [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
        contours =          [contour for slice in sorted_slices for contour in slice]

        student_number = ''
        answers = []

        for row in range(0, total_bubbles, bubbles_columns):
            row_bubbles = contours[row:row+bubbles_columns]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in row_bubbles]
            chosen_bubble_index = areas.index(max(areas))

            student_number += str(chosen_bubble_index)
            answers.append(chosen_bubble_index)

        return answers


def find_student_number(adaptiveFrame,frame_contours,i):
        bubbles_rows = 10
        bubbles_columns = 4
        total_bubbles =40
        
        ovalContours = frame_contours
        if len(ovalContours) < total_bubbles:
            print(f'invalid number of bubbles {len(ovalContours)}__ : img_{i} does not match the expected count. 40')
            ovalContours=fix_missing_contours(ovalContours, bubbles_rows, axis='x')

        
        contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[0])
        
        sliced_contours = [contours[i:i+bubbles_rows] for i in range(0, len(contours), bubbles_rows)]
        


        sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_contours]
        contours = [contour for slice in sorted_slices for contour in slice]
        
        
        

        student_number = ''

        for col in range(0, total_bubbles, bubbles_rows):
            
            column_bubbles = contours[col:col+bubbles_rows]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in column_bubbles]
            chosen_bubble_index = areas.index(max(areas))
            student_number += str(chosen_bubble_index)

        return student_number

def calculate_student_score(student_answers, answer_key):
    score = sum(1 for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans)
    correct_indices = [q_num for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans]
    return score, correct_indices

def get_student_num_student_score(sub_images):
    
    student_answers = []
    student_number = ''
    # Process sub-images
    for i, img in enumerate(sub_images):
        ad_image=getAdaptiveThresh(img)
        counters=getCircularContours(ad_image)
        if i == 0:
            student_number = find_student_number( ad_image,counters, i)
            # print(f"student number : {student_number}")
        else:
            student_answers += find_student_answers(ad_image,counters, i)
            # print(f"block answers {(i-1)*25}-{i*25}: {student_answers[(i-1)*25:i*25]}")
            
    return student_number,student_answers

def get_answers_from_xlsx(path):
    data = pd.read_excel(path)
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    answer_key = {}
    for i, row in data.iterrows():
                q=row[0]
                answer_key[i] = answer_mapping.get(row[1], None)  # Handle invalid answers gracefully
    return answer_key
def write_results_to_csv(student_number, score, correct_indices):
        # Define file path
        file_path = 'student_results.csv'

        # Load existing DataFrame or create a new one
        if os.path.exists(file_path):
            results_df = pd.read_csv(file_path)
        else:
            results_df = pd.DataFrame(columns=['Student Number'] + ['Score from 100'] + [f'Q{i+1}' for i in range(100)])

        # Create a new row for the student
        new_row = pd.DataFrame([[student_number] + [score] + ['True' if i in correct_indices else 'False' for i in range(100)]], columns=results_df.columns)
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
            pdf_images = pdf_to_images(pdf_path)
            pdf_images=resize_images(pdf_images,1200)
            # pdf_images=[cv2.imread("1.jpg")]


            root = tk.Tk()
            root.withdraw()  # Hide the root window

            for i, image in enumerate(pdf_images):
                sub_images = get_sub_images(image)
                student_number, student_answers = get_student_num_student_score(sub_images)
                score, correct_indices = calculate_student_score(student_answers, ANSWER_KEYS)
                write_results_to_csv(student_number, score, correct_indices)
                print(f"Student ( {student_number} ) score : {score}, correct_answers: {correct_indices}")

                display_student_results(student_number, score, root)

            root.mainloop()
