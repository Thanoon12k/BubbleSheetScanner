import cv2
import numpy as np
import os
from PIL import Image
import numpy as np
import fitz  
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def calculate_student_score(student_answers, answer_key):
    student_result = sum(1 for q_num, correct_ans in answer_key.items() if  student_answers[q_num] == correct_ans)
    correct_indices = [q_num for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans]    
    return student_result, correct_indices

def get_answers_from_xlsx(path):
    data = pd.read_excel(path)
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    answer_key = {}
    for i, row in data.iterrows():
                q=row.iloc[0]
                answer_key[i] = answer_mapping.get(row.iloc[1], None)  # Handle invalid answers gracefully
    return answer_key

def write_results_to_csv(student_number, student_result, correct_indices,student_answers):
        # Define file path
        file_path = 'student_results.csv'
        answer_mapping_reverse = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        student_answers_mapped = [answer_mapping_reverse.get(ans, 'N/A') for ans in student_answers]
        # Load existing DataFrame or create a new one
        if os.path.exists(file_path):
            results_df = pd.read_csv(file_path)
        else:
            results_df = pd.DataFrame(columns=['Student Number'] + ['student_result from 100'] + [f'Q{i+1}' for i in range(100)])

        # Create a new row for the student
        new_row = pd.DataFrame([[student_number] + [student_result] + student_answers_mapped], columns=results_df.columns)
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



def get_pdf_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path


def display_student_results1(student_number, score, root):
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



def display_images(images, title="I", scale=100):
    for i, img in enumerate(images):
        if scale != 100:
            width = int(img.shape[1] * scale / 100)
            height = int(img.shape[0] * scale / 100)
            img = cv2.resize(img, (width, height))
        
        cv2.imshow(f'{title}_{i}', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

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
def resize_images(images, width=1200):
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]
        height = int(width / aspect_ratio)
        resized_img = cv2.resize(img, (width, height))
        resized_images.append(resized_img)
    return resized_images
def getBubblesContours(img, adaptiveFrame, expected_count):
        min_ratio=0.99
        max_ratio=1.01  
        contours=[]
        while len(contours)<expected_count:
                contours, hierarchy = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                circular_contours = []

                for contour in contours:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    if len(approx) > 2:  # Assuming circular shapes have more than 3 vertices
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(h) / w
                        condition1= min_ratio <= aspect_ratio <= max_ratio and cv2.contourArea(contour) > 30  #aspect ration = h/w
                        condition2=not ((x >= img.shape[1] // 4 and y <= img.shape[0] // 2.9) or (y <= img.shape[0] // 7)) # not in top right quarter
                                        
                        if condition1 and condition2:  # Remove small dots
                            circular_contours.append(contour)

                contours = circular_contours
                if contours:
                    average_area = sum(cv2.contourArea(contour) for contour in contours) / len(contours)
                else:
                    average_area = 0
                filtered_contours = []
                for cnt in contours:
                    if average_area / 2 <= cv2.contourArea(cnt) <= average_area * 2:
                        filtered_contours.append(cnt)
                contours = filtered_contours 
        
      
                min_ratio-=0.01
                max_ratio+=0.01
        
        return contours


def add_miss_bubbles_to_row(fm,ref,row,normlize_value):
    ref = sorted(ref, key=lambda c: cv2.boundingRect(c)[0])
    row = sorted(row, key=lambda c: cv2.boundingRect(c)[0])
    filtered_row = []
    added_buubles=[]
    first_bubble_y = sorted(row, key=lambda c: cv2.boundingRect(c)[1])
    first_bubble_y = cv2.boundingRect(first_bubble_y[0])[1]
    for c in row:
        y = cv2.boundingRect(c)[1]
        if abs(y - first_bubble_y) < normlize_value:
            filtered_row.append(c)
    row = filtered_row
    # draw_contours_on_frame(fm,ref,color='g',display=True)
    # draw_contours_on_frame(fm,row,color='b',display=True)
    while len(row)<4:
        for i in range(0,3):
            equal_X=abs(cv2.boundingRect(ref[i])[0]-cv2.boundingRect(row[i])[0]) <normlize_value
            if not equal_X:
                new_bubble = ref[i].copy()
                for point in new_bubble:
                    point[0][1] += int(normlize_value * 2.5)        
                row.insert(i, new_bubble)
                added_buubles.append(new_bubble)
                # draw_contours_on_frame(fm,row,color='g',display=True)
                break

    

    return added_buubles

def find_ref_row(fm, cnts, num_columns):
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
    _, _, normlized_value, _ = cv2.boundingRect(cnts[0])
    # display_images([draw_contours_on_frame(fm, cnts, color='g')], scale=70)
    slice = []
    y_first = cv2.boundingRect(cnts[0])[1]
    for c in cnts:
        y = cv2.boundingRect(c)[1]
        if y_first - normlized_value < y < y_first + normlized_value:
            slice.append(c)
        else:
            y_first = y
            # display_images([draw_contours_on_frame(fm, slice, color='g')], scale=33)
            if len(slice) == num_columns:
                slice=sorted(slice, key=lambda c: cv2.boundingRect(c)[1])
                return slice
            slice = [c]

    return None

    


    
def add_answers_miss_bubbles_to_row(fm, row,ref, normlize_value,num_columns):
    _, _, w, _ = cv2.boundingRect(row[0])
    normlize_value = w // 2
    # Sort row based on the x-coordinate of the bounding rectangles
    added_bubbles=[]
    row = sorted(row, key=lambda c: cv2.boundingRect(c)[0])
    
    ref = sorted(ref, key=lambda c: cv2.boundingRect(c)[0])
    
    # Determine the y-coordinate of the first bubble
    first_bubble_y = sorted(row, key=lambda c: cv2.boundingRect(c)[1])
    first_bubble_y = cv2.boundingRect(first_bubble_y[0])[1]

    row = [c for c in row if abs(cv2.boundingRect(c)[1] - first_bubble_y) < normlize_value]
    miss_row=[c for c in row if abs(cv2.boundingRect(c)[1] - first_bubble_y) < normlize_value]

    while len(row)<num_columns:
        for i in range(0,num_columns-1):
            equal_X=abs(cv2.boundingRect(ref[i])[0]-cv2.boundingRect(row[i])[0]) <normlize_value
            
            if not equal_X:
                # print(f'missing bubble [{i+1}] ')
                new_bubble = ref[i].copy()
                y_offset = cv2.boundingRect(row[0])[1] - cv2.boundingRect(new_bubble)[1]
                new_bubble[:, :, 1] += y_offset
                added_bubbles.append(new_bubble)
                row.insert(i, new_bubble)



                
                # draw_contours_on_frame(fm,row,color='g',display=True)
                break

            
    # display_images([draw_contours_on_frame(fm, miss_row, color='b')], scale=33)
    # display_images([draw_contours_on_frame(fm, added_bubbles, color='r')], scale=33)
    return added_bubbles
    # display_images([draw_contours_on_frame(fm,row)],scale=70)

def find_student_number1(img,adaptiveFrame, image_contours, i,analyse=False):
    
    bubbles_rows = 10
    bubbles_columns = 4
    total_bubbles = 40
    chosen_bubbles=[]
    contours = sorted(image_contours, key=lambda c: cv2.boundingRect(c)[0])

    student_number=''
    for col in range(0, total_bubbles, bubbles_rows):
         
  
        column_bubbles = contours[col:col + bubbles_rows]
        column_bubbles=sorted(column_bubbles, key=lambda c: cv2.boundingRect(c)[1])
        areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in column_bubbles]
        chosen_bubble_index = areas.index(max(areas))
        chosen_bubbles.append(column_bubbles[chosen_bubble_index])
        student_number += str(chosen_bubble_index)
        
    return student_number,chosen_bubbles

def find_student_answers1(img,adaptiveFrame,sub_image_counters,i):
        bubbles_rows = 25
        bubbles_columns = 5
        total_bubbles =125
        answers = []
        answers_bubbles=[]
        contours = sorted(sub_image_counters, key=lambda c: cv2.boundingRect(c)[1])

        for row in range(0, total_bubbles, bubbles_columns):
            row_bubbles = contours[row:row+bubbles_columns]
            row_bubbles = sorted(row_bubbles, key=lambda c: cv2.boundingRect(c)[0])
            
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in row_bubbles]
            chosen_bubble_index = areas.index(max(areas))
            answers_bubbles.append(row_bubbles[chosen_bubble_index])
            answers.append(chosen_bubble_index)

        return answers,answers_bubbles

def fix_missing_num_bubbles(bw_img, cnts: list) -> list:
    Refererece_row = []
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
    _, _, w, _ = cv2.boundingRect(cnts[0])
    normlize_value = w // 2

    while len(cnts) < 40:
        for i in range(0, len(cnts) - 4, 4):
            contours_row = sorted(cnts[i:i + 4], key=lambda c: cv2.boundingRect(c)[1])
            y_values = [cv2.boundingRect(c)[1] for c in contours_row]
            is_not_good_row = max(y_values) - min(y_values) > normlize_value

            # draw_contours_on_frame(bw_img, [cnts[i], cnts[i + 1], cnts[i + 2], cnts[i + 3]], display=True,)
            
            if not is_not_good_row and Refererece_row == []:Refererece_row = contours_row
            
            if is_not_good_row:
                miss_bubbles = add_miss_bubbles_to_row(bw_img, Refererece_row, contours_row, normlize_value)
                cnts.extend(miss_bubbles)
                i = 0
                break

        # print(f"i->{i} from 40 new_rows= {len(cnts)}")
    cnts=sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])    
    return cnts


def fix_missing_block_bubbles(bw_img, cnts: list) -> list:
    
    ref_row=find_ref_row(bw_img,cnts,5)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
    _, _, w, _ = cv2.boundingRect(cnts[0])
    normlize_value = w // 2

    while len(cnts) < 125:
        for i in range(0, len(cnts) - 5, 5):
            contours_row = sorted(cnts[i:i + 5], key=lambda c: cv2.boundingRect(c)[1])
            y_values = [cv2.boundingRect(c)[1] for c in contours_row]
            
            is_not_good_row = max(y_values) - min(y_values) > normlize_value
            
            if is_not_good_row:
                miss_bubbles = add_answers_miss_bubbles_to_row(bw_img,contours_row,ref_row, normlize_value,5)
                cnts.extend(miss_bubbles)
                cnts=sorted(cnts, key=lambda c: cv2.boundingRect(c)[1]) 
                i = 0
                break

    
    # display_images([draw_contours_on_frame(bw_img,cnts,color='g')],scale=33)
    return cnts


def remove_not_aligned_bubbls(cnts: list) -> list:
    filterd_cnts=[]
    return cnts
def get_answers_blocks_bubbles(pg,fm,cnts):

    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
    # display_images([draw_contours_on_frame(fm,cnts,color='b')],scale=33)

    cnts = cnts[40:] 
    # display_images([draw_contours_on_frame(fm,cnts)],scale=50)
    bubble_columns=5
    bubbles_rows=25
    # Find the smallest x coordinate
    radious=int(cv2.boundingRect(cnts[0])[2] //2)
    # Get smallest and largest x, y coordinates (assuming 'cnts' is a list of contours)
    x1 = min(cv2.boundingRect(c)[0] for c in cnts)-radious
    y1 = min(cv2.boundingRect(c)[1] for c in cnts)-radious
    x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in cnts)+radious
    y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in cnts)+radious
    width = x2 - x1
    gap = 5*radious
    rect_width = (width-(gap*3)) // 4
    rects = []
    for i in range(5):
        start_x = x1 + (i * (rect_width + gap))
        rect = [start_x, y1, start_x + rect_width, y2]
        rects.append(rect)
        fm = draw_rect_on_frame(fm, rect, 2)
    blocks_cnts = [[] for _ in range(4)]
    for cnt in cnts:
        x, y, _, _ = cv2.boundingRect(cnt)
        for i, rect in enumerate(rects):
            min_x, min_y, max_x, max_y = rect
            if min_x <= x <= max_x and min_y <= y <= max_y :
                blocks_cnts[i].append(cnt)    
    # display_images([fm], scale=44)
    # display_images([draw_contours_on_frame(fm,blocks_cnts[0],color='r')],scale=35)
    # display_images([draw_contours_on_frame(fm,blocks_cnts[1],color='g')],scale=35)
    # display_images([draw_contours_on_frame(fm,blocks_cnts[2],color='r')],scale=35)
    # display_images([draw_contours_on_frame(fm,blocks_cnts[3],color='b')],scale=35)
    
    return blocks_cnts
    # gap = 5*radious
    # rect_width = (width-(gap*3)) // 4
    # rects = []
    # blocks_cnts = [[] for _ in range(4)]
    # for i in range(4):
    #     start_x = x1 + (i * (rect_width + gap))
    #     end_x = start_x + rect_width
    #     for cnt in cnts:
    #         x, _, _, _ = cv2.boundingRect(cnt)
    #         if start_x <= x <= end_x:
    #             blocks_cnts[i].append(cnt)
        
    draw_contours_on_frame(fm,blocks_cnts[3],display=True,scale=30) 
    # cv2.rectangle(pg, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Display the image with the rectangle
    # cv2.imshow("Image with Rectangle", pg)
    # cv2.waitKey(0)
    # display_images([draw_contours_on_frame(fm,first_xs)],scale=25,title='fx')
    # display_images([draw_contours_on_frame(fm,first_ys)],scale=25,title='fy')
    # display_images([draw_contours_on_frame(fm,last_xs)],scale=25,title='lx')
    # display_images([draw_contours_on_frame(fm,last_ys)],scale=25,title='ly')
    # x_first, _, _,_ = cv2.boundingRect(first_four_cnts[0])
    x_last, _, w, _ = cv2.boundingRect(first_four_cnts[-1])

    return []



def get_student_num_bubbles(bw_img,cnts:list)->list:

    number_cnts=[]
    bubble_columns=4
    bubbles_rows=10
    
    cnts=sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])[:40]
    ref=find_ref_row(bw_img,cnts,4)
    
    x_first = cv2.boundingRect(ref[0])[0]
    x_last = cv2.boundingRect(ref[-1])[0]
    
    normlize_value = cv2.boundingRect(ref[0])[2] // 2
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    first_ten_cnts=sorted(cnts[:bubbles_rows], key=lambda c: cv2.boundingRect(c)[1])
    
    _, y_first, _, _ = cv2.boundingRect(first_ten_cnts[0])
    _, y_last, _, _ = cv2.boundingRect(first_ten_cnts[-1])

    for cntr in cnts:
        _,y,_,_=cv2.boundingRect(cntr)
        if y>= (y_first-normlize_value)  and y<=(y_last+normlize_value):
            number_cnts.append(cntr)
    
    if len(number_cnts)<40:
        number_cnts=fix_missing_num_bubbles(bw_img,number_cnts)
                
    return number_cnts

def draw_rect_on_frame(fm, axis=[],thikness=4): #axis=[x1,x2,x3,x4]
    fm=fm.copy()
    if len(fm.shape) == 2:  # Grayscale frames have 2 dimensions
                fm = cv2.cvtColor(fm, cv2.COLOR_GRAY2BGR)
    if len(axis) == 4:
        x1, y1, x2, y2 = axis
        cv2.rectangle(fm, (x1, y1), (x2, y2), (0, 255, 0), thikness)
    return fm

def draw_rect_top_right_quarter(page):
    h, w, _ = page.shape
   
    cv2.rectangle(page,  (w // 4, 0)   , (w, int(h // 2.9)), 255, 14)  
    cv2.rectangle(page, (0,0), (w,h//7), 255, 14)  
    return page
def ggetAdaptiveThresh(frame,maxx=99,minn=9):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, maxx, minn)


def save_images(images, folder_name,image_name_context=''):
    os.makedirs(folder_name, exist_ok=True)
    for i, img in enumerate(images):
        image_name=f"page_[{image_name_context}].jpg"
        filename = os.path.join(folder_name, image_name)
        cv2.imwrite(filename, img)
    print(f"Saved images to folder: {folder_name}\\{image_name}")

def get_pdf_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Make the dialog appear on top
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path
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
