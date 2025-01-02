from bubbleScanner import BubbleSheetScanner
from bubbles_caculations import find_student_answers,find_student_number
import cv2
import pandas as pd
import os
from pdf2image import convert_from_path
from image_preprocessing import *

def calculate_student_score(student_answers, answer_key):
    score = sum(1 for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans)
    correct_indices = [q_num for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans]
    return score, correct_indices

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
    
    # Append the new row to the DataFrame and save
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(file_path, index=False)

    print("Results saved to student_results.csv")



def get_student_num_student_score(sub_images):
    
    student_answers = []
    student_number = ''
    # Process sub-images
    for i, img in enumerate(sub_images):
        ad_image=getAdaptiveThresh(img)
        counters=getCircularContours(ad_image)
        if i != 0:
            student_answers += find_student_answers(ad_image,counters, i)
            # print(f"block answers {(i-1)*25}-{i*25}: {student_answers[(i-1)*25:i*25]}")
        else:
            student_number = find_student_number( ad_image,counters, i)
            # print(f"student number : {student_number}")
    return student_number,student_answers



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
        save_images([ad_image,drawed_bubbles],f'page_{i}_',f'_({bubbles_count})_')  
 

# Initialize the scanner
scanner = BubbleSheetScanner()

pdf_path = 'aa.pdf'
pdf_images=pdf_to_images(pdf_path)
pdf_images=resize_images(pdf_images,1200)
# pdf_images=[cv2.imread("1.jpg")]
   
for i,image in enumerate(pdf_images):
    sub_images=get_sub_images(image)
    analyse_sub_images(sub_images,i)
    display_images([image],"hh",60)
    student_number,student_answers=get_student_num_student_score(sub_images)
    

    score, correct_indices = calculate_student_score(student_answers, scanner.ANSWER_KEY)
    write_results_to_csv(student_number, score, correct_indices)
    print(f"Student ( {student_number} ) score : {score}, correct_answers: {correct_indices}")
 
        
    # std_number,std_score=get_student_num_student_score(page_blocks)
    # print(f"({std_number}) :: score: {std_score}")
    
#     ad_image=getAdaptiveThresh(image)
#     bubbles=getCircularContours(ad_image)
#     bubbles_count=len(bubbles)
#     drawed_bubbles=draw_contours_on_frame(image.copy(),bubbles,blue=255)
#     # display_images([image,ad_image,drawed_bubbles],'original',)
#     print(f'image_{i}    {bubbles_count}')
#     save_images([image,ad_image,drawed_bubbles],f'neew_idea_page_({bubbles_count})_o{i}')
#     i=i+1

# print('SN:',student_num)
# print(f"answers:: {answers}")
# num_of_contours_thio=4*10 + 100*5
# num_of_contours=len(contours)


contour_areas = [cv2.contourArea(contour) for contour in contours]
sorted_contours = sorted(contours, key=cv2.contourArea)

# Sort contours by their y-axis (top to bottom)
# sorted_contours = sorted(sorted_contours, key=lambda c: cv2.boundingRect(c)[0]) #sort by x axis

# sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])        # sort by y axis
# Draw the identified corner contours
first_left_contour = sorted_contours[0]
last_right_contour = sorted_contours[-1]

# Draw a red box from the first to the last contour
x1, y1, _, _ = cv2.boundingRect(first_left_contour)
x2, y2, w2, h2 = cv2.boundingRect(last_right_contour)

rec2_p1 = (int((x1 + x2) / 3), y1)
rec2_p2 = ((x2), int((y2+y1)/3 ))
in_rect_counters=[]
# Draw green circles around each contour within the rectangle
for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)
    if rec2_p1[0] <= x <= rec2_p2[0] and rec2_p1[1] <= y <= rec2_p2[1]:
        center = (int(x + w / 2), int(y + h / 2))
        radius = int(min(w, h) / 2)
        in_rect_counters.append(contour)
        
# cv2.rectangle(pdf_images[0], top_left, bottom_right, (0, 0, 255), 2)
cv2.rectangle(pdf_images[0], rec2_p1, rec2_p2, (255, 0, 0), 2)

drawn_frame = draw_contours_on_frame(pdf_images[0], contours,blue=255)
drawn_frame = draw_contours_on_frame(pdf_images[0], in_rect_counters,red=255)


display_images([drawn_frame], 'd', 50)
save_images([drawn_frame], 'adaptive_drawn')

display_images([adaptive_images[0]],"A_")
save_images([adaptive_images[0]],'adapti')

# Draw the identified contours with red color
# drawn_frame = draw_contours_on_frame(pdf_images[0], corner_contours, red=255)
# smallest_five_contours = sorted_contours[:1]
# Identify contours that are not in any blocks
# sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

# save_images(pdf_images,'originall')
# save_images(canny_images,'canny')

# display_images(pdf_images,"R_")
# display_images(canny_images,"C_")

print('1')

# resised_images=resize_images(pdf_images)

# cornered_imgs=[add_for_squared_in_image_corners(img) for img in resised_images]
    
    # Add the function call in the main code


# display_images(pdf_images,"O_")
# display_images(cornered_imgs,"R_")
[display_images(get_sub_images(su),f'S_{i}') for i,su in enumerate(pdf_images)]
[display_images(get_sub_images(su),f'R_{i}') for i,su in enumerate(resised_images)]

for image in pdf_images:
    frame = cv2.resize(image, (1200, int(round(1200 * image.shape[0] / image.shape[1]))), interpolation=cv2.INTER_LANCZOS4)
    sub_images = scanner.getSubImages(frame)
    # for idx, sub_img in enumerate(sub_images):
    #     cv2.imshow(f'Sub Image {idx}', sub_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    student_number,student_answers=get_student_num_student_score(sub_images)
    score, correct_indices = calculate_student_score(student_answers, scanner.ANSWER_KEY)
    write_results_to_csv(student_number, score, correct_indices)
    print(f"Student ( {student_number} ) score : {score}")
