import cv2
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import shutil

from onefile import *
from utils import *

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

# def get_answers_from_xlsx(path):
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

            # ANSWER_KEYS=get_answers_from_xlsx('answers.xlsx')
            student_number=0000
            score=0
            answers_bubbles=500
            std_number_bubbles=40
            total_bubbles=540
            # pdf_path = get_pdf_path()
            # pdf_images,output_folder = pdf_to_images(pdf_path)
            # pdf_images=[cv2.imread('20_students/page_1.png'),cv2.imread('20_students/page_2.png'),cv2.imread('20_students/page_3.png'),cv2.imread('20_students/page_4.png'),cv2.imread('20_students/page_5.png'),cv2.imread('20_students/page_6.png'),cv2.imread('20_students/page_7.png'),cv2.imread('20_students/page_8.png'),cv2.imread('20_students/page_9.png'),cv2.imread('20_students/page_10.png'),cv2.imread('20_students/page_11.png'),cv2.imread('20_students/page_12.png'),cv2.imread('20_students/page_13.png'),cv2.imread('20_students/page_14.png'),cv2.imread('20_students/page_15.png'),cv2.imread('20_students/page_16.png'),cv2.imread('20_students/page_17.png'),cv2.imread('20_students/page_18.png'),cv2.imread('20_students/page_19.png'),cv2.imread('20_students/page_20.png'),cv2.imread('20_students/page_21.png'),cv2.imread('20_students/page_22.png')]
            pdf_images=[cv2.imread('20_students/page_8.png')]
            # pdf_images=[cv2.imread('blank.png')]
            # pdf_images=resize_images(pdf_images,1200)
            

            for  i,page in enumerate(pdf_images):
                student_answers = []
                student_number = ''
                min_ratio=0.99
                max_ratio=1.01
                adaptive_frame=ggetAdaptiveThresh(page,maxx=55,minn=20)
                cnts_length=0
                    
                while cnts_length<total_bubbles and min_ratio>0:
                        cnts=getBubblesContours(page,adaptive_frame,540,min_ratio,max_ratio)
                        
                        cnts_length=len(cnts)
                        min_ratio-=0.01
                        max_ratio+=0.01
                number_cnts=get_number_bubbles(adaptive_frame,cnts)
                if len(number_cnts)<40:
                    fix_missing_num_bubbles(adaptive_frame,number_cnts)
                # print(len(number_cnts))
                # aligned_cnts=remove_not_aligned_counters(adaptive_frame,cnts)
                print(f'image  [{i+1}]    - from 40  found {len(number_cnts)}  min_ratio={min_ratio},max_ratio={max_ratio}')
                # page=draw_contours_on_frame(adaptive_frame,cnts,add_colors=True)
                # page=draw_rect_top_right_quarter(page)
                
                # save_images   ([page],'22',f"[{i+1}]")
                # display_images([page],"adaptive page",scale=50)
                # display_images([draw_contours_on_frame(page,aligned_cnts,color='b')],"adaptive page",scale=50)
                
                # student_number,choosen_bubbles_number = find_student_number(img,adaptive,counters, i,analyse=False)
                # print(f"student number : {student_number}")
                # student_answers,choosen_bubbles_answers=find_student_answers(img,adaptive,counters, i)
                # print(f"student answers {student_answers}")
                
                # score, correct_indices = calculate_student_score(student_answers, ANSWER_KEYS)
                # write_results_to_csv(student_number, score, correct_indices,student_answers)
                # print(f"Student ( {student_number} ) score : {score}, correct_answers: {correct_indices}")

                # display_student_results(student_number, score, root)
                
            print(f"found results for  {len(pdf_images)} students succsessfully !!")