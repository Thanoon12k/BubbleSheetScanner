from bubbles_caculations import find_student_answers, find_student_number
import cv2
import pandas as pd
import os
from GUI import get_pdf_path, display_student_results
# from image_preprocessing import getAdaptiveThresh, getCircularContours, get_sub_images, pdf_to_images, resize_images,get_sub_images_from_4_points,find_four_squares
from image_preprocessing import *
import tkinter as tk
from tkinter import filedialog
import shutil


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
