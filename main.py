import cv2
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import shutil
from utils import *
import time
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


if __name__ == "__main__":
                
                start_time = time.time()
            
                ANSWER_KEYS=get_answers_from_xlsx('answers.xlsx')
            
                answers_bubbles_count=500
                student_number_bubbles_count=40
                total_bubbles_count=answers_bubbles_count+student_number_bubbles_count
                output_folder=''
                # pdf_path = get_pdf_path()
                pdf_images,output_folder = pdf_to_images('20.pdf')

                # pdf_images=[cv2.imread('20_students/page_1.png'),cv2.imread('20_students/page_2.png'),cv2.imread('20_students/page_3.png'),cv2.imread('20_students/page_4.png'),cv2.imread('20_students/page_5.png'),cv2.imread('20_students/page_6.png'),cv2.imread('20_students/page_7.png'),cv2.imread('20_students/page_8.png'),cv2.imread('20_students/page_9.png'),cv2.imread('20_students/page_10.png'),cv2.imread('20_students/page_11.png'),cv2.imread('20_students/page_12.png'),cv2.imread('20_students/page_13.png'),cv2.imread('20_students/page_14.png'),cv2.imread('20_students/page_15.png'),cv2.imread('20_students/page_16.png'),cv2.imread('20_students/page_17.png'),cv2.imread('20_students/page_18.png'),cv2.imread('20_students/page_19.png'),cv2.imread('20_students/page_20.png'),cv2.imread('20_students/page_21.png'),cv2.imread('20_students/page_22.png')]
                # pdf_images=[cv2.imread('20_students/page_8.png')]
                # pdf_images=[cv2.imread('blank.png')]
                # pdf_images=resize_images(pdf_images,1200)
                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder)

                for  i,img in enumerate(pdf_images):
                    student_result=0
                    student_answers = []
                    student_num_bubbles=[]
                    answers_bubbles=[]
                    student_number = ''
                    adaptive_frame=ggetAdaptiveThresh(img,maxx=55,minn=20)

                    total_bubbles=getBubblesContours(img,adaptive_frame,total_bubbles_count)
                
                    student_num_bubbles=get_student_num_bubbles(adaptive_frame,total_bubbles)

                    blocks_answers_bubbles=get_answers_blocks_bubbles(img,adaptive_frame,total_bubbles)
                
                
                    for i, block in enumerate(blocks_answers_bubbles):
                            blocks_answers_bubbles[i]=fix_missing_block_bubbles(adaptive_frame,block)
                            sub_ans,sub_ans_bubbles= find_student_answers1(img,adaptive_frame,blocks_answers_bubbles[i],i)
                            answers_bubbles+=sub_ans_bubbles
                            student_answers+=sub_ans

                    all_answers_bubbles = [bubble for block in blocks_answers_bubbles for bubble in block] + student_num_bubbles
                    all_bubbles = all_answers_bubbles+ student_num_bubbles
                
                
                    # print(f'fix numbers image  [{i+1}]    - from {len(student_num_bubbles)} - {len(answers_bubbles)} = 580  found {len(total_bubbles)}  ')
                
                    student_number,std_num_bubbles=find_student_number1(img,adaptive_frame,student_num_bubbles,i)
                
                    student_result, correct_indices = calculate_student_score(student_answers, ANSWER_KEYS)
                    all_choosen_frame=draw_contours_on_frame(img,answers_bubbles+std_num_bubbles,color='b')
                    cv2.putText(all_choosen_frame, f"Student: {student_number}, student_result: {student_result}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
                
                    # display_images([all_choosen_frame],"adaptive img",scale=37)
                    save_images   ([draw_contours_on_frame(img,answers_bubbles+student_num_bubbles,color='b')],f'{output_folder}_archive',f"[{i+1}]")
    
                    write_results_to_csv(student_number, student_result, correct_indices,student_answers)                
                    print(f"Student ( {student_number} ) student_result : {student_result}, correct_answers: {correct_indices}")
            
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Total execution time: {execution_time:.2f} seconds")