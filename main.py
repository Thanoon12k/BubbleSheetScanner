import cv2
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import shutil
from utils import *

def calculate_student_score(student_answers, answer_key):

    # Calculate score by comparing student answers with answer key
    # Sum 1 point for each correct answer where:
    # - q_num is less than total student answers (to avoid index errors)
    # - student answer matches correct answer from key
    score = sum(1 for q_num, correct_ans in answer_key.items() if  student_answers[q_num] == correct_ans)

    # Get list of question numbers (indices) where student answered correctly
    # Uses same comparison logic as score calculation
    correct_indices = [q_num for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans]    
    return score, correct_indices

def get_answers_from_xlsx(path):
    data = pd.read_excel(path)
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    answer_key = {}
    for i, row in data.iterrows():
                q=row.iloc[0]
                answer_key[i] = answer_mapping.get(row.iloc[1], None)  # Handle invalid answers gracefully
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


            # ANSWER_KEYS=get_answers_from_xlsx('answers.xlsx')
            student_number=0000
            score=0
            answers_bubbles=500
            std_number_bubbles=40
            total_bubbles=540
            output_folder=''
            # pdf_path = get_pdf_path()
            # pdf_images,output_folder = pdf_to_images('20.pdf')
            # pdf_images=[cv2.imread('20_students/page_1.png'),cv2.imread('20_students/page_2.png'),cv2.imread('20_students/page_3.png'),cv2.imread('20_students/page_4.png'),cv2.imread('20_students/page_5.png'),cv2.imread('20_students/page_6.png'),cv2.imread('20_students/page_7.png'),cv2.imread('20_students/page_8.png'),cv2.imread('20_students/page_9.png'),cv2.imread('20_students/page_10.png'),cv2.imread('20_students/page_11.png'),cv2.imread('20_students/page_12.png'),cv2.imread('20_students/page_13.png'),cv2.imread('20_students/page_14.png'),cv2.imread('20_students/page_15.png'),cv2.imread('20_students/page_16.png'),cv2.imread('20_students/page_17.png'),cv2.imread('20_students/page_18.png'),cv2.imread('20_students/page_19.png'),cv2.imread('20_students/page_20.png'),cv2.imread('20_students/page_21.png'),cv2.imread('20_students/page_22.png')]
            pdf_images=[cv2.imread('20_students/page_8.png')]
            # pdf_images=[cv2.imread('blank.png')]
            # pdf_images=resize_images(pdf_images,1200)
            

            for  i,page in enumerate(pdf_images):
                student_answers = []
                std_number_bubbles=[]
                answers_bubbles=[]
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
                
                std_num_cnts     =get_student_num_bubbles(adaptive_frame,cnts)
                
                if len(std_num_cnts)<40:
                    print(f'image  [{i+1}]    - from 40  found {len(std_num_cnts)}  ')
                    std_num_cnts=fix_missing_num_bubbles(adaptive_frame,std_num_cnts)
                
                student_number,std_num_bubbles=find_student_number1(page,adaptive_frame,std_num_cnts,i)
                print('student_number',student_number)
                # display_images([draw_contours_on_frame(adaptive_frame,std_num_bubbles)],scale=80)
                answers_blocks=get_answers_blocks_bubbles(page,adaptive_frame,cnts)
                new_blocks=[]
                for blk in answers_blocks:
                        bb=fix_missing_block_bubbles(adaptive_frame,blk)
                        new_blocks.append(bb)
                
                               # Flatten all blocks into a single list
                for block in new_blocks:

                    sub_ans,sub_ans_bubbles= find_student_answers1(page,adaptive_frame,block,i)
                    answers_bubbles+=sub_ans_bubbles
                    student_answers+=sub_ans
                # display_images([draw_contours_on_frame(adaptive_frame,answers_bubbles)],scale=35)

                answers_cnts = [bubble for block in new_blocks for bubble in block] + std_num_cnts
                all_bubbles = answers_cnts+ std_num_cnts
                
                
                print(f'fix numbers image  [{i+1}]    - from {len(std_num_cnts)} - {len(answers_cnts)} = 580  found {len(all_bubbles)}  ')
                
                # print(f'fix numbers image  [{i+1}]    - from 40  found {len(std_num_cnts)}  ')
                # for c in std_num_cnts:
                #                   draw_contours_on_frame(adaptive_frame,[c],display=True,color='r')
  
   
                # print(len(std_num_cnts))
                # aligned_cnts=remove_not_aligned_counters(adaptive_frame,cnts)
                # print(f'image  [{i+1}]    - from 40  found {len(std_num_cnts)}  min_ratio={min_ratio},max_ratio={max_ratio}')
                # page=draw_contours_on_frame(adaptive_frame,cnts)
                # page=draw_rect_top_right_quarter(page)
                
                score, correct_indices = calculate_student_score(student_answers, ANSWER_KEYS)
                ans_num_frame=draw_contours_on_frame(page,answers_bubbles+std_num_bubbles,color='b')
                
                cv2.putText(ans_num_frame, f"Student: {student_number}, Score: {score}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
                
                display_images([ans_num_frame],"adaptive page",scale=37)
                save_images   ([draw_contours_on_frame(page,answers_bubbles+std_num_bubbles,color='b')],f'{output_folder}_archive',f"[{i+1}]")
    
                write_results_to_csv(student_number, score, correct_indices,student_answers)                
                print(f"Student ( {student_number} ) score : {score}, correct_answers: {correct_indices}")

                # display_student_results(student_number, score, root)
                
            # print(f"found results for  {len(pdf_images)} students succsessfully !!")