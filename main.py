import cv2
import pandas as pd
import os

import shutil
from utils import *



if __name__ == "__main__":
                
            
            
                answers_bubbles_count=500
                student_number_bubbles_count=40
                total_bubbles_count=answers_bubbles_count+student_number_bubbles_count
                output_folder=''
                pdf_path = get_pdf_path()
                pdf_images,output_folder = pdf_to_images(pdf_path)

                # pdf_images=[cv2.imread('20_students/page_1.png'),cv2.imread('20_students/page_2.png'),cv2.imread('20_students/page_3.png'),cv2.imread('20_students/page_4.png'),cv2.imread('20_students/page_5.png'),cv2.imread('20_students/page_6.png'),cv2.imread('20_students/page_7.png'),cv2.imread('20_students/page_8.png'),cv2.imread('20_students/page_9.png'),cv2.imread('20_students/page_10.png'),cv2.imread('20_students/page_11.png'),cv2.imread('20_students/page_12.png'),cv2.imread('20_students/page_13.png'),cv2.imread('20_students/page_14.png'),cv2.imread('20_students/page_15.png'),cv2.imread('20_students/page_16.png'),cv2.imread('20_students/page_17.png'),cv2.imread('20_students/page_18.png'),cv2.imread('20_students/page_19.png'),cv2.imread('20_students/page_20.png'),cv2.imread('20_students/page_21.png'),cv2.imread('20_students/page_22.png')]
                # pdf_images=[cv2.imread('20_students/page_8.png')]
                # pdf_images=[cv2.imread('blank.png')]
                # pdf_images=resize_images(pdf_images,1200)
                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder)
                
                root = tk.Tk()
                root.withdraw()  # Hide the root window

                for  i,img in enumerate(pdf_images):
                    student_result=0
                    student_answers = []
                    student_number = ''
                    choosen_std_num_bubbles=[]
                    choosen_answers_bubbles=[]
                    
                    adaptive_frame=ggetAdaptiveThresh(img,maxx=55,minn=20)

                    total_bubbles=getBubblesContours(img,adaptive_frame,total_bubbles_count)
                
                    student_num_bubbles=get_student_num_bubbles(adaptive_frame,total_bubbles)

                    student_number,choosen_std_num_bubbles=find_student_number1(img,adaptive_frame,student_num_bubbles,i)
                    blocks_answers_bubbles=get_answers_blocks_bubbles(img,adaptive_frame,total_bubbles)
                
                
                    for j, block in enumerate(blocks_answers_bubbles):
                            if (len(block) < answers_bubbles_count):
                                blocks_answers_bubbles[j]=fix_missing_block_bubbles(adaptive_frame,block)
                            sub_ans,sub_ans_bubbles= find_student_answers1(img,adaptive_frame,blocks_answers_bubbles[j],j)
                            choosen_answers_bubbles+=sub_ans_bubbles
                            student_answers+=sub_ans

                    all_image_bubbles = [bubble for block in blocks_answers_bubbles for bubble in block] + student_num_bubbles
                
                
                    # print(f'fix numbers image  [{i+1}]    - from {len(student_num_bubbles)} - {len(answers_bubbles)} = 580  found {len(total_bubbles)}  ')
                
                    ANSWER_KEYS=get_answers_from_xlsx('answers.xlsx')

                    student_result, correct_indices = calculate_student_score(student_answers, ANSWER_KEYS)
                    all_choosen_frame=draw_contours_on_frame(img,choosen_answers_bubbles+choosen_std_num_bubbles,color='b')
                    cv2.putText(all_choosen_frame, f"Student: {student_number}, student_result: {student_result}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
                    save_images   ([all_choosen_frame],f'{output_folder}_archive',i+1)
    
                    write_results_to_csv(student_number, student_result, correct_indices,student_answers)                
                    print(f"Student ( {student_number} ) student_result : {student_result}, correct_answers: {correct_indices}")
                    display_student_results1(student_number, student_result, root)
                # root.update()
                root.mainloop()
                    
                    # root.mainloop()  # Start the Tkinter event loop
