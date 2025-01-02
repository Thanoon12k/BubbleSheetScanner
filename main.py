from bubbles_caculations import find_student_answers, find_student_number
import cv2
import pandas as pd
import os
from image_preprocessing import *
from GUI import display_student_results, get_pdf_path
# Install the required packages using the following command:
# pip install opencv-python pandas pillow openpyxl




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
        if i != 0:
            student_answers += find_student_answers(ad_image,counters, i)
            # print(f"block answers {(i-1)*25}-{i*25}: {student_answers[(i-1)*25:i*25]}")
        else:
            student_number = find_student_number( ad_image,counters, i)
            # print(f"student number : {student_number}")
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
    
    # Append the new row to the DataFrame and save
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(file_path, index=False)

    # print("Results saved to student_results.csv")





ANSWER_KEYS=get_answers_from_xlsx('answers.xlsx')
student_number=0000
score=0

pdf_path = get_pdf_path()
pdf_images = pdf_to_images(pdf_path)
pdf_images=resize_images(pdf_images,1200)
# pdf_images=[cv2.imread("1.jpg")]
   
for i,image in enumerate(pdf_images):
    
    sub_images=get_sub_images(image)
    # analyse_sub_images(sub_images,i)
    student_number,student_answers=get_student_num_student_score(sub_images)
    score, correct_indices = calculate_student_score(student_answers, ANSWER_KEYS)
    write_results_to_csv(student_number, score, correct_indices)
    print(f"Student ( {student_number} ) score : {score}, correct_answers: {correct_indices}")


# Display the results
display_student_results(student_number, score)