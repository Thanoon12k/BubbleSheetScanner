import cv2
import numpy as np

def add_bubbles(column_contours):
    contours = sorted(column_contours, key=lambda c: cv2.boundingRect(c)[1])  # sort vertically
    while len(contours) < 10:
            diffs = []
            for i in range(0, len(contours)-1):
                x1 = cv2.boundingRect(contours[i])[1]
                x2 = cv2.boundingRect(contours[i + 1])[1]
                diffs.append( x2 - x1)
            
            avg_space = sum(diffs) // len(diffs)
            max_diff=max(diffs)
            max_diffs_index=diffs.index(max_diff)
            
            new_contour = np.copy(contours[max_diffs_index])
            new_contour[:, :, 1] += average_space
            contours.insert(max_diffs_index,new_contour)
            
    return new_contours
    
def fix_missing_contours(ovalContours, expected_count, bubbles_collection_direction='x'):
   if bubbles_collection_direction == 'y':
        contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[0])
        x_first, y, w, h = cv2.boundingRect(contours[0])
        bubble_Radios=int(w//2)
        #sort for answers
        all_colunms=[]
        column_contours=[]
        new_column_contours=[]
        for  c in contours:
            x, y, w, h = cv2.boundingRect(c)
            diff=abs(x-x_first)
            if diff<bubble_Radios:
                column_contours.append(c)
            else:
                cc_length=len(column_contours)
                if cc_length < 10:
                    new_column_contours=add_bubbles(column_contours)
                
                x_first=x
                all_colunms.append(new_column_contours)
                column_contours=[]

        
        return all_colunms
        
def find_student_answers(adaptiveFrame,sub_image_counters,i):
        bubbles_rows = 25
        bubbles_columns = 5
        total_bubbles =125

        ovalContours = sub_image_counters
        if len(ovalContours) < total_bubbles:
            print(f'invalid answers of bubbles {len(ovalContours)}__ : img_{i} does not match the expected count. 40')
            # ovalContours=fix_missing_counters(ovalContours,5)
            ovalContours=fix_missing_contours(ovalContours, bubbles_columns, bubbles_collection_direction='x')
        # Convert the adaptive frame to color to display colored contours
        

        contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[0])
        sliced_contours =   [contours[i:i+bubbles_columns] for i in range(0, len(contours), bubbles_columns)]
        sorted_slices =     [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
        contours =          [contour for slice in sorted_slices for contour in slice]

        student_number = ''
        answers = []
        answers_bubbles=[]
        for row in range(0, total_bubbles, bubbles_columns):
            row_bubbles = contours[row:row+bubbles_columns]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in row_bubbles]
            chosen_bubble_index = areas.index(max(areas))
            answers_bubbles.append(row_bubbles[chosen_bubble_index])
            student_number += str(chosen_bubble_index)
            answers.append(chosen_bubble_index)

        return [answers,answers_bubbles]


def find_student_number(adaptiveFrame, sub_image_counters, i,analyse=False):
    fm = cv2.cvtColor(adaptiveFrame.copy(), cv2.COLOR_GRAY2BGR)
    bubbles_rows = 10
    bubbles_columns = 4
    total_bubbles = 40

    ovalContours = sub_image_counters
    length_column=len(ovalContours)
    if length_column < total_bubbles:
        print(f'invalid number of bubbles {len(ovalContours)}__ : img_{i} does not match the expected count. 40')
        ovalContours = fix_missing_contours(ovalContours, bubbles_rows, bubbles_collection_direction='y')

    contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[0])
    sliced_contours = [contours[i:i + bubbles_rows] for i in range(0, len(contours), bubbles_rows)]
    sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_contours]
    contours = [contour for slice in sorted_slices for contour in slice]

    student_number = ''
   
    for i,c in enumerate(contours):
            print(f"COLUMN ({i}) ")
            x, y, w, h = cv2.boundingRect(c)
            ff=cv2.circle(fm.copy(),(int(x+w//2),int(y+h//2)),int(w//2),(0,0,255))
            cv2.imshow(f'_{i}', ff)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
