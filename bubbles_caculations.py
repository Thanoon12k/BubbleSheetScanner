import cv2
import numpy as np
from image_preprocessing import getAdaptiveThresh,OrigingetAdaptiveThresh,getCircularContours,display_images,draw_contours_on_frame


def add_missing_bubbles(cnts, expected_count,sorted_slices):
        cnts_length = len(cnts)
        bubble_radius = cv2.boundingRect(cnts[0])[2]

        if expected_count == 125:
            for slice in sorted_slices:
                y_first = cv2.boundingRect(slice[0])[1]
                for c in slice:
                    y = cv2.boundingRect(c)[1]
                    diff = abs(y - y_first)
                    if diff > bubble_radius:
                        missing_bubble = np.copy(c)
                        missing_bubble[:, :, 1] -= diff
                        cnts.append(missing_bubble)
                        return cnts
        elif expected_count == 40:
            for slice in sorted_slices:
                x_first = cv2.boundingRect(slice[0])[0]
                for c in slice:
                    x = cv2.boundingRect(c)[0]
                    diff = abs(x - x_first)
                    if diff > bubble_radius:
                        missing_bubble = np.copy(c)
                        missing_bubble[:, :, 0] -= diff
                        cnts.append(missing_bubble)
                        return cnts
        return "error"
        
def find_student_answers(img,adaptiveFrame,sub_image_counters,i):
        bubbles_rows = 25
        bubbles_columns = 5
        total_bubbles =125
        ad=getAdaptiveThresh(img)
        ad2=OrigingetAdaptiveThresh(img)

        cnts=[]
        cnts1 = getCircularContours(ad)
        cnts2 = getCircularContours(ad2)

        cnts = cnts1 if len(cnts1) == 40 else cnts2
            
        cnts_lenth=len(cnts)
        # display_images([draw_contours_on_frame(img,cnts)],scale=70)
        
        
        while len(cnts) < total_bubbles:
            print(f'invalid answers of bubbles {len(cnts)}__ : img_{i} does not match the expected count. 40')
            # cnts=fix_missing_counters(cnts,5)
            cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
            sliced_contours = [cnts[i:i+bubbles_columns] for i in range(0, cnts_lenth, bubbles_columns)]
            sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
            cnts = [contour for slice in sorted_slices for contour in slice]

            cnts=add_missing_bubbles(cnts, 125,sorted_slices )
        # Convert the adaptive frame to color to display colored contours
        
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
        sliced_contours = [cnts[i:i+bubbles_columns] for i in range(0, cnts_lenth, bubbles_columns)]
        sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
        cnts = [contour for slice in sorted_slices for contour in slice]

        student_number = ''
        answers = []
        answers_bubbles=[]
        for row in range(0, total_bubbles, bubbles_columns):
            row_bubbles = cnts[row:row+bubbles_columns]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in row_bubbles]
            chosen_bubble_index = areas.index(max(areas))
            answers_bubbles.append(row_bubbles[chosen_bubble_index])
            student_number += str(chosen_bubble_index)
            answers.append(chosen_bubble_index)

        return [answers,answers_bubbles]


def find_student_number(img,adaptiveFrame, image_contours, i,analyse=False):
    
    fm = cv2.cvtColor(adaptiveFrame.copy(), cv2.COLOR_GRAY2BGR)

    bubbles_rows = 10
    bubbles_columns = 4
    total_bubbles = 40
    ad=getAdaptiveThresh(img)
    ad2=OrigingetAdaptiveThresh(img)

    cnts=[]
    cnts1 = getCircularContours(ad)
    cnts2 = getCircularContours(ad2)

    cnts = cnts1 if len(cnts1) == 40 else cnts2
        
    cnts_lenth=len(cnts)
    # draw_contours_on_frame(img,cnts,display=True)
    while len(cnts) < total_bubbles:
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        sliced_cnts = [cnts[i:i + bubbles_rows] for i in range(0, len(cnts), bubbles_rows)]
        sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_cnts]
        cnts = [contour for slice in sorted_slices for contour in slice]

        print(f'invalid number of bubbles {len(cnts)}__ : img_{i} does not match the expected count. 40')
        cnts = add_missing_bubbles(cnts, 40,sorted_slices )

    contours = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
    sliced_contours = [contours[i:i + bubbles_rows] for i in range(0, len(contours), bubbles_rows)]
    sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_contours]
    contours = [contour for slice in sorted_slices for contour in slice]

    student_number = ''
   
    # for i,c in enumerate(contours):
    #         print(f"COLUMN ({i}) ")
    #         x, y, w, h = cv2.boundingRect(c)
    #         ff=cv2.circle(fm.copy(),(int(x+w//2),int(y+h//2)),int(w//2),(0,0,255))
    #         cv2.imshow(f'_{i}', ff)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
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
