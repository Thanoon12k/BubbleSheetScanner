import cv2
import numpy as np

    
def find_student_answers(adaptiveFrame,counters,i):
        bubbles_rows = 25
        bubbles_columns = 5
        total_bubbles =125

        ovalContours = counters
        if len(ovalContours) != total_bubbles:
            print(f'Student Answers {i}__ bubbles: {len(ovalContours)} does not match the expected count.')
            return None

        contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[0])
        sliced_contours =   [contours[i:i+bubbles_columns] for i in range(0, len(contours), bubbles_columns)]
        sorted_slices =     [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
        contours =          [contour for slice in sorted_slices for contour in slice]

        student_number = ''
        answers = []

        for row in range(0, total_bubbles, bubbles_columns):
            row_bubbles = contours[row:row+bubbles_columns]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in row_bubbles]
            chosen_bubble_index = areas.index(max(areas))

            student_number += str(chosen_bubble_index)
            answers.append(chosen_bubble_index)

        # cv2.imshow(f'B_{i}', warpedFrame)
        # cv2.waitKey(0)
        # cv2.imwrite(f'results/block_{i}_answers.jpg', warpedFrame)
        # cv2.destroyAllWindows()

        return answers


def find_student_number(adaptiveFrame,frame_contours,i):
        bubbles_rows = 10
        bubbles_columns = 4
        total_bubbles =40
        
        ovalContours = frame_contours
        

        if len(ovalContours) != total_bubbles:
            print(f'Student number {i}__ bubbles: {len(ovalContours)} does not match the expected count.')

            return None

        contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[0])
        sliced_contours = [contours[i:i+bubbles_rows] for i in range(0, len(contours), bubbles_rows)]
        sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_contours]
        contours = [contour for slice in sorted_slices for contour in slice]


        student_number = ''

        for col in range(0, total_bubbles, bubbles_rows):
            
            column_bubbles = contours[col:col+bubbles_rows]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in column_bubbles]
            chosen_bubble_index = areas.index(max(areas))
            student_number += str(chosen_bubble_index)
        # cv2.imshow('ad',adaptiveFrame)
        # cv2.waitKey(0)
        return student_number
        # cv2.imshow(f'B_{i}', warpedFrame)
        # cv2.waitKey(0)
        # cv2.imwrite(f'results/B_{i}_answers.jpg', warpedFrame)
        # cv2.destroyAllWindows()
