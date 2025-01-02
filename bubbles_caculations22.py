import cv2
import numpy as np
def find_student_answers(i,adaptiveFrame,contours,total_bubbles=50):
        bubbles_columns=5

        for cnt in contours:
            cv2.drawContours(adaptiveFrame, [cnt], -1, (0, 255, 0), 2)
        cv2.imshow(f'Oval Contours {i}', adaptiveFrame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(contours) != total_bubbles:
            print('Invalid frame: Number of detected bubbles does not match the expected count.')
            return None
        
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        sliced_contours =   [contours[i:i+bubbles_columns] for i in range(0, len(contours), bubbles_columns)]
        sorted_slices =     [sorted(slice, key=lambda c: cv2.boundingRect(c)[0]) for slice in sliced_contours]
        contours =          [contour for slice in sorted_slices for contour in slice]

        student_number = ''
        answers = []

        for row in range(0, total_bubbles, bubbles_columns):
            row_bubbles = ovalContours[row:row+bubbles_columns]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in row_bubbles]
            chosen_bubble_index = areas.index(max(areas))

            cv2.drawContours(warpedFrame, [row_bubbles[chosen_bubble_index]], -1, (0, 0, 255), 2)
            student_number += str(chosen_bubble_index)
            answers.append(chosen_bubble_index)

        # cv2.imshow(f'B_{i}', warpedFrame)
        # cv2.waitKey(0)
        cv2.imwrite(f'results/block_{i}_answers.jpg', warpedFrame)
        # cv2.destroyAllWindows()

        return answers

def find_student_number(i, adaptiveFrame,contours,total_bubbles=40):
        bubbles_rows = 10
        bubbles_columns = 4
        

        if len(contours) != total_bubbles:
            print('Invalid frame: Number of detected bubbles does not match the expected count.')
            return None

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        sliced_contours = [contours[i:i+bubbles_rows] for i in range(0, len(contours), bubbles_rows)]
        sorted_slices = [sorted(slice, key=lambda c: cv2.boundingRect(c)[1]) for slice in sliced_contours]
        contours = [contour for slice in sorted_slices for contour in slice]

        student_number = ''

        for col in range(0, total_bubbles, bubbles_rows):
            column_bubbles = contours[col:col+bubbles_rows]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in column_bubbles]
            chosen_bubble_index = areas.index(max(areas))

            cv2.drawContours(adaptiveFrame, [column_bubbles[chosen_bubble_index]], -1, (0, 0, 255), 2)
            student_number += str(chosen_bubble_index)
        
        # cv2.imshow(f'B_{i}', warpedFrame)
        # cv2.waitKey(0)
        cv2.imwrite(f'B_{i}_answers.jpg', adaptiveFrame)
        # cv2.destroyAllWindows()
        return student_number,adaptiveFrame
