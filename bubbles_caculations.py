import cv2
import numpy as np

def fix_missing_contours(ovalContours, expected_count, axis='x'):
    # Determine sorting axis
    axis_index = 0 if axis == 'x' else 1
    # Sort contours by the specified axis
    contours = sorted(ovalContours, key=lambda c: cv2.boundingRect(c)[axis_index])
    
    # Separate contours into groups based on axis distance
    groups = []
    current_group = [contours[0]]
    for contour in contours[1:]:
        if cv2.boundingRect(contour)[axis_index] - cv2.boundingRect(current_group[-1])[axis_index] <= 18:
            current_group.append(contour)
        else:
            groups.append(current_group)
            current_group = [contour]
    groups.append(current_group)
    
    # Add missing contours to groups with less than expected_count
    for group in groups:
        while len(group) < expected_count:
            # Duplicate the last contour in the group
            last_contour = group[-1]
            x, y, w, h = cv2.boundingRect(last_contour)
            if axis == 'x':
                new_contour = np.array([[[x, y + h + 1]]])  # Slightly offset the new contour
            else:
                new_contour = np.array([[[x + w + 1, y]]])  # Slightly offset the new contour
            group.append(new_contour)
    
    # Flatten the list of groups back into a single list of contours
    fixed_contours = [contour for group in groups for contour in group]
    return fixed_contours
    
def find_student_answers(adaptiveFrame,frame_contours,i):
        bubbles_rows = 25
        bubbles_columns = 5
        total_bubbles =125

        ovalContours = frame_contours
        if len(ovalContours) < total_bubbles:
            print(f'invalid answers of bubbles {len(ovalContours)}__ : img_{i} does not match the expected count. 40')
            # ovalContours=fix_missing_counters(ovalContours,5)
            ovalContours=fix_missing_contours(ovalContours, bubbles_columns, axis='Y')
        # Convert the adaptive frame to color to display colored contours
        # color_frame = cv2.cvtColor(adaptiveFrame.copy(), cv2.COLOR_GRAY2BGR)
        
        # # Draw contours on the frame for visualization
        # for contour in ovalContours:
        #     cv2.drawContours(color_frame, [contour], -1, (0, 255, 0), 2)
        
        # # Display the frame with contours
        # cv2.imshow(f'Contours_{i}', color_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

        return answers


def find_student_number(adaptiveFrame,frame_contours,i):
        bubbles_rows = 10
        bubbles_columns = 4
        total_bubbles =40
        
        ovalContours = frame_contours
        if len(ovalContours) < total_bubbles:
            print(f'invalid number of bubbles {len(ovalContours)}__ : img_{i} does not match the expected count. 40')
            ovalContours=fix_missing_contours(ovalContours, bubbles_rows, axis='x')

        
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

        return student_number
