import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pandas as pd


class BubbleSheetScanner:
    """
    A class for scanning and grading bubble sheet multiple-choice exams.
    """
    bubbles_blocks = 4
    bubbles_rows = 25
    bubbles_columns = 5
    student_number_rows = 10
    student_number_columns = 4
    sqrAvrArea = 0
    bubbleWidthAvr = 0
    bubbleHeightAvr = 0
    total_bubbles = bubbles_rows * bubbles_columns
    total_student_num_bubbles = student_number_rows * student_number_columns

    ANSWER_KEY = {i: 2 for i in range(100)}

    def __init__(self):
        pass

    def getCannyFrame(self, frame, s1=127, s2=255):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        return cv2.Canny(gray, s1, s2)

    def getSubImages(self, warpedFrame):
        height, width = warpedFrame.shape[:2]
        coordinates = [
            {'top_left': (8, 22), 'bottom_right': (27, 41)},  # student number image
            {'top_left': (10, 40), 'bottom_right': (30, 90)},  # questions 1 to 25
            {'top_left': (33, 40), 'bottom_right': (52, 90)},  # questions 26 to 50
            {'top_left': (56, 40), 'bottom_right': (74, 90)},  # questions 51 to 75
            {'top_left': (79, 40), 'bottom_right': (96, 90)},  # questions 76 to 100
        ]

        sub_images = []
        for coord in coordinates:
            x1 = int(coord['top_left'][0] * width / 100)
            y1 = int(coord['top_left'][1] * height / 100)
            x2 = int(coord['bottom_right'][0] * width / 100)
            y2 = int(coord['bottom_right'][1] * height / 100)
            sub_image = warpedFrame[y1:y2, x1:x2]
            sub_images.append(sub_image)

        return sub_images

    def getFourPoints(self, canny):
        squareContours = []
        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fourPoints = []

        for cnt in contours:
            epsilon = 0.04 * cv2.arcLength(cnt, False)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            if len(approx) == 4 and 0.9 <= aspect_ratio <= 1.1 and w > 12 and h > 12:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                fourPoints.append((cx, cy))
                squareContours.append(cnt)

        return fourPoints, squareContours

    def getWarpedFrame(self, cannyFrame, frame):
        fourPoints, squareContours = self.getFourPoints(cannyFrame)
        fourPoints = np.array(fourPoints, dtype="float32")

        if len(fourPoints) >= 4:
            newFourPoints = [fourPoints[0], fourPoints[1], fourPoints[-2], fourPoints[-1]]
            newFourPoints = np.array(newFourPoints, dtype="float32")
            return four_point_transform(frame, newFourPoints)
        else:
            return None

    def x_cord_contour(self, ovalContour):
        x, y, w, h = cv2.boundingRect(ovalContour)
        return y + x * self.bubbleHeightAvr

    def y_cord_contour(self, ovalContour):
        x, y, w, h = cv2.boundingRect(ovalContour)
        return x + y * self.bubbleWidthAvr

    def getOvalContours(self, adaptiveFrame):
        contours, _ = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ovalContours = []
        total_width = 0
        total_height = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if 0.8 <= aspect_ratio <= 1.4 and w > 20 and h > 20:
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)

                if defects is not None and len(defects) > 5:
                    area = cv2.contourArea(contour)
                    hull_area = cv2.contourArea(cv2.convexHull(contour))
                    solidity = float(area) / hull_area

                    if solidity > 0.9:
                        ovalContours.append(contour)
                        total_width += w
                        total_height += h

        if ovalContours:
            self.bubbleWidthAvr = total_width / len(ovalContours)
            self.bubbleHeightAvr = total_height / len(ovalContours)

        return ovalContours

    def getAdaptiveThresh(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 99, 9)
    
    def find_student_answers(self, warpedFrame, adaptiveFrame,i):
        bubbles_rows = self.bubbles_rows
        bubbles_columns = self.bubbles_columns
        total_bubbles = bubbles_columns * bubbles_rows

        ovalContours = self.getOvalContours(adaptiveFrame)
        for contour in ovalContours:
            cv2.drawContours(warpedFrame, [contour], -1, (0, 255, 0), 2)
        cv2.imshow(f'Oval Contours {i}', warpedFrame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(ovalContours) != total_bubbles:
            print('Invalid frame: Number of detected bubbles does not match the expected count.')
            return None

        ovalContours = sorted(ovalContours, key=self.y_cord_contour, reverse=False)
        sliced_contours = [ovalContours[i:i+bubbles_columns] for i in range(0, len(ovalContours), bubbles_columns)]
        sorted_slices = [sorted(slice, key=self.x_cord_contour) for slice in sliced_contours]
        ovalContours = [contour for slice in sorted_slices for contour in slice]

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

    def find_student_number(self, warpedFrame, adaptiveFrame,i):
        bubbles_rows = self.student_number_rows
        bubbles_columns = self.student_number_columns
        total_bubbles = bubbles_columns * bubbles_rows
        # cv2.imshow(f'BB_{i}', warpedFrame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ovalContours = self.getOvalContours(adaptiveFrame)
        # for contour in ovalContours:
        #     cv2.drawContours(warpedFrame, [contour], -1, (0, 255, 0), 2)
        # cv2.imshow(f'Oval Contours {i}', warpedFrame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if len(ovalContours) != total_bubbles:
            print('Invalid frame: Number of detected bubbles does not match the expected count.')
            return None

        ovalContours = sorted(ovalContours, key=self.x_cord_contour, reverse=False)
        sliced_contours = [ovalContours[i:i+bubbles_rows] for i in range(0, len(ovalContours), bubbles_rows)]
        sorted_slices = [sorted(slice, key=self.y_cord_contour, reverse=False) for slice in sliced_contours]
        ovalContours = [contour for slice in sorted_slices for contour in slice]

        student_number = ''

        for col in range(0, total_bubbles, bubbles_rows):
            column_bubbles = ovalContours[col:col+bubbles_rows]
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=cv2.drawContours(np.zeros(adaptiveFrame.shape, dtype="uint8"), [bubble], -1, 255, -1))) for bubble in column_bubbles]
            chosen_bubble_index = areas.index(max(areas))

            cv2.drawContours(warpedFrame, [column_bubbles[chosen_bubble_index]], -1, (0, 0, 255), 2)
            student_number += str(chosen_bubble_index)
        
        # cv2.imshow(f'B_{i}', warpedFrame)
        # cv2.waitKey(0)
        cv2.imwrite(f'results/B_{i}_answers.jpg', warpedFrame)
        # cv2.destroyAllWindows()
        return student_number
