# Install required packages:
# pip install opencv-python numpy imutils scipy

import cv2
import numpy as np
from imutils.perspective import four_point_transform
import scipy
import main_student_num
class BubbleSheetScanner:
        """
        A class for scanning and grading bubble sheet multiple-choice exams.
        """
        bubbles_blocks=4
        bubbles_rows = 25
        bubbles_columns = 5
        student_number_rows = 10  # Total number of questions
        student_number_columns = 4  # Total number of questions
        sqrAvrArea = 0      # Average area of square markers
        bubbleWidthAvr = 0  # Average width of bubbles
        bubbleHeightAvr = 0 # Average height of bubbles
        total_bubbles=bubbles_rows*bubbles_columns
        total_student_num_bubbles=student_number_rows*student_number_columns

        # Answer key for grading

        ANSWER_KEY = {  0: 2,  1: 2,  2: 2,  3: 2,  4: 2,  5: 2,  6: 2,  7: 2,  8: 2,
                    9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2, 40: 2, 40: 2, 41: 2, 42: 2, 43: 2, 44: 2, 45: 2, 46: 2, 47: 2, 48: 2, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2, 64: 2, 65: 2, 66: 2, 67: 2, 68: 2, 69: 2, 70: 2, 71: 2, 72: 2, 73: 2, 74: 2, 75: 2, 76: 2, 77: 2, 78: 2, 79: 2, 80: 2,
                    81: 2,82: 2,83: 2,84: 2,85: 2,86: 2,87: 2,88: 2,89: 2,90: 2,91: 2,92: 2,
                    93: 2,94: 2,95: 2,96: 2,97: 2,98: 2,99: 2,
   }
        def __init__(sel2):
            pass

        def getCannyFrame(self, frame,s1=127,s2=255):
            """
            Apply Canny edge detection to the input frame.
            """
            gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            frame = cv2.Canny(gray, s1, s2)
            return frame

        
        def get_sub_images(self, warpedFrame):
            height, width = warpedFrame.shape[:2]

            # Coordinates for each frame (as percentages of image dimensions)
            coordinates = [
                {'top_left': (3.14, 1.6), 'bottom_right': (20.07, 23.27)},
                {'top_left': (5.47, 28.089), 'bottom_right': (30.11, 93.098)},
                {'top_left': (30.11, 28.089), 'bottom_right': (48.358, 93.098)},
                {'top_left': (54.744, 28.089), 'bottom_right': (72.99, 93.098)},
                {'top_left': (80.29, 28.089), 'bottom_right': (98.54, 93.098)}
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
            """
            Find four corner points of the bubble sheet in the image.
            """
            squareContours = []
            contours, hie = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                fourPoints = []
                i = 0
                for cnt in contours:
                    (x, y), (MA, ma), angle = cv2.minAreaRect(cnt)
                    epsilon = 0.04 * cv2.arcLength(cnt, False)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    approx_Length = len(approx)
                    if approx_Length == 4 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1 and (w > 12 and h > 12):
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        fourPoints.append((cx, cy))
                        squareContours.append(cnt)
                        i += 1
                return fourPoints, squareContours

        def getWarpedFrame(self, cannyFrame, frame):
            """
            Perform perspective transform to get a top-down view of the bubble sheet.
            """
            fourPoints, squareContours = bubbleSheetScanner.getFourPoints(cannyFrame)
            fourPoints = np.array(fourPoints, dtype="float32")

            # Draw the contours on the original frame
            cv2.drawContours(frame, squareContours, -1, (0, 255, 0), 2)
            cv2.imwrite("results/Four_Points_Frame.jpg", frame)
            fourContours = bubbleSheetScanner.getFourPoints(cannyFrame)[1]

            if len(fourPoints) >= 4:
                newFourPoints = []
                newFourPoints.append(fourPoints[0])
                newFourPoints.append(fourPoints[1])
                newFourPoints.append(fourPoints[len(fourPoints) - 2])
                newFourPoints.append(fourPoints[len(fourPoints) - 1])

                newSquareContours = []
                newSquareContours.append(fourContours[0])
                newSquareContours.append(fourContours[1])
                newSquareContours.append(fourContours[len(fourContours) - 2])
                newSquareContours.append(fourContours[len(fourContours) - 1])

                for cnt in newSquareContours:
                    area = cv2.contourArea(cnt)
                    self.sqrAvrArea += area

                self.sqrAvrArea = int(self.sqrAvrArea / 4)

                newFourPoints = np.array(newFourPoints, dtype="float32")

                return four_point_transform(frame, newFourPoints)
            else:
                return None

       
        def x_cord_contour(self, ovalContour):
            """
            Calculate the x-coordinate for sorting contours.
            """
            x, y, w, h = cv2.boundingRect(ovalContour)
            return y + x * self.bubbleHeightAvr

        def y_cord_contour(self, ovalContour):
            """
            Calculate the y-coordinate for sorting contours.
            """
            x, y, w, h = cv2.boundingRect(ovalContour)
            return x + y * self.bubbleWidthAvr
        
        def getOvalContours(self, adaptiveFrame):
            """
            Detect and return contours of oval-shaped bubbles in the image.
            """
            contours, _ = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ovalContours = []
            total_width = 0
            total_height = 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # Check if the contour is approximately oval-shaped
                if 0.8 <= aspect_ratio <= 1.2 and(w>20 and h>20):
                    # Use convexity defects to better identify oval shapes
                    hull = cv2.convexHull(contour, returnPoints=False)
                    defects = cv2.convexityDefects(contour, hull)
                    
                    if defects is not None and len(defects) > 5:
                        # Calculate solidity as an additional check
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
                    """
                    Apply adaptive thresholding to the input frame with improved parameters.
                    """
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    adaptiveFrame = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 99, 9)
                    return adaptiveFrame


# Create an instance of BubbleSheetScanner
Scanner = BubbleSheetScanner()

# Read and resize the input image
image = cv2.imread('4q.jpg')
h = int(round(600 * image.shape[0] / image.shape[1]))
frame = cv2.resize(image, (600, h), interpolation=cv2.INTER_LANCZOS4)

# Apply Canny edge detection
cannyFrame = Scanner.getCannyFrame(frame)

# Perform perspective transform
warpedFrame =frame
sub_images = Scanner.get_sub_images(warpedFrame)

# warpedFrame=sub_images[0]
# cv2.imshow(f'Sub Image {1}', warpedFrame)
# cv2.waitKey(0)
# Apply adaptive thresholding
adaptiveFrame = Scanner.mygetAdaptiveThresh(warpedFrame)

ovalContours = Scanner.mygetOvalContours(adaptiveFrame)

contour_image = cv2.cvtColor(adaptiveFrame, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, ovalContours, -1, (0, 255, 0), 2)

# Display the image with contours
# cv2.imshow('Contours on Adaptive Frame', contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the frame
# cv2.imwrite('results/adaptive_counters.jpg', contour_image)

def find_student_answers(Scanner, warpedFrame, adaptiveFrame):
    bubbles_rows = 25
    bubbles_columns = 5
    total_bubbls=bubbles_columns*bubbles_rows

    # Detect oval contours
    ovalContours = Scanner.mygetOvalContours(adaptiveFrame)

    # Check if the count matches the expected number of bubbles
    if len(ovalContours) != (bubbles_rows * bubbles_columns):
        print('Invalid frame: Number of detected bubbles does not match the expected count.')
        return None

    ovalContours = sorted(ovalContours, key=Scanner.y_cord_contour, reverse=False)
    sliced_contours = [ovalContours[i:i+bubbles_columns] for i in range(0, len(ovalContours), bubbles_columns)]
    sorted_slices = [sorted(slice, key=Scanner.x_cord_contour) for slice in sliced_contours]
    ovalContours = [contour for slice in sorted_slices for contour in slice]
    student_number = ''
    answers=[]
    for row in range(0,total_bubbls,bubbles_columns):
        row_pointer=row/5
        positions=[]
        indexes=[]
        areas=[]
        # Extract contours for each column    
        row_bubbles = ovalContours[row:row+bubbles_columns]
        max_intensity = 0
        chosen_bubble_index = -1
        for j, bubble in enumerate(row_bubbles):
            positions.append([bubble[0][0][0],bubble[0][0][1]])
            indexes.append([row,j])
            mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            masked_region = cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=mask)
            total_intensity = cv2.countNonZero(masked_region)
            areas.append(total_intensity)
            # # Draw the image after processing
            # cv2.imshow('Masked Region', masked_region)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        chosen_bubble_index=areas.index(max(areas))  
             
        cv2.drawContours(adaptiveFrame, [row_bubbles[chosen_bubble_index]], -1, (0, 0, 255), 2)
        
        if chosen_bubble_index != -1:
            student_number += str(chosen_bubble_index)
            answers.append(chosen_bubble_index) 

            # Draw blue edges around the chosen bubble for visualization
            cv2.drawContours(warpedFrame, [row_bubbles[chosen_bubble_index]], -1, (255, 0, 0), 2)

    # Display the final result
    cv2.imshow('Student Number Result', warpedFrame)
    cv2.waitKey(0)
    cv2.imwrite('results/adaptive_counters.jpg', warpedFrame)

    cv2.destroyAllWindows()
    print("answers: ",answers)

    return answers
student_number=main_student_num.find_student_number(Scanner,warpedFrame,adaptiveFrame)
student_answers=find_student_answers(Scanner,warpedFrame,adaptiveFrame)
print("student number:  ",student_number)
print("student answers: ",student_answers)
print("student corrected answers: ",student_answers)

correct_answers=Scanner.ANSWER_KEY

