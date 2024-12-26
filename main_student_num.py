# Install required packages:
# pip install opencv-python numpy imutils scipy

import cv2
import numpy as np
from imutils.perspective import four_point_transform
import scipy


class BubbleSheetScanner:
        """
        A class for scanning and grading bubble sheet multiple-choice exams.
        """
        student_num_digits = 10
        student_number_bubbles_count = 4
        questionCount = 40  # Total number of questions
        bubbleCount = 4     # Number of bubbles (options) per question
        sqrAvrArea = 0      # Average area of square markers
        bubbleWidthAvr = 0  # Average width of bubbles
        bubbleHeightAvr = 0 # Average height of bubbles
        RealBubblesCount = questionCount * bubbleCount  # Total number of ovals (bubbles) on the sheet
        TotalSNBubbles = student_num_digits * student_number_bubbles_count

        # Answer key for grading

        ANSWER_KEY = {
            0: 1, 1: 4, 2: 2, 3: 0, 4: 1, 5: 3, 6: 2, 7: 4, 8: 0, 9: 3,
            10: 1, 11: 1, 12: 4, 13: 0, 14: 3, 15: 1, 16: 1, 17: 4, 18: 0, 19: 3,
            20: 1, 21: 1, 22: 3, 23: 4, 24: 0, 25: 3, 26: 1, 27: 1, 28: 4, 29: 0,
            30: 3, 31: 1, 32: 1, 33: 4, 34: 0, 35: 3, 36: 1, 37: 1, 38: 4, 39: 0,
            40: 3
        }

        def __init__(self):
            pass

        def getCannyFrame(self, frame,s1=127,s2=255):
            """
            Apply Canny edge detection to the input frame.
            """
            gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            frame = cv2.Canny(gray, s1, s2)
            return frame

        def getAdaptiveThresh(self, frame):
            """
            Apply adaptive thresholding to the input frame.
            """
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
            return adaptiveFrame

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

        def getOvalContours(self, adaptiveFrame):
            """
            Detect and return contours of oval-shaped bubbles in the image.
            """
            contours, hierarchy = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ovalContours = []

            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0, True)
                ret = 0
                x, y, w, h = cv2.boundingRect(contour)

                # Eliminating non-oval shapes by approximation length and aspect ratio
                if (len(approx) > 15 and w / h <= 1.2 and w / h >= 0.8):
                    mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    ret = cv2.matchShapes(mask, contour, 1, 0.0)

                    if (ret < 1):
                        ovalContours.append(contour)
                        self.bubbleWidthAvr += w
                        self.bubbleHeightAvr += h

            self.bubbleWidthAvr = self.bubbleWidthAvr / len(ovalContours)
            self.bubbleHeightAvr = self.bubbleHeightAvr / len(ovalContours)

            return ovalContours

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
        
        def mygetOvalContours(self, adaptiveFrame):
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
                if 0.8 <= aspect_ratio <= 1.2:
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
        def mygetAdaptiveThresh(self, frame):
                    """
                    Apply adaptive thresholding to the input frame with improved parameters.
                    """
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    adaptiveFrame = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 99,1)
                    return adaptiveFrame
# Create an instance of BubbleSheetScanner
bubbleSheetScanner = BubbleSheetScanner()

image = cv2.imread('3st.jpg')

# Read and resize the input image
h = int(round(600 * image.shape[0] / image.shape[1]))
frame = cv2.resize(image, (600, h), interpolation=cv2.INTER_LANCZOS4)

# Apply Canny edge detection
cannyFrame = bubbleSheetScanner.getCannyFrame(frame)

# Perform perspective transform
warpedFrame =frame
sub_images = bubbleSheetScanner.get_sub_images(warpedFrame)

# warpedFrame=sub_images[0]
# cv2.imshow(f'Sub Image {1}', warpedFrame)
# cv2.waitKey(0)
# Apply adaptive thresholding
adaptiveFrame = bubbleSheetScanner.mygetAdaptiveThresh(warpedFrame)
# cv2.imshow('result', adaptiveFrame)
# cv2.waitKey(0)

# Detect oval contours
# Detect oval contours
ovalContours = bubbleSheetScanner.mygetOvalContours(adaptiveFrame)
# ovalContours = sorted(ovalContours, key=cv2.contourArea, reverse=True)
# bubble_areas = [cv2.contourArea(contour) for contour in ovalContours]
# bubble_top_left = [tuple(contour[contour[:,:,1].argmin()][0]) for contour in ovalContours]
# bubble_diagonals = [np.linalg.norm(np.array(tuple(contour[contour[:,:,1].argmin()][0])) - np.array(tuple(contour[contour[:,:,1].argmax()][0]))) for contour in ovalContours]

# ovalContours = ovalContours[:BubbleSheetScanner.TotalSNBubbles]

# # Draw green edges around all contours
# contour_image = warpedFrame.copy()
# cv2.drawContours(contour_image, ovalContours, -1, (0, 255, 0), 2)


# # Display the image with contours
# cv2.imshow('Contours', contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np


student_number=find_student_number(bubbleSheetScanner,warpedFrame,adaptiveFrame)
print(f'student number ,{student_number}')