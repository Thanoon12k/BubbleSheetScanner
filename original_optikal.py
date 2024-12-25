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

    questionCount = 40  # Total number of questions
    bubbleCount = 5     # Number of bubbles (options) per question
    sqrAvrArea = 0      # Average area of square markers
    bubbleWidthAvr = 0  # Average width of bubbles
    bubbleHeightAvr = 0 # Average height of bubbles
    ovalCount = questionCount * bubbleCount  # Total number of ovals (bubbles) on the sheet

    # Answer key for grading
    ANSWER_KEY = {
        0: 1,
         1: 4,
          2: 2,
           3: 0,
            4: 1,
             5: 3,
              6: 2,
               7: 4,
                8: 0,
                 9: 3,
        10: 1,
         11: 1,
          12: 4,
           13: 0,
            14: 3,
                          15: 1,
                          16: 1,
                          17: 4,
                          18: 0,
                          19: 3,
                         
                        20: 1,
                      21: 1,
                      22: 3,
                      23: 4,
                      24: 0,
                      25: 3,
                      26: 1,
                      27: 1,
                      28: 4,
                      29: 0,
                     
                    30: 3,
                      31: 1,
                      32: 1,
                      33: 4,
                      34: 0,
                      35: 3,
                      36: 1,
                      37: 1,
                      38: 4,
                      39: 0,
                     
                        40: 3
    }

    def __init__(self):
        pass

    def getCannyFrame(self, frame):
        """
        Apply Canny edge detection to the input frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        frame = cv2.Canny(gray, 127, 255)
        return frame

    def getAdaptiveThresh(self, frame):
        """
        Apply adaptive thresholding to the input frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
        return adaptiveFrame

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
                approx_Length=len(approx)
                if approx_Length == 4 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
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
        fourPoints = np.array(bubbleSheetScanner.getFourPoints(cannyFrame)[0], dtype="float32")
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


# Create an instance of BubbleSheetScanner
bubbleSheetScanner = BubbleSheetScanner()

# Read and resize the input image
image = cv2.imread('optik1.jpg')
h = int(round(600 * image.shape[0] / image.shape[1]))
frame = cv2.resize(image, (600, h), interpolation=cv2.INTER_LANCZOS4)

# Display the original image
cv2.imshow('result', frame)
cv2.waitKey(0)

# Apply Canny edge detection
cannyFrame = bubbleSheetScanner.getCannyFrame(frame)
cv2.imshow('result', cannyFrame)
cv2.waitKey(0)

# Perform perspective transform
warpedFrame = bubbleSheetScanner.getWarpedFrame(cannyFrame, frame)
cv2.imshow('result', warpedFrame)
cv2.waitKey(0)

# Apply adaptive thresholding
adaptiveFrame = bubbleSheetScanner.getAdaptiveThresh(warpedFrame)
cv2.imshow('result', adaptiveFrame)
cv2.waitKey(0)

# Detect oval contours
ovalContours = bubbleSheetScanner.getOvalContours(adaptiveFrame)

# Process the detected ovals if the count matches the expected number
if (len(ovalContours) == bubbleSheetScanner.ovalCount):
    # Sort contours by y-coordinate (question order)
    ovalContours = sorted(ovalContours, key=bubbleSheetScanner.y_cord_contour, reverse=False)

    score = 0
    for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleSheetScanner.bubbleCount)):
        # Sort bubbles for each question by x-coordinate
        bubbles = sorted(ovalContours[i:i + bubbleSheetScanner.bubbleCount], key=bubbleSheetScanner.x_cord_contour,
                         reverse=False)

        for (j, c) in enumerate(bubbles):
            area = cv2.contourArea(c)

            # Create a mask to isolate the bubble
            mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=mask)
            total = cv2.countNonZero(mask)
            answer = bubbleSheetScanner.ANSWER_KEY[q]

            x, y, w, h = cv2.boundingRect(c)

            # Check if the bubble is filled
            isBubbleSigned = ((float)(total) / (float)(area)) > 1
            if (isBubbleSigned):
                if (answer == j):
                    # Calculate score for correct answer
                    score += 100 / bubbleSheetScanner.questionCount
                    # Mark correct answer with green
                    cv2.drawContours(warpedFrame, bubbles, j, (0, 255, 0), 2)
                else:
                    # Mark incorrect answer with red
                    cv2.drawContours(warpedFrame, bubbles, j, (0, 0, 255), 1)

            # Mark the correct answer with blue
            cv2.drawContours(warpedFrame, bubbles, answer, (255, 0, 0), 1)

    # Add score to the image
    warpedFrame = cv2.putText(warpedFrame, 'Score:' + str(score),
                              (100, 12),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 255, 255),
                              1
                              )

    # Display the final result
    cv2.imshow('result', warpedFrame)
    print('hit6')
    cv2.waitKey(0)
else:
    print('Invalid frame')
