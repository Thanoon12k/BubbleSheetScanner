import cv2
import numpy as np

class BubbleSheetScanner:
    questionCount = 100  # Total number of questions
    bubbleCount = 5      # Number of bubbles (options) per question
    ovalCount = questionCount * bubbleCount  # Total number of ovals (bubbles) on the sheet
    
    def __init__(self):
        self.bubbleWidthAvr = 0
        self.bubbleHeightAvr = 0

    
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
        x, y, w, h = cv2.boundingRect(ovalContour)
        return x

    def y_cord_contour(self, ovalContour):
        x, y, w, h = cv2.boundingRect(ovalContour)
        return y

# Create an instance of BubbleSheetScanner
bubbleSheetScanner = BubbleSheetScanner()

# Read and resize the input image
image = cv2.imread('lower_answers.jpg')  # Replace with the path to your bubble sheet image
h = int(round(600 * image.shape[0] / image.shape[1]))
frame = cv2.resize(image, (600, h), interpolation=cv2.INTER_LANCZOS4)

# Apply adaptive thresholding
adaptiveFrame = bubbleSheetScanner.getAdaptiveThresh(frame)
cv2.imshow('Adaptive Frame', adaptiveFrame)
cv2.waitKey(0)

# Detect oval contours
ovalContours = bubbleSheetScanner.getOvalContours(adaptiveFrame)

# Define the correct answers for comparison
correct_answers = [
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E',
    'A', 'B', 'C', 'D', 'E'
]  # Example: 100 questions

# Process the detected ovals and determine student answers
student_answers = [''] * bubbleSheetScanner.questionCount
if len(ovalContours) == bubbleSheetScanner.ovalCount:
    ovalContours = sorted(ovalContours, key=bubbleSheetScanner.y_cord_contour, reverse=False)

    for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleSheetScanner.bubbleCount)):
        bubbles = sorted(ovalContours[i:i + bubbleSheetScanner.bubbleCount], key=bubbleSheetScanner.x_cord_contour, reverse=False)

        for (j, c) in enumerate(bubbles):
            x, y, w, h = cv2.boundingRect(c)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            
            # Draw red circles around all detected bubbles
            cv2.circle(frame, center, radius, (0, 0, 255), 2)

            mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
            cv2.circle(mask, center, radius, 255, -1)
            masked_data = cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=mask)
            total = cv2.countNonZero(masked_data)

            if total > 0.5 * (np.pi * (radius ** 2)):  # Adjust the threshold as needed
                student_answers[q] = chr(65 + j)  # Convert index to A, B, C, D, or E
                cv2.circle(frame, center, radius, (255, 0, 0), 2)  # Blue circle for filled answers

# Calculate the student's score
score = sum(1 for student_answer, correct_answer in zip(student_answers, correct_answers) if student_answer == correct_answer)
print("Student's Answers:", student_answers)
print("Correct Answers:", correct_answers)
print("Student's Score:", score)

# Display the final result
cv2.imshow('Result', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
