import cv2
import numpy as np

def find_student_number(bubbleSheetScanner, warpedFrame, adaptiveFrame):
    """
    Extract the student number (4 digits) from the bubble sheet scanner.

    Each student has a 4-digit number (0000-9999). The sheet contains 4 columns and 10 rows (0-9). 
    The student must fill one bubble per column to indicate their number.

    Parameters:
        bubbleSheetScanner: Object containing helper functions like `mygetOvalContours`, `x_cord_contour`, `y_cord_contour`.
        warpedFrame: The image frame of the warped bubble sheet for visualization.
        adaptiveFrame: The thresholded image frame for detecting filled bubbles.

    Returns:
        str: The detected student number as a string, or None if invalid frame.
    """
    bubbles_rows = 10
    bubbles_columns = 4

    # Detect oval contours
    ovalContours = bubbleSheetScanner.mygetOvalContours(adaptiveFrame)

    # Check if the count matches the expected number of bubbles
    if len(ovalContours) != (bubbles_rows * bubbles_columns):
        print('Invalid frame: Number of detected bubbles does not match the expected count.')
        return None

    # Sort contours by x-coordinate (column order)
    ovalContours = sorted(ovalContours, key=bubbleSheetScanner.x_cord_contour, reverse=False)

    student_number = ''

    for col in range(bubbles_columns):
        # Extract contours for each column
        column_bubbles = [ovalContours[i] for i in range(col * bubbles_rows, (col + 1) * bubbles_rows)]
        
        # Sort bubbles for each column by y-coordinate
        column_bubbles = sorted(column_bubbles, key=bubbleSheetScanner.y_cord_contour, reverse=False)

        max_area = 0
        chosen_bubble_index = -1

        for j, bubble in enumerate(column_bubbles):
            # Calculate the area of the bubble
            area = cv2.contourArea(bubble)

            # Create a mask to isolate the bubble
            mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
            cv2.drawContours(mask, [bubble], -1, 255, -1)
            
            # Count the non-zero pixels within the mask
            total = cv2.countNonZero(mask)

            # Check if the bubble is the largest filled bubble
            if total > max_area:
                max_area = total
                chosen_bubble_index = j

        if chosen_bubble_index != -1:
            student_number += str(chosen_bubble_index)

            # Draw blue edges around the chosen bubble for visualization
            cv2.drawContours(warpedFrame, [column_bubbles[chosen_bubble_index]], -1, (255, 0, 0), 2)

    # Display the final result
    cv2.putText(warpedFrame, f"Student Number: {student_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Student Number Result', warpedFrame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return student_number

# Example usage
class BubbleSheetScanner:
    bubbleCount = 10      # Number of bubbles (options) per column
    TotalSNBubbles = 4 * bubbleCount  # 4 columns, each with 10 bubbles
    
    def __init__(self):
        self.bubbleWidthAvr = 0
        self.bubbleHeightAvr = 0
    
    def mygetOvalContours(self, adaptiveFrame):
        kernel = np.ones((3, 3), np.uint8)
        dilatedFrame = cv2.dilate(adaptiveFrame, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilatedFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ovalContours = []

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(contour)

            if len(approx) > 8 and 0.8 <= w / h <= 1.2 and w > 10 and h > 10:
                ovalContours.append(contour)
                self.bubbleWidthAvr += w
                self.bubbleHeightAvr += h

        if len(ovalContours) > 0:
            self.bubbleWidthAvr /= len(ovalContours)
            self.bubbleHeightAvr /= len(ovalContours)

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

# Find student number
student_number = find_student_number(bubbleSheetScanner, frame, adaptiveFrame)
print("Student Number:", student_number)
