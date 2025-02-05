import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import fitz

class OMRProcessor:
    def __init__(self):
        self.student_number_bubbles = 40
        self.answers_bubbles = 500
        self.total_bubbles = self.student_number_bubbles + self.answers_bubbles

    def get_longest_horizontal_lines(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=image.shape[1] // 2, maxLineGap=50)
        if lines is None:
            return []
        
        horizontal_lines = sorted([line[0] for line in lines if abs(line[0][1] - line[0][3]) < 10], key=lambda l: l[1])
        return horizontal_lines[:1] + horizontal_lines[-1:] if len(horizontal_lines) >= 2 else []

class ImageProcessor:
    def __init__(self):
        self.threshold_block_size = 55
        self.threshold_C = 20

    def get_adaptive_thresh(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, self.threshold_block_size, self.threshold_C)

    def draw_contours(self, frame, contours, color=(0, 0, 255)):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(frame, contours, -1, color, 2)
        return frame

    def save_image(self, image, folder, name):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        cv2.imwrite(path, image)
        print(f"Saved: {path}")

class BubbleDetector:
    def get_bubbles(self, img, adaptive_frame, expected_count):
        contours, _ = cv2.findContours(adaptive_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sorted([c for c in contours if self.is_circular(c)], key=cv2.contourArea, reverse=True)[:expected_count]

    def is_circular(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        return 0.9 <= w / h <= 1.1 and cv2.contourArea(contour) > 30



class StudentNumberProcessor:
    def __init__(self):
        self.bubble_columns = 4
        self.bubbles_rows = 10
        
    def get_student_num_bubbles(self, bw_img, cnts):
        number_cnts = []
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])[:40]
        ref = None
        
        # Find reference row
        for i in range(len(cnts)-3):
            row = sorted(cnts[i:i + 4], key=lambda c: cv2.boundingRect(c)[1])
            y_values = [cv2.boundingRect(c)[1] for c in row]
            if max(y_values) - min(y_values) < 20:
                ref = row
                break
                
        if ref is None:
            return []
            
        x_first = cv2.boundingRect(ref[0])[0]
        x_last = cv2.boundingRect(ref[-1])[0]
        normlize_value = cv2.boundingRect(ref[0])[2] // 2
        
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        
        first_ten_cnts = sorted(cnts[:self.bubbles_rows], 
                              key=lambda c: cv2.boundingRect(c)[1])
        _, y_first, _, _ = cv2.boundingRect(first_ten_cnts[0])
        _, y_last, _, _ = cv2.boundingRect(first_ten_cnts[-1])
        
        for cntr in cnts:
            _, y, _, _ = cv2.boundingRect(cntr)
            if y >= (y_first-normlize_value) and y <= (y_last+normlize_value):
                number_cnts.append(cntr)
                
        if len(number_cnts) < 40:
            number_cnts = self.fix_missing_num_bubbles(bw_img, number_cnts)
            
        return number_cnts
        
    def fix_missing_num_bubbles(self, bw_img, cnts):
        if len(cnts) < 40:
            print(f'bad fixing stdnt number - snb {len(cnts)}')
            return cnts
            
        ref_row = []
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])
        _, _, w, _ = cv2.boundingRect(cnts[0])
        normlize_value = w // 2
        
        max_num_missing_bubbles = 10
        while len(cnts) < 40 and max_num_missing_bubbles > 0:
            max_num_missing_bubbles -= 1
            
            for i in range(0, len(cnts)-4, 4):
                contours_row = sorted(cnts[i:i + 4], key=lambda c: cv2.boundingRect(c)[1])
                y_values = [cv2.boundingRect(c)[1] for c in contours_row]
                
                if max(y_values) - min(y_values) > normlize_value:
                    if not ref_row:
                        ref_row = contours_row
                        
                    miss_bubbles = self.add_miss_bubbles_to_row(bw_img, ref_row, 
                                                              contours_row, normlize_value)
                    cnts.extend(miss_bubbles)
                    
        return sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

class AnswerProcessor:
    def __init__(self):
        self.blocks_answers_bubbles = []
        self.bubble_columns = 5
        self.bubbles_rows = 25
        
    def get_answers_blocks_bubbles(self, img, adaptive_frame, total_bubbles):
        cnts = sorted(total_bubbles, key=lambda c: cv2.boundingRect(c)[1])
        cnts = cnts[40:]
        
        radious = int(cv2.boundingRect(cnts[0])[2] // 2)
        x1 = min(cv2.boundingRect(c)[0] for c in cnts)-radious
        y1 = min(cv2.boundingRect(c)[1] for c in cnts)-radious
        x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in cnts)+radious
        y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in cnts)+radious
        
        width = x2 - x1
        gap = 5*radious
        rect_width = (width-(gap*3)) // 4
        rects = []
        
        for i in range(5):
            start_x = x1 + (i * (rect_width + gap))
            rect = [start_x, y1, start_x + rect_width, y2]
            rects.append(rect)
            
        blocks_cnts = [[] for _ in range(4)]
        for cnt in cnts:
            x, y, _, _ = cv2.boundingRect(cnt)
            for i, rect in enumerate(rects):
                min_x, min_y, max_x, max_y = rect
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    blocks_cnts[i].append(cnt)
                    
        return blocks_cnts
        
    def find_student_answers(self,  adaptive_frame, blocks_answers_bubbles):
        answers = []
        answers_bubbles = []
        
        for block in blocks_answers_bubbles:
            row_bubbles = sorted(block, key=lambda c: cv2.boundingRect(c)[1])
            areas = [cv2.countNonZero(cv2.bitwise_and(adaptive_frame, adaptive_frame,
                                                     mask=cv2.drawContours(np.zeros(adaptive_frame.shape,
                                                                                   dtype="uint8"), [bubble], -1, 255, -1)))
                    for bubble in row_bubbles]
            
            chosen_bubble_index = areas.index(max(areas))
            answers_bubbles.append(row_bubbles[chosen_bubble_index])
            answers.append(chosen_bubble_index)
            
        return answers, answers_bubbles


class ResultProcessor:
    def __init__(self, answer_key_path):
        self.answer_key = self.load_answer_key(answer_key_path)

    def load_answer_key(self, path):
        df = pd.read_excel(path)
        return {i: ord(row.iloc[1]) - ord('A') for i, row in df.iterrows()}


    def calculate_student_score(self, student_answers):
        answer_key=self.answer_key
        score = sum(1 for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans)
        correct_indices = [q_num for q_num, correct_ans in answer_key.items() if q_num < len(student_answers) and student_answers[q_num] == correct_ans]
        return score, correct_indices


    def save_results(self, student_number, score, answers):
        df = pd.DataFrame([[student_number, score] + answers], columns=["Student Number", "Score"] + [f"Q{i+1}" for i in range(len(answers))])
        df.to_csv("student_results.csv", mode='a', header=not os.path.exists("student_results.csv"), index=False)

class OMRSystem:
    def __init__(self, answer_key_path):
        self.omr_processor = OMRProcessor()
        self.image_processor = ImageProcessor()
        self.bubble_detector = BubbleDetector()
        self.answer_processor = AnswerProcessor()
        self.result_processor = ResultProcessor(answer_key_path)

    def process_pdf(self, pdf_path):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_folder = f"{pdf_name}_processed"
        os.makedirs(output_folder, exist_ok=True)

        pdf_document = fitz.open(pdf_path)
        print(f"plaese wait, converting pdf pages[{pdf_document.page_count}] to images .....................")
        images = [self.convert_pdf_page(page, output_folder, i) for i, page in enumerate(pdf_document)]
        print(f"please wait, getting {len(images)} students results ...................... ")
        for i, img in enumerate(images):
            adaptive_frame = self.image_processor.get_adaptive_thresh(img)
            bubbles = self.bubble_detector.get_bubbles(img, adaptive_frame, self.omr_processor.total_bubbles)
            student_answers = self.answer_processor.find_student_answers(adaptive_frame, bubbles[self.omr_processor.student_number_bubbles:])
            student_number = self.extract_student_number(adaptive_frame, bubbles[:self.omr_processor.student_number_bubbles])
            score,_ = self.result_processor.calculate_student_score(student_answers)
            self.result_processor.save_results(student_number, score, student_answers)
            result_image = self.image_processor.draw_contours(img, student_answers)
            self.image_processor.save_image(result_image, output_folder, f"result_{i+1}.jpg")
            print(f"student {i+1} from {len(images)} results saved successfully")
        

    def convert_pdf_page(self, page, output_folder, page_num):
        pix = page.get_pixmap(dpi=200)
        img_path = os.path.join(output_folder, f'page_{page_num + 1}.png')
        pix.save(img_path)
        return np.array(Image.open(img_path))

    def extract_student_number(self, adaptive_frame, student_bubbles):
        return "".join(str(i) for i, bubble in enumerate(student_bubbles) if cv2.countNonZero(cv2.drawContours(np.zeros(adaptive_frame.shape, dtype="uint8"), [bubble], -1, 255, -1)) > 100)

if __name__ == "__main__":
    processor = OMRSystem("answers.xlsx")
    processor.process_pdf("1.pdf")