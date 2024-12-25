# Bubble Sheet Scanner

This project implements a bubble sheet scanner for grading multiple-choice exams and extracting student numbers.

## Features

- Detects and extracts the student number from a bubble sheet
- Scans and grades multiple-choice answers
- Handles perspective transformation and image preprocessing
- Supports different bubble sheet layouts

## Usage

1. Install required packages:


pip install opencv-python numpy imutils scipy


2. Run the main script:


python main.py


3. The script will process the input image (default: '4q.jpg') and output:
   - Student number
   - Student answers
   - Corrected answers

## Files

- `main.py`: Main script for processing the full bubble sheet
- `main_student_num.py`: Module for extracting the student number

## Customization

You can modify the following parameters in `main.py`:

- `bubbles_blocks`: Number of answer blocks
- `bubbles_rows`: Number of rows in each answer block
- `bubbles_columns`: Number of columns in each answer block
- `ANSWER_KEY`: Dictionary containing correct answers

## Output

The script generates the following output files in the `results` folder:

- `Four_Points_Frame.jpg`: Image showing detected corner points
- `adaptive_counters.jpg`: Image with detected bubbles and student answers

## Note

Ensure that the input image is clear and well-aligned for optimal results.
