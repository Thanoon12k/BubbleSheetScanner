
import tkinter as tk
from tkinter import filedialog

def get_pdf_path():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    return file_path



def display_student_results(student_number, score, root):
    result_window = tk.Toplevel(root)
    result_window.title("Student Results")

    # Center the window on the screen
    window_width = 300
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    result_window.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    # Set background color
    result_window.configure(bg='lightblue')

    # Create and pack the labels and button with colors
    tk.Label(result_window, text=f"Student Number: {student_number}", font=("Helvetica", 16), bg='lightblue', fg='darkblue').pack(pady=10)
    tk.Label(result_window, text=f"Score: {score}", font=("Helvetica", 16), bg='lightblue', fg='darkblue').pack(pady=10)
    tk.Button(result_window, text="OK", font=("Helvetica", 14), bg='darkblue', fg='white', command=result_window.destroy).pack(pady=10)

