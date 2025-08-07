"""
GUI for the ID Photo Validator application.
"""

import os
import tkinter as tk
import cv2
from tkinter import filedialog, ttk
import threading
import time
from PIL import Image, ImageTk

from .config import (
    FACE_PROTO, FACE_MODEL, LANDMARK_MODEL,
    FACE_PROTO_URL, FACE_MODEL_URL, LANDMARK_MODEL_URL
)
from .utils import download_file
from .validator import validate_id_photo

class IDPhotoValidatorGUI:
    """
    The main GUI class for the ID Photo Validator application.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ID Photo Validator - Refactored Edition")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.current_image_path: str = ""
        self.models_downloading = False

        self._setup_styles()
        self._create_widgets()
        self._check_and_download_models()

    def _setup_styles(self):
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10)
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel: Image Upload and Display ---
        left_panel = ttk.Frame(main_frame, width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(left_panel, text="Original Photo", style="Header.TLabel").pack(pady=10)
        self.image_label = ttk.Label(left_panel, text="Upload an image to begin", relief="solid", borderwidth=2)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(pady=20, fill=tk.X)

        self.upload_btn = ttk.Button(btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, expand=True, padx=5)

        self.validate_btn = ttk.Button(btn_frame, text="Validate Photo", command=self.validate_image, state=tk.DISABLED)
        self.validate_btn.pack(side=tk.RIGHT, expand=True, padx=5)

        # --- Right Panel: Validation Results ---
        right_panel = ttk.Frame(main_frame, width=600)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        ttk.Label(right_panel, text="Validation Results", style="Header.TLabel").pack(pady=10)
        self.result_image_label = ttk.Label(right_panel, text="Validation pending...", relief="solid", borderwidth=2)
        self.result_image_label.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(right_panel, height=10, wrap=tk.WORD, font=("Helvetica", 11), relief="flat", bg="#ffffff")
        self.result_text.pack(pady=10, fill=tk.X)
        self.result_text.tag_configure("success", foreground="#28a745", font=("Helvetica", 12, "bold"))
        self.result_text.tag_configure("failure", foreground="#dc3545", font=("Helvetica", 12, "bold"))
        self.result_text.tag_configure("reason", foreground="#6c757d", font=("Helvetica", 11))
        self.result_text.tag_configure("time", foreground="#17a2b8", font=("Helvetica", 10, "italic"))
        self.result_text.config(state=tk.DISABLED)

    def _check_and_download_models(self):
        models_to_check = {
            "Face Protoxt": (FACE_PROTO_URL, FACE_PROTO),
            "Face Model": (FACE_MODEL_URL, FACE_MODEL),
            "Landmark Model": (LANDMARK_MODEL_URL, LANDMARK_MODEL)
        }

        missing_models = {name: (url, path) for name, (url, path) in models_to_check.items() if not os.path.exists(path)}

        if missing_models:
            self.models_downloading = True
            self.show_result_message("Downloading required models... Please wait.", "failure")
            self.upload_btn.config(state=tk.DISABLED)
            self.validate_btn.config(state=tk.DISABLED)

            download_thread = threading.Thread(target=self._download_worker, args=(missing_models,), daemon=True)
            download_thread.start()

    def _download_worker(self, models_to_download):
        try:
            for name, (url, path) in models_to_download.items():
                download_file(url, path, desc=name)
            self.show_result_message("Models downloaded successfully. You can now upload an image.", "success")
        except Exception as e:
            self.show_result_message(f"Failed to download models: {e}. Please check your connection and restart.", "failure")
        finally:
            self.models_downloading = False
            self.upload_btn.config(state=tk.NORMAL)

    def upload_image(self):
        if self.models_downloading:
            self.show_result_message("Please wait for models to finish downloading.", "failure")
            return

        path = filedialog.askopenfilename(filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ])
        if not path:
            return

        self.current_image_path = path
        self.display_image(self.image_label, path)
        self.validate_btn.config(state=tk.NORMAL)
        self.show_result_message("Image loaded. Click 'Validate Photo' to proceed.", "reason")
        self.result_image_label.configure(image='')
        self.result_image_label.image = None

    def validate_image(self):
        if not self.current_image_path:
            return

        start_time = time.time()
        try:
            is_valid, reasons, annotated_img = validate_id_photo(self.current_image_path, return_annotated=True)
            end_time = time.time()
            processing_time = end_time - start_time

            self.display_results(is_valid, reasons, processing_time)
            if annotated_img is not None:
                self.display_image(self.result_image_label, annotated_img, is_cv2_img=True)
            else:
                self.result_image_label.configure(image='')
                self.result_image_label.image = None

        except Exception as e:
            end_time = time.time()
            self.display_results(False, [f"An unexpected error occurred: {e}"], end_time - start_time)

    def display_image(self, label, image_source, is_cv2_img=False):
        try:
            if is_cv2_img:
                # Ensure the array is uint8, which is required by Pillow
                rgb_img = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_img.astype('uint8'))
            else:
                img = Image.open(image_source)
            
            img.thumbnail((500, 500))  # Use a fixed size for the thumbnail
            photo = ImageTk.PhotoImage(img)
            
            label.configure(image=photo)
            label.image = photo
        except Exception as e:
            error_message = f"Error displaying image:\n{e}"
            label.configure(text=error_message, image='')
            label.image = None

    def display_results(self, is_valid, reasons, duration):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        if is_valid:
            self.result_text.insert(tk.END, "Validation Passed!\n", "success")
            self.result_text.insert(tk.END, "The photo meets all requirements.", "reason")
        else:
            self.result_text.insert(tk.END, "Validation Failed\n", "failure")
            for reason in reasons:
                self.result_text.insert(tk.END, f"- {reason}\n", "reason")
        
        self.result_text.insert(tk.END, f"\n\nProcessing time: {duration:.2f} seconds", "time")
        self.result_text.config(state=tk.DISABLED)

    def show_result_message(self, message, tag):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, message, tag)
        self.result_text.config(state=tk.DISABLED)
