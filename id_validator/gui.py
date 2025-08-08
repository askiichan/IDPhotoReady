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
from .validation_config import ValidationConfig, STRICT_CONFIG, BASIC_CONFIG, LENIENT_CONFIG

class IDPhotoValidatorGUI:
    """
    The main GUI class for the ID Photo Validator application.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ID Photo Validator - Refactored Edition")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.current_image_path = None
        self.models_downloading = False
        self.validation_config = ValidationConfig()  # Default config
        
        # Validation configuration variables
        self.face_sizing_var = tk.BooleanVar(value=True)
        self.landmark_analysis_var = tk.BooleanVar(value=True)
        self.eye_validation_var = tk.BooleanVar(value=True)
        self.obstruction_detection_var = tk.BooleanVar(value=True)
        self.mouth_validation_var = tk.BooleanVar(value=True)
        self.quality_assessment_var = tk.BooleanVar(value=True)
        self.background_validation_var = tk.BooleanVar(value=True)

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
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Configuration frame (middle)
        config_frame = ttk.Frame(main_frame)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # --- Left Panel: Image Upload and Display ---
        left_panel = ttk.Frame(left_frame, width=600)
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

        # --- Configuration Panel ---
        self.setup_config_panel(config_frame)

        # --- Right Panel: Validation Results ---
        right_panel = ttk.Frame(right_frame, width=600)
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
            is_valid, reasons, annotated_img = validate_id_photo(self.current_image_path, return_annotated=True, config=self.validation_config)
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
    
    def setup_config_panel(self, parent):
        """Setup the validation configuration panel."""
        # Configuration panel header
        config_label = ttk.Label(parent, text="Validation Settings", font=("Arial", 12, "bold"))
        config_label.pack(pady=(0, 10))
        
        # Preset configurations frame
        preset_frame = ttk.LabelFrame(parent, text="Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(preset_frame, text="Strict (All)", command=self.apply_strict_config).pack(fill=tk.X, pady=2)
        ttk.Button(preset_frame, text="Basic", command=self.apply_basic_config).pack(fill=tk.X, pady=2)
        ttk.Button(preset_frame, text="Lenient", command=self.apply_lenient_config).pack(fill=tk.X, pady=2)
        
        # Individual validation categories
        categories_frame = ttk.LabelFrame(parent, text="Validation Categories", padding=10)
        categories_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Core validations (always enabled)
        ttk.Label(categories_frame, text="Core (Always On):", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(categories_frame, text="• File Handling", foreground="gray").pack(anchor=tk.W, padx=10)
        ttk.Label(categories_frame, text="• Face Detection", foreground="gray").pack(anchor=tk.W, padx=10)
        
        ttk.Separator(categories_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Configurable validations
        ttk.Label(categories_frame, text="Configurable:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        ttk.Checkbutton(categories_frame, text="Face Sizing", variable=self.face_sizing_var, 
                       command=self.update_config).pack(anchor=tk.W, padx=10)
        
        ttk.Checkbutton(categories_frame, text="Landmark Analysis", variable=self.landmark_analysis_var, 
                       command=self.update_config).pack(anchor=tk.W, padx=10)
        
        ttk.Checkbutton(categories_frame, text="Eye Validation", variable=self.eye_validation_var, 
                       command=self.update_config).pack(anchor=tk.W, padx=10)
        
        ttk.Checkbutton(categories_frame, text="Obstruction Detection", variable=self.obstruction_detection_var, 
                       command=self.update_config).pack(anchor=tk.W, padx=10)
        
        ttk.Checkbutton(categories_frame, text="Mouth Validation", variable=self.mouth_validation_var, 
                       command=self.update_config).pack(anchor=tk.W, padx=10)
        
        ttk.Checkbutton(categories_frame, text="Quality Assessment", variable=self.quality_assessment_var, 
                       command=self.update_config).pack(anchor=tk.W, padx=10)
        
        ttk.Checkbutton(categories_frame, text="Background Validation", variable=self.background_validation_var, 
                       command=self.update_config).pack(anchor=tk.W, padx=10)
        
        # Current config display
        self.config_status_label = ttk.Label(parent, text="", font=("Arial", 8), foreground="blue")
        self.config_status_label.pack(pady=(10, 0))
        
        # Update initial status
        self.update_config_status()
    
    def apply_strict_config(self):
        """Apply strict validation configuration."""
        self.face_sizing_var.set(True)
        self.landmark_analysis_var.set(True)
        self.eye_validation_var.set(True)
        self.obstruction_detection_var.set(True)
        self.mouth_validation_var.set(True)
        self.quality_assessment_var.set(True)
        self.background_validation_var.set(True)
        self.update_config()
    
    def apply_basic_config(self):
        """Apply basic validation configuration."""
        self.face_sizing_var.set(True)
        self.landmark_analysis_var.set(False)
        self.eye_validation_var.set(True)
        self.obstruction_detection_var.set(False)
        self.mouth_validation_var.set(False)
        self.quality_assessment_var.set(False)
        self.background_validation_var.set(True)
        self.update_config()
    
    def apply_lenient_config(self):
        """Apply lenient validation configuration."""
        self.face_sizing_var.set(False)
        self.landmark_analysis_var.set(False)
        self.eye_validation_var.set(False)
        self.obstruction_detection_var.set(False)
        self.mouth_validation_var.set(False)
        self.quality_assessment_var.set(False)
        self.background_validation_var.set(False)
        self.update_config()
    
    def update_config(self):
        """Update the validation configuration based on checkbox states."""
        self.validation_config = ValidationConfig(
            face_sizing=self.face_sizing_var.get(),
            landmark_analysis=self.landmark_analysis_var.get(),
            eye_validation=self.eye_validation_var.get(),
            obstruction_detection=self.obstruction_detection_var.get(),
            mouth_validation=self.mouth_validation_var.get(),
            quality_assessment=self.quality_assessment_var.get(),
            background_validation=self.background_validation_var.get()
        )
        self.update_config_status()
    
    def update_config_status(self):
        """Update the configuration status display."""
        enabled = self.validation_config.get_enabled_categories()
        if len(enabled) == 7:
            status = "Strict Mode (All)"
        elif len(enabled) == 0:
            status = "Lenient Mode (Core Only)"
        else:
            status = f"Custom ({len(enabled)}/7 enabled)"
        
        self.config_status_label.config(text=f"Current: {status}")
