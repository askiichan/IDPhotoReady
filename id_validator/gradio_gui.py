"""
Gradio GUI for the ID Photo Validator application.
"""

import os
import cv2
import time
import threading
from typing import List, Tuple, Optional
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import requests

from .config import (
    FACE_PROTO, FACE_MODEL, LANDMARK_MODEL,
    FACE_PROTO_URL, FACE_MODEL_URL, LANDMARK_MODEL_URL
)
from .utils import download_file
from .validator import validate_id_photo
from .validation_config import ValidationConfig, STRICT_CONFIG, BASIC_CONFIG, LENIENT_CONFIG

class IDPhotoValidatorGradio:
    """
    The main Gradio GUI class for the ID Photo Validator application.
    """
    def __init__(self):
        self.models_downloading = False
        self.validation_config = ValidationConfig()
        self.batch_results = []
        
        # Check and download models if needed
        self._check_and_download_models()
        
        # Create the Gradio interface
        self.demo = self._create_interface()

    def _check_and_download_models(self):
        """Check if models exist and download if missing."""
        models_to_check = {
            "Face Protoxt": (FACE_PROTO_URL, FACE_PROTO),
            "Face Model": (FACE_MODEL_URL, FACE_MODEL),
            "Landmark Model": (LANDMARK_MODEL_URL, LANDMARK_MODEL)
        }

        missing_models = {name: (url, path) for name, (url, path) in models_to_check.items() if not os.path.exists(path)}

        if missing_models:
            self.models_downloading = True
            print("Downloading required models... Please wait.")
            
            try:
                for name, (url, path) in missing_models.items():
                    download_file(url, path, desc=name)
                print("Models downloaded successfully.")
            except Exception as e:
                print(f"Failed to download models: {e}. Please check your connection and restart.")
            finally:
                self.models_downloading = False

    def _update_config_from_presets(self, preset: str) -> dict:
        """Update validation configuration based on selected preset."""
        if preset == "Strict (All)":
            config = STRICT_CONFIG
        elif preset == "Basic":
            config = BASIC_CONFIG
        elif preset == "Lenient":
            config = LENIENT_CONFIG
        else:  # Custom
            return {}
            
        return {
            "face_sizing": config.face_sizing,
            "landmark_analysis": config.landmark_analysis,
            "eye_validation": config.eye_validation,
            "obstruction_detection": config.obstruction_detection,
            "mouth_validation": config.mouth_validation,
            "quality_assessment": config.quality_assessment,
            "background_validation": config.background_validation
        }

    def _create_validation_config(self, 
                                 face_sizing: bool,
                                 landmark_analysis: bool,
                                 eye_validation: bool,
                                 obstruction_detection: bool,
                                 mouth_validation: bool,
                                 quality_assessment: bool,
                                 background_validation: bool) -> ValidationConfig:
        """Create validation configuration from individual settings."""
        return ValidationConfig(
            face_sizing=face_sizing,
            landmark_analysis=landmark_analysis,
            eye_validation=eye_validation,
            obstruction_detection=obstruction_detection,
            mouth_validation=mouth_validation,
            quality_assessment=quality_assessment,
            background_validation=background_validation
        )

    def validate_single_image(self, 
                              image_path: str,
                              face_sizing: bool,
                              landmark_analysis: bool,
                              eye_validation: bool,
                              obstruction_detection: bool,
                              mouth_validation: bool,
                              quality_assessment: bool,
                              background_validation: bool) -> Tuple[str, str, Optional[np.ndarray], float]:
        """Validate a single image and return results."""
        if self.models_downloading:
            return "Please wait for models to finish downloading.", "", None, 0.0

        if not image_path:
            return "Please upload an image.", "", None, 0.0

        # Always use explicit user configuration (presets removed)
        config = self._create_validation_config(
            face_sizing, landmark_analysis, eye_validation,
            obstruction_detection, mouth_validation,
            quality_assessment, background_validation
        )
        
        self.validation_config = config

        start_time = time.time()
        try:
            is_valid, reasons, annotated_img = validate_id_photo(image_path, return_annotated=True, config=config)
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Format results
            if is_valid:
                result_text = "Validation Passed!\n\nThe photo meets all requirements."
            else:
                result_text = "Validation Failed\n\n" + "\n".join([f"- {reason}" for reason in reasons])
            
            result_text += f"\n\nProcessing time: {processing_time:.2f} seconds"
            
            return result_text, "success" if is_valid else "failure", annotated_img, processing_time
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            result_text = f"Validation Failed\n\nAn unexpected error occurred: {str(e)}"
            result_text += f"\n\nProcessing time: {processing_time:.2f} seconds"
            return result_text, "failure", None, processing_time

    def process_batch(self, 
                      folder_path: str,
                      face_sizing: bool,
                      landmark_analysis: bool,
                      eye_validation: bool,
                      obstruction_detection: bool,
                      mouth_validation: bool,
                      quality_assessment: bool,
                      background_validation: bool) -> Tuple[str, List]:
        """Process all images in a folder and return batch results."""
        if self.models_downloading:
            return "Please wait for models to finish downloading.", []

        if not folder_path:
            return "Please select a folder.", []

        # Get all image files in the folder
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions) and 
                      os.path.isfile(os.path.join(folder_path, f))]

        if not image_files:
            return "No image files found in the selected folder.", []

        # Always use explicit user configuration
        config = self._create_validation_config(
            face_sizing, landmark_analysis, eye_validation,
            obstruction_detection, mouth_validation,
            quality_assessment, background_validation
        )
        
        self.validation_config = config

        # Process each image
        results = []
        total_files = len(image_files)
        passed_count = 0
        failed_count = 0
        
        for i, filename in enumerate(image_files):
            file_path = os.path.join(folder_path, filename)
            
            # Validate image
            try:
                start_time = time.time()
                is_valid, reasons, annotated_img = validate_id_photo(file_path, return_annotated=True, config=config)
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Store results
                status = "PASSED" if is_valid else "FAILED"
                reason_text = "\n".join([f"• {reason}" for reason in reasons]) if not is_valid and reasons else ""
                
                results.append((file_path, status, reason_text, processing_time, annotated_img))
                
                if is_valid:
                    passed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                results.append((file_path, "ERROR", f"Error: {str(e)}", 0, None))
                failed_count += 1
        
        # Create summary
        summary = f"Processed {total_files} files: {passed_count} passed, {failed_count} failed"
        return summary, results

    def _create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="ID Photo Validator",
            theme=gr.themes.Soft(),
            css="""
            /* Highlight primary validate button */
            #validate-btn button {
                font-size: 1.05rem !important;
                font-weight: 600 !important;
                letter-spacing: .5px;
                padding: 0.95rem 1.4rem !important;
                border: 2px solid #8b5cf6 !important;
                background: linear-gradient(90deg,#6366F1,#8B5CF6,#6366F1) !important;
                background-size: 200% 100%;
                animation: validateGradient 6s linear infinite, validatePulse 2.4s ease-in-out infinite;
                box-shadow: 0 0 0 0 rgba(139,92,246,.55), 0 4px 14px -4px rgba(0,0,0,.55);
                transition: transform .25s ease, box-shadow .25s ease;
            }
            #validate-btn button:hover {
                transform: translateY(-2px) scale(1.02);
                animation: validateGradient 4s linear infinite, validatePulseHover 1.6s ease-in-out infinite;
                box-shadow: 0 0 0 4px rgba(139,92,246,.25), 0 8px 20px -6px rgba(0,0,0,.65);
            }
            #validate-btn button:active { transform: translateY(0) scale(.99); }
            #validate-btn button:disabled { filter: grayscale(.4) brightness(.8); animation: none; opacity:.65; }
            @keyframes validateGradient { 0%{background-position:0 0}100%{background-position:200% 0} }
            @keyframes validatePulse { 0%{box-shadow:0 0 0 0 rgba(139,92,246,.55)} 70%{box-shadow:0 0 0 14px rgba(139,92,246,0)} 100%{box-shadow:0 0 0 0 rgba(139,92,246,0)} }
            @keyframes validatePulseHover { 0%{box-shadow:0 0 0 0 rgba(139,92,246,.55)} 60%{box-shadow:0 0 0 18px rgba(139,92,246,0)} 100%{box-shadow:0 0 0 0 rgba(139,92,246,0)} }
            /* Optional subtle arrow cue */
            #validate-btn button::after { content:" →"; font-weight:700; }
            /* Process Folder button mirrors styling */
            #process-btn button {
                font-size: 1.0rem !important;
                font-weight: 600 !important;
                letter-spacing: .5px;
                padding: 0.85rem 1.3rem !important;
                border: 2px solid #10b981 !important;
                background: linear-gradient(90deg,#059669,#10b981,#059669) !important;
                background-size: 200% 100%;
                animation: processGradient 6s linear infinite, processPulse 2.8s ease-in-out infinite;
                box-shadow: 0 0 0 0 rgba(16,185,129,.55), 0 4px 14px -4px rgba(0,0,0,.55);
                transition: transform .25s ease, box-shadow .25s ease;
            }
            #process-btn button:hover {
                transform: translateY(-2px) scale(1.02);
                animation: processGradient 4s linear infinite, processPulseHover 1.8s ease-in-out infinite;
                box-shadow: 0 0 0 4px rgba(16,185,129,.25), 0 8px 20px -6px rgba(0,0,0,.65);
            }
            #process-btn button:active { transform: translateY(0) scale(.99); }
            #process-btn button:disabled { filter: grayscale(.4) brightness(.85); animation:none; opacity:.65; }
            #process-btn button::after { content:" ↻"; font-weight:700; }
            @keyframes processGradient { 0%{background-position:0 0}100%{background-position:200% 0} }
            @keyframes processPulse { 0%{box-shadow:0 0 0 0 rgba(16,185,129,.55)} 70%{box-shadow:0 0 0 14px rgba(16,185,129,0)} 100%{box-shadow:0 0 0 0 rgba(16,185,129,0)} }
            @keyframes processPulseHover { 0%{box-shadow:0 0 0 0 rgba(16,185,129,.55)} 60%{box-shadow:0 0 0 18px rgba(16,185,129,0)} 100%{box-shadow:0 0 0 0 rgba(16,185,129,0)} }
            """
        ) as demo:
            gr.Markdown("# ID Photo Validator")
            gr.Markdown("Validate ID photos against institutional standards with configurable criteria.")
            
            with gr.Tabs():
                # Single Image Validation Tab
                with gr.TabItem("Single Image Validation"):
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(type="filepath", label="Upload Image")
                            validate_btn = gr.Button("Validate Image", variant="primary", elem_id="validate-btn", interactive=False)
                            
                            # Validation Options (presets removed; direct selection)
                            gr.Markdown("### Validation Options")
                            with gr.Group() as custom_config:
                                face_sizing_cb = gr.Checkbox(label="Face Sizing", value=True)
                                landmark_analysis_cb = gr.Checkbox(label="Landmark Analysis", value=True)
                                eye_validation_cb = gr.Checkbox(label="Eye Validation", value=True)
                                obstruction_detection_cb = gr.Checkbox(label="Obstruction Detection", value=True)
                                mouth_validation_cb = gr.Checkbox(label="Mouth Validation", value=True)
                                quality_assessment_cb = gr.Checkbox(label="Quality Assessment", value=True)
                                background_validation_cb = gr.Checkbox(label="Background Validation", value=False)
                            
                        with gr.Column():
                            result_output = gr.Textbox(label="Validation Result", lines=10, interactive=False)
                            annotated_output = gr.Image(label="Annotated Image")
                
                    # Validation function
                    # Enable button only when an image is selected
                    def enable_validate(img_path):
                        return gr.update(interactive=bool(img_path))

                    image_input.change(enable_validate, inputs=[image_input], outputs=[validate_btn])

                    validate_btn.click(
                        self.validate_single_image,
                        inputs=[
                            image_input,
                            face_sizing_cb,
                            landmark_analysis_cb,
                            eye_validation_cb,
                            obstruction_detection_cb,
                            mouth_validation_cb,
                            quality_assessment_cb,
                            background_validation_cb
                        ],
                        outputs=[result_output, gr.State(), annotated_output, gr.State()]
                    )
                
                # Batch Processing Tab
                with gr.TabItem("Batch Processing"):
                    with gr.Row():
                        with gr.Column():
                            folder_input = gr.Textbox(label="Folder Path")
                            folder_btn = gr.Button("Select Folder")
                            
                            # Function to handle folder selection
                            def select_folder():
                                try:
                                    # Try to use tkinter for folder selection dialog
                                    import tkinter as tk
                                    from tkinter import filedialog
                                    root = tk.Tk()
                                    root.withdraw()  # Hide the main window
                                    root.attributes('-topmost', True)  # Make dialog appear on top
                                    folder_path = filedialog.askdirectory(title="Select Folder for Batch Processing")
                                    root.destroy()
                                    if folder_path:
                                        return folder_path
                                    else:
                                        return gr.update()
                                except Exception:
                                    # Fallback to returning an empty update if tkinter is not available
                                    return gr.update()
                            
                            # Attach the folder selection function to the button
                            folder_btn.click(
                                select_folder,
                                inputs=[],
                                outputs=[folder_input]
                            )
                            
                            # Validation Options for batch
                            gr.Markdown("### Validation Options")
                            with gr.Group() as batch_custom_config:
                                batch_face_sizing_cb = gr.Checkbox(label="Face Sizing", value=True)
                                batch_landmark_analysis_cb = gr.Checkbox(label="Landmark Analysis", value=True)
                                batch_eye_validation_cb = gr.Checkbox(label="Eye Validation", value=True)
                                batch_obstruction_detection_cb = gr.Checkbox(label="Obstruction Detection", value=True)
                                batch_mouth_validation_cb = gr.Checkbox(label="Mouth Validation", value=True)
                                batch_quality_assessment_cb = gr.Checkbox(label="Quality Assessment", value=True)
                                batch_background_validation_cb = gr.Checkbox(label="Background Validation", value=False)
                            
                            process_btn = gr.Button("Process Folder", variant="primary", elem_id="process-btn", interactive=False)
                            progress_output = gr.Textbox(label="Progress", interactive=False)
                            
                        with gr.Column():
                            gallery_output = gr.Gallery(label="Batch Results", columns=3, object_fit="contain", height="auto")
                            results_table = gr.Dataframe(
                                headers=["File", "Status", "Reasons", "Time (s)"],
                                datatype=["str", "str", "str", "number"],
                                row_count=(0, "dynamic"),
                                col_count=(4, "fixed"),
                                interactive=False,
                                label="Detailed Results"
                            )
                    
                    # Streaming batch processor to show progress and detailed results
                    def stream_batch_results(folder_path,
                                              face_sizing, landmark_analysis, eye_validation,
                                              obstruction_detection, mouth_validation,
                                              quality_assessment, background_validation):
                        if self.models_downloading:
                            yield "Please wait for models to finish downloading.", [], []
                            return
                        if not folder_path:
                            yield "Please select a folder.", [], []
                            return

                        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                        image_files = [f for f in os.listdir(folder_path)
                                       if f.lower().endswith(image_extensions) and
                                       os.path.isfile(os.path.join(folder_path, f))]
                        if not image_files:
                            yield "No image files found in the selected folder.", [], []
                            return

                        # Config (direct from user choices)
                        config = self._create_validation_config(
                            face_sizing, landmark_analysis, eye_validation,
                            obstruction_detection, mouth_validation,
                            quality_assessment, background_validation
                        )
                        self.validation_config = config

                        total = len(image_files)
                        gallery_items = []
                        table_rows = []
                        passed = 0
                        failed = 0

                        # Initial yield
                        yield f"Starting batch: {total} files", gallery_items, table_rows

                        prog = gr.Progress(track_tqdm=False)
                        for idx, filename in enumerate(image_files):
                            prog((idx + 1) / total, desc=f"Processing {filename}")
                            file_path = os.path.join(folder_path, filename)
                            try:
                                start = time.time()
                                is_valid, reasons, annotated_img = validate_id_photo(file_path, return_annotated=True, config=config)
                                proc_time = time.time() - start
                                status = "PASSED" if is_valid else "FAILED"
                                if is_valid:
                                    passed += 1
                                else:
                                    failed += 1
                                reason_text = "; ".join(reasons) if reasons else ""
                            except Exception as e:
                                status = "ERROR"
                                reason_text = f"Error: {str(e)}"
                                proc_time = 0
                                annotated_img = None
                                failed += 1

                            # Prepare image for gallery
                            display_img = None
                            if annotated_img is not None:
                                if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                                    display_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                                else:
                                    display_img = Image.fromarray(annotated_img)
                            else:
                                try:
                                    original_img = cv2.imread(file_path)
                                    if original_img is not None:
                                        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                        display_img = Image.fromarray(original_img)
                                except Exception:
                                    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                                    placeholder[:, :, 2] = 255
                                    display_img = Image.fromarray(placeholder)

                            # Overlay status badge (top-right corner)
                            if display_img is not None:
                                try:
                                    img_with_badge = display_img.copy()
                                    draw = ImageDraw.Draw(img_with_badge)
                                    font = ImageFont.load_default()
                                    badge_text = "OK" if status == "PASSED" else ("ERR" if status == "ERROR" else "FAIL")
                                    # Colors
                                    if status == "PASSED":
                                        bg_color = (34, 139, 34, 230)  # green
                                    elif status == "FAILED":
                                        bg_color = (220, 20, 60, 230)  # crimson
                                    else:
                                        bg_color = (255, 140, 0, 230)  # dark orange
                                    text_color = (255, 255, 255, 255)
                                    # Measure text
                                    text_w, text_h = draw.textbbox((0,0), badge_text, font=font)[2:]
                                    pad = 4
                                    badge_w = text_w + pad * 2
                                    badge_h = text_h + pad * 2
                                    W, H = img_with_badge.size
                                    # Rectangle coordinates (top-right)
                                    rect_xy = [W - badge_w - 5, 5, W - 5, 5 + badge_h]
                                    # Draw rectangle
                                    draw.rectangle(rect_xy, fill=bg_color)
                                    # Draw text centered
                                    text_x = rect_xy[0] + pad
                                    text_y = rect_xy[1] + pad - 1
                                    draw.text((text_x, text_y), badge_text, font=font, fill=text_color)
                                    display_img = img_with_badge
                                except Exception:
                                    pass

                            short_reason = reason_text.split(';')[0][:60] + ('...' if reason_text and len(reason_text.split(';')[0]) > 60 else '')
                            caption = f"{filename}\n{status} ({proc_time:.2f}s)" + (f"\n{short_reason}" if short_reason else "")
                            gallery_items.append((display_img, caption))
                            table_rows.append([filename, status, reason_text, round(proc_time, 2)])

                            summary = f"Processing {idx + 1}/{total} | Passed: {passed} Failed: {failed}"
                            yield summary, gallery_items, table_rows

                        final_summary = f"Processed {total} files: {passed} passed, {failed} failed"
                        yield final_summary, gallery_items, table_rows

                    # Hook streaming processor
                    # Enable process button only when folder path present
                    def enable_process(path):
                        return gr.update(interactive=bool(path and path.strip()))

                    folder_input.change(enable_process, inputs=[folder_input], outputs=[process_btn])

                    process_btn.click(
                        stream_batch_results,
                        inputs=[
                            folder_input,
                            batch_face_sizing_cb,
                            batch_landmark_analysis_cb,
                            batch_eye_validation_cb,
                            batch_obstruction_detection_cb,
                            batch_mouth_validation_cb,
                            batch_quality_assessment_cb,
                            batch_background_validation_cb
                        ],
                        outputs=[progress_output, gallery_output, results_table]
                    )
        
        return demo

    def launch(self, *args, **kwargs):
        """Launch the Gradio interface."""
        self.demo.launch(*args, **kwargs)

# Create a simple function for Gradio to use directly
def create_gradio_interface():
    """Create and return the Gradio interface."""
    validator = IDPhotoValidatorGradio()
    return validator.demo

# For direct execution
if __name__ == "__main__":
    validator = IDPhotoValidatorGradio()
    validator.launch()
