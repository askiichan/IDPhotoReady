"""
Gradio GUI for the ID Photo Validator application.
"""

import os
import cv2
import time
import threading
from typing import List, Tuple, Optional, Any
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
        self.batch_results: List = []
        # Centralized image extension tuple (lowercase)
        self.IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
        # Ensure required models are present
        self._check_and_download_models()
        # Build UI
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

    # -------------------------- Internal Utility Helpers --------------------------
    def _build_config_from_flags(self, *flags: bool) -> ValidationConfig:
        """Convenience: accepts ordered bool flags matching ValidationConfig fields."""
        return self._create_validation_config(*flags)

    def _add_status_badge(self, image_rgb: np.ndarray, status: str) -> np.ndarray:
        """Overlay a small status badge (OK/FAIL/ERR) top-right on an RGB image.

        Silently returns input image if anything unexpected occurs.
        """
        try:
            if image_rgb is None or not isinstance(image_rgb, np.ndarray):
                return image_rgb
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                return image_rgb
            pil_img = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.load_default()
            badge_text = ("OK" if status == "PASSED" else ("ERR" if status == "ERROR" else "FAIL"))
            if status == "PASSED":
                bg_color = (34, 139, 34, 230)
            elif status == "FAILED":
                bg_color = (220, 20, 60, 230)
            else:
                bg_color = (255, 140, 0, 230)
            text_color = (255, 255, 255, 255)
            text_w, text_h = draw.textbbox((0, 0), badge_text, font=font)[2:]
            pad = 4
            badge_w = text_w + pad * 2
            badge_h = text_h + pad * 2
            W, _ = pil_img.size
            rect_xy = [W - badge_w - 5, 5, W - 5, 5 + badge_h]
            draw.rectangle(rect_xy, fill=bg_color)
            draw.text((rect_xy[0] + pad, rect_xy[1] + pad - 1), badge_text, font=font, fill=text_color)
            return np.asarray(pil_img)
        except Exception:
            return image_rgb

    def _validate_image(self, file_path: str, config: ValidationConfig):
        """Core single-image validation logic shared by single + batch flows.

        Returns tuple: (status(str), reasons(list[str]), processing_time(float), annotated_img(np.ndarray|None))
        status in {PASSED, FAILED, ERROR}
        """
        start_time = time.time()
        try:
            is_valid, reasons, annotated_img = validate_id_photo(file_path, return_annotated=True, config=config)
            processing_time = time.time() - start_time
            status = "PASSED" if is_valid else "FAILED"
            return status, reasons, processing_time, annotated_img
        except Exception as e:
            return "ERROR", [f"Unexpected error: {e}"], time.time() - start_time, None

    def _prepare_display_image(self, file_path: str, annotated_img: Optional[np.ndarray], status: str) -> Image.Image:
        """Return a PIL.Image for gallery/display (RGB) with badge.

        Falls back to original image or placeholder if needed.
        """
        # Determine base image
        base_rgb = None
        try:
            if annotated_img is not None:
                if annotated_img.ndim == 3 and annotated_img.shape[2] == 3:
                    # annotated assumed BGR from OpenCV
                    base_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            if base_rgb is None:
                original = cv2.imread(file_path)
                if original is not None:
                    if original.ndim == 3 and original.shape[2] == 3:
                        base_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            if base_rgb is None:
                placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                placeholder[:, :, 2] = 255
                base_rgb = placeholder
        except Exception:
            placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder[:, :, 2] = 255
            base_rgb = placeholder

        # Add badge
        base_rgb = self._add_status_badge(base_rgb, status)
        return Image.fromarray(base_rgb)

    def validate_single_image(self, 
                              image_path: str,
                              face_sizing: bool,
                              landmark_analysis: bool,
                              eye_validation: bool,
                              obstruction_detection: bool,
                              mouth_validation: bool,
                              quality_assessment: bool,
                              background_validation: bool) -> Tuple[str, Optional[np.ndarray], List[List[Any]]]:
        """Validate a single image and return results."""
        if self.models_downloading:
            return "Please wait for models to finish downloading.", None, []

        if not image_path:
            return "Please upload an image.", None, []

        # Build config and store
        config = self._build_config_from_flags(
            face_sizing, landmark_analysis, eye_validation,
            obstruction_detection, mouth_validation,
            quality_assessment, background_validation
        )
        self.validation_config = config

        status, reasons, processing_time, annotated_img = self._validate_image(image_path, config)

        # Prepare annotated display (RGB + badge) for PASSED/FAILED only
        if annotated_img is not None and status in {"PASSED", "FAILED"}:
            try:
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                annotated_img = self._add_status_badge(annotated_rgb, status)
            except Exception:
                pass

        summary = f"File: {os.path.basename(image_path)} - {status} ({processing_time:.2f}s)"
        table_rows = [[
            os.path.basename(image_path),
            status,
            "; ".join(reasons) if reasons else "",
            round(processing_time, 2)
        ]]
        # If error, drop image output
        if status == "ERROR":
            annotated_img = None
        return summary, annotated_img, table_rows

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
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(self.IMAGE_EXTENSIONS) and
                       os.path.isfile(os.path.join(folder_path, f))]

        if not image_files:
            return "No image files found in the selected folder.", []

        # Always use explicit user configuration
        config = self._build_config_from_flags(
            face_sizing, landmark_analysis, eye_validation,
            obstruction_detection, mouth_validation,
            quality_assessment, background_validation
        )
        self.validation_config = config

        # Process each image
        results: List[Tuple[str, str, str, float, Optional[np.ndarray]]] = []
        total_files = len(image_files)
        passed_count = 0
        failed_count = 0

        for filename in image_files:
            file_path = os.path.join(folder_path, filename)
            status, reasons, processing_time, annotated_img = self._validate_image(file_path, config)
            reason_text = "\n".join([f"• {r}" for r in reasons]) if reasons else ""
            results.append((file_path, status, reason_text, processing_time, annotated_img))
            if status == "PASSED":
                passed_count += 1
            else:
                failed_count += 1

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
                            single_summary = gr.Textbox(label="Result Summary", lines=2, interactive=False)
                            annotated_output = gr.Image(label="Annotated Image")
                            single_table = gr.Dataframe(
                                headers=["File", "Status", "Reasons", "Time (s)"],
                                datatype=["str", "str", "str", "number"],
                                row_count=(0, "dynamic"),
                                col_count=(4, "fixed"),
                                interactive=False,
                                label="Detailed Result"
                            )
                
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
                        outputs=[single_summary, annotated_output, single_table]
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

                        image_files = [f for f in os.listdir(folder_path)
                                       if f.lower().endswith(self.IMAGE_EXTENSIONS) and
                                       os.path.isfile(os.path.join(folder_path, f))]
                        if not image_files:
                            yield "No image files found in the selected folder.", [], []
                            return

                        # Config (direct from user choices)
                        config = self._build_config_from_flags(
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
                            status, reasons, proc_time, annotated_img = self._validate_image(file_path, config)
                            if status == "PASSED":
                                passed += 1
                            else:
                                failed += 1
                            reason_text = "; ".join(reasons) if reasons else ""

                            # Prepare display image (annotated preferred) with badge
                            display_img = self._prepare_display_image(file_path, annotated_img, status)

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
