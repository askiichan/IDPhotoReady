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
from PIL import Image
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
                              preset: str,
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

        # Create validation config
        if preset != "Custom":
            # Use preset configuration
            config_dict = self._update_config_from_presets(preset)
            if config_dict:
                config = ValidationConfig(**config_dict)
            else:
                config = self.validation_config
        else:
            # Use custom configuration
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
                      preset: str,
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

        # Create validation config
        if preset != "Custom":
            # Use preset configuration
            config_dict = self._update_config_from_presets(preset)
            if config_dict:
                config = ValidationConfig(**config_dict)
            else:
                config = self.validation_config
        else:
            # Use custom configuration
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
        with gr.Blocks(title="ID Photo Validator", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# ID Photo Validator")
            gr.Markdown("Validate ID photos against institutional standards with configurable criteria.")
            
            with gr.Tabs():
                # Single Image Validation Tab
                with gr.TabItem("Single Image Validation"):
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(type="filepath", label="Upload Image")
                            validate_btn = gr.Button("Validate Image", variant="primary")
                            
                            # Preset configurations
                            gr.Markdown("### Validation Presets")
                            preset_radio = gr.Radio(
                                choices=["Strict (All)", "Basic", "Lenient", "Custom"],
                                value="Strict (All)",
                                label="Configuration Preset"
                            )
                            
                            # Custom configuration
                            with gr.Group(visible=False) as custom_config:
                                gr.Markdown("### Custom Configuration")
                                face_sizing_cb = gr.Checkbox(label="Face Sizing", value=True)
                                landmark_analysis_cb = gr.Checkbox(label="Landmark Analysis", value=True)
                                eye_validation_cb = gr.Checkbox(label="Eye Validation", value=True)
                                obstruction_detection_cb = gr.Checkbox(label="Obstruction Detection", value=True)
                                mouth_validation_cb = gr.Checkbox(label="Mouth Validation", value=True)
                                quality_assessment_cb = gr.Checkbox(label="Quality Assessment", value=True)
                                background_validation_cb = gr.Checkbox(label="Background Validation", value=True)
                            
                            # Update visibility of custom config based on preset
                            def update_config_visibility(preset):
                                return gr.Group(visible=(preset == "Custom"))
                            
                            preset_radio.change(
                                update_config_visibility,
                                inputs=[preset_radio],
                                outputs=[custom_config]
                            )
                            
                        with gr.Column():
                            result_output = gr.Textbox(label="Validation Result", lines=10, interactive=False)
                            annotated_output = gr.Image(label="Annotated Image")
                
                    # Validation function
                    validate_btn.click(
                        self.validate_single_image,
                        inputs=[
                            image_input,
                            preset_radio,
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
                            
                            # Preset configurations
                            gr.Markdown("### Validation Presets")
                            batch_preset_radio = gr.Radio(
                                choices=["Strict (All)", "Basic", "Lenient", "Custom"],
                                value="Strict (All)",
                                label="Configuration Preset"
                            )
                            
                            # Custom configuration
                            with gr.Group(visible=False) as batch_custom_config:
                                gr.Markdown("### Custom Configuration")
                                batch_face_sizing_cb = gr.Checkbox(label="Face Sizing", value=True)
                                batch_landmark_analysis_cb = gr.Checkbox(label="Landmark Analysis", value=True)
                                batch_eye_validation_cb = gr.Checkbox(label="Eye Validation", value=True)
                                batch_obstruction_detection_cb = gr.Checkbox(label="Obstruction Detection", value=True)
                                batch_mouth_validation_cb = gr.Checkbox(label="Mouth Validation", value=True)
                                batch_quality_assessment_cb = gr.Checkbox(label="Quality Assessment", value=True)
                                batch_background_validation_cb = gr.Checkbox(label="Background Validation", value=True)
                            
                            # Update visibility of custom config based on preset
                            batch_preset_radio.change(
                                update_config_visibility,
                                inputs=[batch_preset_radio],
                                outputs=[batch_custom_config]
                            )
                            
                            process_btn = gr.Button("Process Folder", variant="primary")
                            progress_output = gr.Textbox(label="Progress", interactive=False)
                            
                        with gr.Column():
                            gallery_output = gr.Gallery(label="Batch Results", columns=3, object_fit="contain", height="auto")
                    
                    # Wrapper function to format results for gallery
                    def format_batch_results(*args):
                        summary, results = self.process_batch(*args)
                        
                        # Format results for gallery: (image, caption) tuples
                        gallery_items = []
                        for file_path, status, reason_text, processing_time, annotated_img in results:
                            if annotated_img is not None:
                                # Convert numpy array to PIL Image for Gradio
                                # Gradio expects RGB format
                                if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                                    # Convert BGR to RGB
                                    rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                                else:
                                    rgb_img = annotated_img
                                pil_img = Image.fromarray(rgb_img)
                                caption = f"{os.path.basename(file_path)} - {status} ({processing_time:.2f}s)"
                                gallery_items.append((pil_img, caption))
                            else:
                                # If no annotated image, use the original image
                                try:
                                    original_img = cv2.imread(file_path)
                                    if original_img is not None:
                                        if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                                            rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                                        else:
                                            rgb_img = original_img
                                        pil_img = Image.fromarray(rgb_img)
                                        caption = f"{os.path.basename(file_path)} - {status} ({processing_time:.2f}s)"
                                        gallery_items.append((pil_img, caption))
                                except Exception:
                                    # If we can't load the image, just add a text placeholder
                                    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                                    placeholder[:, :, 2] = 255  # Red placeholder
                                    pil_img = Image.fromarray(placeholder)
                                    caption = f"{os.path.basename(file_path)} - {status} ({processing_time:.2f}s)"
                                    gallery_items.append((pil_img, caption))
                        
                        return summary, gallery_items
                    
                    # Process function
                    process_btn.click(
                        format_batch_results,
                        inputs=[
                            folder_input,
                            batch_preset_radio,
                            batch_face_sizing_cb,
                            batch_landmark_analysis_cb,
                            batch_eye_validation_cb,
                            batch_obstruction_detection_cb,
                            batch_mouth_validation_cb,
                            batch_quality_assessment_cb,
                            batch_background_validation_cb
                        ],
                        outputs=[progress_output, gallery_output]
                    )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("### Validation Categories")
            gr.Markdown("""
            - **Core (Always On)**: File Handling, Face Detection
            - **Configurable**: 
              - Face Sizing
              - Landmark Analysis
              - Eye Validation
              - Obstruction Detection
              - Mouth Validation
              - Quality Assessment
              - Background Validation
            """)
        
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
