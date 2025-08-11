"""
Main entry point for the ID Photo Validator application.
"""

from id_validator.gradio_gui import IDPhotoValidatorGradio

if __name__ == "__main__":
    # Create and launch the Gradio interface
    validator = IDPhotoValidatorGradio()
    validator.launch()
