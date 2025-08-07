"""
ID Photo Validator Package

A professional ID photo validation system using OpenCV and machine learning.
"""

__version__ = "1.0.0"
__author__ = "ID Photo Validator Team"

from .validator import validate_id_photo
from .gui import IDPhotoValidatorGUI
from .models import ValidationResponse, ValidationRequest, HealthResponse, ErrorResponse

__all__ = ["validate_id_photo", "IDPhotoValidatorGUI", "ValidationResponse", "ValidationRequest", "HealthResponse", "ErrorResponse"]
