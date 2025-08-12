"""
Configurable validation settings for ID Photo Validator.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ValidationConfig:
    """Configuration class for validation categories."""
    
    # Core validation (always enabled)
    file_handling: bool = True  # Cannot be disabled
    face_detection: bool = True  # Cannot be disabled
    
    # Configurable validation categories
    face_sizing: bool = True
    landmark_analysis: bool = True
    eye_validation: bool = True
    obstruction_detection: bool = True
    mouth_validation: bool = True
    quality_assessment: bool = True
    background_validation: bool = True
    shoulder_balance_validation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'face_sizing': self.face_sizing,
            'landmark_analysis': self.landmark_analysis,
            'eye_validation': self.eye_validation,
            'obstruction_detection': self.obstruction_detection,
            'mouth_validation': self.mouth_validation,
            'quality_assessment': self.quality_assessment,
            'background_validation': self.background_validation
            , 'shoulder_balance_validation': self.shoulder_balance_validation
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ValidationConfig':
        """Create config from dictionary."""
        return cls(
            face_sizing=config_dict.get('face_sizing', True),
            landmark_analysis=config_dict.get('landmark_analysis', True),
            eye_validation=config_dict.get('eye_validation', True),
            obstruction_detection=config_dict.get('obstruction_detection', True),
            mouth_validation=config_dict.get('mouth_validation', True),
            quality_assessment=config_dict.get('quality_assessment', True),
            background_validation=config_dict.get('background_validation', True)
            , shoulder_balance_validation=config_dict.get('shoulder_balance_validation', True)
        )
    
    def get_enabled_categories(self) -> list:
        """Get list of enabled validation categories."""
        enabled = []
        if self.face_sizing:
            enabled.append("Face Sizing")
        if self.landmark_analysis:
            enabled.append("Landmark Analysis")
        if self.eye_validation:
            enabled.append("Eye Validation")
        if self.obstruction_detection:
            enabled.append("Obstruction Detection")
        if self.mouth_validation:
            enabled.append("Mouth Validation")
        if self.quality_assessment:
            enabled.append("Quality Assessment")
        if self.background_validation:
            enabled.append("Background Validation")
        if self.shoulder_balance_validation:
            enabled.append("Shoulder Balance Validation")
        return enabled

# Default configuration
DEFAULT_CONFIG = ValidationConfig()
