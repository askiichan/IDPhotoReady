"""
Pydantic models for API request/response handling.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ValidationResponse(BaseModel):
    """Response model for photo validation."""
    
    is_valid: bool = Field(..., description="Whether the photo passes validation")
    reasons: List[str] = Field(default=[], description="List of validation failure reasons")
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence: Optional[float] = Field(None, description="Face detection confidence score")
    face_size_ratio: Optional[float] = Field(None, description="Face size as ratio of total image")
    landmarks_detected: Optional[int] = Field(None, description="Number of facial landmarks detected")
    annotated_image: Optional[str] = Field(None, description="Base64 encoded annotated image (data URI)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ValidationRequest(BaseModel):
    """Request model for photo validation."""
    
    return_annotated: bool = Field(default=False, description="Whether to return annotated image")
    strict_mode: bool = Field(default=True, description="Use strict validation criteria")

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether AI models are loaded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
