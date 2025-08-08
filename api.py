"""
FastAPI application for ID Photo Validator.
"""

import os
import io
import time
import base64
import tempfile
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from id_validator.validator import validate_id_photo
from id_validator.models import ValidationResponse, ValidationRequest, HealthResponse, ErrorResponse
from id_validator.validation_config import ValidationConfig, STRICT_CONFIG, BASIC_CONFIG, LENIENT_CONFIG

# Initialize FastAPI app
app = FastAPI(
    title="ID Photo Validator API",
    description="Professional ID photo validation system using OpenCV and advanced computer vision",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model status
models_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global models_loaded
    try:
        # Test model loading by running a dummy validation
        # This will trigger model downloads if needed
        dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
        temp_path = "temp_startup_test.jpg"
        cv2.imwrite(temp_path, dummy_image)
        
        # Try to load models
        validate_id_photo(temp_path, return_annotated=False)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        models_loaded = True
        print("✅ ID Photo Validator API started successfully!")
        print("📊 Models loaded and ready for validation")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        models_loaded = False

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ID Photo Validator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "validate": "/validate"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        version="1.0.0",
        models_loaded=models_loaded
    )

@app.post("/validate", response_model=ValidationResponse)
async def validate_photo(
    file: UploadFile = File(..., description="Image file to validate"),
    return_annotated: bool = Form(default=False, description="Return annotated image"),
    validation_preset: str = Form(default="strict", description="Validation preset: strict, basic, or lenient"),
    face_sizing: bool = Form(default=True, description="Enable face sizing validation"),
    landmark_analysis: bool = Form(default=True, description="Enable landmark analysis"),
    eye_validation: bool = Form(default=True, description="Enable eye validation"),
    obstruction_detection: bool = Form(default=True, description="Enable obstruction detection"),
    mouth_validation: bool = Form(default=True, description="Enable mouth validation"),
    quality_assessment: bool = Form(default=True, description="Enable quality assessment"),
    background_validation: bool = Form(default=True, description="Enable background (white) validation")
):
    """
    Validate an ID photo for compliance with standard requirements.
    
    - **file**: Image file (JPG, PNG, BMP, TIFF)
    - **return_annotated**: Whether to return the annotated image with landmarks
    - **strict_mode**: Use strict validation criteria (recommended for ID photos)
    """
    
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Please try again later."
        )
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {allowed_types}"
        )
    
    # Validate file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # Create validation configuration
        if validation_preset.lower() == "strict":
            config = STRICT_CONFIG
        elif validation_preset.lower() == "basic":
            config = BASIC_CONFIG
        elif validation_preset.lower() == "lenient":
            config = LENIENT_CONFIG
        else:
            # Custom configuration from individual parameters
            config = ValidationConfig(
                face_sizing=face_sizing,
                landmark_analysis=landmark_analysis,
                eye_validation=eye_validation,
                obstruction_detection=obstruction_detection,
                mouth_validation=mouth_validation,
                quality_assessment=quality_assessment,
                background_validation=background_validation
            )
        
        # Perform validation
        start_time = time.time()
        is_valid, reasons, annotated_image = validate_id_photo(
            temp_path, 
            return_annotated=return_annotated,
            config=config
        )
        processing_time = time.time() - start_time
        
        # Extract additional metadata
        image = cv2.imread(temp_path)
        if image is not None:
            # Get basic image info
            h, w = image.shape[:2]
            image_area = h * w
            
            # Try to get face detection confidence (simplified)
            face_confidence = None
            face_size_ratio = None
            landmarks_count = None
            
            # You could enhance this to return more detailed metrics
            # For now, we'll use basic validation results
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        response_data = {
            "is_valid": is_valid,
            "reasons": reasons,
            "processing_time": processing_time,
            "confidence": face_confidence,
            "face_size_ratio": face_size_ratio,
            "landmarks_detected": landmarks_count
        }
        
        # If annotated image is requested and available
        if return_annotated and annotated_image is not None:
            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        return ValidationResponse(**response_data)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )

@app.post("/validate-base64", response_model=ValidationResponse)
async def validate_photo_base64(
    image_data: str = Form(..., description="Base64 encoded image string"),
    return_annotated: bool = Form(default=False, description="Return annotated image"),
    validation_preset: str = Form(default="strict", description="Validation preset: strict, basic, or lenient"),
    face_sizing: bool = Form(default=True, description="Enable face sizing validation"),
    landmark_analysis: bool = Form(default=True, description="Enable landmark analysis"),
    eye_validation: bool = Form(default=True, description="Enable eye validation"),
    obstruction_detection: bool = Form(default=True, description="Enable obstruction detection"),
    mouth_validation: bool = Form(default=True, description="Enable mouth validation"),
    quality_assessment: bool = Form(default=True, description="Enable quality assessment"),
    background_validation: bool = Form(default=True, description="Enable background (white) validation")
):
    """
    Validate an ID photo from base64 encoded image data.
    
    - **image_data**: Base64 encoded image string
    - **return_annotated**: Whether to return the annotated image with landmarks
    - **strict_mode**: Use strict validation criteria (recommended for ID photos)
    """
    
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Please try again later."
        )
    
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Validate image size
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        # Create validation configuration
        if validation_preset.lower() == "strict":
            config = STRICT_CONFIG
        elif validation_preset.lower() == "basic":
            config = BASIC_CONFIG
        elif validation_preset.lower() == "lenient":
            config = LENIENT_CONFIG
        else:
            # Custom configuration from individual parameters
            config = ValidationConfig(
                face_sizing=face_sizing,
                landmark_analysis=landmark_analysis,
                eye_validation=eye_validation,
                obstruction_detection=obstruction_detection,
                mouth_validation=mouth_validation,
                quality_assessment=quality_assessment,
                background_validation=background_validation
            )
        
        # Perform validation
        start_time = time.time()
        is_valid, reasons, annotated_image = validate_id_photo(
            temp_path, 
            return_annotated=return_annotated,
            config=config
        )
        processing_time = time.time() - start_time
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        response_data = {
            "is_valid": is_valid,
            "reasons": reasons,
            "processing_time": processing_time,
            "confidence": None,
            "face_size_ratio": None,
            "landmarks_detected": None
        }
        
        # If annotated image is requested and available
        if return_annotated and annotated_image is not None:
            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        return ValidationResponse(**response_data)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"HTTP {exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
