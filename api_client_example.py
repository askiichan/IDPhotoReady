"""
Example client code for ID Photo Validator API.
"""

import requests
import base64
import json
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"

class IDPhotoValidatorClient:
    """Client for ID Photo Validator API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def validate_image_file(self, image_path: str, return_annotated: bool = False, validation_preset: str = "strict", **config_options):
        """Validate an image file using the API."""
        url = f"{self.base_url}/validate"
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {
                'return_annotated': return_annotated,
                'validation_preset': validation_preset,
                **config_options
            }
            
            response = self.session.post(url, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def validate_base64_image(self, image_base64: str, return_annotated: bool = False, validation_preset: str = "strict", **config_options):
        """Validate a base64 encoded image using the API."""
        url = f"{self.base_url}/validate-base64"
        
        data = {
            'image_data': image_base64,
            'return_annotated': return_annotated,
            'validation_preset': validation_preset,
            **config_options
        }
        
        response = self.session.post(url, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def validate_file_to_base64(self, file_path: str, return_annotated: bool = False) -> dict:
        """
        Convert file to base64 and validate.
        
        Args:
            file_path: Path to image file
            return_annotated: Whether to return annotated image
            
        Returns:
            Validation response dictionary
        """
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        return self.validate_base64_image(image_data, return_annotated)

def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = IDPhotoValidatorClient()
    
    print(" ID Photo Validator API Client Example")
    print("=" * 50)
    
    # Health check
    try:
        health = client.health_check()
        print(f" API Status: {health['status']}")
        print(f" Models Loaded: {health['models_loaded']}")
        print(f" Version: {health['version']}")
        print()
    except Exception as e:
        print(f" API Health Check Failed: {e}")
        print("Make sure the API server is running: python start_api.py")
        return
    
    # Example 1: Basic validation with strict preset
    print("\n=== Example 1: Strict Validation ===")
    result = client.validate_image_file("test_image.jpg", validation_preset="strict")
    if result:
        print(f"Valid: {result['is_valid']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        if not result['is_valid']:
            print("Issues found:")
            for reason in result['reasons']:
                print(f"  - {reason}")
    
    # Example 1b: Basic preset validation
    print("\n=== Example 1b: Basic Validation ===")
    result = client.validate_image_file("test_image.jpg", validation_preset="basic")
    if result:
        print(f"Valid: {result['is_valid']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        if not result['is_valid']:
            print("Issues found:")
            for reason in result['reasons']:
                print(f"  - {reason}")
    
    # Example 1c: Lenient preset validation
    print("\n=== Example 1c: Lenient Validation ===")
    result = client.validate_image_file("test_image.jpg", validation_preset="lenient")
    if result:
        print(f"Valid: {result['is_valid']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        if not result['is_valid']:
            print("Issues found:")
            for reason in result['reasons']:
                print(f"  - {reason}")
    
    # Example 2: Custom validation configuration
    print("\n=== Example 2: Custom Validation Configuration ===")
    result = client.validate_image_file(
        "test_image.jpg", 
        validation_preset="custom",
        face_sizing=True,
        landmark_analysis=False,
        eye_validation=True,
        obstruction_detection=False,
        mouth_validation=False,
        quality_assessment=True
    )
    if result:
        print(f"Valid: {result['is_valid']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        if not result['is_valid']:
            print("Issues found:")
            for reason in result['reasons']:
                print(f"  - {reason}")
    
    # Example 3: Validation with annotated image
    print("\n=== Example 3: Validation with Annotated Image ===")
    result = client.validate_image_file("test_image.jpg", return_annotated=True, validation_preset="strict")
    if result and result.get('annotated_image'):
        # Save annotated image
        img_data = result['annotated_image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        with open("annotated_result.jpg", "wb") as f:
            f.write(base64.b64decode(img_data))
        print("Annotated image saved as 'annotated_result.jpg'")
    
    # Example 4: Create a test image and validate it
    print("\n=== Example 4: Test Image Creation and Validation ===")
    try:
        import numpy as np
        import cv2
        
        # Create a simple test image (this will likely fail validation)
        test_image = np.zeros((400, 300, 3), dtype=np.uint8)
        test_image.fill(128)  # Gray background
        
        # Save test image
        test_path = "test_image.jpg"
        cv2.imwrite(test_path, test_image)
        
        # Validate test image
        result = client.validate_image_file(test_path)
        
        print(f"Test Image Valid: {result['is_valid']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        if not result['is_valid']:
            print("Expected Failure Reasons:")
            for reason in result['reasons']:
                print(f"   • {reason}")
        
        # Clean up
        Path(test_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"Test Image Error: {e}")
    
    print("\n🎯 API Client Example Complete!")
    print("📖 For more details, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
