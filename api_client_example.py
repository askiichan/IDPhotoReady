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
    
    def validate_file(self, file_path: str, return_annotated: bool = False) -> dict:
        """
        Validate an image file.
        
        Args:
            file_path: Path to image file
            return_annotated: Whether to return annotated image
            
        Returns:
            Validation response dictionary
        """
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'image/jpeg')}
            data = {'return_annotated': return_annotated}
            
            response = self.session.post(
                f"{self.base_url}/validate",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def validate_base64(self, image_base64: str, return_annotated: bool = False) -> dict:
        """
        Validate a base64 encoded image.
        
        Args:
            image_base64: Base64 encoded image string
            return_annotated: Whether to return annotated image
            
        Returns:
            Validation response dictionary
        """
        data = {
            'image_data': image_base64,
            'return_annotated': return_annotated
        }
        
        response = self.session.post(
            f"{self.base_url}/validate-base64",
            data=data
        )
        response.raise_for_status()
        return response.json()
    
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
        
        return self.validate_base64(image_data, return_annotated)

def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = IDPhotoValidatorClient()
    
    print("🔍 ID Photo Validator API Client Example")
    print("=" * 50)
    
    # Health check
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Models Loaded: {health['models_loaded']}")
        print(f"🔢 Version: {health['version']}")
        print()
    except Exception as e:
        print(f"❌ API Health Check Failed: {e}")
        print("Make sure the API server is running: python start_api.py")
        return
    
    # Example 1: Validate a file (you'll need to provide an actual image path)
    image_path = input("Enter path to an image file (or press Enter to skip): ").strip()
    
    if image_path and Path(image_path).exists():
        try:
            print(f"🖼️  Validating file: {image_path}")
            result = client.validate_file(image_path, return_annotated=True)
            
            print(f"✅ Valid: {result['is_valid']}")
            print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
            
            if not result['is_valid']:
                print("❌ Failure Reasons:")
                for reason in result['reasons']:
                    print(f"   • {reason}")
            
            if 'annotated_image' in result:
                print("🎨 Annotated image included in response (base64)")
            
            print()
            
        except Exception as e:
            print(f"❌ Validation Error: {e}")
    
    # Example 2: Create a test image and validate it
    print("🧪 Creating test image for validation...")
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
        result = client.validate_file(test_path)
        
        print(f"🧪 Test Image Valid: {result['is_valid']}")
        print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
        
        if not result['is_valid']:
            print("❌ Expected Failure Reasons:")
            for reason in result['reasons']:
                print(f"   • {reason}")
        
        # Clean up
        Path(test_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Test Image Error: {e}")
    
    print("\n🎯 API Client Example Complete!")
    print("📖 For more details, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
