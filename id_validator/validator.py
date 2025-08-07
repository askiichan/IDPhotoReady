"""
Core ID photo validation logic.
"""

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional, Dict, Any

from .config import (
    FACE_PROTO, FACE_MODEL, LANDMARK_MODEL,
    MIN_FACE_SIZE_RATIO, MAX_FACE_SIZE_RATIO, MIN_FACE_CONFIDENCE,
    CARTOON_THRESHOLD, EAR_THRESHOLD,
    MIN_SKIN_PERCENTAGE, MAX_UNIFORM_BLOCK_RATIO, UNIFORM_COLOR_STD_THRESHOLD,
    MIN_EDGE_DENSITY, MIN_COLOR_VARIANCE, MAX_DARK_PIXEL_RATIO, MAX_BRIGHT_PIXEL_RATIO,
    DARK_PIXEL_THRESHOLD, BRIGHT_PIXEL_THRESHOLD
)
from .validation_config import ValidationConfig, DEFAULT_CONFIG

# --- Model Loading ---
try:
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    landmark_facemark = cv2.face.createFacemarkLBF()
    landmark_facemark.loadModel(LANDMARK_MODEL)
    models_loaded = True
except cv2.error as e:
    models_loaded = False
    print(f"Error loading models: {e}. Please run the main script to download them.")

def validate_id_photo(image_path: str, return_annotated: bool = False, config: ValidationConfig = None) -> Tuple[bool, List[str], Optional[np.ndarray]]:
    """
    Validates a student ID photo based on configurable validation rules.

    Args:
        image_path (str): The path to the image file.
        return_annotated (bool): If True, returns the annotated image.
        config (ValidationConfig): Configuration for validation categories.

    Returns:
        A tuple containing:
        - bool: True if the photo is valid, False otherwise.
        - list[str]: A list of failure reasons.
        - np.ndarray | None: The annotated image if requested, otherwise None.
    """
    if not models_loaded:
        return False, ["Models are not loaded. Please ensure they are downloaded correctly."], None

    # Use default config if none provided
    if config is None:
        config = ValidationConfig()

    reasons = []
    annotated_image = None

    # 1. Load Image
    if not os.path.exists(image_path):
        return False, ["Image file not found."], None
        
    image = cv2.imread(image_path)
    if image is None:
        return False, ["Could not read the image file."], None
        
    (h, w) = image.shape[:2]
    image_area = h * w
    
    if return_annotated:
        annotated_image = image.copy()

    # 2. Face Detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    face_confidences = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > MIN_FACE_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
            face_confidences.append(confidence)

    if len(faces) != 1:
        reasons.append(f"Found {len(faces)} faces. Exactly one is required.")
        return False, reasons, annotated_image

    (startX, startY, endX, endY) = faces[0]
    face_confidence = face_confidences[0]
    face_w, face_h = endX - startX, endY - startY
    face_area = face_w * face_h
    face_ratio = face_area / image_area

    # Enhanced validation checks
    if face_confidence < 0.85:  # Even higher confidence requirement
        reasons.append(f"Face detection confidence too low: {face_confidence:.2f} (minimum 0.85 required for ID photos).")

    # Face Sizing Validation (configurable)
    if config.face_sizing and not (MIN_FACE_SIZE_RATIO <= face_ratio <= MAX_FACE_SIZE_RATIO):
        reasons.append(f"Face size is {face_ratio:.1%}, outside the acceptable range of {MIN_FACE_SIZE_RATIO:.0%}-{MAX_FACE_SIZE_RATIO:.0%}.")

    # 3. Quality Assessment and Obstruction Detection (configurable)
    face_roi = image[startY:endY, startX:endX]
    if face_roi.size > 0:
        # Quality Assessment - Cartoon/Drawing Detection
        if config.quality_assessment:
            pixels = cv2.resize(face_roi, (100, 100)).reshape(-1, 3)
            kmeans = KMeans(n_clusters=CARTOON_THRESHOLD, random_state=42, n_init=10).fit(pixels)
            if len(np.unique(kmeans.labels_)) < CARTOON_THRESHOLD / 2:
                reasons.append("Image appears to be a cartoon or drawing due to low color complexity.")
        
        # Enhanced Obstruction Detection
        if config.obstruction_detection:
            face_roi_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # 1. Skin color detection
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(face_roi_hsv, lower_skin, upper_skin)
            skin_percentage = np.sum(skin_mask > 0) / (face_roi.shape[0] * face_roi.shape[1])
            
            # 2. Detect uniform color blocks (artificial obstructions)
            # Check for large areas of uniform color that shouldn't be in a natural face
            face_roi_blur = cv2.GaussianBlur(face_roi_gray, (5, 5), 0)
            _, binary = cv2.threshold(face_roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours of uniform regions
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for large uniform blocks (like black rectangles)
            face_area_roi = face_roi.shape[0] * face_roi.shape[1]
            large_uniform_blocks = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # If a uniform block covers more than threshold of face area, it's suspicious
                if area > MAX_UNIFORM_BLOCK_RATIO * face_area_roi:
                    # Check if it's a very uniform color (low standard deviation)
                    mask = np.zeros(face_roi_gray.shape, np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    masked_region = cv2.bitwise_and(face_roi_gray, face_roi_gray, mask=mask)
                    non_zero_pixels = masked_region[masked_region > 0]
                    
                    if len(non_zero_pixels) > 0:
                        color_std = np.std(non_zero_pixels)
                        # Very low standard deviation indicates uniform color (artificial obstruction)
                        if color_std < UNIFORM_COLOR_STD_THRESHOLD:
                            large_uniform_blocks += 1
            
            # 3. Edge density analysis - natural faces have more edges than artificial obstructions
            edges = cv2.Canny(face_roi_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (face_roi.shape[0] * face_roi.shape[1])
            
            # 4. Color variance analysis - natural faces have more color variation
            color_variance = np.var(face_roi_gray)
            
            # 5. Detect very dark or very bright regions (common in artificial obstructions)
            very_dark_pixels = np.sum(face_roi_gray < DARK_PIXEL_THRESHOLD) / face_area_roi
            very_bright_pixels = np.sum(face_roi_gray > BRIGHT_PIXEL_THRESHOLD) / face_area_roi
            
            # Apply obstruction detection rules
            obstruction_detected = False
            obstruction_reasons = []
            
            # Rule 1: Insufficient skin visible
            if skin_percentage < MIN_SKIN_PERCENTAGE:
                obstruction_detected = True
                obstruction_reasons.append(f"insufficient skin visible ({skin_percentage:.1%})")
            
            # Rule 2: Large uniform color blocks detected
            if large_uniform_blocks > 0:
                obstruction_detected = True
                obstruction_reasons.append(f"artificial uniform color blocks detected ({large_uniform_blocks} blocks)")
            
            # Rule 3: Very low edge density (too smooth/artificial)
            if edge_density < MIN_EDGE_DENSITY:
                obstruction_detected = True
                obstruction_reasons.append(f"unnaturally smooth surface detected (edge density: {edge_density:.3f})")
            
            # Rule 4: Very low color variance (too uniform)
            if color_variance < MIN_COLOR_VARIANCE:
                obstruction_detected = True
                obstruction_reasons.append(f"unnaturally uniform coloring detected (variance: {color_variance:.1f})")
            
            # Rule 5: Excessive dark or bright regions
            if very_dark_pixels > MAX_DARK_PIXEL_RATIO:
                obstruction_detected = True
                obstruction_reasons.append(f"excessive dark regions detected ({very_dark_pixels:.1%})")
            
            if very_bright_pixels > MAX_BRIGHT_PIXEL_RATIO:
                obstruction_detected = True
                obstruction_reasons.append(f"excessive bright regions detected ({very_bright_pixels:.1%})")
            
            # Report obstruction if detected
            if obstruction_detected:
                reason_text = ", ".join(obstruction_reasons)
                reasons.append(f"Face obstruction detected: {reason_text}. Face may be covered by hands, objects, or artificial elements.")

    # Always draw the face rectangle if we have an annotated image
    if annotated_image is not None:
        cv2.rectangle(annotated_image, (startX, startY), (endX, endY), (0, 255, 255), 2)  # Cyan rectangle
        cv2.putText(annotated_image, "Face Detected", (startX, startY-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 4. Facial Landmark Detection (Now Required for ID Photos)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rect = np.array([[startX, startY, face_w, face_h]], dtype=np.int32)
    ok, landmarks = landmark_facemark.fit(gray, faces_rect)

    # Landmark Analysis (configurable)
    if config.landmark_analysis and (not ok or len(landmarks) == 0):
        reasons.append("Facial landmarks could not be detected. Face may be obstructed, covered, or of poor quality.")
    elif ok and len(landmarks) > 0:
        # Check landmark quality - ensure sufficient landmarks are detected
        landmark_points = landmarks[0][0] if len(landmarks[0].shape) == 3 else landmarks[0]
        num_landmarks = len(landmark_points)
        
        # Detailed landmark analysis (configurable)
        if config.landmark_analysis and num_landmarks < 68:
            reasons.append(f"Incomplete facial landmarks detected: {num_landmarks}/68. Face may be partially obstructed.")
        
        # Advanced facial obstruction detection
        if num_landmarks >= 68:
            # Check facial symmetry - compare left and right sides
            face_center_x = (startX + endX) / 2
            
            # Get key facial points
            jaw_points = landmark_points[0:17]  # Jaw line
            left_eye_landmarks = landmark_points[42:48]  # Left eye
            right_eye_landmarks = landmark_points[36:42]  # Right eye
            nose_points = landmark_points[27:36]  # Nose
            mouth_points = landmark_points[48:68]  # Mouth
            
            # Obstruction Detection - Check landmark distribution
            if config.obstruction_detection:
                left_side_points = landmark_points[landmark_points[:, 0] < face_center_x]
                right_side_points = landmark_points[landmark_points[:, 0] >= face_center_x]
                
                if len(left_side_points) < 20 or len(right_side_points) < 20:
                    reasons.append("Facial landmarks are unevenly distributed. Face may be partially covered.")
            
            # Eye Validation (configurable)
            if config.eye_validation:
                def calculate_eye_aspect_ratio(eye_landmarks):
                    """Calculate Eye Aspect Ratio (EAR) to detect closed eyes."""
                    # Vertical eye landmarks
                    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # |p2-p6|
                    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # |p3-p5|
                    # Horizontal eye landmark
                    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # |p1-p4|
                    # EAR formula
                    ear = (A + B) / (2.0 * C)
                    return ear
                
                # Calculate EAR for both eyes
                left_ear = calculate_eye_aspect_ratio(left_eye_landmarks)
                right_ear = calculate_eye_aspect_ratio(right_eye_landmarks)
                
                # EAR threshold for open eyes (typically > 0.2 for open eyes)
                EAR_THRESHOLD = 0.25
                
                if left_ear < EAR_THRESHOLD:
                    reasons.append(f"Left eye appears to be closed or nearly closed (EAR: {left_ear:.3f}).")
                
                if right_ear < EAR_THRESHOLD:
                    reasons.append(f"Right eye appears to be closed or nearly closed (EAR: {right_ear:.3f}).")
                
                # Additional eye dimension checks
                left_eye_width = np.max(left_eye_landmarks[:, 0]) - np.min(left_eye_landmarks[:, 0])
                right_eye_width = np.max(right_eye_landmarks[:, 0]) - np.min(right_eye_landmarks[:, 0])
                left_eye_height = np.max(left_eye_landmarks[:, 1]) - np.min(left_eye_landmarks[:, 1])
                right_eye_height = np.max(right_eye_landmarks[:, 1]) - np.min(right_eye_landmarks[:, 1])
                
                # Eyes should have reasonable dimensions
                if left_eye_width < 15 or right_eye_width < 15:
                    reasons.append("Eyes are not clearly visible or may be obstructed.")
                
                if left_eye_height < 3:  # Lowered threshold since closed eyes have very small height
                    reasons.append("Left eye landmarks indicate eye may be closed.")
                    
                if right_eye_height < 3:
                    reasons.append("Right eye landmarks indicate eye may be closed.")
            
            # Mouth Validation (configurable)
            if config.mouth_validation:
                mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
                mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
                
                if mouth_width < 20 or mouth_height < 8:
                    reasons.append("Mouth region is not clearly visible or may be obstructed.")
            
            # Advanced Obstruction Detection (configurable)
            if config.obstruction_detection:
                # Check for abnormal landmark clustering (hands/objects covering face)
                landmark_std_x = np.std(landmark_points[:, 0])
                landmark_std_y = np.std(landmark_points[:, 1])
                
                # If landmarks are too clustered, it might indicate incorrect detection
                expected_face_width = face_w * 0.6  # Expected landmark spread
                if landmark_std_x < expected_face_width * 0.15:
                    reasons.append("Facial landmarks appear abnormally clustered. Face may be obstructed by hands or objects.")
                
                # Check nose bridge continuity
                nose_bridge = landmark_points[27:31]
                if len(nose_bridge) >= 4:
                    nose_bridge_length = np.linalg.norm(nose_bridge[-1] - nose_bridge[0])
                    if nose_bridge_length < 15:
                        reasons.append("Nose bridge landmarks are too close together. Face may be obstructed.")
        
        # Draw landmarks only if detection succeeds
        if annotated_image is not None:
            cv2.face.drawFacemarks(annotated_image, landmarks[0], (0, 255, 255))  # Cyan landmarks
            
            # Add validation status overlay
            if len(reasons) > 0:
                cv2.putText(annotated_image, "VALIDATION FAILED", (startX, endY+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red warning
                # Add number of issues
                cv2.putText(annotated_image, f"{len(reasons)} issues detected", (startX, endY+65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(annotated_image, "VALIDATION PASSED", (startX, endY+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green success

    is_valid = len(reasons) == 0
    return is_valid, reasons, annotated_image
