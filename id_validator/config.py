"""
Configuration constants for the ID Photo Validator.
"""

import os

# --- File Paths and URLs ---
# Base directory for storing models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Ensure models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Face detection model (Caffe)
FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Facial landmark detection model
LANDMARK_MODEL = os.path.join(MODEL_DIR, "lbfmodel.yaml")

# URLs for downloading models if they are not found locally
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
LANDMARK_MODEL_URL = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# --- Validation Thresholds ---

# Minimum and maximum allowable face size as a ratio of the total image area
MIN_FACE_SIZE_RATIO = 0.05  # 5% of the image area
MAX_FACE_SIZE_RATIO = 0.80  # 80% of the image area

# Eye Aspect Ratio threshold for closed eye detection
EAR_THRESHOLD = 0.25

# Enhanced Obstruction Detection Parameters
MIN_SKIN_PERCENTAGE = 0.4  # Minimum skin percentage required
MAX_UNIFORM_BLOCK_RATIO = 0.15  # Maximum ratio of face area that can be uniform color
UNIFORM_COLOR_STD_THRESHOLD = 15  # Standard deviation threshold for uniform color detection
MIN_EDGE_DENSITY = 0.02  # Minimum edge density for natural faces
MIN_COLOR_VARIANCE = 100  # Minimum color variance for natural faces
MAX_DARK_PIXEL_RATIO = 0.2  # Maximum ratio of very dark pixels
MAX_BRIGHT_PIXEL_RATIO = 0.2  # Maximum ratio of very bright pixels
DARK_PIXEL_THRESHOLD = 30  # Pixel value threshold for "very dark"
BRIGHT_PIXEL_THRESHOLD = 220  # Pixel value threshold for "very bright"

# Confidence threshold for face detection (increased for ID photos)
MIN_FACE_CONFIDENCE = 0.8

# K-means clustering threshold for cartoon/drawing detection
# A lower number of clusters suggests a simpler color palette, like a cartoon.
CARTOON_THRESHOLD = 12
