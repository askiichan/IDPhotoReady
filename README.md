# ID Photo Validator

A professional-grade ID photo validation system using OpenCV and machine learning to ensure photos meet standard requirements for official documents.

## Features

### Core Validation

- **Face Detection**: Uses OpenCV DNN with SSD MobileNet for accurate face detection
- **Facial Landmarks**: 68-point landmark detection for detailed facial analysis
- **Eye Validation**: Advanced Eye Aspect Ratio (EAR) calculation to detect closed eyes
- **Quality Assessment**: Cartoon/drawing detection using color clustering analysis
- **Size Validation**: Ensures face occupies appropriate portion of image (5-80%)
- **Obstruction Detection**: Identifies hands, objects, or other obstructions covering the face
- **Mouth Validation**: Checks for proper mouth visibility and positioning
- **Background Validation**: Verifies that the photo background is mostly white and uniform using HSV analysis on a ring around the detected face (annotates sampled region)

### Configurable Validation Categories

- **Face Sizing**: Validates face size ratio within image
- **Landmark Analysis**: Comprehensive facial landmark detection and analysis
- **Eye Validation**: Closed eye detection using Eye Aspect Ratio (EAR)
- **Obstruction Detection**: Skin color analysis to detect face coverings
- **Mouth Validation**: Mouth visibility and positioning checks
- **Quality Assessment**: Cartoon/drawing detection and image quality analysis
- **Background Validation**: White/neutral, uniform background check (toggleable)

### User Interfaces

- **Professional GUI**: User-friendly interface with configurable validation settings
- **REST API**: FastAPI-based web service with configurable validation presets
- **Annotated Results**: Visual feedback showing detected landmarks and validation issues
- **Preset Configurations**: Strict, Basic, and Lenient validation modes

## Usage

### GUI Application

Run the desktop application:

```bash
python main.py
```

The GUI provides:

- **Image Upload**: Drag & drop or browse for image files
- **Validation Configuration**: Choose from preset modes or customize individual validation categories:
  - **Strict Mode**: All validation categories enabled (default)
  - **Basic Mode**: Essential validations only (face sizing, eye validation)
  - **Lenient Mode**: Core validations only (file handling, face detection) — background check off
  - **Custom Mode**: Individual control over each validation category
- **Real-time Results**: Instant validation feedback with detailed reasons
- **Annotated Images**: Visual display with facial landmarks and validation overlays
- **Processing Time**: Performance metrics for each validation

**Supported formats**: JPG, PNG, BMP, TIFF

## Installation

### Using Virtual Environment (Recommended)

```bash
# Clone or download the project
cd protrait-validation-OpenCVYOLO

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.8 or higher
- OpenCV with contrib modules
- scikit-learn
- Pillow (PIL)
- tkinter (usually included with Python)
- NumPy (dependency of OpenCV)
- FastAPI and Uvicorn (for API)
- Requests (for API client)

### REST API

Start the API server:

```bash
python start_api.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

#### Endpoints

- `POST /validate` - Upload and validate an image file with configurable validation
- `POST /validate-base64` - Validate a base64 encoded image with configurable validation
- `GET /health` - Check API health status

#### Validation Configuration

The API supports three validation presets:

- **strict** (default): All validation categories enabled
- **basic**: Essential validations (face sizing, eye validation)
- **lenient**: Core validations only (face detection)
- **custom**: Individual control via parameters

#### Example Usage

```python
import requests

# Strict validation (default)
with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/validate',
        files={'file': f},
        data={
            'return_annotated': True,
            'validation_preset': 'strict'  # background_validation defaults to true in presets
        }
    )

# Custom validation configuration
with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/validate',
        files={'file': f},
        data={
            'validation_preset': 'custom',
            'face_sizing': True,
            'eye_validation': True,
            'landmark_analysis': False,
            'obstruction_detection': False,
            'mouth_validation': False,
            'quality_assessment': True,
            'background_validation': True,  # toggle background check
            'return_annotated': True
        }
    )
    result = response.json()
    print(f"Valid: {result['is_valid']}")
```

## Validation Criteria

The system validates photos against configurable criteria organized into categories:

### Core Requirements (Always Active)

- **File Handling**: JPEG, PNG, or other common image formats
- **Face Detection**: Must detect exactly one face with ≥85% confidence

### Configurable Validation Categories

#### Face Sizing

- Face must occupy 5-80% of the total image area
- Ensures proper framing for ID photo standards

#### Landmark Analysis

- All 68 facial landmarks must be detectable
- Facial symmetry and landmark distribution checks
- Nose bridge continuity validation
- Landmark clustering analysis

#### Eye Validation

- Both eyes must be open (Eye Aspect Ratio ≥ 0.25)
- Advanced EAR calculation for accurate closed eye detection

#### Obstruction Detection

- Face must not be covered by hands or objects
- Skin color analysis (≥40% skin visible in face region)
- HSV color space analysis for obstruction detection

#### Mouth Validation

- Mouth area must be clearly visible
- Proper mouth positioning checks

#### Background Validation

- Background should be mostly white/neutral and reasonably uniform
- Robust sampling: evaluates a ring-shaped region around the detected face (avoids clothing/edges)
- HSV-based decision with configurable thresholds; visually annotates sampled region when `return_annotated=True`
- Toggle via GUI checkbox or API param `background_validation` (default enabled; disabled in lenient preset)

#### Quality Assessment

- Image must not be a cartoon or drawing
- Color complexity analysis using KMeans clustering
- Image quality and clarity validation

## Project Structure

```
protrait-validation-OpenCVYOLO/
├── main.py                    # GUI application entry point
├── api.py                     # FastAPI REST API application
├── start_api.py              # API server startup script
├── api_client_example.py     # API usage examples
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── id_validator/             # Main validation package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration constants
│   ├── utils.py              # Utility functions
│   ├── validator.py          # Core validation logic
│   ├── validation_config.py  # Validation configuration classes
│   ├── models.py             # Pydantic models for API
│   └── gui.py                # GUI application
├── models/                   # AI model files (auto-downloaded)
│   ├── deploy.prototxt       # Face detection model architecture
│   ├── lbfmodel.yaml         # Facial landmark detection model
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Face detection weights
└── venv/                     # Virtual environment (if created)
```

## Technical Details

### Validation Architecture

- **Modular Design**: Configurable validation categories
- **Preset Configurations**: Strict, Basic, and Lenient modes
- **Custom Configuration**: Individual control over validation rules

### Models Used

- **Face Detection**: OpenCV DNN with Caffe SSD MobileNet
- **Landmark Detection**: LBF (Local Binary Features) 68-point model
- **Eye Analysis**: Eye Aspect Ratio (EAR) calculation
- **Quality Assessment**: KMeans clustering for cartoon detection
- **Skin Detection**: HSV color space analysis

### Configuration System

- `ValidationConfig` class for flexible validation control
- Preset configurations: `STRICT_CONFIG`, `BASIC_CONFIG`, `LENIENT_CONFIG`
- Runtime configuration through GUI checkboxes or API parameters

#### Background Validation thresholds

The following constants in `id_validator/config.py` control background analysis:

- `BG_SAMPLE_BORDER_PCT` — size of the ring sampled around the detected face (default: 0.08 ≈ 8% of min(image dimension)).
- `BG_MIN_MEAN_V` — minimum acceptable brightness (V channel) for a white-ish background (default: 190).
- `BG_MAX_MEAN_S` — maximum acceptable saturation (S channel) (default: 60).
- `BG_MAX_V_STD` — target maximum brightness standard deviation for uniformity (used for diagnostics; background logic is now robust to small dark areas) (default: 60).

Decision rule (implemented in `id_validator/validator.py`):

- Pass if the ring's white-ish coverage is high (≥ 60%)
  OR if overall V is high (≥ 190) and S is low (≤ 70).

Tips for tuning:

- If clean white backgrounds are failing: lower `BG_MIN_MEAN_V` (e.g., 185) or increase `BG_MAX_MEAN_S` (e.g., 70).
- If colored backgrounds are slipping through: raise `BG_MIN_MEAN_V` or lower `BG_MAX_MEAN_S` back toward stricter values.
- Increase `BG_SAMPLE_BORDER_PCT` slightly (e.g., 0.10) if the sampled ring is too close to hair/shoulders; decrease if images are tightly cropped.

### Dependencies

- OpenCV (cv2) with contrib modules
- NumPy for numerical operations
- scikit-learn for KMeans clustering
- Tkinter for GUI (included with Python)
- FastAPI and Uvicorn for REST API
- Pillow for image handling
- Requests for HTTP operations

## Troubleshooting

### Model Download Issues

If models fail to download automatically:

1. Check internet connection
2. Manually download models to `models/` directory:
   - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
   - [lbfmodel.yaml](https://github.com/spmallick/GSOC2017/raw/master/data/lbfmodel.yaml)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This application is designed for educational and development purposes. For production use in official ID systems, additional security and compliance measures may be required.
