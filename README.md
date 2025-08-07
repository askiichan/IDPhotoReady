# ID Photo Validator

A professional-grade ID photo validation system using OpenCV, machine learning, and advanced computer vision techniques to ensure compliance with standard ID photo requirements.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Features

### Core Validation Checks

- ✅ **Face Detection**: Ensures exactly one face is present in the image
- ✅ **Face Size Validation**: Verifies face occupies appropriate portion of image (5%-80%)
- ✅ **High Confidence Requirement**: Minimum 85% face detection confidence
- ✅ **Facial Landmark Detection**: Requires all 68 facial landmarks to be detected
- ✅ **Obstruction Detection**: Identifies when face is covered by hands or objects
- ✅ **Eye Visibility Check**: Ensures both eyes are clearly visible and properly shaped
- ✅ **Mouth Region Analysis**: Verifies mouth area is unobstructed
- ✅ **Skin Color Analysis**: Detects insufficient skin visibility (hands covering face)
- ✅ **Landmark Clustering Analysis**: Detects incorrect landmark placement

### User Interface

- 🖥️ **GUI**: Clean, intuitive Tkinter-based interface
- 🖼️ **Image Preview**: Side-by-side original and annotated image display
- 📊 **Detailed Results**: Comprehensive validation feedback with specific failure reasons
- ⏱️ **Processing Time**: Real-time performance metrics
- 🎨 **Visual Annotations**: Cyan-colored facial landmarks and bounding boxes
- ⚠️ **Error Indicators**: Clear visual warnings for validation failures

## 📋 Requirements

- Python 3.8 or higher
- OpenCV with contrib modules
- scikit-learn
- Pillow (PIL)
- tkinter (usually included with Python)
- NumPy (dependency of OpenCV)

## 🚀 Installation & Setup

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

## 📁 Project Structure

```
protrait-validation-OpenCVYOLO/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── id_validator/             # Main package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration constants
│   ├── utils.py              # Utility functions (model downloader)
│   ├── validator.py          # Core validation logic
│   └── gui.py                # GUI application
├── models/                   # AI model files (auto-downloaded)
│   ├── deploy.prototxt       # Face detection model architecture
│   ├── lbfmodel.yaml         # Facial landmark detection model
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Face detection weights
└── venv/                     # Virtual environment (if created)
```

## 🔧 Configuration

The application uses several configurable parameters in `id_validator/config.py`:

```python
# Face size requirements (as percentage of total image area)
MIN_FACE_SIZE_RATIO = 0.05  # 5%
MAX_FACE_SIZE_RATIO = 0.80  # 80%

# Face detection confidence threshold
MIN_FACE_CONFIDENCE = 0.8   # 80%

# Cartoon detection sensitivity
CARTOON_THRESHOLD = 12
```

## 📖 Usage

### GUI Application

1. **Activate virtual environment**:

   ```bash
   venv\Scripts\activate
   ```

2. **Run the GUI application**:

   ```bash
   python main.py
   ```

3. **Upload and validate**: Click "Upload Image" → Select photo → Click "Validate Photo"

**Supported formats**: JPG, PNG, BMP, TIFF

### REST API

1. **Activate virtual environment**:

   ```bash
   venv\Scripts\activate
   ```

2. **Start API server**:

   ```bash
   python start_api.py
   ```

3. **Access API**:
   - **Interactive Docs**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

#### Key Endpoints

- **POST `/validate`** - Upload image file for validation
- **POST `/validate-base64`** - Validate base64 encoded image
- **GET `/health`** - Check API status

**Response format:**

```json
{
  "is_valid": true,
  "reasons": [],
  "processing_time": 1.23
}
```

#### Quick Test

**Python:**

```python
import requests

# Validate image
with open("photo.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/validate", files=files)
    print(response.json())
```

**cURL:**

```bash
curl -X POST "http://localhost:8000/validate" -F "file=@photo.jpg"
```

## 🛠️ Troubleshooting

### Model Download Issues

If models fail to download automatically:

1. Check internet connection
2. Manually download models to `models/` directory:
   - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
   - [lbfmodel.yaml](https://github.com/spmallick/GSOC2017/raw/master/data/lbfmodel.yaml)

## 🔬 Technical Details

### AI Models Used

1. **Face Detection**: OpenCV DNN with Caffe SSD MobileNet
2. **Facial Landmarks**: LBF (Local Binary Features) model with 68-point detection
3. **Color Analysis**: K-means clustering for cartoon detection
4. **Skin Detection**: HSV color space analysis

### Processing Pipeline

1. Image loading and validation
2. Face detection using deep neural network
3. Face size and confidence validation
4. Facial landmark detection (68 points)
5. Advanced obstruction analysis
6. Skin color and symmetry checks
7. Quality assessment and final decision

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This application is designed for educational and development purposes. For production use in official ID systems, additional security and compliance measures may be required.
