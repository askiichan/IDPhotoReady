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
- ✅ **Facial Symmetry**: Checks for even landmark distribution on both face sides
- ✅ **Skin Color Analysis**: Detects insufficient skin visibility (hands covering face)
- ✅ **Cartoon/Drawing Detection**: Identifies non-photographic images
- ✅ **Landmark Clustering Analysis**: Detects incorrect landmark placement

### User Interface
- 🖥️ **Professional GUI**: Clean, intuitive Tkinter-based interface
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

### Option 1: Using Virtual Environment (Recommended)

#### Windows
```bash
# Clone or download the project
cd protrait-validation-OpenCVYOLO

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

#### macOS/Linux
```bash
# Clone or download the project
cd protrait-validation-OpenCVYOLO

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Option 2: Using Conda Environment

```bash
# Create conda environment
conda create -n id-validator python=3.9

# Activate environment
conda activate id-validator

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Option 3: System-wide Installation

```bash
# Install dependencies globally (not recommended for production)
pip install -r requirements.txt

# Run the application
python main.py
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

### Starting the Application

1. **Activate your virtual environment** (if using one):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **First-time setup**: The application will automatically download required AI models (~67MB total) on first launch.

### Using the Validator

1. **Upload Image**: Click "Upload Image" and select a photo file (JPG, PNG, BMP, TIFF)
2. **Validate Photo**: Click "Validate Photo" to run the analysis
3. **Review Results**: 
   - ✅ **Green "Validation Passed"**: Photo meets all ID requirements
   - ❌ **Red "Validation Failed"**: Photo has issues (detailed reasons provided)
4. **View Annotations**: The right panel shows the analyzed image with facial landmarks

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## 🔍 Validation Criteria

### ✅ Valid ID Photo Requirements
- Exactly one face detected with high confidence (≥85%)
- Face size between 5% and 80% of total image area
- All 68 facial landmarks successfully detected
- Both eyes clearly visible and properly shaped
- Mouth region unobstructed
- Facial features evenly distributed (no partial covering)
- Sufficient skin visibility (≥40% in face region)
- Real photograph (not cartoon/drawing)

### ❌ Common Rejection Reasons
- Multiple faces or no face detected
- Face too small or too large in frame
- Hand or object covering part of face
- Eyes closed, covered, or not clearly visible
- Poor image quality or low resolution
- Cartoon, drawing, or heavily filtered image
- Insufficient facial landmarks detected

## 🛠️ Troubleshooting

### Model Download Issues
If models fail to download automatically:
1. Check internet connection
2. Manually download models to `models/` directory:
   - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
   - [res10_300x300_ssd_iter_140000.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
   - [lbfmodel.yaml](https://github.com/spmallick/GSOC2017/raw/master/data/lbfmodel.yaml)

### OpenCV Installation Issues
If you encounter OpenCV-related errors:
```bash
# Install OpenCV with contrib modules
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Virtual Environment Issues
If virtual environment creation fails:
```bash
# Windows: Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative: Use conda instead of venv
conda create -n id-validator python=3.9
conda activate id-validator
```

### Performance Issues
- Ensure sufficient RAM (minimum 4GB recommended)
- Close other applications during processing
- Use smaller image files (under 5MB) for faster processing

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the validation criteria
3. Ensure all dependencies are properly installed
4. Verify your Python version is 3.8 or higher

## 🔄 Version History

- **v1.0.0** - Initial release with comprehensive validation features
  - Professional GUI interface
  - Advanced obstruction detection
  - 68-point facial landmark analysis
  - Skin color and symmetry validation
  - Automated model downloading

---

**Note**: This application is designed for educational and development purposes. For production use in official ID systems, additional security and compliance measures may be required.
