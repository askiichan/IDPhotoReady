# IDPhotoReady

A program that checks if a photo is suitable for use on an ID document.

## 📋 Table of Contents

- [Key Features](#✨-key-features)
- [Installation](#⚙️-installation)
- [Usage](#🚀-usage)
  - [Web GUI](#🖥️-web-gui)
  - [REST API](#🔌-rest-api)
- [Validation Options](#✔️-validation-options)
- [Configuration & Tuning](#🔧-configuration--tuning)
- [Project Structure](#📂-project-structure)
- [Troubleshooting](#🚑-troubleshooting)
- [License](#📄-license)

## ✨ Key Features

- **🖥️ Dual Interface:** A user-friendly **Web GUI** for easy uploads and a **REST API** for system integration.
- **✅ Flexible Validation:** Checks for face position, eye state (open/closed), obstructions, background uniformity, shoulder balance, and image quality.
- **🖼️ Annotated Results:** Provides clear visual feedback on the image, highlighting detected landmarks and specific validation failures.
- **🚀 Built with Power:** Leverages **OpenCV**, **MediaPipe**, and **scikit-learn** for accurate and reliable analysis.

## ⚙️ Installation

A virtual environment is recommended.

```bash
# 1. Clone the repository
git clone https://github.com/askiichan/IDPhotoReady.git
cd IDPhotoReady

# 2. Create and activate a virtual environment
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

The required AI models will be downloaded automatically on first run. For manual downloads, see [Troubleshooting](#🚑-troubleshooting).

## 🚀 Usage

You can run this project as a self-contained web application or as a backend API.

### 🖥️ Web GUI

The easiest way to get started. Ideal for validating single photos or batches.

1.  **Start the application:**
    ```bash
    python main.py
    ```
2.  **Open your browser** and navigate to the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

_The interface provides detailed failure reasons and an annotated image._

### 🔌 REST API

For programmatic access and integration with other services.

1.  **Start the API server:**
    ```bash
    python start_api.py
    ```
2.  **Access the interactive documentation** in your browser at `http://localhost:8000/docs`. You can test all endpoints directly from this page.

#### API Endpoints

- `POST /validate`: Upload an image file for validation.
- `POST /validate-base64`: Submit a base64-encoded image string.
- `GET /health`: Check if the API is running.

#### Example: cURL

```bash
curl -X POST http://localhost:8000/validate \
  -H "Accept: application/json" \
  -F "file=@/path/to/photo.jpg;type=image/jpeg" \
  -F "return_annotated=true" \
  -F "validation_preset=custom" \
  -F "face_sizing=true" \
  -F "landmark_analysis=true" \
  -F "eye_validation=true" \
  -F "obstruction_detection=true" \
  -F "mouth_validation=true" \
  -F "quality_assessment=true" \
  -F "background_validation=true" \
  -F "shoulder_balance_validation=true"
```

## ✔️ Validation Options

The system performs a series of checks. Most can be enabled or disabled.

| Category                  | What it Checks                                               | Default     |
| :------------------------ | :----------------------------------------------------------- | :---------- |
| **Face Detection**        | Detects exactly one face with high confidence.               | Always On   |
| **Face Sizing**           | Face occupies a reasonable portion (5-80%) of the image.     | On          |
| **Landmark Analysis**     | All 68 facial landmarks are visible and properly positioned. | On          |
| **Eye Validation**        | Both eyes are open using the Eye Aspect Ratio (EAR).         | On          |
| **Obstruction Detection** | Face is not covered by hands, masks, or other objects.       | On          |
| **Mouth Validation**      | Mouth is visible and not unnaturally obscured.               | On          |
| **Image Quality**         | Image is not a cartoon or drawing (using color analysis).    | On          |
| **Background**            | Background is uniform and neutral (e.g., white).             | On          |
| **Shoulder Balance**      | Shoulders are visible, level, and framed correctly.          | Strict Only |

## 🔧 Configuration & Tuning

You can fine-tune the validation logic by editing the constants in `id_validator/config.py`. This is for advanced users who need to adjust the validator's sensitivity.

**Background Validation Thresholds**

- `BG_MIN_MEAN_V`: Minimum brightness (0-255). Lower to allow darker backgrounds.
- `BG_MAX_MEAN_S`: Maximum saturation (0-255). Raise to allow more colorful backgrounds.
- `BG_SAMPLE_BORDER_PCT`: Size of the ring around the face used for sampling.

**Shoulder Balance Thresholds**

- `SHOULDER_VISIBILITY_THRESHOLD`: Minimum confidence score (0.0-1.0) for detecting shoulders.
- `MAX_SHOULDER_TILT_DEG`: Maximum allowed angle for shoulder tilt.
- `MIN_SHOULDER_WIDTH_TO_FACE_RATIO`: Minimum shoulder width relative to face width.

## 📂 Project Structure

```
.
├── main.py               # Gradio Web GUI entrypoint
├── api.py                # FastAPI server entrypoint
├── requirements.txt      # Dependencies
├── id_validator/         # Core library
│   ├── validator.py      # Validation pipeline
│   ├── config.py         # Thresholds & constants
│   ├── validation_config.py  # Preset option grouping
│   ├── models.py         # Pydantic models
│   └── gradio_gui.py     # GUI wiring
├── models/               # Downloaded model weights
└── test_imgs/            # Sample images
```

## 🚑 Troubleshooting

If the ML models do not download automatically, please check your internet connection. You can also download them manually from the links below and place them in the `models/` directory.

- [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
- [res10_300x300_ssd_iter_140000.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
- [lbfmodel.yaml](https://github.com/spmallick/GSOC2017/raw/master/data/lbfmodel.yaml)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

**Disclaimer**: This tool is intended for educational and development purposes. For official use, ensure compliance with all local regulations.
