#                                           Diabetic Retinopathy Detection System

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)


> Automating diabetic retinopathy screening using deep learning to reduce diagnosis time from weeks to hours

---

## 📌 Project Overview

This end-to-end deep learning system detects and classifies diabetic retinopathy from retinal fundus images across five severity levels. The solution processes both eyes simultaneously using a custom CNN architecture with dual-eye fusion, reducing diagnostic turnaround from 2-4 weeks to real-time results. Built with a Flask REST API backend and PDF report generation for clinical documentation.

### 🎯 Key Highlights

- Achieved ~0.70 kappa score with initial model, improved with 512×512 architecture
- Processed 106,386 images after strategic augmentation from 35GB raw dataset
- Dual-eye fusion architecture combining left and right eye features before classification
- Production-ready Flask API with real-time predictions and PDF report generation

---

## 💡 Business Problem

Diabetic retinopathy affects over 93 million people globally. In India alone, 65 million individuals have diabetes, with 1 in 10 developing retinopathy. Rural health camps collect thousands of retinal scans that wait weeks for specialist review—delays that can mean the difference between treatable and permanent vision loss.

### Current Workflow vs. Solution

| Current Process | This Solution |
|----------------|---------------|
| Manual screening by ophthalmologist | Automated CNN-based classification |
| 2-4 weeks turnaround | Real-time results |
| Limited scalability | Handles thousands of scans |
| Requires specialist availability | Available 24/7 via API |

---

## 🔬 Classification Levels

| Stage | Condition | Clinical Description |
|-------|-----------|---------------------|
| 0 | No DR | Healthy retina with no disease markers |
| 1 | Mild | Microaneurysms present in blood vessels |
| 2 | Moderate | Blood vessel damage beyond microaneurysms |
| 3 | Severe | Significant blockage in retinal blood supply |
| 4 | Proliferative | Abnormal new vessel growth, high blindness risk |

---

## 📊 Dataset

**Source:** [Kaggle Diabetic Retinopathy Detection Competition (2015)](https://www.kaggle.com/c/diabetic-retinopathy-detection)

| Metric | Value |
|--------|-------|
| Original Images | 35,126 |
| Dataset Size | 35 GB |
| After Augmentation | 106,386 |
| Black Images Removed | 403 |
| Input Resolution | 512×512 / 256×256 |

### Class Distribution

**Original Dataset (Imbalanced):**

| Class | Count | Percentage |
|-------|-------|------------|
| No DR | 25,810 | 73.5% |
| Mild | 2,443 | 7.0% |
| Moderate | 5,292 | 15.1% |
| Severe | 873 | 2.5% |
| Proliferative | 708 | 2.0% |
| **Total** | **35,126** | |

**After Strategic Augmentation:**

| Class | Count | Augmentation Strategy |
|-------|-------|----------------------|
| No DR | 50,976 | Mirror only (2×) |
| Mild | 14,622 | Mirror + Rotations 90°/120°/180°/270° (6×) |
| Moderate | 31,350 | Mirror + Rotations (6×) |
| Severe | 5,190 | Mirror + Rotations (6×) |
| Proliferative | 4,248 | Mirror + Rotations (6×) |
| **Total** | **106,386** | |

---

## ⚙️ Preprocessing Pipeline

Raw fundus images vary significantly in size, lighting, and quality. I implemented a multi-stage preprocessing pipeline combining three approaches:

### 1. Gregwchase Approach

```
Raw Image → Center Crop (1800×1800) → Resize (512×512 or 256×256) → Remove Black Images
```

- Removed 403 corrupted images with no valid color space

### 2. Contrast Enhancement (Ms. Sheetal/Prof. Renke Approach)

```python
# Denoising + CLAHE for enhanced vessel visibility
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(denoised)
```

### 3. Ben Graham Method (Python 2.7)

- Rescale all images to fixed radius (300 or 500 pixels)
- Subtract local average color, remap to 50% gray
- Clip to 90% size to remove boundary effects

### Pipeline Scripts

| Script | Function |
|--------|----------|
| `1_crop_and_resize.py` | Crop to 1800×1800, resize to 512×512/256×256 |
| `2_find_black_images.py` | Identify and remove corrupted files |
| `3_rotate_images.py` | Rotate DR images (90°/120°/180°/270°) + mirror; mirror only for No DR |
| `4_reconcile_label.py` | Update CSV with augmented image labels |
| `5_image_to_array.py` | Convert images to NumPy arrays |
| `6_Denoise_and_CLAHE.py` | Apply denoising and CLAHE contrast enhancement |

---

## 🧠 Model Architecture

### Development Iterations

| Model | Input Size | Batch Size | Kappa Score | Notes |
|-------|------------|------------|-------------|-------|
| v1 | 120×120 | 32 | ~0.70 | Resolution too low for microaneurysm detection |
| v2 (Final) | 512×512 | 64 | Improved | Dual-eye fusion architecture |

### First Model Architecture (120×120)

Used cyclic layers from the Deep Sea team with Leaky ReLU (α=0.3) and SVD orthogonal initialization.

### Final Model Architecture (512×512)

```
Input: 64 × 3 × 512 × 512
         │
         ▼
  Conv2D(32, 7×7, stride=2) → MaxPool(3×3, stride=2)
         │
         ▼
  Conv2D(32, 3×3) × 2 → MaxPool(3×3, stride=2)
         │
         ▼
  Conv2D(64, 3×3) × 2 → MaxPool(3×3, stride=2)
         │
         ▼
  Conv2D(128, 3×3) × 4 → MaxPool(3×3, stride=2)
         │
         ▼
  Conv2D(256, 3×3) × 4 → MaxPool(3×3, stride=2) → Dropout
         │
         ▼
  Maxout (2-pool) → 512 units
         │
         ▼
  Concat with image dimensions → 514
         │
         ▼
┌─────────────────────────────────┐
│       DUAL-EYE FUSION           │
│  Reshape: (64, 514) → (32, 1028)   │
│  Merge left + right eye features   │
└─────────────────────────────────┘
         │
         ▼
  Dropout → Maxout(512) → Dropout → Dense(10) → Reshape(64, 5) → Softmax
```

### Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Activation | Leaky ReLU (α=0.5) | α=0.3 or lower gave significantly lower scores |
| Dense Layers | Maxout (2-pool) | Same or better accuracy with fewer parameters |
| Weight Init | SVD Orthogonal | Based on Saxe et al., improved training stability |
| Batch Size | 64 | GPU memory constraint at 512×512 resolution |

### Training Strategy

**Two-Phase Sampling for Class Imbalance:**

1. **Phase 1:** Oversample minority classes to uniform distribution for stable updates
2. **Phase 2:** Switch back to original distribution to prevent overfitting on repeatedly sampled images

---

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| Kappa Score (v1 - 120×120) | ~0.70 |
| Training Images | 106,386 |
| Test Images | 53,576 |

### Test Set Distribution

| Class | Count |
|-------|-------|
| No DR | 39,533 |
| Mild | 3,762 |
| Moderate | 7,861 |
| Severe | 1,214 |
| Proliferative | 1,206 |
| **Total** | **53,576** |

---

## 🌐 Web Application

Flask REST API with image upload interface and PDF report generation for clinical use.

### Running the Server

```bash
python app.py
```

### API Endpoint

**POST** `/predict`

```
Content-Type: multipart/form-data
Fields: left_eye (image), right_eye (image)
```

### Sample Response

```json
{
  "left_eye": {
    "class": "Moderate",
    "confidence": 0.87,
    "probabilities": [0.02, 0.05, 0.87, 0.04, 0.02]
  },
  "right_eye": {
    "class": "Mild",
    "confidence": 0.91,
    "probabilities": [0.04, 0.91, 0.03, 0.01, 0.01]
  }
}
```

### Features

- Upload retinal images for both eyes
- Real-time prediction with probability bar charts
- PDF report generation for doctors to reference later

---

## 🛠️ Technical Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Primary language | 3.7+ |
| **TensorFlow/Keras** | Deep learning framework | 2.0+ |
| **OpenCV** | Image preprocessing, CLAHE, denoising | 4.0+ |
| **NumPy** | Numerical computations | Latest |
| **Pandas** | Data manipulation | Latest |
| **Flask** | REST API backend | Latest |
| **Scikit-Image** | Image processing | Latest |
| **Matplotlib** | Visualization | Latest |

---

## 📁 Project Structure

```
Diabetic-Retinopathy-Detection/
│
├── src/
│   ├── Model/
│   │   └── [trained model files]
│   ├── Preprocessing_Scripts/
│   │   ├── Train/
│   │   │   ├── 1_crop_and_resize.py
│   │   │   ├── 2_find_black_images.py
│   │   │   ├── 3_rotate_images.py
│   │   │   ├── 4_reconcile_label.py
│   │   │   ├── 5_image_to_array.py
│   │   │   ├── 6_Denoise_and_CLAHE.py
│   │   │   └── Ben_Graham/
│   │   │       └── 1_remove_boundary_effects.py
│   │   └── Test/
│   └── miscellaneous_scripts/
│
├── images/
│   └── readme/
│
├── data/
│   └── [dataset files]
│
├── docs/
│   ├── data_dictionary.md
│   └── methodology.md
│
├── app.py
├── train.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager
- GPU recommended for training (CUDA compatible)

### Installation

**1. Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download dataset**

Download from [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) and place in `data/` folder.

### Quick Start

**Preprocess Images:**

```bash
python src/Preprocessing_Scripts/Train/1_crop_and_resize.py
python src/Preprocessing_Scripts/Train/2_find_black_images.py
python src/Preprocessing_Scripts/Train/3_rotate_images.py
python src/Preprocessing_Scripts/Train/4_reconcile_label.py
```

**Train Model:**

```bash
python train.py --input_size 512 --batch_size 64 --epochs 100
```

**Run Inference:**

```python
from model import DRClassifier

classifier = DRClassifier.load('src/Model/dr_model.h5')
result = classifier.predict('left_eye.jpg', 'right_eye.jpg')

print(f"Left Eye: {result['left']['class']} ({result['left']['confidence']:.1%})")
print(f"Right Eye: {result['right']['class']} ({result['right']['confidence']:.1%})")
```

---

## 🔑 Key Technical Contributions

### Dual-Eye Fusion

Most approaches classify each eye independently. I merge feature representations from both eyes before the final classification layers because diabetic retinopathy often manifests asymmetrically—patterns across both eyes carry diagnostic signals that single-eye models miss.

### Selective Augmentation

Applied aggressive augmentation (rotations + flips) only to underrepresented disease classes. Healthy images received only mirroring to avoid skewing the natural distribution excessively.

### Maxout Networks

Replaced traditional fully connected layers with Maxout units, which provide equivalent or better accuracy with fewer trainable parameters.

### Two-Phase Sampling

Training begins with uniform class distribution to ensure adequate exposure to rare cases, then transitions to natural class frequencies to prevent overfitting on repeatedly sampled images.

### Higher Leakiness

Using Leaky ReLU with α=0.5 instead of the standard 0.3 made a significant difference in model performance.

---

## 📚 References

- [Kaggle DR Detection Competition (2015)](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- [Ben Graham's Preprocessing Method](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801)
- [Ben Graham's SparseConvNet](https://github.com/btgraham/SparseConvNet)
- [Gregwchase DSI Capstone](https://github.com/gregwchase/dsi-capstone)
- [OpenCV Denoising](https://docs.opencv.org/3.3.0/d5/d69/tutorial_py_non_local_means.html)
- [OpenCV CLAHE](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
- [Ms. Sheetal/Prof. Renke - DR Detection Paper](https://www.ripublication.com/irph/ijert_spl17/ijertv10n1spl_96.pdf)
- [Orthogonal Weight Initialization - Saxe et al.](https://arxiv.org/abs/1312.6120)
- [Deep Sea Team - Cyclic Layers](http://benanne.github.io/2015/03/17/plankton.html)
- [Google AI - DR Detection](https://ai.googleblog.com/2018/12/improving-effectiveness-of-diabetic.html)
- [Google - Seeing Potential](https://about.google/stories/seeingpotential/)

---

## 👩‍💻 Author

**Keerthi Samhitha Kadaveru**

- 📧 Email: [k.samhitha23@gmail.com](mailto:k.samhitha23@gmail.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

[Back to Top](#-diabetic-retinopathy-detection-system)
