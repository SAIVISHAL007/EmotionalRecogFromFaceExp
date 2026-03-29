# 🎭 Real-Time Facial Emotion Recognition System

**Academic Project: Neural Networks & Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-red)](https://opencv.org/)

---

## 📋 Overview

A **production-grade, full-stack facial emotion recognition system** that combines deep learning, computer vision, and modern web technologies. This system detects **multiple faces** in real-time from a webcam and predicts emotions using a CNN trained on FER-2013, with a landmark-based Random Forest classifier as the high-accuracy alternative.

### 🎯 Key Features

- ✅ **CNN Deep Learning Model** — 4-block ConvNet trained on 35,000+ FER-2013 images
- ✅ **YuNet DNN Face Detector** — OpenCV's built-in DNN detector (~97% accuracy, replaces Haar)
- ✅ **Multi-Face Support** — Up to 10 simultaneous faces in a single frame, each tracked independently
- ✅ **7 Emotion Classes** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral (Ekman's model)
- ✅ **Per-Face EMA Smoothing** — Stabilises predictions across frames per tracked face
- ✅ **MediaPipe Landmark Pipeline** — 478 3D face landmarks → Random Forest (85%+ accuracy via Colab)
- ✅ **REST API Backend** — FastAPI + Uvicorn
- ✅ **Modern React Frontend** — Live emotion bars, face count, FPS display
- ✅ **Python 3.13 Compatible** — Works without MediaPipe `solutions` API

---

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the CNN model (~30-60 min on CPU)
python model/train.py

# 4. Start backend (Terminal 1)
uvicorn backend.main:app --port 8000

# 5. Start frontend (Terminal 2)
cd frontend && npm install && npm run dev

# 6. Open http://localhost:3000
```

---

## 🏗️ System Architecture

```
┌─────────────┐     📹     ┌──────────────┐     🌐     ┌────────────────────┐
│   Browser   │ ─────────> │   Frontend   │ ─────────> │  FastAPI Backend   │
│  (Webcam)   │            │   (React)    │            │   :8000            │
└─────────────┘            └──────────────┘            └────────────────────┘
                                                                │
                              ┌─────────────────────────────────┤
                              ↓                                 ↓
                    ┌──────────────────┐             ┌──────────────────┐
                    │  YuNet DNN       │             │  CNN Predictor   │
                    │  Face Detector   │─── ROI ───> │  (48×48 crops)   │
                    │  multi-face      │             │  7 emotions      │
                    └──────────────────┘             └──────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **CNN Model** | TensorFlow/Keras | 7-class emotion classification |
| **Face Detector** | OpenCV YuNet DNN | Multi-face detection (~97% accuracy) |
| **Fallback Detector** | OpenCV Haar Cascade | Python 3.13 safety fallback |
| **RF Classifier** | scikit-learn Random Forest | High-accuracy landmark-based inference (Colab) |
| **Landmark Extractor** | MediaPipe Face Mesh (Tasks API) | 478 3D landmarks per face |
| **Backend API** | FastAPI + Uvicorn | RESTful emotion prediction service |
| **Frontend** | React 18 + Vite 5 | Real-time webcam interface |

---

## 📂 Project Structure

```
EmotionalRecogFromFaceExp/
│
├── 📁 model/                          # CNN Model & Training (CORE)
│   ├── config.py                      # Hyperparameters & settings
│   ├── architecture.py                # 4-block CNN architecture
│   ├── train.py                       # Local CPU training pipeline
│   ├── train_rf_colab.ipynb           # 🚀 Colab notebook (RF, 85%+ accuracy)
│   ├── evaluate.py                    # Model evaluation & metrics
│   ├── utils.py                       # Visualization & helpers
│   └── README.md                      # Training documentation
│
├── 📁 realtime/                       # Detection & Prediction
│   ├── yunet_detector.py              # ⭐ YuNet DNN multi-face detector
│   ├── detector.py                    # Haar Cascade detector (fallback)
│   ├── emotion_predictor.py           # CNN emotion predictor wrapper
│   ├── mediapipe_detector.py          # MediaPipe Tasks API (478 landmarks)
│   ├── multi_emotion_predictor.py     # Random Forest predictor
│   ├── webcam_app.py                  # Standalone OpenCV webcam app
│   └── README.md                      # Detection guide
│
├── 📁 backend/                        # FastAPI Backend
│   ├── main.py                        # FastAPI application & routing
│   ├── models.py                      # Pydantic request/response schemas
│   ├── services/
│   │   └── emotion_service.py         # Core inference orchestration
│   └── README.md                      # API documentation
│
├── 📁 frontend/                       # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamFeed.jsx        # Webcam capture & base64 encoding
│   │   │   └── EmotionDisplay.jsx    # Per-face emotion bars
│   │   ├── services/api.js           # Backend API client
│   │   └── App.jsx                   # Main application
│   ├── package.json
│   └── vite.config.js
│
├── 📁 trained_models/                 # Model files (gitignored)
│   ├── emotion_cnn_model.h5           # Trained CNN weights
│   ├── emotion_rf_model.pkl           # Random Forest model (Colab output)
│   └── face_detection_yunet.onnx      # YuNet ONNX weights
│
├── 📁 data/                           # Dataset directory
├── 📄 requirements.txt                # Python dependencies
├── 📄 QUICKSTART.md                   # Full setup guide
├── 📄 UPGRADE_GUIDE.md                # YuNet / MediaPipe upgrade notes
└── 📄 README.md                       # This file
```

---

## 🧠 CNN Model Architecture

```
Input: 48×48×1 (Grayscale)
    ↓
Conv Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv Block 4: Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(7) → Softmax → Output (7 emotion probabilities)
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 0.001) |
| Loss | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 50 (with EarlyStopping) |
| Input | 48×48 grayscale |
| Dataset | FER-2013 (~28K train / 3.5K val) |
| Val Accuracy | **~55% (CPU local)** |

> **Note**: FER-2013 has a known human accuracy ceiling of ~65±5%. The CNN reaches 55% locally. For 85%+ accuracy, run `model/train_rf_colab.ipynb` on Google Colab (MediaPipe landmarks → Random Forest).

---

## 👁️ Face Detection Pipeline

### YuNet DNN Detector (Primary)

`realtime/yunet_detector.py` — OpenCV's built-in `cv2.FaceDetectorYN`:

- **Accuracy**: ~97% on WIDER FACE benchmark (vs ~60% Haar)
- **Multi-face**: Detects up to 10 simultaneous faces
- **Full-face bbox**: Returns proper head enclosure, not just eye region
- **Decoupled ROI**: Tight CNN crop (inference) vs expanded display box (visuals)
- **Model**: `trained_models/face_detection_yunet.onnx` (~350KB)

```
Detection → Tight ROI (48×48) → CNN Inference
         → Expanded Display Box → Label overlay on screen
```

### Fallback Chain

```
YuNet DNN → Haar Cascade → Error
```

---

## 🎓 Academic Context

This project is designed for **Neural Networks & Deep Learning** coursework and covers:

### Deep Learning Concepts
- Convolutional Neural Networks (CNNs) — spatial feature extraction
- Backpropagation & gradient descent — Adam optimizer
- Activation functions — ReLU (hidden), Softmax (output)
- Regularization — Dropout + Batch Normalization

### Computer Vision
- Face detection algorithm comparison (Haar vs DNN)
- Image preprocessing — grayscale, normalization, resize
- Real-time video frame processing pipeline
- Landmark extraction — MediaPipe 478 3D face points

### Model Evaluation
- Training/validation/test split strategy
- Confusion matrix analysis per class
- Precision, Recall, F1 metrics
- Dataset imbalance effects (Disgust: 547 vs Happy: 7,215 samples)

### Production Engineering
- REST API design with FastAPI
- Client-server full-stack architecture
- Singleton service management
- EMA smoothing for stable real-time predictions

---

## 📊 Dataset: FER-2013

| Property | Value |
|----------|-------|
| Total images | 35,887 grayscale |
| Image size | 48×48 pixels |
| Classes | 7 emotions |
| Train split | ~28,000 |
| Val split | ~3,500 |
| Human accuracy | ~65±5% |
| CNN baseline | ~60% |

**Class Distribution (imbalanced):**

| Emotion | Samples |
|---------|---------|
| 😠 Angry | 3,995 |
| 🤢 Disgust | **547** ← most underrepresented |
| 😨 Fear | 4,097 |
| 😊 Happy | 7,215 |
| 😢 Sad | 4,830 |
| 😲 Surprise | 3,171 |
| 😐 Neutral | 4,965 |

**Source**: [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 🛠️ Technology Stack

### Backend
- **Python 3.13** — fully compatible
- **TensorFlow 2.x** — CNN training & inference
- **OpenCV 4.13** — YuNet DNN + Haar fallback + video processing
- **FastAPI 0.104** — REST API framework
- **Uvicorn** — ASGI server
- **scikit-learn** — Random Forest classifier
- **NumPy** — array operations

### Frontend
- **React 18** — UI framework
- **Vite 5** — development build tool
- **Axios** — HTTP client for API calls
- **HTML5 Canvas** — webcam frame capture

### ML & CV
- **YuNet ONNX** — DNN face detector
- **MediaPipe 0.10 (Tasks API)** — 478 3D face landmarks
- **Random Forest** — landmark-based classifier (Colab path)
- **CNN** — pixel-based classifier (local path)

---

## 🌐 API Reference

```
GET  /api/health          → Service status + model_loaded flag
GET  /api/model-info      → Model input/output shape, emotion labels
POST /api/predict         → Predict from uploaded image file
POST /api/predict-base64  → Predict from base64-encoded frame
```

**Swagger UI**: http://localhost:8000/docs

---

## 📈 Accuracy Roadmap

| Setup | Detector | Classifier | Expected Accuracy |
|-------|---------|------------|-------------------|
| Local CPU (current) | YuNet | CNN | ~55% |
| Local GPU | YuNet | CNN | ~62% |
| **Google Colab** | MediaPipe | **Random Forest** | **~85%+** |

To reach 85%+ accuracy: run `model/train_rf_colab.ipynb` on Colab → download `emotion_rf_model.pkl` → place in `trained_models/`.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model Not Found` badge | Run `python model/train.py` first |
| Port 3000 in use | Frontend auto-switches to 3001 |
| Backend unhealthy | Restart `uvicorn backend.main:app --port 8000` |
| YuNet ONNX missing | Run `python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', 'trained_models/face_detection_yunet.onnx')"` |
| Webcam not working | Grant browser camera permission; try Chrome |
| `mediapipe.solutions` error | Expected on Python 3.13 — YuNet handles detection instead |

---

## 📝 Future Enhancements

- [x] Multi-face tracking across frames
- [x] YuNet DNN face detection (done)
- [x] Per-face EMA smoothing (done)
- [ ] Emotion history timeline chart
- [ ] Transfer learning (VGG-Face, ArcFace)
- [ ] Mobile / Edge deployment (ONNX export)
- [ ] Cloud deployment (Docker + AWS)
- [ ] Audio + facial emotion fusion

---

## 🤝 Research References

- Ciraolo et al., *"Facial expression recognition based on emotional AI for tele-rehabilitation"*, Biomedical Signal Processing and Control, 2024
- Goodfellow et al., *"Challenges in Representation Learning: A report on three machine learning contests"*, ICML 2013 (FER-2013)
- MediaPipe Face Mesh: Kartynnik et al., 2019

---

## 📄 License

MIT License — free to use for educational purposes.

---

## 👨‍🎓 Author

**Academic Project**  
Course: Neural Networks & Deep Learning  
Year: 2026  
GitHub: [SAIVISHAL007/EmotionalRecogFromFaceExp](https://github.com/SAIVISHAL007/EmotionalRecogFromFaceExp)
