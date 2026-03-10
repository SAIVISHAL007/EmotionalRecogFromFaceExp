# 🎭 Real-Time Facial Emotion Recognition System

**Academic Project: Neural Networks & Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)

---

## 📋 Overview

A **production-grade, full-stack facial emotion recognition system** that combines deep learning, computer vision, and modern web technologies. This system detects faces in real-time from webcam video and predicts emotions using a Convolutional Neural Network trained on the FER-2013 dataset.

### 🎯 Key Features

- ✅ **CNN Deep Learning Model** trained on 35,000+ facial images
- ✅ **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- ✅ **Real-time Detection** from webcam video stream
- ✅ **REST API Backend** with FastAPI
- ✅ **Modern React Frontend** with live visualization
- ✅ **Production-Ready** code with proper architecture
- ✅ **Academic Viva-Ready** with clear documentation

---

## 🚀 Quick Start

**Want to get running quickly?** See **[QUICKSTART.md](QUICKSTART.md)** for complete setup instructions!

### TL;DR

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python data/download_fer2013.py

# 3. Train model (30-90 min)
python model/train.py

# 4. Start backend (Terminal 1)
python backend/main.py

# 5. Start frontend (Terminal 2)
cd frontend && npm install && npm run dev

# 6. Open http://localhost:3000
```

---

## 🏗️ System Architecture

```
┌─────────────┐     📹     ┌──────────────┐     🌐     ┌────────────┐
│   Browser   │ ─────────> │   Frontend   │ ─────────> │  Backend   │
│  (Webcam)   │            │   (React)    │            │  (FastAPI) │
└─────────────┘            └──────────────┘            └────────────┘
                                                              │
                                                              ↓
                                                        ┌────────────┐
                                                        │    CNN     │
                                                        │   Model    │
                                                        └────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **CNN Model** | TensorFlow/Keras | 7-class emotion classification |
| **Face Detection** | OpenCV (Haar Cascade) | Detect faces in frames |
| **Backend API** | FastAPI + Uvicorn | RESTful emotion prediction service |
| **Frontend** | React + Vite | Real-time webcam interface |
| **Real-time App** | OpenCV + Python | Standalone webcam application |

---

## 📂 Project Structure

```
EmotionalRecogFromFaceExp/
│
├── 📁 model/                          # CNN Model & Training (ACADEMIC CORE)
│   ├── config.py                      # Hyperparameters & settings
│   ├── architecture.py                # CNN architecture (Conv2D, ReLU, etc.)
│   ├── train.py                       # Training pipeline
│   ├── evaluate.py                    # Model evaluation & metrics
│   ├── utils.py                       # Visualization & helpers
│   └── README.md                      # Training documentation
│
├── 📁 data/                           # Dataset Management
│   ├── download_fer2013.py            # Dataset download script
│   ├── preprocess.py                  # Data preprocessing utilities
│   └── README.md                      # Dataset information
│
├── 📁 realtime/                       # Real-time Detection (OpenCV)
│   ├── detector.py                    # Face detection (Haar Cascade)
│   ├── emotion_predictor.py           # Emotion prediction wrapper
│   ├── webcam_app.py                  # Standalone webcam app
│   └── README.md                      # Real-time detection guide
│
├── 📁 backend/                        # FastAPI Backend
│   ├── main.py                        # FastAPI application
│   ├── models.py                      # Pydantic request/response models
│   ├── services/
│   │   └── emotion_service.py         # Business logic
│   ├── requirements.txt               # Backend dependencies
│   └── README.md                      # API documentation
│
├── 📁 frontend/                       # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamFeed.jsx        # Webcam capture component
│   │   │   └── EmotionDisplay.jsx    # Results display component
│   │   ├── services/
│   │   │   └── api.js                # Backend API client
│   │   ├── App.jsx                   # Main application
│   │   └── main.jsx                  # Entry point
│   ├── package.json                   # Node dependencies
│   ├── vite.config.js                # Vite configuration
│   └── README.md                      # Frontend documentation
│
├── 📁 trained_models/                 # Model checkpoints (.h5 files)
├── 📁 logs/                          # Training logs & history
│
├── 📄 requirements.txt                # Python dependencies
├── 📄 QUICKSTART.md                   # Quick setup guide
├── 📄 .gitignore                     # Git ignore rules
└── 📄 README.md                       # This file
```

---

## 🧠 CNN Model Architecture

### Network Design

```
Input (48x48x1 Grayscale)
    ↓
┌─────────────────────────────────┐
│ Conv Block 1: 32 filters        │
│ Conv2D → BatchNorm → ReLU       │
│ MaxPool2D → Dropout(0.25)       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Conv Block 2: 64 filters        │
│ Conv2D → BatchNorm → ReLU       │
│ MaxPool2D → Dropout(0.25)       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Conv Block 3: 128 filters       │
│ Conv2D → BatchNorm → ReLU       │
│ MaxPool2D → Dropout(0.25)       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Conv Block 4: 256 filters       │
│ Conv2D → BatchNorm → ReLU       │
│ MaxPool2D → Dropout(0.25)       │
└─────────────────────────────────┘
    ↓
Flatten
    ↓
Dense(512) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    ↓
Dense(7) → Softmax
    ↓
Output (7 emotion probabilities)
```

### Key Components

- **Activation**: ReLU (Rectified Linear Unit)
- **Pooling**: MaxPooling2D (2×2)
- **Regularization**: Dropout + Batch Normalization
- **Loss**: Categorical Cross-Entropy
- **Optimizer**: Adam (learning rate: 0.001)
- **Output**: Softmax (7 classes)

### Performance

- **Training Accuracy**: ~70-75%
- **Validation Accuracy**: ~65-70%
- **Test Accuracy**: ~60-68%
- **Training Time**: 30-90 minutes (CPU) / 10-30 minutes (GPU)

---

## 🎓 Academic Focus

This project is designed for **Neural Networks & Deep Learning** coursework and covers:

### 1. Deep Learning Concepts
- Convolutional Neural Networks (CNNs)
- Backpropagation and gradient descent
- Activation functions (ReLU, Softmax)
- Loss functions (Categorical Cross-Entropy)
- Regularization techniques (Dropout, Batch Normalization)

### 2. Computer Vision
- Image preprocessing and normalization
- Face detection algorithms (Haar Cascade)
- Real-time video processing
- Feature extraction from images

### 3. Model Training & Evaluation
- Train/validation/test split
- Learning curves and overfitting
- Confusion matrix analysis
- Precision, Recall, F1-score metrics
- Hyperparameter tuning

### 4. Production Deployment
- REST API design
- Client-server architecture
- Real-time inference optimization
- Full-stack web development

---

## 📊 Dataset: FER-2013

- **Total Images**: 35,887 grayscale images
- **Image Size**: 48×48 pixels
- **Classes**: 7 emotions
  - 😠 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😊 Happy
  - 😢 Sad
  - 😲 Surprise
  - 😐 Neutral
- **Split**: ~28K train, ~3.5K validation, ~3.5K test

**Source**: [Kaggle FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **TensorFlow 2.15** - Deep learning framework
- **Keras** - High-level neural networks API
- **OpenCV 4.8** - Computer vision library
- **FastAPI 0.104** - Modern web framework
- **Uvicorn** - ASGI server
- **NumPy, Pandas** - Data processing

### Frontend
- **React 18** - UI framework
- **Vite 5** - Build tool
- **Axios** - HTTP client
- **HTML5 Canvas** - Frame capture
- **MediaDevices API** - Webcam access

### ML & CV
- **CNN** - Convolutional Neural Network
- **Haar Cascade** - Face detection
- **Adam Optimizer** - Training optimization
- **Data Augmentation** - Training enhancement

---

## 🎯 Usage

### 1. Train the CNN Model

```bash
python model/train.py
```

**Output:**
- Trained model: `trained_models/emotion_cnn_model.h5`
- Training history: `logs/training_YYYYMMDD_HHMMSS/`
- Performance plots: Loss/accuracy curves

### 2. Evaluate Model Performance

```bash
python model/evaluate.py
```

**Output:**
- Confusion matrix
- Classification report
- Per-class accuracy
- Sample predictions visualization

### 3. Test Real-time Detection (Standalone)

```bash
python realtime/webcam_app.py
```

**Controls:**
- `q` - Quit
- `s` - Save screenshot
- `p` - Pause/Resume
- `f` - Toggle FPS display
- `d` - Debug mode (show top-3 predictions)

### 4. Run Full Web Application

**Terminal 1 (Backend):**
```bash
python backend/main.py
```
Access API at: http://localhost:8000  
Swagger docs: http://localhost:8000/docs

**Terminal 2 (Frontend):**
```bash
cd frontend
npm install  # First time only
npm run dev
```
Access app at: http://localhost:3000

---

## 🌐 Web Application Features

### Frontend Interface
- 📹 Live webcam video feed
- 🎯 Real-time face detection overlays
- 📊 Emotion probability bars for each face
- ⚡ FPS counter and performance stats
- 🎨 Color-coded emotion visualization
- 📱 Responsive design for all devices

### Backend API
- `GET /api/health` - Service health check
- `GET /api/model-info` - Model architecture info
- `POST /api/predict` - Predict from image file
- `POST /api/predict-base64` - Predict from base64 image

### Real-time Workflow
1. Frontend captures webcam frame
2. Converts frame to base64
3. Sends to backend API
4. Backend detects faces
5. CNN predicts emotions
6. Results returned as JSON
7. Frontend displays predictions
8. Repeat every 500ms

---

## 📈 Performance Optimization

### Training
- **GPU Acceleration**: Automatically used if available
- **Early Stopping**: Stops when validation plateaus
- **Learning Rate Decay**: Reduces LR on plateau
- **Data Augmentation**: Increases training diversity

### Inference
- **Batch Processing**: Process multiple faces simultaneously
- **Model Caching**: Load model once, reuse for all requests
- **Frame Throttling**: Capture at 2 FPS (adjustable)
- **Async Processing**: Non-blocking API requests

---

## 🐛 Troubleshooting

### Common Issues

**1. Model not found**
```bash
# Solution: Train the model
python model/train.py
```

**2. Dataset not found**
```bash
# Solution: Download dataset
python data/download_fer2013.py
```

**3. Backend connection failed**
```bash
# Solution: Ensure backend is running
python backend/main.py
# Check: curl http://localhost:8000/api/health
```

**4. Webcam not working**
- Grant browser camera permissions
- Close other apps using camera
- Try different browser (Chrome recommended)
- Ensure localhost or HTTPS

See **[QUICKSTART.md](QUICKSTART.md)** for more troubleshooting tips.

---

## 🎤 For Academic Presentation

### Key Discussion Points

1. **CNN Architecture**
   - Why convolutional layers?
   - Role of pooling and dropout
   - Batch normalization benefits

2. **Training Process**
   - Loss function choice
   - Optimizer comparison
   - Preventing overfitting

3. **Real-world Challenges**
   - Low-resolution images (48×48)
   - Class imbalance in dataset
   - Lighting and pose variations

4. **Performance Analysis**
   - Confusion matrix interpretation
   - Which emotions are hardest?
   - Improvements and future work

5. **Deployment**
   - API design choices
   - Real-time optimization
   - Scalability considerations

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Complete setup guide
- **[model/README.md](model/README.md)** - CNN training guide
- **[realtime/README.md](realtime/README.md)** - Real-time detection
- **[backend/README.md](backend/README.md)** - API documentation
- **[frontend/README.md](frontend/README.md)** - Frontend guide
- **[data/README.md](data/README.md)** - Dataset information

---

## 📝 Future Enhancements

- [ ] Multi-face tracking across frames
- [ ] Emotion history visualization
- [ ] MTCNN for better face detection
- [ ] Transfer learning from pre-trained models
- [ ] Mobile app development
- [ ] Cloud deployment (AWS/Azure)
- [ ] Real-time emotion heatmaps
- [ ] Voice/audio emotion integration

---

## 🤝 Contributing

This is an academic project. Suggestions and improvements welcome!

---

## 📄 License

MIT License - Feel free to use for educational purposes.

---

## 👨‍🎓 Author

**Academic Project**  
Course: Neural Networks & Deep Learning  
Year: 2026

---

## 🙏 Acknowledgments

- **FER-2013 Dataset**: Kaggle community
- **TensorFlow/Keras**: Google Brain team
- **OpenCV**: Open Source Computer Vision Library
- **FastAPI**: Sebastián Ramírez
- **React**: Meta/Facebook

---

## 📞 Support

For issues or questions:
1. Check individual README files in each directory
2. Review [QUICKSTART.md](QUICKSTART.md) troubleshooting section
3. Examine error messages carefully
4. Ensure all dependencies are installed

---

## ⭐ Project Highlights

✨ **Production-Grade Code**  
✨ **Complete Full-Stack System**  
✨ **Real-time Performance**  
✨ **Academic Viva-Ready**  
✨ **Comprehensive Documentation**  
✨ **Modern Tech Stack**

---

**Ready to start? See [QUICKSTART.md](QUICKSTART.md) for complete setup instructions!** 🚀
