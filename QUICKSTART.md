# 🚀 Quick Start Guide

## Complete Setup & Run Instructions

This guide will help you get the entire emotion recognition system running from scratch.

## Prerequisites

✅ **Python 3.8+**  
✅ **Node.js 16+**  
✅ **Webcam**  
✅ **Git** (optional)

---

## 📋 Step-by-Step Setup

### **STEP 1: Install Python Dependencies**

```bash
# From project root directory
pip install -r requirements.txt
```

**This installs:**
- TensorFlow/Keras (Deep Learning)
- OpenCV (Computer Vision)
- FastAPI (Backend)
- NumPy, Matplotlib, etc.

---

### **STEP 2: Download FER-2013 Dataset**

```bash
python data/download_fer2013.py
```

**Follow the instructions to:**
1. Visit Kaggle and download FER-2013
2. Extract to `data/` directory
3. Or use Kaggle API if configured

**Dataset Structure:**
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    └── (same structure)
```

---

### **STEP 3: Train the CNN Model**

```bash
python model/train.py
```

**Training Details:**
- Duration: 30-90 minutes (CPU) or 10-30 minutes (GPU)
- Epochs: Up to 50 (with early stopping)
- Output: `trained_models/emotion_cnn_model.h5`

**Monitor Progress:**
- Watch console for accuracy/loss
- Training plots saved automatically
- Model auto-saves best weights

---

### **STEP 4: (Optional) Test Real-time Detection**

```bash
python realtime/webcam_app.py
```

**Controls:**
- `q` - Quit
- `s` - Save screenshot
- `p` - Pause/Resume
- `f` - Toggle FPS
- `d` - Debug mode

This tests the standalone OpenCV application before web integration.

---

### **STEP 5: Start Backend API**

```bash
# Terminal 1
python backend/main.py
```

**Backend will run on:** http://localhost:8000

**Verify it's working:**
- Open http://localhost:8000 in browser
- You should see the API landing page
- Check http://localhost:8000/docs for Swagger UI

---

### **STEP 6: Start Frontend**

```bash
# Terminal 2 (new terminal)
cd frontend
npm install  # First time only
npm run dev
```

**Frontend will run on:** http://localhost:3000

---

### **STEP 7: Use the Application**

1. Open http://localhost:3000 in your browser
2. Grant camera permissions when prompted
3. Click **"Start Detection"**
4. Face the camera and see your emotions detected in real-time!

---

## 🎯 Quick Commands Reference

| Task | Command |
|------|---------|
| Train Model | `python model/train.py` |
| Evaluate Model | `python model/evaluate.py` |
| Test Webcam (Standalone) | `python realtime/webcam_app.py` |
| Start Backend | `python backend/main.py` |
| Start Frontend | `cd frontend && npm run dev` |
| Backend API Docs | http://localhost:8000/docs |
| Frontend App | http://localhost:3000 |

---

## 🔧 Troubleshooting

### Model Not Found

**Error:** `Model not found at: trained_models/emotion_cnn_model.h5`

**Solution:**
```bash
python model/train.py
```

Wait for training to complete.

---

### Dataset Not Found

**Error:** `Dataset not found at data/train`

**Solution:**
```bash
python data/download_fer2013.py
```

Follow instructions to download from Kaggle.

---

### Backend Connection Failed

**Error:** `Failed to connect to backend`

**Solution:**
1. Ensure backend is running: `python backend/main.py`
2. Check it's on http://localhost:8000
3. Test with: `curl http://localhost:8000/api/health`

---

### Webcam Not Working

**Error:** `Failed to access webcam`

**Solutions:**
- Grant browser camera permissions
- Close other apps using the camera
- Try different browser (Chrome recommended)
- Ensure HTTPS or localhost
- Check camera with: `python realtime/detector.py`

---

### Port Already in Use

**Error:** `Address already in use: port 8000`

**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or change port
uvicorn backend.main:app --port 8001
```

---

### CORS Errors

**Error:** `CORS policy blocked`

**Solution:**
- Backend automatically allows all origins in dev mode
- Check frontend is using correct backend URL
- Verify in `frontend/.env`: `VITE_API_URL=http://localhost:8000`

---

## 📊 Expected Performance

### Training (Step 3)
- **CPU:** 30-90 minutes
- **GPU:** 10-30 minutes
- **Expected Accuracy:** 60-70%

### Real-time Detection (Steps 4-7)
- **FPS:** 15-30 (depends on hardware)
- **Latency:** 30-60ms per frame
- **Faces:** Multiple face support

---

## 🎓 For Academic Presentation

### Key Points to Demonstrate

1. **CNN Training Process**
   - Show training curves
   - Explain architecture
   - Discuss hyperparameters

2. **Real-time Detection**
   - Run standalone webcam app
   - Explain face detection
   - Show preprocessing steps

3. **Full-stack Integration**
   - Demonstrate web interface
   - Show API requests in browser DevTools
   - Explain data flow

4. **Performance Analysis**
   - Show confusion matrix
   - Discuss challenging emotions
   - Explain accuracy metrics

---

## 📁 Project Structure Overview

```
EmotionalRecogFromFaceExp/
├── model/           # CNN training & evaluation
├── data/            # Dataset management
├── realtime/        # OpenCV real-time detection
├── backend/         # FastAPI REST API
├── frontend/        # React web interface
├── trained_models/  # Saved models
└── logs/            # Training logs
```

---

## 🌐 Web Application Workflow

```
User's Browser (React)
    ↓
📹 Webcam Capture
    ↓
🖼️ Frame to Base64
    ↓
📤 POST to Backend API
    ↓
🔍 Face Detection (Haar Cascade)
    ↓
🧠 CNN Emotion Prediction
    ↓
📨 JSON Response
    ↓
✨ Display Results
```

---

## ⚡ Performance Tips

### Speed Up Training
```bash
# Use GPU if available
# TensorFlow will automatically detect and use GPU
```

### Improve FPS
- Increase `captureInterval` in frontend (e.g., 1000ms)
- Reduce video resolution
- Use GPU for inference

### Reduce Latency
- Deploy backend and frontend on same machine
- Use smaller batch sizes
- Optimize MTCNN instead of Haar Cascade

---

## 🚀 Production Deployment

### Backend (FastAPI)
```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend (React)
```bash
cd frontend
npm run build
# Deploy dist/ folder to Netlify, Vercel, or any static hosting
```

---

## 📞 Support

### Common Issues
- Check individual README files in each directory
- Review error messages in console
- Ensure all dependencies are installed
- Verify dataset is downloaded

### Resources
- Model README: `model/README.md`
- Backend README: `backend/README.md`
- Frontend README: `frontend/README.md`
- Data README: `data/README.md`

---

## ✅ Verification Checklist

Before running the full application:

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and extracted to `data/`
- [ ] Model trained successfully (`.h5` file exists)
- [ ] Backend starts without errors
- [ ] Frontend dependencies installed (`npm install`)
- [ ] Frontend starts without errors
- [ ] Webcam permissions granted
- [ ] Both terminals running (backend + frontend)

---

## 🎉 Success!

If everything is working:
- Backend shows "✅ Service initialized successfully"
- Frontend shows "✅ Connected" badge
- Start detection button is enabled
- Webcam video appears
- Emotions are detected and displayed

**Congratulations! Your emotion recognition system is fully operational!** 🎭✨
