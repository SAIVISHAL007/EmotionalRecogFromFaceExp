# 🎯 Real-Time Detection Module

This module handles face detection and emotion prediction for the live webcam pipeline.

---

## Detectors

### YuNet DNN Detector (`yunet_detector.py`) ⭐ Primary

OpenCV's built-in `cv2.FaceDetectorYN` powered by a ONNX neural network.

| Feature | Value |
|---------|-------|
| Max faces | Up to 10 simultaneously |
| Bounding box | Full face (forehead to chin) |
| Python support | 3.8 — 3.13 ✅ |
| Model file | `trained_models/face_detection_yunet.onnx` |

**Key design**: The detector returns a **tight inference ROI** (fed directly to CNN) and a separately computed **expanded display bbox** (shown on screen). This ensures the CNN receives the same tight crop it was trained on while the user sees a proper full-face rectangle.

```python
from realtime.yunet_detector import YuNetFaceDetector

detector = YuNetFaceDetector(score_threshold=0.6, max_faces=10)
faces = detector.detect_faces(frame)  # List of (x, y, w, h)

for bbox in faces:
    roi = detector.extract_face_roi(frame, bbox)          # tight → CNN
    disp = detector.get_display_bbox(bbox, fw, fh)        # expanded → draw
```

### Haar Cascade Detector (`detector.py`) — Fallback

OpenCV Haar Cascade — used only when YuNet ONNX is unavailable.

- Detects the eye/nose region (not full face)
- Padding applied for display bbox, tight crop for CNN
- `max_faces=5` by default

### MediaPipe Face Mesh (`mediapipe_detector.py`) — RF Pipeline

Optional alternate pipeline used with the Random Forest classifier.

- Uses MediaPipe Face Mesh landmarks and a lightweight feature extractor
- Returns 478 3D landmarks per face
- Connected to `multi_emotion_predictor.py` for landmark-based RF inference

---

## Predictors

### CNN Predictor (`emotion_predictor.py`)

Wraps the TensorFlow CNN model for single-face inference.

```python
from realtime.emotion_predictor import EmotionPredictor

predictor = EmotionPredictor()
emotion, confidence, all_probs = predictor.predict_emotion(face_roi)
# emotion: str, confidence: float, all_probs: ndarray(7,)
```

**Important**: No probability calibration is applied — raw softmax output is returned directly. The `_ema_decay` smoothing in `emotion_service.py` stabilises predictions across frames.

### Random Forest Predictor (`multi_emotion_predictor.py`)

Used with MediaPipe landmarks in the optional alternate path.

```python
from realtime.multi_emotion_predictor import MultiEmotionPredictor

predictor = MultiEmotionPredictor(model_path='trained_models/emotion_rf_model.pkl')
emotion, confidence, probs = predictor._predict_rf(landmarks)  # landmarks: (478, 3)
```

---

## Standalone Webcam App (`webcam_app.py`)

Legacy standalone webcam app. Run without the web interface:

```bash
python realtime/webcam_app.py
```

**Controls:**
- `q` — Quit
- `s` — Save screenshot
- `p` — Pause/Resume
- `f` — Toggle FPS display

---

## Detection Flow

```
Frame (BGR)
    │
    ▼
YuNetFaceDetector.detect_faces()
    │
    ├── For each face bbox:
    │       │
    │       ├── extract_face_roi()  →  48×48 grayscale crop  →  CNN
    │       └── get_display_bbox()  →  expanded box          →  draw on frame
    │
    ▼
EmotionPredictor.predict_emotion(roi)
    │
    ▼
Per-face EMA smoothing  →  Stable emotion label + probabilities
```
