# Real-Time Emotion Detection

This module provides real-time facial emotion recognition using webcam video streams.

## Components

### 1. Face Detector (`detector.py`)
- **Method**: Haar Cascade Classifier (default)
- **Alternative**: MTCNN (optional, better accuracy)
- **Features**:
  - Face detection in video frames
  - Bounding box extraction
  - ROI (Region of Interest) extraction
  - Configurable detection parameters

### 2. Emotion Predictor (`emotion_predictor.py`)
- Wraps the trained CNN model
- Preprocesses face images
- Predicts emotions with confidence scores
- Provides top-k predictions
- Color-coded emotion visualization

### 3. Webcam Application (`webcam_app.py`)
- Complete standalone application
- Real-time video processing
- Interactive controls
- FPS monitoring
- Statistics tracking

## Quick Start

### Prerequisites
Ensure you have trained the model:
```bash
python model/train.py
```

### Run Webcam Application
```bash
python realtime/webcam_app.py
```

### Test Components Individually

**Test Face Detector:**
```bash
python realtime/detector.py
```

**Test Emotion Predictor:**
```bash
python realtime/emotion_predictor.py
```

## Webcam Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame |
| `p` | Pause/Resume |
| `f` | Toggle FPS display |
| `d` | Toggle debug mode (show top-3 predictions) |

## How It Works

```
Video Frame → Face Detection → Face Extraction → 
Preprocessing → CNN Prediction → Emotion Label + Confidence
```

### Step-by-Step Process

1. **Frame Capture**: Get frame from webcam
2. **Face Detection**: Detect faces using Haar Cascade
3. **ROI Extraction**: Extract 48x48 grayscale face region
4. **Preprocessing**: Normalize pixel values [0,1]
5. **CNN Inference**: Predict emotion probabilities
6. **Visualization**: Draw bounding box and emotion label
7. **Display**: Show annotated frame

## Technical Details

### Face Detection (Haar Cascade)
- **Algorithm**: Violin-Jones object detection
- **Scale Factor**: 1.1 (how much image size is reduced at each scale)
- **Min Neighbors**: 5 (quality threshold)
- **Min Size**: 30x30 pixels (minimum face size)

### Preprocessing Pipeline
```python
Face ROI → Grayscale → Resize (48x48) → 
Normalize (/255) → Reshape (1,48,48,1)
```

### Emotion Colors
- 🔴 **Angry**: Red
- 🟢 **Disgust**: Dark Green
- 🟣 **Fear**: Purple
- 🟡 **Happy**: Yellow
- 🔵 **Sad**: Blue
- 🟠 **Surprise**: Orange
- ⚪ **Neutral**: Gray

## Performance Considerations

### FPS Optimization
- **Expected FPS**: 15-30 (depending on hardware)
- **Bottleneck**: CNN inference time (~30-50ms per face)
- **Optimization**: Use GPU acceleration if available

### Multiple Faces
The system can detect and predict emotions for multiple faces simultaneously.

## Advanced Usage

### Custom Camera
```bash
python realtime/webcam_app.py 1  # Use camera ID 1
```

### Custom Model
```bash
python realtime/webcam_app.py 0 path/to/custom_model.h5
```

### Using MTCNN (Better Accuracy)

Install MTCNN:
```bash
pip install mtcnn
```

Modify `webcam_app.py`:
```python
self.face_detector = FaceDetector(method='mtcnn')
```

## Troubleshooting

**Camera Not Opening?**
- Check camera permissions
- Try different camera ID: `python realtime/webcam_app.py 1`
- Verify camera with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

**Low FPS?**
- Increase `scale_factor` in detector (trades accuracy for speed)
- Reduce video resolution
- Use GPU for model inference

**No Faces Detected?**
- Ensure good lighting
- Face camera directly
- Adjust `min_neighbors` parameter (lower = more detections)
- Check `scale_factor` (lower = more sensitive)

**Model Not Found?**
- Train model first: `python model/train.py`
- Check path: `trained_models/emotion_cnn_model.h5`

## Academic Discussion Points

For your viva/presentation:

1. **Why Haar Cascade?**
   - Fast, real-time performance
   - Pre-trained, no additional training needed
   - Trade-off: accuracy vs speed

2. **Real-time Challenges**
   - Frame rate vs accuracy
   - Multiple face handling
   - Lighting variations
   - Pose variations

3. **Preprocessing Importance**
   - Consistent input format
   - Normalization for model stability
   - Grayscale reduces computational load

4. **Confidence Thresholding**
   - Low confidence predictions
   - When to show "uncertain"?
   - User experience considerations

## Output

The application provides:
- Real-time emotion labels on video
- Confidence scores
- FPS counter
- Session statistics
- Saved screenshots (optional)

## Integration

This module can be integrated with:
- Backend API (next step)
- Web interface via video streaming
- Mobile applications
- IoT devices with cameras

## Next Steps

After testing real-time detection:
1. Implement backend API: `python backend/main.py`
2. Create web interface with React
3. Full-stack integration
