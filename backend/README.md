# FastAPI Backend for Emotion Recognition

This backend provides REST API endpoints for real-time facial emotion recognition.

## Overview

The backend integrates:
- Trained CNN model
- Face detection (YuNet DNN primary, Haar fallback)
- Optional MediaPipe + Random Forest alternate path
- FastAPI web framework
- RESTful API endpoints
- CORS support for frontend integration

The runtime defaults to the YuNet + CNN pipeline. Set `EMOTION_PIPELINE=mediapipe` only if you want the alternate landmark-based path.

## Architecture

```
Client Request → FastAPI → YuNet Face Detector → CNN Model → Emotion Prediction → JSON Response
```

## API Endpoints

### 1. Health Check
```
GET /api/health
```
Returns service status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "num_classes": 7,
    "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
  },
  "timestamp": "2026-02-11T10:30:00"
}
```

### 2. Model Information
```
GET /api/model-info
```
Get detailed CNN model information.

**Response:**
```json
{
  "model_path": "trained_models/emotion_cnn_model.h5",
  "input_shape": [null, 48, 48, 1],
  "output_shape": [null, 7],
  "num_classes": 7,
  "emotion_labels": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
  "total_parameters": 1234567
}
```

### 3. Predict from File Upload
```
POST /api/predict
Content-Type: multipart/form-data
```

**Request:**
- `file`: Image file (JPEG/PNG)

**Response:**
```json
{
  "success": true,
  "num_faces": 1,
  "faces": [
    {
      "bbox": [100, 150, 200, 200],
      "emotion": "happy",
      "confidence": 0.87,
      "probabilities": {
        "angry": 0.05,
        "disgust": 0.02,
        "fear": 0.08,
        "happy": 0.73,
        "sad": 0.03,
        "surprise": 0.06,
        "neutral": 0.03
      }
    }
  ],
  "processing_time": 45.2,
  "timestamp": "2026-02-11T10:30:00",
  "message": "Successfully detected 1 face(s)"
}
```

### 4. Predict from Base64
```
POST /api/predict-base64
Content-Type: application/json
```

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:** Same as `/api/predict`

## Installation

### Prerequisites
```bash
# Ensure main dependencies are installed
pip install -r requirements.txt
```

### Backend-Specific Dependencies
```bash
cd backend
pip install -r requirements.txt
```

## Running the Server

### Development Mode
```bash
# From project root
python backend/main.py
```

Or using uvicorn directly:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Server URLs

- **API Root**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing the API

### Using cURL

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Predict from File:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@path/to/image.jpg"
```

### Using Python

```python
import requests

# Health check
response = requests.get('http://localhost:8000/api/health')
print(response.json())

# Predict from file
files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/api/predict', files=files)
print(response.json())
```

### Using JavaScript (Fetch API)

```javascript
// Predict from base64
fetch('http://localhost:8000/api/predict-base64', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    image: base64ImageString
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Project Structure

```
backend/
├── main.py                  # FastAPI application
├── models.py                # Pydantic models (request/response)
├── services/
│   ├── __init__.py
│   └── emotion_service.py   # Business logic
└── requirements.txt         # Backend dependencies
```

Legacy and optional runtime helpers live in `realtime/`:

- `realtime/yunet_detector.py` for the primary detector
- `realtime/detector.py` for Haar fallback
- `realtime/mediapipe_detector.py` and `realtime/multi_emotion_predictor.py` for the optional alternate pipeline

## CORS Configuration

CORS is configured to allow all origins for development:
```python
allow_origins=["*"]
```

**For production**, update to specific origins:
```python
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

## Error Handling

The API returns consistent error responses:

```json
{
  "success": false,
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2026-02-11T10:30:00"
}
```

**Common Error Codes:**
- `400` - Bad Request (invalid image, missing fields)
- `404` - Not Found (invalid endpoint)
- `500` - Internal Server Error
- `503` - Service Unavailable (model not loaded)

## Performance

- **Average Response Time**: 30-60ms per image (CPU)
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Max Image Size**: Recommended < 5MB
- **Supported Formats**: JPEG, PNG

## Security Considerations

**For Production:**
1. Add authentication (JWT tokens)
2. Rate limiting
3. Input validation
4. HTTPS only
5. Restrict CORS origins
6. File size limits
7. Content type validation

## Academic Discussion Points

For your viva/presentation:

1. **Why FastAPI?**
   - Modern, fast Python web framework
   - Automatic API documentation
   - Type validation with Pydantic
   - Async support

2. **RESTful Design**
   - Clear endpoint structure
   - Standard HTTP methods
   - JSON responses
   - Status codes

3. **Service Architecture**
   - Separation of concerns
   - Business logic in service layer
   - API layer handles HTTP
   - Singleton pattern for model loading

4. **Real-world Deployment**
   - Load balancing
   - Caching predictions
   - GPU acceleration
   - Containerization (Docker)

## Integration with Frontend

The React frontend will:
1. Capture webcam frames
2. Convert to base64
3. POST to `/api/predict-base64`
4. Display results in real-time

## Troubleshooting

**Model not loading?**
- Ensure model is trained: `python model/train.py`
- Check path in config.py

**Port already in use?**
- Change port: `--port 8001`
- Kill existing process

**CORS errors?**
- Check allow_origins configuration
- Ensure frontend uses correct backend URL

**Slow predictions?**
- Use GPU if available
- Reduce image size
- Increase scale_factor in detector

## Next Steps

After backend is running:
1. Test with Swagger UI (`/docs`)
2. Implement React frontend
3. Integrate webcam capture
4. Deploy full-stack application
