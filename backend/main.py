"""
FastAPI Backend for Facial Emotion Recognition

This backend provides REST API endpoints for emotion prediction from images.
It integrates the trained CNN model with face detection for real-time inference.

API Endpoints:
    - POST /api/predict - Predict emotions from uploaded image
    - POST /api/predict-base64 - Predict from base64 encoded image
    - GET /api/health - Health check
    - GET /api/model-info - Get model information
    - GET / - API documentation

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Academic Focus:
    This demonstrates how the trained CNN model can be deployed
    as a web service for practical applications.
"""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, cast

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import (
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    EmotionProbabilities,
    FaceDetectionResult
)
from backend.services.emotion_service import get_emotion_service


# ==================
# FastAPI Application
# ==================

app = FastAPI(
    title="Facial Emotion Recognition API",
    description="Real-time emotion recognition from facial images using CNN",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ==================
# CORS Middleware
# ==================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================
# Global Service
# ==================

emotion_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize the emotion recognition service on startup."""
    global emotion_service
    
    print("\n" + "="*70)
    print("FACIAL EMOTION RECOGNITION API")
    print("="*70)
    
    try:
        emotion_service = get_emotion_service()
        
        if emotion_service.is_ready():
            print("\n✅ Service initialized successfully")
            loaded_info = emotion_service.get_model_info()
            if loaded_info is not None:
                print(f"   Model: {loaded_info['num_classes']} emotions")
                print(f"   Emotions: {', '.join(loaded_info['emotion_labels'])}")
        else:
            print("\n⚠️  Service initialized but not ready")
    
    except Exception as e:
        print(f"\n❌ Failed to initialize service: {e}")
        print("   Make sure the model is trained: python model/train.py")
    
    print("="*70 + "\n")


# ==================
# API Endpoints
# ==================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Recognition API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }
            h1 { margin-top: 0; }
            .endpoint {
                background: rgba(255,255,255,0.2);
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }
            .method {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
                margin-right: 10px;
            }
            .get { background: #61affe; }
            .post { background: #49cc90; }
            a { color: #ffd700; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Facial Emotion Recognition API</h1>
            <p>Real-time emotion detection powered by CNN deep learning</p>
            
            <h2>📚 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/predict</strong><br>
                <small>Upload an image file for emotion prediction</small>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/predict-base64</strong><br>
                <small>Send base64 encoded image for prediction</small>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/health</strong><br>
                <small>Check service health status</small>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/model-info</strong><br>
                <small>Get information about the CNN model</small>
            </div>
            
            <h2>📖 Documentation</h2>
            <p>
                <a href="/docs" target="_blank">📄 Swagger UI Documentation</a><br>
                <a href="/redoc" target="_blank">📘 ReDoc Documentation</a>
            </p>
            
            <h2>🎓 Academic Project</h2>
            <p>Neural Networks & Deep Learning<br>Real-Time Facial Emotion Recognition System</p>
            
            <h2>🚀 Emotions Detected</h2>
            <p>😠 Angry • 🤢 Disgust • 😨 Fear • 😊 Happy • 😢 Sad • 😲 Surprise • 😐 Neutral</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and model information.
    """
    if emotion_service is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_info=None,
            timestamp=datetime.now().isoformat()
        )
    
    is_ready = emotion_service.is_ready()
    model_info = None
    
    if is_ready:
        full_info = emotion_service.get_model_info()
        if full_info is None:
            raise HTTPException(status_code=503, detail="Model info unavailable")
        model_info = {
            "num_classes": full_info['num_classes'],
            "emotions": full_info['emotion_labels']
        }
    
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_info=model_info,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get detailed model information.
    
    Returns:
        ModelInfoResponse: Detailed model architecture and configuration
    """
    if emotion_service is None or not emotion_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Model not loaded."
        )
    
    info = emotion_service.get_model_info()
    if info is None:
        raise HTTPException(
            status_code=503,
            detail="Model info unavailable"
        )
    typed_info = cast(Dict[str, Any], info)
    return ModelInfoResponse(**typed_info)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotions from an uploaded image file.
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        PredictionResponse: Emotion predictions for all detected faces
    """
    start_time = time.time()
    
    # Validate service
    if emotion_service is None or not emotion_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Model not loaded."
        )
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Use JPEG or PNG."
        )
    
    try:
        # Read and decode image
        image_data = await file.read()
        image = emotion_service.decode_image(image_data)
        
        # Predict emotions
        results, annotated_image = emotion_service.predict_emotions(image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create response
        faces = []
        for result in results:
            faces.append(FaceDetectionResult(
                bbox=result['bbox'],
                emotion=result['emotion'],
                confidence=result['confidence'],
                probabilities=EmotionProbabilities(**result['probabilities'])
            ))
        
        return PredictionResponse(
            success=True,
            num_faces=len(faces),
            faces=faces,
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            message=f"Successfully detected {len(faces)} face(s)" if faces else "No faces detected"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict-base64", response_model=PredictionResponse)
async def predict_emotion_base64(request: Request):
    """
    Predict emotions from a base64 encoded image.
    
    Expects JSON body with 'image' field containing base64 string.
    
    Returns:
        PredictionResponse: Emotion predictions
    """
    start_time = time.time()
    
    # Validate service
    if emotion_service is None or not emotion_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Model not loaded."
        )
    
    try:
        # Parse request body
        body = cast(Dict[str, Any], await request.json())
        
        if 'image' not in body:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image' field in request body"
            )
        
        base64_string = body['image']
        
        # Decode image
        image = emotion_service.decode_base64_image(base64_string)
        
        # Predict emotions
        results, annotated_image = emotion_service.predict_emotions(image)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create response
        faces = []
        for result in results:
            faces.append(FaceDetectionResult(
                bbox=result['bbox'],
                emotion=result['emotion'],
                confidence=result['confidence'],
                probabilities=EmotionProbabilities(**result['probabilities'])
            ))
        
        return PredictionResponse(
            success=True,
            num_faces=len(faces),
            faces=faces,
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat(),
            message=f"Successfully detected {len(faces)} face(s)" if faces else "No faces detected"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ==================
# Error Handlers
# ==================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "detail": f"The endpoint {request.url.path} does not exist",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================
# Run Application
# ==================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING EMOTION RECOGNITION API SERVER")
    print("="*70)
    print("\nServer will be available at:")
    print("  → http://localhost:8000")
    print("  → http://localhost:8000/docs (Swagger UI)")
    print("  → http://localhost:8000/redoc (ReDoc)")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
