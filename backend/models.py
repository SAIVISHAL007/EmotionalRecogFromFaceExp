"""
Pydantic Models for API Request/Response Validation

These models define the structure of data exchanged between
the frontend and backend API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class EmotionPrediction(BaseModel):
    """
    Single emotion prediction result.
    """
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "emotion": "happy",
                "confidence": 0.87
            }
        }


class EmotionProbabilities(BaseModel):
    """
    Probabilities for all emotion classes.
    """
    angry: float = Field(..., ge=0.0, le=1.0)
    disgust: float = Field(..., ge=0.0, le=1.0)
    fear: float = Field(..., ge=0.0, le=1.0)
    happy: float = Field(..., ge=0.0, le=1.0)
    sad: float = Field(..., ge=0.0, le=1.0)
    surprise: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "angry": 0.05,
                "disgust": 0.02,
                "fear": 0.08,
                "happy": 0.73,
                "sad": 0.03,
                "surprise": 0.06,
                "neutral": 0.03
            }
        }


class FaceDetectionResult(BaseModel):
    """
    Result for a single detected face with emotion prediction.
    """
    bbox: List[int] = Field(..., description="Bounding box [x, y, width, height]")
    emotion: str = Field(..., description="Predicted emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: EmotionProbabilities = Field(..., description="All emotion probabilities")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """
    Complete response for emotion prediction request.
    """
    success: bool = Field(..., description="Whether prediction was successful")
    num_faces: int = Field(..., ge=0, description="Number of faces detected")
    faces: List[FaceDetectionResult] = Field(default=[], description="List of detected faces with emotions")
    processing_time: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Server timestamp")
    message: Optional[str] = Field(None, description="Optional message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "num_faces": 1,
                "faces": [{
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
                }],
                "processing_time": 45.2,
                "timestamp": "2026-02-11T10:30:00",
                "message": "Emotion predicted successfully"
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response model.
    """
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Server timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format",
                "detail": "Image must be in JPEG or PNG format",
                "timestamp": "2026-02-11T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_info: Optional[Dict] = Field(None, description="Model information")
    timestamp: str = Field(..., description="Server timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_info": {
                    "num_classes": 7,
                    "emotions": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
                },
                "timestamp": "2026-02-11T10:30:00"
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Model information response.
    """
    model_path: str
    input_shape: List[Optional[int]]
    output_shape: List[Optional[int]]
    num_classes: int
    emotion_labels: List[str]
    total_parameters: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "trained_models/emotion_cnn_model.h5",
                "input_shape": [None, 48, 48, 1],
                "output_shape": [None, 7],
                "num_classes": 7,
                "emotion_labels": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
                "total_parameters": 1234567
            }
        }
