"""
Emotion Recognition Service

This module provides business logic for emotion prediction,
including face detection and CNN inference.
"""

import os
import sys
import numpy as np
import cv2
import base64
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.detector import FaceDetector
from realtime.emotion_predictor import EmotionPredictor
from model import config


class EmotionRecognitionService:
    """
    Service class for emotion recognition operations.
    
    This class encapsulates the face detection and emotion prediction
    logic for use by the API endpoints.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognition service.
        
        Args:
            model_path (str): Path to trained model
        """
        self.model_path = model_path or config.MODEL_SAVE_PATH
        
        # Initialize components
        self.face_detector: Optional[FaceDetector] = None
        self.emotion_predictor: Optional[EmotionPredictor] = None
        self._emotion_ema: Optional[np.ndarray] = None
        self._ema_decay = 0.35
        
        self._initialize()
    
    def _initialize(self):
        """Initialize face detector and emotion predictor."""
        print("Initializing Emotion Recognition Service...")
        
        # Initialize face detector
        self.face_detector = FaceDetector(
            method='haar',
            scale_factor=1.05,
            min_neighbors=5,
            min_size=(50, 50),
            max_faces=1
        )
        print("✅ Face detector initialized")
        
        # Initialize emotion predictor
        try:
            self.emotion_predictor = EmotionPredictor(model_path=self.model_path)
            print("✅ Emotion predictor initialized")
        except FileNotFoundError as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """
        Check if service is ready to process requests.
        
        Returns:
            bool: True if service is ready
        """
        return (self.face_detector is not None and 
                self.emotion_predictor is not None and
                self.emotion_predictor.model is not None)

    def _require_components(self) -> Tuple[FaceDetector, EmotionPredictor]:
        """Return initialized components or raise a clear error."""
        if self.face_detector is None or self.emotion_predictor is None:
            raise RuntimeError("Emotion service not initialized")
        return self.face_detector, self.emotion_predictor
    
    def decode_image(self, image_data: bytes) -> np.ndarray:
        """
        Decode image from bytes to OpenCV format.
        
        Args:
            image_data (bytes): Image data in bytes
            
        Returns:
            numpy.ndarray: Decoded image
            
        Raises:
            ValueError: If image cannot be decoded
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image. Invalid format.")
        
        return image
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Decode base64 encoded image.
        
        Args:
            base64_string (str): Base64 encoded image string
            
        Returns:
            numpy.ndarray: Decoded image
        """
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        return self.decode_image(image_bytes)
    
    def predict_emotions(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect faces and predict emotions in an image.
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            tuple: (list of face results, annotated image)
                   Each face result is a dict with:
                   - bbox: [x, y, w, h]
                   - emotion: str
                   - confidence: float
                   - probabilities: dict of {emotion: probability}
        """
        face_detector, emotion_predictor = self._require_components()

        # Detect faces
        faces = face_detector.detect_faces(image)
        if not faces:
            self._emotion_ema = None
        
        results = []
        annotated_image = image.copy()
        
        for face_bbox in faces:
            # Extract face ROI
            face_roi = face_detector.extract_face_roi(
                image, face_bbox,
                target_size=(48, 48),
                grayscale=True
            )
            
            # Predict emotion
            _, _, all_probs = emotion_predictor.predict_emotion(face_roi)

            probs = np.asarray(all_probs, dtype=np.float32)
            if self._emotion_ema is None:
                self._emotion_ema = probs
            else:
                self._emotion_ema = self._ema_decay * self._emotion_ema + (1.0 - self._ema_decay) * probs

            # Keep numeric stability for downstream dict serialization.
            probs_sum = float(np.sum(self._emotion_ema))
            smoothed_probs = self._emotion_ema / probs_sum if probs_sum > 0 else probs

            best_idx = int(np.argmax(smoothed_probs))
            confidence = float(smoothed_probs[best_idx])
            emotion = config.EMOTION_LABELS[best_idx]
            
            # Create probabilities dict
            probabilities = {
                label: float(prob)
                for label, prob in zip(config.EMOTION_LABELS, smoothed_probs)
            }
            
            # Add to results
            x, y, w, h = face_bbox
            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'emotion': emotion,
                'confidence': float(confidence),
                'probabilities': probabilities
            })
            
            # Draw on image
            annotated_image = emotion_predictor.draw_emotion_label(
                annotated_image, face_bbox, emotion, confidence,
                show_confidence=True
            )
        
        return results, annotated_image
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if not self.is_ready():
            return None

        _, emotion_predictor = self._require_components()
        info = emotion_predictor.get_model_info()
        if info is None:
            return None
        
        # Convert to JSON-serializable format
        return {
            'model_path': str(info['model_path']),
            'input_shape': [int(x) if x is not None else None for x in info['input_shape']],
            'output_shape': [int(x) if x is not None else None for x in info['output_shape']],
            'num_classes': int(info['num_classes']),
            'emotion_labels': info['emotion_labels'],
            'total_parameters': int(info['total_parameters'])
        }
    
    def encode_image(self, image: np.ndarray, format: str = '.jpg') -> bytes:
        """
        Encode image to bytes.
        
        Args:
            image (numpy.ndarray): Image to encode
            format (str): Image format ('.jpg' or '.png')
            
        Returns:
            bytes: Encoded image data
        """
        success, encoded = cv2.imencode(format, image)
        
        if not success:
            raise ValueError("Failed to encode image")
        
        return encoded.tobytes()
    
    def encode_image_base64(self, image: np.ndarray, format: str = '.jpg') -> str:
        """
        Encode image to base64 string.
        
        Args:
            image (numpy.ndarray): Image to encode
            format (str): Image format
            
        Returns:
            str: Base64 encoded image string
        """
        image_bytes = self.encode_image(image, format)
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        # Add data URL prefix
        mime_type = 'image/jpeg' if format == '.jpg' else 'image/png'
        return f"data:{mime_type};base64,{base64_string}"


# Global service instance (singleton pattern)
_service_instance = None


def get_emotion_service(model_path=None) -> EmotionRecognitionService:
    """
    Get or create the global emotion recognition service instance.
    
    Args:
        model_path (str): Path to model (only used on first call)
        
    Returns:
        EmotionRecognitionService: Service instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = EmotionRecognitionService(model_path=model_path)
    
    return _service_instance
