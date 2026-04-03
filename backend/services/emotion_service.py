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
try:
    from realtime.yunet_detector import YuNetFaceDetector
    YUNET_AVAILABLE = True
except ImportError:
    YUNET_AVAILABLE = False

# Default to the README's primary runtime path: YuNet DNN + CNN.
# MediaPipe + Random Forest remains an explicit alternative.
PIPELINE_MODE = os.getenv("EMOTION_PIPELINE", "yunet").strip().lower()
USE_MEDIAPIPE = PIPELINE_MODE in {"mediapipe", "rf", "random_forest"}
from model import config


class EmotionRecognitionService:
    """Service class for emotion recognition operations."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or config.MODEL_SAVE_PATH
        self.face_detector: Optional[Any] = None
        self.haar_detector: Optional[Any] = None   # backup only
        self.emotion_predictor: Optional[Any] = None
        self._emotion_ema = {}
        self._ema_decay = 0.35
        self._initialize()
    
    def _initialize(self):
        """Initialize face detector and emotion predictor."""
        print("Initializing Emotion Recognition Service...")
        
        if USE_MEDIAPIPE:
            try:
                from realtime.mediapipe_detector import MediaPipeFaceDetector, draw_rectangles_with_labels
                from realtime.multi_emotion_predictor import MultiEmotionPredictor

                self._draw_rectangles_with_labels = draw_rectangles_with_labels
                self.face_detector = MediaPipeFaceDetector(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_faces=10,
                )
                print("✅ MediaPipe Face detector initialized")

                rf_model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'trained_models', 'emotion_rf_model.pkl'
                )
                self.emotion_predictor = MultiEmotionPredictor(
                    model_path=rf_model_path,
                    model_type='random_forest',
                )
                print("✅ RF Emotion predictor initialized")
                return
            except Exception as e:
                print(f"⚠️ MediaPipe pipeline unavailable, falling back to YuNet+CNN: {e}")

        # ---- YuNet + CNN path (README primary runtime path) ----
        if YUNET_AVAILABLE:
            yunet = YuNetFaceDetector(score_threshold=0.6, max_faces=10)
            if yunet.is_available:
                self.face_detector = yunet
                print("✅ YuNet multi-face detector initialized (Primary)")
            else:
                self.face_detector = None

        if self.face_detector is None:
            # Final fallback: Haar
            self.face_detector = FaceDetector(
                method='haar', scale_factor=1.05, min_neighbors=5,
                min_size=(50, 50), max_faces=5
            )
            print("✅ Haar Face detector initialized (Fallback)")

        try:
            self.emotion_predictor = EmotionPredictor(model_path=self.model_path)
            print("✅ CNN Emotion predictor initialized")
        except FileNotFoundError as e:
            print(f"❌ Failed to load CNN model: {e}")
            self.emotion_predictor = None
    
    def is_ready(self) -> bool:
        """
        Check if service is ready to process requests.
        
        Returns:
            bool: True if service is ready
        """
        return (self.face_detector is not None and 
                self.emotion_predictor is not None and
                self.emotion_predictor.model is not None)

    def _require_components(self) -> Tuple[Any, Any]:
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

        # ── YuNet / Haar multi-face CNN path ────────────────────────────────
        if not USE_MEDIAPIPE:
            faces = face_detector.detect_faces(image)
            frame_h, frame_w = image.shape[:2]

            # Prune EMA trackers for faces no longer detected
            active_ids = set(range(len(faces)))
            self._emotion_ema = {k: v for k, v in self._emotion_ema.items() if k in active_ids}

            results = []
            annotated_image = image.copy()

            for face_id, (x, y, w, h) in enumerate(faces):
                face_bbox = (x, y, w, h)

                # Tight crop → CNN (matches FER-2013 48×48 training format)
                face_roi = face_detector.extract_face_roi(
                    image, face_bbox, target_size=(48, 48), grayscale=True
                )
                if face_roi is None or face_roi.size == 0:
                    continue

                _, _, all_probs = emotion_predictor.predict_emotion(face_roi)
                probs = np.asarray(all_probs, dtype=np.float32)

                # Per-face EMA smoothing
                if face_id not in self._emotion_ema:
                    self._emotion_ema[face_id] = probs
                else:
                    self._emotion_ema[face_id] = (
                        self._ema_decay * self._emotion_ema[face_id]
                        + (1.0 - self._ema_decay) * probs
                    )

                probs_sum = float(np.sum(self._emotion_ema[face_id]))
                smoothed_probs = (
                    self._emotion_ema[face_id] / probs_sum if probs_sum > 0 else probs
                )
                best_idx = int(np.argmax(smoothed_probs))
                confidence = float(smoothed_probs[best_idx])
                emotion = config.EMOTION_LABELS[best_idx]
                probabilities = {
                    label: float(prob)
                    for label, prob in zip(config.EMOTION_LABELS, smoothed_probs)
                }

                # Expanded bbox for display — CNN still used tight crop above
                disp_bbox = face_detector.get_display_bbox(face_bbox, frame_w, frame_h)
                results.append({
                    'bbox': list(disp_bbox),
                    'emotion': emotion,
                    'confidence': confidence,
                    'probabilities': probabilities,
                })
                annotated_image = emotion_predictor.draw_emotion_label(
                    annotated_image, disp_bbox, emotion, confidence, show_confidence=True
                )

            return results, annotated_image

        # MediaPipe processing
        faces_dict = face_detector.detect_faces(image)
        if not faces_dict:
            self._emotion_ema = {}
        
        results = []
        annotated_image = image.copy()
        face_predictions = {}
        
        for face_id, face_data in faces_dict.items():
            
            # Predict emotion
            emotion, confidence, all_probs = emotion_predictor._predict_rf(face_data.landmarks)

            probs = np.asarray(all_probs, dtype=np.float32)
            if face_id not in self._emotion_ema:
                self._emotion_ema[face_id] = probs
            else:
                self._emotion_ema[face_id] = self._ema_decay * self._emotion_ema[face_id] + (1.0 - self._ema_decay) * probs

            # Keep numeric stability for downstream dict serialization.
            probs_sum = float(np.sum(self._emotion_ema[face_id]))
            smoothed_probs = self._emotion_ema[face_id] / probs_sum if probs_sum > 0 else probs

            best_idx = int(np.argmax(smoothed_probs))
            confidence = float(smoothed_probs[best_idx])
            emotion = config.EMOTION_LABELS[best_idx]
            
            # Create probabilities dict
            probabilities = {
                label: float(prob)
                for label, prob in zip(config.EMOTION_LABELS, smoothed_probs)
            }
            
            # Add to results
            x, y, w, h = face_data.bbox
            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'emotion': emotion,
                'confidence': float(confidence),
                'probabilities': probabilities
            })
            
            face_predictions[face_id] = (emotion, confidence)
            
        draw_rectangles = getattr(self, '_draw_rectangles_with_labels', None)
        if draw_rectangles is None:
            from realtime.mediapipe_detector import draw_rectangles_with_labels as draw_rectangles
        annotated_image = draw_rectangles(annotated_image, faces_dict, face_predictions)
        
        return results, annotated_image
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if not self.is_ready():
            return None

        return {
            'model_path': str(self.model_path),
            'input_shape': [478, 3],
            'output_shape': [len(config.EMOTION_LABELS)],
            'num_classes': len(config.EMOTION_LABELS),
            'emotion_labels': config.EMOTION_LABELS,
            'total_parameters': 0
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


def reset_emotion_service():
    """Force-reset the singleton so the next call creates a fresh instance."""
    global _service_instance
    _service_instance = None


def get_emotion_service(model_path=None) -> EmotionRecognitionService:
    """
    Get or create the global emotion recognition service instance.
    Always creates a fresh instance to ensure the latest model is loaded.
    """
    global _service_instance
    _service_instance = EmotionRecognitionService(model_path=model_path)
    return _service_instance
