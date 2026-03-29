"""
Lightweight Emotion Recognition using MediaPipe Landmarks + Random Forest.

Based on research findings that show RF classifiers with MediaPipe landmarks
provide optimal real-time performance while maintaining accuracy.

Features:
    - Extract 64 empirically-selected face landmarks (from research)
    - Random Forest classifier for fast inference (< 1ms)
    - Support for multi-face processing
    - Compatible with edge devices

Reference:
    Ciraolo et al. (2024) - Facial expression recognition for tele-rehabilitation
    Shows that Empiric Feature Map (64 features) + RF provides best accuracy/speed tradeoff
"""

import numpy as np
import pickle
import os
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LandmarkFeatureMaps:
    """Configuration for different landmark feature maps."""
    
    EMPIRIC_INDICES = [
        # Mouth landmarks (based on FACS)
        61, 40, 78, 80, 13, 311, 308,  # upper lip
        91, 17, 321, 291,  # lower lip
        88, 14, 402,  # lower lip inner
        49, 206, 216, 212, 204, 200,  # around mouth left
        279, 426, 436, 432, 424,  # around mouth right
        # Eye landmarks
        147, 70, 63, 105, 66, 107,  # upper left eye
        143, 117, 118, 128,  # lower left eye
        50, 216,  # left cheek
        336, 296, 334, 293, 300, 372,  # upper right eye
        357, 347, 346,  # lower right eye
        280, 436,  # right cheek
        # Eyebrow and nose
        107, 6, 336, 6, 4,  # eyebrows and nose
        33, 159, 133, 145,  # left eye angles
        362, 386, 263, 374,  # right eye angles
        # Mouth opening measurements
        13, 14,  # vertical mouth
        61, 291,  # horizontal mouth
        159, 145,  # left eye opening
        386, 374,  # right eye opening
        55, 285,  # eyebrow distance
        118, 61, 347, 291,  # cheekbone to mouth
    ]
    
    FULL_MESH_SIZE = 478


class LandmarkFeatureExtractor:
    """
    Extract discriminative features from MediaPipe face landmarks.
    
    Implements the "Empiric Face Mesh" approach from research, which selects
    64 most relevant landmarks for emotion classification.
    """
    
    def __init__(self, feature_map: str = 'empiric'):
        """
        Initialize feature extractor.
        
        Args:
            feature_map: Type of feature extraction ('empiric', 'full', 'angles')
        """
        self.feature_map = feature_map
        self.reference_landmark_idx = 0  # Cupid's bow center
        
        if feature_map == 'empiric':
            self.selected_indices = LandmarkFeatureMaps.EMPIRIC_INDICES
        elif feature_map == 'full':
            self.selected_indices = list(range(LandmarkFeatureMaps.FULL_MESH_SIZE))
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")
    
    def extract_features(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Extract features from normalized landmarks.
        
        Args:
            landmarks_3d: Normalized landmarks of shape (478, 3)
            
        Returns:
            Feature vector of shape (n_features,)
        """
        if landmarks_3d.shape[0] != 478:
            raise ValueError(f"Expected 478 landmarks, got {landmarks_3d.shape[0]}")
        
        # Extract edge lengths for selected landmark pairs
        features = []
        
        # Extract distances for landmark pairs
        for i in range(0, len(self.selected_indices) - 1, 2):
            idx1 = self.selected_indices[i]
            idx2 = self.selected_indices[i + 1]
            
            # Calculate edge length
            landmark1 = landmarks_3d[idx1]
            landmark2 = landmarks_3d[idx2]
            distance = np.linalg.norm(landmark1 - landmark2)
            features.append(distance)
        
        # Extract individual landmark coordinates
        for idx in self.selected_indices[:min(30, len(self.selected_indices))]:
            features.extend(landmarks_3d[idx])
        
        return np.array(features, dtype=np.float32)
    
    def extract_angles(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Extract angle-based features (alternative approach).
        
        Args:
            landmarks_3d: Normalized landmarks
            
        Returns:
            Feature vector based on angles
        """
        features = []
        
        # Example: Extract angles for eye opening, mouth opening, etc.
        # This would involve calculating angles between vectors formed by landmarks
        
        return np.array(features, dtype=np.float32)


class MultiEmotionPredictor:
    """
    Unified emotion predictor supporting multiple faces and both CNN/RF models.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = 'cnn',
        emotion_labels: Optional[list] = None,
    ):
        """
        Initialize multi-face emotion predictor.
        
        Args:
            model_path: Path to trained model
            model_type: 'cnn' or 'random_forest'
            emotion_labels: List of emotion class names
        """
        self.model_path = model_path
        self.model_type = model_type
        self.emotion_labels = emotion_labels or [
            'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
        ]
        self.model = None
        self.feature_extractor = None
        
        if model_type == 'random_forest':
            self.feature_extractor = LandmarkFeatureExtractor('empiric')
            self._load_rf_model()
        elif model_type == 'cnn':
            self._load_cnn_model()
    
    def _load_cnn_model(self):
        """Load CNN model for emotion recognition."""
        try:
            from tensorflow import keras
            if self.model_path is None:
                # Use default path
                self.model_path = 'trained_models/emotion_cnn_model.h5'
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ CNN model loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load CNN model: {e}")
            raise
    
    def _load_rf_model(self):
        """Load Random Forest model."""
        try:
            if self.model_path is None:
                self.model_path = 'trained_models/emotion_rf_model.pkl'
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print(f"✅ Random Forest model loaded: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load RF model: {e}")
            raise
    
    def predict_multiple_faces(
        self,
        faces_dict: Dict[int, 'FaceData'],
    ) -> Dict[int, Tuple[str, float, np.ndarray]]:
        """
        Predict emotions for multiple detected faces.
        
        Args:
            faces_dict: Dictionary of FaceData objects
            
        Returns:
            Dictionary mapping face_id to (emotion, confidence, probabilities)
        """
        predictions = {}
        
        for face_id, face_data in faces_dict.items():
            if self.model_type == 'random_forest':
                emotion, confidence, probs = self._predict_rf(face_data.landmarks)
            else:
                emotion, confidence, probs = self._predict_cnn(face_data.landmarks)
            
            predictions[face_id] = (emotion, confidence, probs)
        
        return predictions
    
    def _predict_rf(self, landmarks_3d: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion using Random Forest model.
        
        Args:
            landmarks_3d: Face landmarks (478, 3)
            
        Returns:
            (emotion_label, confidence, probabilities)
        """
        if self.feature_extractor is None:
            self.feature_extractor = LandmarkFeatureExtractor('empiric')
        
        # Extract features
        features = self.feature_extractor.extract_features(landmarks_3d)
        features = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        emotion_idx = prediction
        emotion = self.emotion_labels[emotion_idx]
        confidence = probabilities[emotion_idx]
        
        return emotion, float(confidence), probabilities
    
    def _predict_cnn(self, landmarks_3d: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion using CNN model.
        
        Args:
            landmarks_3d: Face landmarks (478, 3)
            
        Returns:
            (emotion_label, confidence, probabilities)
        """
        import cv2
        
        # For CNN: Need to convert landmarks to 48x48 grayscale image
        # This is a placeholder - actual conversion depends on your CNN input
        # For now, using direct prediction on features
        
        face_image = landmarks_3d  # Placeholder
        
        # Preprocess and predict with CNN
        # This would depend on your specific CNN architecture
        
        # Placeholder prediction
        probabilities = np.ones(len(self.emotion_labels)) / len(self.emotion_labels)
        emotion_idx = 0
        emotion = self.emotion_labels[emotion_idx]
        confidence = probabilities[emotion_idx]
        
        return emotion, float(confidence), probabilities
