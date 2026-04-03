"""
Emotion Prediction Module

This module wraps the trained CNN model for real-time emotion prediction.
It loads the model and provides an interface for predicting emotions from face images.

Usage:
    predictor = EmotionPredictor(model_path='trained_models/emotion_cnn_model.h5')
    emotion, confidence = predictor.predict_emotion(face_image)
"""

import os
import sys
import numpy as np
import cv2
from tensorflow import keras
from typing import Any, cast

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import config


class EmotionPredictor:
    """
    Emotion prediction wrapper for the trained CNN model.
    
    This class loads a trained model and provides methods to predict
    emotions from face images in real-time.
    """
    
    def __init__(self, model_path=None, emotion_labels=None):
        """
        Initialize the emotion predictor.
        
        Args:
            model_path (str): Path to trained model file
            emotion_labels (list): List of emotion label names
        """
        if model_path is None:
            model_path = config.MODEL_SAVE_PATH
        
        if emotion_labels is None:
            emotion_labels = config.EMOTION_LABELS
        
        self.model_path = model_path
        self.emotion_labels = emotion_labels
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained CNN model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at: {self.model_path}\n"
                f"Please train the model first: python model/train.py"
            )
        
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("Γ£à Model loaded successfully")
            
            # Verify model output shape
            expected_output = len(self.emotion_labels)
            actual_output = self.model.output_shape[-1]
            
            if actual_output != expected_output:
                print(f"ΓÜá∩╕Å  Warning: Model output ({actual_output}) doesn't match "
                      f"emotion labels ({expected_output})")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_face(self, face_image, target_size=(48, 48)):
        """
        Preprocess face image for model prediction.
        
        Args:
            face_image (numpy.ndarray): Face image (grayscale or RGB)
            target_size (tuple): Target size for model input
            
        Returns:
            numpy.ndarray: Preprocessed image ready for prediction
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_image
        
        # Resize to target size
        face_resized = cv2.resize(face_gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Reshape for model input: (1, height, width, channels)
        face_preprocessed = face_normalized.reshape(1, target_size[0], target_size[1], 1)
        
        return face_preprocessed
    
    def predict_emotion(self, face_image):
        """
        Predict emotion from a face image.
        
        Args:
            face_image (numpy.ndarray): Face image (grayscale or RGB)
            
        Returns:
            tuple: (emotion_label, confidence_score, all_probabilities)
        """
        # Preprocess the face
        preprocessed_face = self.preprocess_face(face_image)
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        # Predict
        predictions = self.model.predict(preprocessed_face, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        emotion_label = self.emotion_labels[predicted_class]
        
        return emotion_label, confidence, predictions[0]
    
    def predict_top_k(self, face_image, k=3):
        """
        Get top-k emotion predictions.
        
        Args:
            face_image (numpy.ndarray): Face image
            k (int): Number of top predictions to return
            
        Returns:
            list: List of (emotion, confidence) tuples
        """
        # Preprocess and predict
        preprocessed_face = self.preprocess_face(face_image)
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        predictions = self.model.predict(preprocessed_face, verbose=0)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        # Create list of (emotion, confidence) tuples
        top_k_predictions = [
            (self.emotion_labels[idx], predictions[idx])
            for idx in top_k_indices
        ]
        
        return top_k_predictions
    
    def get_emotion_color(self, emotion):
        """
        Get color code for visualization based on emotion.
        
        Args:
            emotion (str): Emotion label
            
        Returns:
            tuple: BGR color code
        """
        color_map = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Dark Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255), # Orange
            'neutral': (128, 128, 128) # Gray
        }
        return color_map.get(emotion.lower(), (255, 255, 255))
    
    def draw_emotion_label(self, frame, bbox, emotion, confidence, show_confidence=True):
        """
        Draw emotion label and bounding box on frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            bbox (tuple): Bounding box (x, y, w, h)
            emotion (str): Predicted emotion
            confidence (float): Prediction confidence
            show_confidence (bool): Whether to show confidence score
            
        Returns:
            numpy.ndarray: Frame with drawn label
        """
        x, y, w, h = bbox
        
        # Get emotion color
        color = self.get_emotion_color(emotion)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Prepare label text
        if show_confidence:
            label = f"{emotion}: {confidence*100:.1f}%"
        else:
            label = emotion
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw text background
        cv2.rectangle(
            frame,
            (x, y - text_h - baseline - 10),
            (x + text_w, y),
            color,
            -1  # Filled
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x, y - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
        
        return frame
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return None
        
        return {
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.emotion_labels),
            'emotion_labels': self.emotion_labels,
            'total_parameters': self.model.count_params()
        }


def test_predictor():
    """Test the emotion predictor with a dummy image."""
    print("\n" + "="*70)
    print("EMOTION PREDICTOR TEST")
    print("="*70 + "\n")
    
    try:
        # Initialize predictor
        predictor = EmotionPredictor()
        
        # Display model info
        info = predictor.get_model_info()
        if info is None:
            raise RuntimeError("Model info is unavailable")
        print("\nModel Information:")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Output shape: {info['output_shape']}")
        print(f"  Number of classes: {info['num_classes']}")
        print(f"  Emotion labels: {info['emotion_labels']}")
        print(f"  Total parameters: {info['total_parameters']:,}")
        
        # Create a dummy face image (48x48 grayscale)
        dummy_face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        
        print("\n" + "-"*70)
        print("Testing with dummy image...")
        print("-"*70)
        
        # Predict emotion
        emotion, confidence, all_probs = predictor.predict_emotion(dummy_face)
        
        print(f"\nPredicted Emotion: {emotion}")
        print(f"Confidence: {confidence*100:.2f}%")
        
        print("\nAll probabilities:")
        for label, prob in zip(predictor.emotion_labels, all_probs):
            print(f"  {label:10s}: {prob*100:5.2f}%")
        
        # Test top-k predictions
        print("\n" + "-"*70)
        print("Top-3 predictions:")
        print("-"*70)
        top_3 = predictor.predict_top_k(dummy_face, k=3)
        for i, (emo, conf) in enumerate(top_3, 1):
            print(f"{i}. {emo:10s}: {conf*100:5.2f}%")
        
        print("\n" + "="*70)
        print("Γ£à Predictor test successful!")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nΓ¥î {e}")
        print("\nPlease train the model first:")
        print("  python model/train.py")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\nΓ¥î Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_predictor()
