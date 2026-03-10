"""
Real-Time Facial Emotion Recognition Webcam Application

This standalone application captures video from webcam, detects faces,
and predicts emotions in real-time using the trained CNN model.

Usage:
    python realtime/webcam_app.py

Controls:
    - 'q': Quit application
    - 's': Save current frame
    - 'd': Toggle debug mode
    - 'p': Pause/resume
    - 'f': Toggle FPS display

Academic Focus:
    This demonstrates the practical application of the trained CNN model
    for real-time emotion recognition from live video streams.
"""

import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.detector import FaceDetector
from realtime.emotion_predictor import EmotionPredictor
from model import config


class EmotionRecognitionApp:
    """
    Real-time emotion recognition application using webcam.
    
    This class combines face detection and emotion prediction
    to create a complete real-time emotion recognition system.
    """
    
    def __init__(self, model_path=None, camera_id=0):
        """
        Initialize the emotion recognition application.
        
        Args:
            model_path (str): Path to trained model
            camera_id (int): Camera device ID
        """
        self.camera_id = camera_id
        self.model_path = model_path or config.MODEL_SAVE_PATH
        
        # Initialize components
        self.face_detector: Optional[FaceDetector] = None
        self.emotion_predictor: Optional[EmotionPredictor] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # App state
        self.paused = False
        self.show_fps = True
        self.debug_mode = False
        self.frame_count = 0
        self.fps = 0
        
        # Statistics
        self.total_faces_detected = 0
        self.emotion_counts = {emotion: 0 for emotion in config.EMOTION_LABELS}
        
        # Initialize
        self._setup()
    
    def _setup(self):
        """Setup face detector and emotion predictor."""
        print("\n" + "="*70)
        print("REAL-TIME EMOTION RECOGNITION SYSTEM")
        print("="*70 + "\n")
        
        # Initialize face detector
        print("Initializing face detector...")
        self.face_detector = FaceDetector(
            method='haar',
            scale_factor=1.05,
            min_neighbors=7,
            min_size=(60, 60),
            max_faces=1
        )
        
        # Initialize emotion predictor
        print("\nInitializing emotion predictor...")
        try:
            self.emotion_predictor = EmotionPredictor(model_path=self.model_path)
        except FileNotFoundError:
            print("\n❌ Model not found!")
            print("Please train the model first: python model/train.py")
            sys.exit(1)
        
        # Display model info
        info = self.emotion_predictor.get_model_info()
        if info is None:
            print("\n❌ Failed to read model info")
            sys.exit(1)
        print(f"\n✅ Model loaded: {info['num_classes']} emotions")
        print(f"   Emotions: {', '.join(info['emotion_labels'])}")
        
        print("\n" + "="*70)
        print("CONTROLS")
        print("="*70)
        print("  'q' - Quit application")
        print("  's' - Save current frame")
        print("  'p' - Pause/Resume")
        print("  'f' - Toggle FPS display")
        print("  'd' - Toggle debug mode")
        print("="*70 + "\n")
    
    def start(self):
        """Start the webcam application."""
        # Open webcam
        print(f"Opening camera (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("❌ Failed to open camera")
            return
        
        print("✅ Camera opened successfully")
        print("\nStarting emotion recognition...\n")
        
        # Main loop
        prev_time = time.time()
        processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print("❌ Failed to read frame")
                        break
                    
                    # Process frame
                    processed_frame = self._process_frame(frame)
                    
                    # Calculate FPS
                    current_time = time.time()
                    self.fps = 1 / (current_time - prev_time)
                    prev_time = current_time
                    
                    # Add overlays
                    processed_frame = self._add_overlays(processed_frame)
                    
                    # Display frame
                    cv2.imshow('Real-Time Emotion Recognition', processed_frame)
                    
                    self.frame_count += 1
                else:
                    # Just show the last frame when paused
                    cv2.imshow('Real-Time Emotion Recognition', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting application...")
                    break
                elif key == ord('s'):
                    self._save_frame(processed_frame)
                elif key == ord('p'):
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"\n{status}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    status = "ON" if self.debug_mode else "OFF"
                    print(f"\nDebug mode: {status}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self._cleanup()
    
    def _process_frame(self, frame):
        """
        Process a single frame: detect faces and predict emotions.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        if self.face_detector is None or self.emotion_predictor is None:
            return frame

        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Update statistics
        self.total_faces_detected += len(faces)
        
        # Process each detected face
        for face_bbox in faces:
            x, y, w, h = face_bbox
            
            # Extract face ROI
            face_roi = self.face_detector.extract_face_roi(
                frame, face_bbox,
                target_size=(48, 48),
                grayscale=True
            )
            
            # Predict emotion
            emotion, confidence, all_probs = self.emotion_predictor.predict_emotion(face_roi)
            
            # Update emotion counts
            self.emotion_counts[emotion] += 1
            
            # Draw emotion label and bounding box
            frame = self.emotion_predictor.draw_emotion_label(
                frame, face_bbox, emotion, confidence,
                show_confidence=True
            )
            
            # Debug mode: show top-3 predictions
            if self.debug_mode:
                top_3 = self.emotion_predictor.predict_top_k(face_roi, k=3)
                self._draw_debug_info(frame, face_bbox, top_3)
        
        return frame
    
    def _draw_debug_info(self, frame, bbox, top_3_predictions):
        """
        Draw debug information (top-3 predictions) near the face.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            bbox (tuple): Face bounding box
            top_3_predictions (list): Top-3 predictions
        """
        x, y, w, h = bbox
        
        # Position for debug text (to the right of face)
        debug_x = x + w + 10
        debug_y = y + 20
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        for i, (emotion, confidence) in enumerate(top_3_predictions):
            text = f"{i+1}. {emotion}: {confidence*100:.1f}%"
            cv2.putText(
                frame, text,
                (debug_x, debug_y + i * 20),
                font, font_scale, (255, 255, 255), thickness
            )
    
    def _add_overlays(self, frame):
        """
        Add informational overlays to the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with overlays
        """
        # FPS counter
        if self.show_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                frame, fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
        
        # Paused indicator
        if self.paused:
            cv2.putText(
                frame, "PAUSED",
                (frame.shape[1] // 2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )
        
        # Debug mode indicator
        if self.debug_mode:
            cv2.putText(
                frame, "DEBUG",
                (frame.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )
        
        return frame
    
    def _save_frame(self, frame):
        """Save current frame to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_capture_{timestamp}.jpg"
        
        # Create screenshots directory
        screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'screenshots'
        )
        os.makedirs(screenshots_dir, exist_ok=True)
        
        filepath = os.path.join(screenshots_dir, filename)
        cv2.imwrite(filepath, frame)
        
        print(f"📸 Frame saved: {filepath}")
    
    def _cleanup(self):
        """Cleanup resources and display statistics."""
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Display statistics
        print("\n" + "="*70)
        print("SESSION STATISTICS")
        print("="*70)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total faces detected: {self.total_faces_detected}")
        print(f"Average FPS: {self.fps:.1f}")
        
        print("\nEmotion Detection Counts:")
        sorted_emotions = sorted(
            self.emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for emotion, count in sorted_emotions:
            if count > 0:
                percentage = (count / self.total_faces_detected * 100) if self.total_faces_detected > 0 else 0
                print(f"  {emotion.capitalize():10s}: {count:4d} ({percentage:5.1f}%)")
        
        print("="*70 + "\n")


def main():
    """Main function to run the application."""
    # Parse command line arguments (simple)
    camera_id = 0
    model_path = None
    
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera ID: {sys.argv[1]}")
            print("Usage: python webcam_app.py [camera_id] [model_path]")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    
    # Create and start application
    try:
        app = EmotionRecognitionApp(model_path=model_path, camera_id=camera_id)
        app.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
