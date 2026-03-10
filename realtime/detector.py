"""
Face Detection Module using Haar Cascade Classifier.

This module provides face detection functionality for real-time emotion recognition.
It uses OpenCV's pre-trained Haar Cascade classifier for detecting faces in video frames.

For better accuracy, MTCNN can be used as an alternative (optional).

Usage:
    detector = FaceDetector()
    faces = detector.detect_faces(frame)
"""

import cv2
import numpy as np
import os


class FaceDetector:
    """
    Face detector class using Haar Cascade classifier.
    
    This detector identifies faces in images/video frames and returns
    bounding box coordinates for each detected face.
    """
    
    def __init__(self, method='haar', scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Initialize the face detector.
        
        Args:
            method (str): Detection method ('haar' or 'mtcnn')
            scale_factor (float): Haar cascade scale factor
            min_neighbors (int): Minimum neighbors for detection confidence
            min_size (tuple): Minimum face size to detect
        """
        self.method = method.lower()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        if self.method == 'haar':
            self._load_haar_cascade()
        elif self.method == 'mtcnn':
            self._load_mtcnn()
        else:
            raise ValueError(f"Unknown detection method: {method}. Use 'haar' or 'mtcnn'.")
    
    def _load_haar_cascade(self):
        """Load Haar Cascade classifier for face detection."""
        try:
            # Try to load from OpenCV data directory
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise IOError("Failed to load Haar Cascade classifier")
            
            print("✅ Haar Cascade classifier loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading Haar Cascade: {e}")
            raise
    
    def _load_mtcnn(self):
        """Load MTCNN face detector (optional, requires mtcnn package)."""
        try:
            from mtcnn import MTCNN
            self.mtcnn_detector = MTCNN()
            print("✅ MTCNN detector loaded successfully")
        except ImportError:
            print("❌ MTCNN not installed. Install with: pip install mtcnn")
            print("   Falling back to Haar Cascade...")
            self.method = 'haar'
            self._load_haar_cascade()
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame.
        
        Args:
            frame (numpy.ndarray): Input image/frame (BGR format)
            
        Returns:
            list: List of face bounding boxes [(x, y, w, h), ...]
        """
        if self.method == 'haar':
            return self._detect_faces_haar(frame)
        elif self.method == 'mtcnn':
            return self._detect_faces_mtcnn(frame)
    
    def _detect_faces_haar(self, frame):
        """
        Detect faces using Haar Cascade.
        
        Args:
            frame (numpy.ndarray): Input frame (BGR)
            
        Returns:
            list: List of bounding boxes [(x, y, w, h), ...]
        """
        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    def _detect_faces_mtcnn(self, frame):
        """
        Detect faces using MTCNN.
        
        Args:
            frame (numpy.ndarray): Input frame (BGR)
            
        Returns:
            list: List of bounding boxes [(x, y, w, h), ...]
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.mtcnn_detector.detect_faces(rgb_frame)
        
        # Extract bounding boxes
        faces = []
        for detection in detections:
            x, y, w, h = detection['box']
            # Ensure positive dimensions
            x, y = max(0, x), max(0, y)
            w, h = max(0, w), max(0, h)
            faces.append((x, y, w, h))
        
        return faces
    
    def extract_face_roi(self, frame, bbox, target_size=(48, 48), grayscale=True):
        """
        Extract Face Region of Interest (ROI) from frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            bbox (tuple): Bounding box (x, y, w, h)
            target_size (tuple): Size to resize face to
            grayscale (bool): Whether to convert to grayscale
            
        Returns:
            numpy.ndarray: Extracted and preprocessed face ROI
        """
        x, y, w, h = bbox
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale if needed
        if grayscale and len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        face_roi = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        
        return face_roi
    
    def draw_face_boxes(self, frame, faces, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected faces.
        
        Args:
            frame (numpy.ndarray): Input frame
            faces (list): List of face bounding boxes
            color (tuple): Box color (BGR)
            thickness (int): Box line thickness
            
        Returns:
            numpy.ndarray: Frame with drawn boxes
        """
        frame_copy = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, thickness)
        
        return frame_copy
    
    def get_detection_info(self):
        """
        Get information about the detector configuration.
        
        Returns:
            dict: Detector configuration
        """
        return {
            'method': self.method,
            'scale_factor': self.scale_factor,
            'min_neighbors': self.min_neighbors,
            'min_size': self.min_size
        }


def test_detector():
    """Test the face detector with webcam."""
    print("\n" + "="*70)
    print("FACE DETECTOR TEST")
    print("="*70)
    print("Testing face detection with webcam...")
    print("Press 'q' to quit\n")
    
    # Initialize detector
    detector = FaceDetector(method='haar')
    print(f"Detector config: {detector.get_detection_info()}\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("✅ Webcam opened successfully")
    print("="*70 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw bounding boxes
        if faces:
            frame = detector.draw_face_boxes(frame, faces, color=(0, 255, 0), thickness=2)
            
            # Display face count
            cv2.putText(
                frame, 
                f"Faces: {len(faces)}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Show frame
        cv2.imshow('Face Detection Test', frame)
        
        frame_count += 1
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test complete. Processed {frame_count} frames.")


if __name__ == "__main__":
    test_detector()
