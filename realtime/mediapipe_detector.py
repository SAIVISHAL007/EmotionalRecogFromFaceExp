"""
Enhanced Face Detection using MediaPipe Face Mesh and Multiple Face Support.

This module provides multi-face detection and mesh extraction capabilities
optimized for real-time performance on edge devices, based on research patterns
from facial expression recognition studies.

Features:
    - Detect multiple faces in a single frame
    - Extract 478 3D face landmarks per face using MediaPipe
    - Robust bounding box calculation
    - Confidence-based filtering
    - Optimized for edge deployment

Usage:
    detector = MediaPipeFaceDetector()
    results = detector.detect_faces(frame)
    
    for face_id, face_data in results.items():
        bbox = face_data['bbox']  # (x, y, w, h)
        landmarks = face_data['landmarks']  # 478 x 3 array
        confidence = face_data['confidence']
"""

import cv2
import numpy as np
MEDIAPIPE_AVAILABLE = False
try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    mp_face_mesh = solutions.face_mesh
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        MEDIAPIPE_AVAILABLE = True
    except (ImportError, AttributeError):
        mp_face_mesh = None
        mp_drawing = None
        mp_drawing_styles = None
        MEDIAPIPE_AVAILABLE = False

if not MEDIAPIPE_AVAILABLE:
    raise ImportError("MediaPipe not available on this runtime")

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FaceData:
    """Data structure for detected face information."""
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: np.ndarray  # (478, 3) - x, y, z coordinates
    confidence: float
    mesh_points: List[Tuple[int, int]]  # 2D pixel coordinates for drawing


class MediaPipeFaceDetector:
    """
    Multi-face detector using MediaPipe Face Mesh.
    
    This detector identifies multiple faces in images/video frames and returns
    3D face landmarks for each detected face, optimized for real-time emotion recognition.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 10,
        static_image_mode: bool = False,
        refine_landmarks: bool = True,
    ):
        """
        Initialize MediaPipe Face Mesh detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold for face detection (0-1)
            min_tracking_confidence: Minimum confidence for face tracking (0-1)
            max_num_faces: Maximum number of faces to detect (1-10)
            static_image_mode: Whether to process frames as static images
            refine_landmarks: Use Attention Mesh for refined eye/mouth landmarks
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_faces = max_num_faces
        self.static_image_mode = static_image_mode
        self.refine_landmarks = refine_landmarks
        
        # Initialize MediaPipe FaceMesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Store drawing utilities
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        self.frame_width = None
        self.frame_height = None
        self._face_id_counter = 0
        
        print("✅ MediaPipe Face Mesh initialized successfully")
        print(f"   - Max faces: {max_num_faces}")
        print(f"   - Detection confidence: {min_detection_confidence}")
        print(f"   - Tracking confidence: {min_tracking_confidence}")
        print(f"   - Refined landmarks: {refine_landmarks}")
    
    def detect_faces(self, frame: np.ndarray) -> Dict[int, FaceData]:
        """
        Detect all faces in the frame with landmarks.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Dict mapping face_id to FaceData containing landmarks and bounding box
        """
        if frame is None or frame.size == 0:
            return {}
        
        # Get frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.face_mesh.process(frame_rgb)
        
        faces_dict = {}
        
        if results.multi_face_landmarks and results.multi_face_confidences:
            for face_idx, (face_landmarks, confidence) in enumerate(
                zip(results.multi_face_landmarks, results.multi_face_confidences)
            ):
                # Skip low confidence detections
                if confidence < self.min_detection_confidence:
                    continue
                
                # Extract 3D landmarks
                landmarks_3d = self._extract_landmarks_3d(face_landmarks)
                
                # Calculate bounding box from landmarks
                bbox = self._calculate_bbox(landmarks_3d)
                
                # Convert landmarks to 2D pixel coordinates for visualization
                mesh_points = self._landmarks_to_2d_pixels(landmarks_3d)
                
                # Create face data
                face_data = FaceData(
                    face_id=face_idx,
                    bbox=bbox,
                    landmarks=landmarks_3d,
                    confidence=float(confidence),
                    mesh_points=mesh_points,
                )
                
                faces_dict[face_idx] = face_data
        
        return faces_dict
    
    def _extract_landmarks_3d(self, face_landmarks) -> np.ndarray:
        """
        Extract 3D landmarks from MediaPipe results.
        
        Args:
            face_landmarks: MediaPipe face landmarks object
            
        Returns:
            Array of shape (478, 3) containing x, y, z coordinates
        """
        landmarks_3d = []
        
        for landmark in face_landmarks.landmark:
            landmarks_3d.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks_3d, dtype=np.float32)
    
    def _calculate_bbox(self, landmarks_3d: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from 3D landmarks.
        
        Args:
            landmarks_3d: Array of shape (478, 3)
            
        Returns:
            Tuple of (x, y, w, h) in pixel coordinates
        """
        # Get normalized coordinates
        x_coords = landmarks_3d[:, 0]
        y_coords = landmarks_3d[:, 1]
        
        # Calculate min/max in normalized space
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add margin (5% padding)
        margin_x = (x_max - x_min) * 0.05
        margin_y = (y_max - y_min) * 0.05
        
        x_min = max(0, x_min - margin_x)
        x_max = min(1, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(1, y_max + margin_y)
        
        # Convert to pixel coordinates
        x = int(x_min * self.frame_width)
        y = int(y_min * self.frame_height)
        w = int((x_max - x_min) * self.frame_width)
        h = int((y_max - y_min) * self.frame_height)
        
        return (x, y, w, h)
    
    def _landmarks_to_2d_pixels(self, landmarks_3d: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert normalized 3D landmarks to 2D pixel coordinates.
        
        Args:
            landmarks_3d: Array of shape (478, 3)
            
        Returns:
            List of (x, y) tuples in pixel coordinates
        """
        mesh_points = []
        
        for landmark in landmarks_3d:
            x = int(landmark[0] * self.frame_width)
            y = int(landmark[1] * self.frame_height)
            mesh_points.append((x, y))
        
        return mesh_points
    
    def draw_face_mesh(self, frame: np.ndarray, faces_dict: Dict[int, FaceData]) -> np.ndarray:
        """
        Draw face mesh landmarks and bounding boxes on frame.
        
        Args:
            frame: Input frame (BGR)
            faces_dict: Dictionary of detected faces
            
        Returns:
            Frame with drawn annotations
        """
        annotated_frame = frame.copy()
        
        for face_id, face_data in faces_dict.items():
            # Draw bounding box
            x, y, w, h = face_data.bbox
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw face ID and confidence
            text = f"Face {face_id+1} ({face_data.confidence:.2f})"
            cv2.putText(
                annotated_frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            
            # Draw mesh points (sparse for performance)
            # Draw every 5th point for cleaner visualization
            for i in range(0, len(face_data.mesh_points), 5):
                pt = face_data.mesh_points[i]
                cv2.circle(annotated_frame, pt, 1, (0, 0, 255), 1)
        
        return annotated_frame
    
    def extract_face_regions(self, frame: np.ndarray, faces_dict: Dict[int, FaceData]) -> Dict[int, np.ndarray]:
        """
        Extract face regions from the frame.
        
        Args:
            frame: Input frame (BGR)
            faces_dict: Dictionary of detected faces
            
        Returns:
            Dictionary mapping face_id to face region image
        """
        face_regions = {}
        
        for face_id, face_data in faces_dict.items():
            x, y, w, h = face_data.bbox
            
            # Ensure bounds are within frame
            x = max(0, x)
            y = max(0, y)
            x_end = min(frame.shape[1], x + w)
            y_end = min(frame.shape[0], y + h)
            
            # Extract face region
            face_region = frame[y:y_end, x:x_end]
            face_regions[face_id] = face_region
        
        return face_regions
    
    def normalize_landmarks(self, landmarks_3d: np.ndarray, reference_point_idx: int = 0) -> np.ndarray:
        """
        Normalize landmarks to zero mean based on reference point.
        
        This follows the normalization approach from the research papers,
        using the center of Cupid's bow (landmark 0) as reference.
        
        Args:
            landmarks_3d: Array of shape (478, 3)
            reference_point_idx: Index of reference landmark (default: 0 for Cupid's bow)
            
        Returns:
            Normalized landmarks in range (-1, 1)
        """
        ref_point = landmarks_3d[reference_point_idx]
        
        # Subtract reference point
        normalized = landmarks_3d - ref_point
        
        # Get max values for normalization
        max_x = np.max(np.abs(normalized[:, 0]))
        max_y = np.max(np.abs(normalized[:, 1]))
        max_z = np.max(np.abs(normalized[:, 2]))
        
        # Avoid division by zero
        max_x = max(max_x, 1e-6)
        max_y = max(max_y, 1e-6)
        max_z = max(max_z, 1e-6)
        
        # Normalize to (-1, 1)
        normalized[:, 0] /= max_x
        normalized[:, 1] /= max_y
        normalized[:, 2] /= max_z
        
        return normalized


def draw_rectangles_with_labels(
    frame: np.ndarray,
    faces_dict: Dict[int, FaceData],
    predictions: Optional[Dict[int, Tuple[str, float]]] = None,
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw rectangles and labels for multiple detected faces.
    
    Args:
        frame: Input frame (BGR)
        faces_dict: Dictionary of detected faces
        predictions: Optional dict mapping face_id to (emotion, confidence)
        thickness: Rectangle line thickness
        color: Rectangle color (BGR)
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    for face_id, face_data in faces_dict.items():
        x, y, w, h = face_data.bbox
        
        # Draw bounding rectangle
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
        
        # Prepare label text
        label = f"Face {face_id + 1}"
        
        if predictions and face_id in predictions:
            emotion, emotion_conf = predictions[face_id]
            label += f" - {emotion} ({emotion_conf:.2f})"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y - 10, label_size[1])
        cv2.rectangle(
            annotated_frame,
            (x, label_y - label_size[1] - 5),
            (x + label_size[0], label_y + 5),
            color,
            -1,
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    
    return annotated_frame
