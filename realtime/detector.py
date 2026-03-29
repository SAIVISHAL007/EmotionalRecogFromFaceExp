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
from typing import Any, List, Tuple


class FaceDetector:
    """
    Face detector class using Haar Cascade classifier.
    
    This detector identifies faces in images/video frames and returns
    bounding box coordinates for each detected face.
    """
    
    def __init__(
        self,
        method='haar',
        scale_factor=1.05,
        min_neighbors=7,
        min_size=(60, 60),
        max_faces=1,
    ):
        """
        Initialize the face detector.
        
        Args:
            method (str): Detection method ('haar' or 'mtcnn')
            scale_factor (float): Haar cascade scale factor
            min_neighbors (int): Minimum neighbors for detection confidence
            min_size (tuple): Minimum face size to detect
            max_faces (int): Maximum number of faces to return (sorted by area)
        """
        self.method = method.lower()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.max_faces = max_faces
        self._tracked_bbox: Tuple[int, int, int, int] | None = None
        self._no_detection_frames: int = 0
        self._max_no_detection_frames: int = 8  # keep last bbox for up to 8 missed frames
        
        if self.method == 'haar':
            self._load_haar_cascade()
        elif self.method == 'mtcnn':
            self._load_mtcnn()
        else:
            raise ValueError(f"Unknown detection method: {method}. Use 'haar' or 'mtcnn'.")
    
    def _load_haar_cascade(self):
        """Load Haar Cascade classifier for face detection."""
        try:
            # Resolve cascade path without relying on cv2 stub attributes.
            cv2_data: Any = getattr(cv2, 'data', None)
            if cv2_data is not None and hasattr(cv2_data, 'haarcascades'):
                cascade_dir = cv2_data.haarcascades
            else:
                cascade_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')

            cascade_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
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
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.
        
        Args:
            frame (numpy.ndarray): Input image/frame (BGR format)
            
        Returns:
            list: List of face bounding boxes [(x, y, w, h), ...]
        """
        if self.method == 'haar':
            faces = self._detect_faces_haar(frame)
        elif self.method == 'mtcnn':
            faces = self._detect_faces_mtcnn(frame)
        else:
            faces = []

        return self._apply_tracking(faces)
    
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
        # CLAHE gives better local contrast than global equalizeHist
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        frame_h, frame_w = gray.shape[:2]
        dynamic_min_w = max(self.min_size[0], int(frame_w * 0.10))
        dynamic_min_h = max(self.min_size[1], int(frame_h * 0.10))
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(dynamic_min_w, dynamic_min_h),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Filter obvious false positives by geometry and area ratio.
        filtered_faces = []
        frame_area = float(frame_w * frame_h)
        for (x, y, w, h) in faces:
            area_ratio = (w * h) / frame_area
            aspect_ratio = w / float(h)
            if 0.02 <= area_ratio <= 0.60 and 0.75 <= aspect_ratio <= 1.33:
                # Keep the Haar detection as-is for CNN inference accuracy
                # (FER-2013 was trained on tight face crops — no padding)
                filtered_faces.append((int(x), int(y), int(w), int(h)))

        # Sort by area descending and keep top-N faces.
        filtered_faces.sort(key=lambda b: b[2] * b[3], reverse=True)
        return filtered_faces[:self.max_faces]

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        pad_x: float = 0.18,
        pad_top: float = 0.15,
        pad_bottom: float = 0.55,
    ) -> Tuple[int, int, int, int]:
        """Expand a face bbox so the ROI contains full forehead/chin context."""
        x, y, w, h = bbox

        x0 = max(0, int(x - w * pad_x))
        y0 = max(0, int(y - h * pad_top))
        x1 = min(frame_w, int(x + w + w * pad_x))
        y1 = min(frame_h, int(y + h + h * pad_bottom))

        new_w = max(1, x1 - x0)
        new_h = max(1, y1 - y0)
        return (x0, y0, new_w, new_h)
    
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
        frame_h, frame_w = frame.shape[:2]
        for detection in detections:
            x, y, w, h = detection['box']
            # Ensure positive dimensions
            x, y = max(0, x), max(0, y)
            w, h = max(0, w), max(0, h)
            faces.append((max(0, x), max(0, y), max(0, w), max(0, h)))
        
        faces.sort(key=lambda b: b[2] * b[3], reverse=True)
        return faces[:self.max_faces]

    def _apply_tracking(self, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Stabilize detections by tracking and smoothing the primary face box."""
        if not faces:
            # Keep returning the last known bbox for a few frames instead of
            # immediately dropping it — this prevents flicker when the detector
            # misses a single frame.
            if self._tracked_bbox is not None and self._no_detection_frames < self._max_no_detection_frames:
                self._no_detection_frames += 1
                return [self._tracked_bbox]
            self._tracked_bbox = None
            self._no_detection_frames = 0
            return []

        self._no_detection_frames = 0

        if self._tracked_bbox is None:
            best = max(faces, key=lambda b: b[2] * b[3])
            self._tracked_bbox = best
            return [best]

        tracked_bbox = self._tracked_bbox
        best = max(faces, key=lambda b: self._iou(b, tracked_bbox))
        if self._iou(best, tracked_bbox) < 0.08:
            best = max(faces, key=lambda b: b[2] * b[3])

        # alpha=0.35 → 35% old, 65% new: more responsive, less lag
        smoothed = self._smooth_bbox(tracked_bbox, best, alpha=0.35)
        self._tracked_bbox = smoothed
        return [smoothed]

    def _smooth_bbox(
        self,
        prev_bbox: Tuple[int, int, int, int],
        curr_bbox: Tuple[int, int, int, int],
        alpha: float = 0.60,
    ) -> Tuple[int, int, int, int]:
        """Exponential smoothing for bbox coordinates to reduce jitter."""
        px, py, pw, ph = prev_bbox
        cx, cy, cw, ch = curr_bbox
        x = int(alpha * px + (1.0 - alpha) * cx)
        y = int(alpha * py + (1.0 - alpha) * cy)
        w = max(1, int(alpha * pw + (1.0 - alpha) * cw))
        h = max(1, int(alpha * ph + (1.0 - alpha) * ch))
        return (x, y, w, h)

    def _iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        """Intersection-over-union between two bboxes."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0

        union = aw * ah + bw * bh - inter_area
        if union <= 0:
            return 0.0
        return inter_area / float(union)
    
    def extract_face_roi(self, frame, bbox, target_size=(48, 48), grayscale=True):
        """
        Extract Face Region of Interest for CNN inference.
        Uses the raw tight Haar bbox — NOT the padded display bbox —
        so the crop matches FER-2013 training data format.
        """
        x, y, w, h = bbox
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            face_roi = frame[max(0,y):max(1,y+h), max(0,x):max(1,x+w)]
        
        # Convert to grayscale if needed
        if grayscale and len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        face_roi = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        
        return face_roi

    def get_display_bbox(self, bbox, frame_w, frame_h):
        """Return an expanded bbox for display only — does NOT affect CNN inference."""
        return self._expand_bbox(
            bbox, frame_w, frame_h,
            pad_x=0.12,
            pad_top=0.28,
            pad_bottom=0.10,
        )
    
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
            'min_size': self.min_size,
            'max_faces': self.max_faces
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
