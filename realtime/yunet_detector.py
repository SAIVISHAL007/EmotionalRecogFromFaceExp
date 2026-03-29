"""
YuNet DNN Multi-Face Detector
Replaces Haar Cascade with OpenCV's built-in cv2.FaceDetectorYN (YuNet).
- Works on Python 3.13 with OpenCV 4.8+
- Detects multiple faces in a single frame
- Returns proper full-face bounding boxes with 5 landmarks
- ~97% accuracy on WIDER FACE benchmark vs ~60% for Haar
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class YuNetFaceDetector:
    """
    Multi-face detector using OpenCV's YuNet DNN model.
    Drop-in upgrade for the Haar cascade FaceDetector.
    """

    def __init__(
        self,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 10,
        max_faces: int = 10,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.max_faces = max_faces
        self._detector = None
        self._input_size = (320, 320)
        self._last_frame_size = (0, 0)
        self._init_detector()

    def _init_detector(self):
        """Initialize YuNet detector using the downloaded ONNX model."""
        import os
        # Find model relative to this file
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, 'trained_models', 'face_detection_yunet.onnx')
        if not os.path.exists(model_path):
            print(f"⚠️ YuNet model not found at {model_path}")
            self._detector = None
            return
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=self._input_size,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
                backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            print("✅ YuNet face detector initialized (multi-face DNN)")
        except Exception as e:
            print(f"⚠️ YuNet init failed: {e}. Falling back to Haar.")
            self._detector = None

    def _update_input_size(self, frame_w: int, frame_h: int):
        """Resize YuNet input to match current frame — avoids rescaling artifacts."""
        if (frame_w, frame_h) != self._last_frame_size:
            if self._detector is not None:
                self._detector.setInputSize((frame_w, frame_h))
            self._last_frame_size = (frame_w, frame_h)

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in the frame.
        Returns list of (x, y, w, h) bounding boxes — tight Haar-equivalent crop
        suitable for FER-2013 trained CNN inference.
        """
        if self._detector is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        self._update_input_size(frame_w, frame_h)

        _, faces = self._detector.detect(frame)
        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Clamp to frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)
            if w > 10 and h > 10:
                results.append((x, y, w, h))

        # Sort largest face first
        results.sort(key=lambda b: b[2] * b[3], reverse=True)
        return results[:self.max_faces]

    def get_display_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        pad_x: float = 0.08,
        pad_top: float = 0.15,
        pad_bottom: float = 0.08,
    ) -> Tuple[int, int, int, int]:
        """
        Expand the tight inference bbox for nice on-screen display.
        YuNet already returns a tighter, more accurate box than Haar,
        so we only need modest padding.
        """
        x, y, w, h = bbox
        dx = int(w * pad_x)
        dt = int(h * pad_top)
        db = int(h * pad_bottom)

        x0 = max(0, x - dx)
        y0 = max(0, y - dt)
        x1 = min(frame_w, x + w + dx)
        y1 = min(frame_h, y + h + db)

        return (x0, y0, x1 - x0, y1 - y0)

    def extract_face_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (48, 48),
        grayscale: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Crop the tight inference bbox from frame and resize for CNN.
        Matches FER-2013 training format exactly.
        """
        x, y, w, h = bbox
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        if grayscale and len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
        return roi

    @property
    def is_available(self) -> bool:
        return self._detector is not None
