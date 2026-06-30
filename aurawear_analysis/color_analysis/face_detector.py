"""M2. FaceLandmarkDetector - Detect face landmarks using OpenCV Haar Cascade."""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceDetectionResult:
    """Face detection output."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: np.ndarray  # (468, 2) FaceMesh landmarks in pixel coords
    pose: Dict[str, float]  # yaw, pitch, roll (degrees)
    confidence: float
    face_index: int


class FaceLandmarkDetector:
    """
    Detect face and facial landmarks using OpenCV Haar Cascade.
    Generates simulated 468-point landmarks for downstream compatibility.
    
    Output:
      - bbox: (x1, y1, x2, y2)
      - landmarks: (468, 2) in pixel coordinates
      - pose: {yaw, pitch, roll}
      - confidence: 0-1
    """
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)

        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    def detect(self, img_bgr: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Detect face landmarks using OpenCV Haar Cascade.
        
        Args:
            img_bgr: BGR image (H, W, 3)
            
        Returns:
            FaceDetectionResult or None if no face detected
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) == 0:
            return None

        # Get largest face
        (x1, y1, w_face, h_face) = max(faces, key=lambda f: f[2] * f[3])
        x2 = x1 + w_face
        y2 = y1 + h_face
        
        # Try to detect eyes inside face bbox to generate more realistic landmarks.
        eye_boxes = self._detect_eyes(gray, (x1, y1, x2, y2))

        # Generate simulated landmarks (for compatibility)
        landmarks_px = self._generate_fake_landmarks(x1, y1, x2, y2, eye_boxes=eye_boxes)
        
        pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        
        return FaceDetectionResult(
            bbox=(x1, y1, x2, y2),
            landmarks=landmarks_px,
            pose=pose,
            confidence=0.80,
            face_index=0,
        )
    
    @staticmethod
    def _generate_fake_landmarks(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        num_landmarks: int = 468,
        eye_boxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate 468 fake landmarks for compatibility.
        Place landmarks at anatomically reasonable positions.
        """
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w_face = x2 - x1
        h_face = y2 - y1
        
        landmarks = np.zeros((num_landmarks, 2), dtype=np.float32)
        
        # Key landmark indices:
        # 33: left eye outer, 263: right eye outer
        # 33-42: left eye, 263-272: right eye
        # 10, 109, 338: forehead points
        # 116, 123, 147: left cheek, 340, 349, 371: right cheek
        # 61, 91: mouth
        
        # Forehead (10, 109, 338)
        landmarks[10] = [cx, y1 + 0.15 * h_face]
        landmarks[109] = [cx - 0.15 * w_face, y1 + 0.15 * h_face]
        landmarks[338] = [cx + 0.15 * w_face, y1 + 0.15 * h_face]
        
        # Eye regions (33-42, 263-272)
        # If we have detected eye boxes, use them; otherwise fall back to proportional placement.
        if eye_boxes is not None and len(eye_boxes) >= 2:
            # eye_boxes are absolute coords: (ex1, ey1, ex2, ey2)
            # Sort left-to-right
            eye_boxes = sorted(eye_boxes, key=lambda b: b[0])
            left_box, right_box = eye_boxes[0], eye_boxes[1]

            lx1, ly1, lx2, ly2 = left_box
            rx1, ry1, rx2, ry2 = right_box

            left_eye_x = (lx1 + lx2) / 2
            left_eye_y = (ly1 + ly2) / 2
            right_eye_x = (rx1 + rx2) / 2
            right_eye_y = (ry1 + ry2) / 2

            left_r = max(4.0, 0.25 * (lx2 - lx1))
            right_r = max(4.0, 0.25 * (rx2 - rx1))
        else:
            left_eye_x = cx - 0.25 * w_face
            left_eye_y = cy - 0.15 * h_face
            right_eye_x = cx + 0.25 * w_face
            right_eye_y = cy - 0.15 * h_face
            left_r = right_r = 0.06 * w_face

        for i in range(33, 43):
            angle = ((i - 33) / 10) * 2 * np.pi
            landmarks[i] = [left_eye_x + left_r * np.cos(angle), left_eye_y + left_r * np.sin(angle)]

        for i in range(263, 273):
            angle = ((i - 263) / 10) * 2 * np.pi
            landmarks[i] = [right_eye_x + right_r * np.cos(angle), right_eye_y + right_r * np.sin(angle)]
        
        # Left cheek (116, 123, 147)
        landmarks[116] = [cx - 0.25 * w_face, cy + 0.05 * h_face]
        landmarks[123] = [cx - 0.2 * w_face, cy + 0.15 * h_face]
        landmarks[147] = [cx - 0.1 * w_face, cy + 0.2 * h_face]
        
        # Right cheek (340, 349, 371)
        landmarks[340] = [cx + 0.25 * w_face, cy + 0.05 * h_face]
        landmarks[349] = [cx + 0.2 * w_face, cy + 0.15 * h_face]
        landmarks[371] = [cx + 0.1 * w_face, cy + 0.2 * h_face]
        
        # Nose (1)
        landmarks[1] = [cx, cy - 0.05 * h_face]
        
        # Mouth (61, 91)
        landmarks[61] = [cx - 0.1 * w_face, y2 - 0.15 * h_face]
        landmarks[91] = [cx + 0.1 * w_face, y2 - 0.15 * h_face]
        
        # Fill remaining landmarks with reasonable positions
        for i in range(num_landmarks):
            if landmarks[i].sum() == 0:  # Not yet set
                # Random position within face
                u = i / num_landmarks
                landmarks[i] = [
                    x1 + u * w_face,
                    y1 + ((i % 4) / 4) * h_face
                ]
        
        return landmarks

    def _detect_eyes(
        self,
        gray: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Detect eyes inside face bbox using OpenCV haarcascade_eye."""
        if self.eye_cascade is None:
            return np.empty((0, 4), dtype=np.int32)

        x1, y1, x2, y2 = face_bbox
        # Use upper half of the face for eye search
        roi_y2 = y1 + int(0.6 * (y2 - y1))
        roi = gray[y1:roi_y2, x1:x2]
        if roi.size == 0:
            return np.empty((0, 4), dtype=np.int32)

        eyes = self.eye_cascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
        )

        if len(eyes) == 0:
            return np.empty((0, 4), dtype=np.int32)

        # Convert to absolute coords (x1,y1,x2,y2)
        eye_boxes = []
        for (ex, ey, ew, eh) in eyes:
            ax1 = x1 + ex
            ay1 = y1 + ey
            ax2 = ax1 + ew
            ay2 = ay1 + eh
            eye_boxes.append((ax1, ay1, ax2, ay2))

        # Keep top 2 largest
        eye_boxes = sorted(eye_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:2]
        return np.array(eye_boxes, dtype=np.int32)
