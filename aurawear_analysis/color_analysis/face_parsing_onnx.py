"""Optional face parsing (semantic segmentation) via ONNX Runtime.

Goal: produce cleaner ROI masks for hair/skin/eyes than heuristic geometry.

This module is intentionally optional:
- If `onnxruntime` isn't installed or the model file doesn't exist, it returns None.
- The rest of the pipeline will fall back to the existing heuristic ROI builder.

Expected model: a face-parsing network trained on CelebAMask-HQ (19 classes).
Typical output is logits shaped like (1, 19, H, W) or (1, H, W, 19).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# CelebAMask-HQ (19 classes) common label mapping.
# This is the de-facto mapping used by many public face-parsing implementations.
CELEBAMASKHQ_LABELS: Dict[str, int] = {
    "background": 0,
    "skin": 1,
    "l_brow": 2,
    "r_brow": 3,
    "l_eye": 4,
    "r_eye": 5,
    "eye_g": 6,
    "l_ear": 7,
    "r_ear": 8,
    "ear_r": 9,
    "nose": 10,
    "mouth": 11,
    "u_lip": 12,
    "l_lip": 13,
    "neck": 14,
    "neck_l": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}


@dataclass(frozen=True)
class FaceParsingMasks:
    parsing_map: np.ndarray  # (H, W) int labels
    skin_mask: np.ndarray  # (H, W) uint8 {0,1}
    hair_mask: np.ndarray  # (H, W) uint8 {0,1}
    eye_mask: np.ndarray  # (H, W) uint8 {0,1}


def _default_model_path() -> Path:
    env_path = os.environ.get("FACE_MODEL_PATH", "").strip()
    if env_path:
        return Path(env_path)
    pkg_root = Path(__file__).resolve().parents[1]
    return pkg_root / "assets" / "models" / "face_parsing.onnx"


class FaceParsingONNX:
    """ONNX Runtime face parsing wrapper."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        input_size: int = 512,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        face_padding_ratio: float = 0.15,
    ):
        self.model_path = Path(model_path) if model_path is not None else _default_model_path()
        self.input_size = int(input_size)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.face_padding_ratio = float(face_padding_ratio)

        self._ort = None
        self._session = None

    def is_available(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import onnxruntime as ort  # noqa: F401

            return True
        except Exception:
            return False

    def _ensure_session(self):
        if self._session is not None:
            return

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self._ort = ort
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)

    @staticmethod
    def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return x1, y1, x2, y2

    def parse(self, img_bgr: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[FaceParsingMasks]:
        """Run face parsing and return full-image masks.

        Args:
            img_bgr: (H,W,3) BGR uint8
            face_bbox: (x1,y1,x2,y2)
        """
        if not self.is_available():
            return None

        self._ensure_session()

        import cv2

        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = face_bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = int(bw * self.face_padding_ratio)
        pad_y = int(bh * self.face_padding_ratio)
        cx1, cy1, cx2, cy2 = self._clip_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)

        crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
        if crop_bgr.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        # NCHW
        x = np.transpose(x, (2, 0, 1))[None, ...]

        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        out = self._session.run([output_name], {input_name: x})[0]

        # Handle (1,19,H,W) or (1,H,W,19)
        if out.ndim != 4:
            return None
        if out.shape[1] == 19:
            logits = out[0]  # (19,H,W)
            parsing_small = np.argmax(logits, axis=0).astype(np.uint8)
        elif out.shape[-1] == 19:
            logits = out[0]  # (H,W,19)
            parsing_small = np.argmax(logits, axis=-1).astype(np.uint8)
        else:
            return None

        crop_h, crop_w = crop_bgr.shape[:2]
        parsing_crop = cv2.resize(parsing_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        parsing_map = np.zeros((h, w), dtype=np.uint8)
        parsing_map[cy1:cy2, cx1:cx2] = parsing_crop

        skin_id = CELEBAMASKHQ_LABELS["skin"]
        hair_id = CELEBAMASKHQ_LABELS["hair"]
        l_eye_id = CELEBAMASKHQ_LABELS["l_eye"]
        r_eye_id = CELEBAMASKHQ_LABELS["r_eye"]

        skin_mask = (parsing_map == skin_id).astype(np.uint8)
        hair_mask = (parsing_map == hair_id).astype(np.uint8)
        eye_mask = ((parsing_map == l_eye_id) | (parsing_map == r_eye_id)).astype(np.uint8)

        return FaceParsingMasks(
            parsing_map=parsing_map,
            skin_mask=skin_mask,
            hair_mask=hair_mask,
            eye_mask=eye_mask,
        )
