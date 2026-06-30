"""M1. ImageLoader - Load and preprocess images with EXIF handling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


class ImageLoader:
    """
    Load images from file or bytes, handle EXIF rotation, and resize.
    
    Output:
      - img_bgr: BGR image (H, W, 3) as np.ndarray
      - meta: dict with image_id, hash, original_size, etc.
    """
    
    def __init__(self, max_side: int = 900):
        self.max_side = max_side
    
    def load(self, image_path_or_bytes: str | Path | bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load image and return BGR array + metadata.
        
        Args:
            image_path_or_bytes: path, Path, or bytes
            
        Returns:
            (img_bgr, meta) where meta includes image_id, hash, orig_size, final_size
        """
        # Load image
        if isinstance(image_path_or_bytes, bytes):
            image_bytes = image_path_or_bytes
            img_pil = Image.open(image_path_or_bytes if isinstance(image_path_or_bytes, bytes) else None)
            # 如果是 bytes，需要用 BytesIO 包裝
            from io import BytesIO
            img_pil = Image.open(BytesIO(image_path_or_bytes))
        else:
            path = Path(image_path_or_bytes)
            img_pil = Image.open(path)
        
        # Get original size before rotation
        orig_w, orig_h = img_pil.size
        
        # Handle EXIF rotation
        img_pil = self._fix_exif_rotation(img_pil)
        
        # Convert to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        # Resize maintaining aspect ratio
        h, w = img_bgr.shape[:2]
        if max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Generate image_id and hash
        if isinstance(image_path_or_bytes, bytes):
            image_hash = hashlib.md5(image_path_or_bytes).hexdigest()[:12]
            image_id = f"img_{image_hash}"
        else:
            image_hash = hashlib.md5(img_bgr.tobytes()).hexdigest()[:12]
            image_id = f"img_{image_hash}"
        
        meta = {
            "image_id": image_id,
            "hash": image_hash,
            "original_size": (orig_h, orig_w),
            "final_size": img_bgr.shape[:2],
            "exif_rotated": True,
        }
        
        return img_bgr, meta
    
    @staticmethod
    def _fix_exif_rotation(img_pil: Image.Image) -> Image.Image:
        """
        Handle EXIF orientation tag (common in selfies).
        
        Orientation values:
          1: Normal
          3: 180°
          6: 90° CCW (rotated left)
          8: 90° CW (rotated right)
        """
        try:
            exif_data = img_pil._getexif()
            if exif_data is None:
                return img_pil
            
            for tag, value in exif_data.items():
                if TAGS.get(tag) == "Orientation":
                    if value == 1:
                        return img_pil
                    elif value == 3:
                        return img_pil.rotate(180, expand=True)
                    elif value == 6:
                        return img_pil.rotate(270, expand=True)
                    elif value == 8:
                        return img_pil.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        
        return img_pil
