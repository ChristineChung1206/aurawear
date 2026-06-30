"""M5. PixelFilterAndNormalize - Filter and normalize pixels in Lab color space."""

from __future__ import annotations

from typing import Tuple, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PixelFilterResult:
    """Result of pixel filtering and normalization."""
    pixels_lab: np.ndarray  # (N, 3) valid pixels in Lab color space
    pixels_bgr: np.ndarray  # (N, 3) valid pixels in BGR (uint8)
    valid_ratio: float  # fraction of valid pixels
    original_count: int
    valid_count: int
    lighting_stats: Dict[str, float]  # L_mean, L_std, etc.


class PixelFilterAndNormalize:
    """
    Filter pixels by lighting conditions and normalize L channel.
    
    Filters:
      - Shadow filter: L < L_low (p5)
      - Overexposed filter: L > L_high (p95)
      - Skin-like filter (for hair ROI): remove pixels too similar to skin
      - Highlight filter (for eye ROI): remove very bright pixels (V > threshold and S < threshold in HSV)
    
    Normalization:
      - L channel: clamp to (p5, p95) then scale to fixed range
      - a*, b* channels: no normalization
    """
    
    def __init__(self):
        self.L_low_percentile = 5
        self.L_high_percentile = 95
        self.skin_like_threshold = 20.0  # 膚色相似度閾值
        self.hair_highlight_threshold = 240  # 反光過濾：亮度門檻
        self.eye_highlight_saturation = 40  # 眼睛反光：飽和度門檻
        self.skin_L_percentile_low = 10  # 膚色亮度：p10（而不是 p5）- 保留更多像素
        self.skin_L_percentile_high = 90  # 膚色亮度：p90（而不是 p95）
    
    def filter_and_normalize(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        roi_name: str = "skin",
        skin_lab_ref: np.ndarray = None,
    ) -> PixelFilterResult:
        """
        Filter pixels from ROI and normalize.
        
        Args:
            img_bgr: BGR image (H, W, 3)
            mask: binary mask (H, W) indicating ROI
            roi_name: 'skin', 'hair', or 'eye'
            skin_lab_ref: reference skin color in Lab (for hair ROI)
            
        Returns:
            PixelFilterResult with filtered and normalized pixels
        """
        h, w = img_bgr.shape[:2]
        
        # Extract pixels from ROI
        roi_pixels_bgr = img_bgr[mask > 0].reshape(-1, 3)
        original_count = len(roi_pixels_bgr)
        
        if original_count == 0:
            # Empty ROI
            return PixelFilterResult(
                pixels_lab=np.empty((0, 3), dtype=np.float32),
                pixels_bgr=np.empty((0, 3), dtype=np.uint8),
                valid_ratio=0.0,
                original_count=0,
                valid_count=0,
                lighting_stats={},
            )
        
        # Convert to Lab
        roi_pixels_bgr_uint8 = roi_pixels_bgr.astype(np.uint8).reshape(1, -1, 3)
        # OpenCV Lab: L in [0,255], a in [0,255] with 128 as 0, b in [0,255] with 128 as 0
        roi_pixels_lab_cv = cv2.cvtColor(roi_pixels_bgr_uint8, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        # Convert to CIE Lab: L in [0,100], a/b in roughly [-128,127]
        roi_pixels_lab = roi_pixels_lab_cv.copy()
        roi_pixels_lab[:, 0] = roi_pixels_lab[:, 0] * (100.0 / 255.0)
        roi_pixels_lab[:, 1] = roi_pixels_lab[:, 1] - 128.0
        roi_pixels_lab[:, 2] = roi_pixels_lab[:, 2] - 128.0
        
        # Extract L, a*, b* channels
        L = roi_pixels_lab[:, 0]
        a_star = roi_pixels_lab[:, 1]
        b_star = roi_pixels_lab[:, 2]
        
        # Lighting filter: remove shadows and overexposed
        # Use different percentiles for skin (more conservative) vs hair (more aggressive)
        if roi_name == "skin":
            L_low = np.percentile(L, self.skin_L_percentile_low)
            L_high = np.percentile(L, self.skin_L_percentile_high)
        else:
            L_low = np.percentile(L, self.L_low_percentile)
            L_high = np.percentile(L, self.L_high_percentile)
        
        lighting_mask = (L >= L_low) & (L <= L_high)
        
        # Apply specific filters per ROI
        if roi_name == "hair" and skin_lab_ref is not None:
            # Skin-like filter for hair: remove pixels similar to skin
            delta_E = np.sqrt((L - skin_lab_ref[0])**2 + 
                              (a_star - skin_lab_ref[1])**2 + 
                              (b_star - skin_lab_ref[2])**2)
            skin_like_mask = delta_E >= self.skin_like_threshold
            valid_mask = lighting_mask & skin_like_mask
        
        elif roi_name == "eye":
            # Eye ROI is easy to contaminate with skin/sclera. Use:
            # 1) mild highlight removal
            # 2) adaptive chroma threshold (keep more "iris-like" pixels)
            # 3) optional skin-like removal if skin reference is provided
            roi_pixels_hsv = cv2.cvtColor(roi_pixels_bgr_uint8, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
            S = roi_pixels_hsv[:, 1]
            V = roi_pixels_hsv[:, 2]

            # Specular highlights are very bright and low saturation
            highlight_mask = ~((V > 240) & (S < 60))

            chroma = np.sqrt((a_star**2 + b_star**2))
            # Adaptive: keep top chroma pixels within the ROI (iris tends to be higher chroma than sclera)
            c_thr = float(np.percentile(chroma, 60))
            c_thr = max(5.0, c_thr)
            chroma_mask = chroma >= c_thr

            valid_mask = lighting_mask & highlight_mask & chroma_mask

            # Exclude very dark pixels (pupil/shadows) using adaptive L* threshold
            L_thr = float(np.percentile(L, 30))
            L_thr = max(15.0, L_thr)
            valid_mask = valid_mask & (L >= L_thr)

            if skin_lab_ref is not None:
                delta_E = np.sqrt((L - skin_lab_ref[0])**2 +
                                  (a_star - skin_lab_ref[1])**2 +
                                  (b_star - skin_lab_ref[2])**2)
                # For eyes, be stricter about removing skin-like pixels
                skin_like_mask = delta_E >= max(10.0, self.skin_like_threshold)
                valid_mask = valid_mask & skin_like_mask
        
        else:
            # Default (skin): just use lighting filter
            valid_mask = lighting_mask
        
        # Extract valid pixels
        valid_pixels_lab = roi_pixels_lab[valid_mask]
        valid_pixels_bgr = roi_pixels_bgr[valid_mask].astype(np.uint8)
        valid_count = len(valid_pixels_lab)

        # Keep absolute CIE Lab values (do NOT normalize L per ROI).
        # Per-ROI normalization changes actual colors and harms hex output + eye/season rules.
        if valid_count > 0:
            L_valid = valid_pixels_lab[:, 0]
            lighting_stats = {
                "L_mean": float(L_valid.mean()),
                "L_std": float(L_valid.std()),
                "L_p5": float(np.percentile(L_valid, 5)),
                "L_p95": float(np.percentile(L_valid, 95)),
            }
        else:
            lighting_stats = {}
        
        valid_ratio = valid_count / original_count if original_count > 0 else 0.0
        
        return PixelFilterResult(
            pixels_lab=valid_pixels_lab.astype(np.float32),
            pixels_bgr=valid_pixels_bgr,
            valid_ratio=valid_ratio,
            original_count=original_count,
            valid_count=valid_count,
            lighting_stats=lighting_stats,
        )
