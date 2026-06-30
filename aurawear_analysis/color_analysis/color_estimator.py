"""M6. RepresentativeColorEstimator - Estimate representative colors using KMeans."""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class ColorEstimationResult:
    """Result of color estimation for a ROI."""
    color_lab: np.ndarray  # (3,) representative color in Lab
    color_rgb: np.ndarray  # (3,) in 0-255 range
    color_hex: str  # hex string #RRGGBB
    confidence: float  # 0-1 based on cluster quality
    cluster_info: Dict[str, Any]  # debug info


class RepresentativeColorEstimator:
    """
    Estimate representative color from filtered pixels using KMeans.
    
    Algorithm:
      1. If pixel count >= N_min: use KMeans(k=2~3)
      2. Otherwise: use median
      3. Sanity checks on resulting cluster
    """
    
    def __init__(self, n_min: int = 50, k_cluster: int = 2):  # 進一步降低 n_min（100→50）
        self.n_min = n_min
        self.k_cluster = k_cluster
        self.kmeans_n_init = 20  # 增加初始化次數以提高穩定性
    
    def estimate(
        self,
        pixels_lab: np.ndarray,
        pixels_bgr: Optional[np.ndarray] = None,
        roi_name: str = "skin",
        valid_ratio: float = 1.0,
    ) -> Optional[ColorEstimationResult]:
        """
        Estimate representative color from Lab pixels.
        
        Args:
            pixels_lab: (N, 3) pixels in Lab
            roi_name: 'skin', 'hair', or 'eye'
            valid_ratio: fraction of valid pixels (for confidence)
            
        Returns:
            ColorEstimationResult or None if estimation fails
        """
        if len(pixels_lab) == 0:
            return None
        
        # Choose between KMeans and median
        if len(pixels_lab) >= self.n_min:
            # Use KMeans
            representative_lab = self._kmeans_estimation(pixels_lab, roi_name, pixels_bgr=pixels_bgr)
        else:
            # Use median (fallback)
            representative_lab = np.median(pixels_lab, axis=0)
        
        if representative_lab is None:
            return None
        
        # Sanity check
        if not self._sanity_check(representative_lab, roi_name, pixels_lab):
            # Use median as fallback
            representative_lab = np.median(pixels_lab, axis=0)
        
        # Convert Lab to RGB to Hex
        # Internal Lab is CIE Lab: L in [0,100], a/b centered at 0.
        # OpenCV expects Lab 8-bit: L in [0,255], a/b in [0,255] with 128 offset.
        L_star, a_star, b_star = representative_lab.astype(np.float32)
        L_cv = np.clip(L_star * (255.0 / 100.0), 0, 255)
        a_cv = np.clip(a_star + 128.0, 0, 255)
        b_cv = np.clip(b_star + 128.0, 0, 255)
        lab_8bit = np.array([[[L_cv, a_cv, b_cv]]], dtype=np.uint8)
        bgr_8bit = cv2.cvtColor(lab_8bit, cv2.COLOR_LAB2BGR)[0, 0]
        rgb_8bit = bgr_8bit[::-1]
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb_8bit[0]), int(rgb_8bit[1]), int(rgb_8bit[2]))
        
        # Compute confidence
        confidence = self._compute_confidence(
            representative_lab,
            pixels_lab,
            len(pixels_lab),
            valid_ratio,
            roi_name,
        )
        
        result = ColorEstimationResult(
            color_lab=representative_lab,
            color_rgb=rgb_8bit,
            color_hex=hex_color,
            confidence=float(confidence),
            cluster_info={
                "method": "kmeans" if len(pixels_lab) >= self.n_min else "median",
                "pixel_count": len(pixels_lab),
                "valid_ratio": float(valid_ratio),
            },
        )
        
        return result
    
    def _kmeans_estimation(
        self,
        pixels_lab: np.ndarray,
        roi_name: str,
        pixels_bgr: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Use KMeans to find representative color.
        ROI-specific selection:
          - skin: choose cluster whose L* is closest to median L* (avoids shadows/highlights)
          - hair: prefer brighter cluster (helps blonde) while avoiding extreme highlights
          - eye: choose cluster with highest saturation (iris) if BGR is available
        """
        try:
            n_clusters = self.k_cluster
            if roi_name == "eye":
                n_clusters = max(n_clusters, 3)
            kmeans = KMeans(n_clusters=n_clusters, n_init=self.kmeans_n_init, random_state=42)
            kmeans.fit(pixels_lab)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_
            
            # Select cluster with most pixels (largest by count)
            labels = kmeans.labels_
            cluster_counts = np.bincount(labels, minlength=n_clusters)

            # Default: largest cluster
            best_cluster = int(np.argmax(cluster_counts))

            if roi_name == "skin":
                L_all = pixels_lab[:, 0]
                L_med = float(np.median(L_all))
                # pick cluster center closest to median lightness
                best_cluster = int(np.argmin(np.abs(centers[:, 0] - L_med)))

            elif roi_name == "hair":
                # Prefer brighter cluster (blonde/light hair), but avoid extreme highlights.
                L_centers = centers[:, 0]
                # Penalize L* too high (likely specular)
                penalty = np.where(L_centers > 92.0, (L_centers - 92.0) * 5.0, 0.0)
                score = L_centers - penalty
                best_cluster = int(np.argmax(score))

            elif roi_name == "eye" and pixels_bgr is not None and len(pixels_bgr) == len(pixels_lab):
                # Iris tends to have higher saturation than sclera/skin.
                # Choose cluster by mean saturation (HSV S) with mild brightness constraints.
                pixels_bgr_u8 = pixels_bgr.astype(np.uint8).reshape(-1, 1, 3)
                pixels_hsv = cv2.cvtColor(pixels_bgr_u8, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                S = pixels_hsv[:, 1].astype(np.float32)
                V = pixels_hsv[:, 2].astype(np.float32)

                best_score = -1.0
                best_idx = best_cluster
                for c in range(n_clusters):
                    idx = labels == c
                    if idx.sum() < 5:
                        continue
                    s_mean = float(S[idx].mean())
                    v_mean = float(V[idx].mean())
                    # avoid very dark (pupil) and very bright (sclera highlights)
                    if v_mean < 35 or v_mean > 245:
                        continue
                    # combine saturation with a slight preference for mid brightness
                    score = s_mean - 0.2 * abs(v_mean - 140)
                    if score > best_score:
                        best_score = score
                        best_idx = int(c)
                best_cluster = best_idx

            representative = centers[best_cluster]
            return representative
        
        except Exception as e:
            print(f"KMeans estimation failed: {e}")
            return None
    
    @staticmethod
    def _sanity_check(representative_lab: np.ndarray, roi_name: str, pixels_lab: np.ndarray) -> bool:
        """
        Check if estimated color is reasonable.
        
        Rules per ROI:
          - skin: L in reasonable range (not too dark/bright)
          - hair: not too similar to grey (a* and b* not too close to 0)
          - eye: reasonable saturation (a*, b* not too close to 0)
        """
        L, a_star, b_star = representative_lab
        
        if roi_name == "skin":
            # Skin L should be in reasonable range
            if L < 20 or L > 90:
                return False
            return True
        
        elif roi_name == "hair":
            # Hair should have some chroma (not grey)
            chroma = np.sqrt(a_star**2 + b_star**2)
            if chroma < 5:
                return False
            return True
        
        elif roi_name == "eye":
            # Eye should have reasonable chroma
            chroma = np.sqrt(a_star**2 + b_star**2)
            if chroma < 3:
                return False
            return True
        
        return True
    
    @staticmethod
    def _compute_confidence(
        representative_lab: np.ndarray,
        pixels_lab: np.ndarray,
        pixel_count: int,
        valid_ratio: float,
        roi_name: str,
    ) -> float:
        """
        Compute confidence score based on:
          1. Pixel count (more = higher confidence)
          2. Valid ratio (higher = better)
          3. Color spread (lower spread = higher confidence)
        """
        # Pixel count confidence
        if pixel_count >= 500:
            count_conf = 1.0
        elif pixel_count >= 300:
            count_conf = 0.85
        elif pixel_count >= 100:
            count_conf = 0.7
        else:
            count_conf = 0.5
        
        # Valid ratio confidence
        ratio_conf = valid_ratio
        
        # Color spread confidence (1 - normalized std dev)
        distances = np.linalg.norm(pixels_lab - representative_lab, axis=1)
        spread_ratio = distances.mean() / (distances.max() + 1e-6)
        spread_conf = 1.0 - min(spread_ratio, 1.0)
        
        # Combine: average of three factors
        confidence = (count_conf + ratio_conf + spread_conf) / 3.0
        
        return float(np.clip(confidence, 0.0, 1.0))
