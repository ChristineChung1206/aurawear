"""M7. SeasonFeatureExtractor - Extract undertone/value/chroma/contrast features."""

from __future__ import annotations

from typing import Dict, Any
from dataclasses import dataclass

import numpy as np


@dataclass
class SeasonFeatures:
    """Extracted seasonal color features."""
    undertone: str  # "cool", "warm", "neutral"
    value: str  # "light", "medium", "deep"
    chroma: str  # "soft", "true", "bright"
    contrast: str  # "low", "medium", "high"
    features_dict: Dict[str, Any]  # all numeric features
    confidence: float  # 0-1


class SeasonFeatureExtractor:
    """
    Extract seasonal color features from representative colors.
    
    Features:
      1. Undertone (cool/warm/neutral): from a* and b* channels
      2. Value (light/medium/deep): from L channel average
      3. Chroma (soft/true/bright): from saturation (chroma = sqrt(a*^2 + b*^2))
      4. Contrast (low/medium/high): from difference in L values
    """
    
    def __init__(self):
        # Feature thresholds (tunable)
        self.undertone_threshold = 5.0  # for a* and b*
        # Value: CIE Lab skin L* typically 35-80.
        # Adjusted so medium sits in the realistic 50-65 range.
        self.value_thresholds = [50, 65]  # deep < 50, medium 50-65, light > 65
        self.chroma_thresholds = [15, 35]  # soft < 15, true 15-35, bright > 35
        self.contrast_thresholds = [10, 20]  # low < 10, medium 10-20, high > 20
    
    def extract(
        self,
        skin_lab: np.ndarray,
        skin_conf: float,
        hair_lab: np.ndarray = None,
        hair_conf: float = 0.0,
        eye_lab: np.ndarray = None,
        eye_conf: float = 0.0,
        pose: Dict[str, float] = None,
        lighting_quality: float = 0.8,
    ) -> SeasonFeatures:
        """
        Extract season features from skin/hair/eye colors.
        
        Args:
            skin_lab: (3,) skin color in Lab
            skin_conf: confidence 0-1
            hair_lab: (3,) hair color in Lab, optional
            hair_conf: confidence 0-1
            eye_lab: (3,) eye color in Lab, optional
            eye_conf: confidence 0-1
            pose: dict with yaw/pitch/roll (degrees)
            lighting_quality: 0-1 overall lighting quality
            
        Returns:
            SeasonFeatures with extracted features and classifications
        """
        L_s, a_s, b_s = skin_lab
        
        # === Undertone ===
        undertone, undertone_score = self._classify_undertone(a_s, b_s)
        
        # === Value ===
        # Combine skin L with hair if available
        if hair_lab is not None and hair_conf > 0.3:
            L_h = hair_lab[0]
            value_combo = 0.7 * L_s + 0.3 * L_h
        else:
            value_combo = L_s
        
        value, value_score = self._classify_value(value_combo)
        
        # === Chroma ===
        # Use skin chroma primarily
        chroma_s = np.sqrt(a_s**2 + b_s**2)
        chroma, chroma_score = self._classify_chroma(chroma_s)
        
        # === Contrast ===
        if hair_lab is not None and hair_conf > 0.3:
            L_h = hair_lab[0]
            delta_L = abs(L_s - L_h)
        elif eye_lab is not None and eye_conf > 0.3:
            L_e = eye_lab[0]
            delta_L = abs(L_s - L_e)
        else:
            # Default: assume some contrast
            delta_L = 20
        
        contrast, contrast_score = self._classify_contrast(delta_L)
        
        # === Overall Confidence ===
        # Weighted average (NOT multiplicative) so medium features don't crush score.
        # Each component contributes proportionally rather than multiplying.
        w_avail, w_consist, w_pose, w_light = 0.40, 0.25, 0.15, 0.20
        availability_score = (
            skin_conf
            + (hair_conf if hair_lab is not None else 0.5)
            + (eye_conf if eye_lab is not None else 0.5)
        ) / 3.0
        consistency_score = self._compute_consistency(skin_lab, hair_lab, eye_lab)
        pose_quality = self._pose_quality(pose)

        overall_conf = (
            w_avail * availability_score
            + w_consist * consistency_score
            + w_pose * pose_quality
            + w_light * lighting_quality
        )
        overall_conf = np.clip(overall_conf, 0.0, 1.0)
        
        features_dict = {
            "L_skin": float(L_s),
            "a_skin": float(a_s),
            "b_skin": float(b_s),
            "chroma_skin": float(chroma_s),
            "undertone_score": float(undertone_score),
            "value_score": float(value_score),
            "chroma_score": float(chroma_score),
            "contrast_score": float(contrast_score),
            "availability_score": float(availability_score),
            "consistency_score": float(consistency_score),
            "pose_quality": float(pose_quality),
            "lighting_quality": float(lighting_quality),
        }
        
        if hair_lab is not None:
            L_h, a_h, b_h = hair_lab
            features_dict.update({
                "L_hair": float(L_h),
                "a_hair": float(a_h),
                "b_hair": float(b_h),
                "chroma_hair": float(np.sqrt(a_h**2 + b_h**2)),
            })
        
        if eye_lab is not None:
            L_e, a_e, b_e = eye_lab
            features_dict.update({
                "L_eye": float(L_e),
                "a_eye": float(a_e),
                "b_eye": float(b_e),
                "chroma_eye": float(np.sqrt(a_e**2 + b_e**2)),
            })
        
        return SeasonFeatures(
            undertone=undertone,
            value=value,
            chroma=chroma,
            contrast=contrast,
            features_dict=features_dict,
            confidence=float(overall_conf),
        )
    
    def _classify_undertone(self, a_star: float, b_star: float) -> tuple:
        """
        Classify undertone from a* and b* channels.

        Personal color analysis convention:
          - b* > 0 → yellowish warmth; the higher, the warmer
          - b* < 0 → bluish coolness
          - a* > 0 → reddish (warm pink vs cool pink depends on b*)

        We use a weighted index: warmth = 0.35*a* + 0.65*b*
          warmth >  threshold → warm
          warmth < -threshold → cool
          else                → neutral
        """
        threshold = self.undertone_threshold

        # Warmth index: b* dominates (yellow-blue axis) with a* contribution
        warmth = 0.35 * a_star + 0.65 * b_star

        if warmth > threshold:
            undertone = "warm"
            undertone_score = min(warmth / 25.0, 1.0)
        elif warmth < -threshold:
            undertone = "cool"
            undertone_score = min(abs(warmth) / 25.0, 1.0)
        else:
            undertone = "neutral"
            undertone_score = 0.5

        return undertone, undertone_score
    
    def _classify_value(self, L: float) -> tuple:
        """Classify value (lightness) from L channel.
        
        CIE Lab: L*=0 is black (deep), L*=100 is white (light).
        So low L → deep, high L → light.
        """
        L_low, L_high = self.value_thresholds
        
        if L < L_low:
            value = "deep"
            value_score = 1.0 - (L / L_low)
        elif L > L_high:
            value = "light"
            value_score = (L - L_high) / (100 - L_high)
        else:
            value = "medium"
            value_score = 0.5
        
        return value, value_score
    
    def _classify_chroma(self, chroma: float) -> tuple:
        """Classify chroma (saturation) from sqrt(a*^2 + b*^2)."""
        C_low, C_high = self.chroma_thresholds
        
        if chroma < C_low:
            chroma_class = "soft"
            chroma_score = 1.0 - (chroma / C_low)
        elif chroma > C_high:
            chroma_class = "bright"
            chroma_score = (chroma - C_high) / 50
        else:
            chroma_class = "true"
            chroma_score = 0.5
        
        return chroma_class, chroma_score
    
    def _classify_contrast(self, delta_L: float) -> tuple:
        """Classify contrast from L difference."""
        C_low, C_high = self.contrast_thresholds
        
        if delta_L < C_low:
            contrast = "low"
            contrast_score = delta_L / C_low
        elif delta_L > C_high:
            contrast = "high"
            contrast_score = min(delta_L / 40, 1.0)
        else:
            contrast = "medium"
            contrast_score = 0.5
        
        return contrast, contrast_score
    
    @staticmethod
    def _compute_consistency(skin_lab: np.ndarray, hair_lab: np.ndarray = None, eye_lab: np.ndarray = None) -> float:
        """
        Check consistency between skin, hair, eye colors.
        
        High score if colors are in consistent undertone/value range.
        """
        if hair_lab is None and eye_lab is None:
            return 1.0
        
        consistency_scores = []
        
        if hair_lab is not None:
            # Check if hair and skin undertones are similar
            delta_ab = np.sqrt((skin_lab[1] - hair_lab[1])**2 + (skin_lab[2] - hair_lab[2])**2)
            consistency = 1.0 - min(delta_ab / 30, 1.0)
            consistency_scores.append(consistency)
        
        if eye_lab is not None:
            delta_ab = np.sqrt((skin_lab[1] - eye_lab[1])**2 + (skin_lab[2] - eye_lab[2])**2)
            consistency = 1.0 - min(delta_ab / 40, 1.0)
            consistency_scores.append(consistency)
        
        if consistency_scores:
            return float(np.mean(consistency_scores))
        return 1.0
    
    @staticmethod
    def _pose_quality(pose: Dict[str, float] = None) -> float:
        """
        Rate pose quality for color analysis.
        
        Penalty for extreme head turns (yaw > 30°) or tilts.
        """
        if pose is None:
            return 0.8
        
        yaw = abs(pose.get("yaw", 0))
        pitch = abs(pose.get("pitch", 0))
        roll = abs(pose.get("roll", 0))
        
        quality = 1.0
        if yaw > 30:
            quality *= (1.0 - (yaw - 30) / 60)
        if pitch > 30:
            quality *= (1.0 - (pitch - 30) / 60)
        if roll > 20:
            quality *= (1.0 - (roll - 20) / 60)
        
        return float(np.clip(quality, 0.0, 1.0))
