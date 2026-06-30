"""M8. Season12RuleEngine - Classify to 12-season type."""

from __future__ import annotations

from typing import Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class Season12Result:
    """12-season classification result."""
    season_12: str  # e.g., "Light Spring", "True Summer", "Deep Autumn", "Bright Winter"
    macro_season: str  # Spring/Summer/Autumn/Winter
    sub_season: str  # Light/True/Soft/Bright/Deep
    confidence: float  # 0-1
    diagnostics: Dict[str, Any]


class Season12RuleEngine:
    """
    Classify to 12-season type using features and rules.
    
    12-Season Types:
      Spring: Light Spring, True Spring, Bright Spring
      Summer: Light Summer, True Summer, Soft Summer
      Autumn: Soft Autumn, True Autumn, Deep Autumn
      Winter: Bright Winter, True Winter, Deep Winter
    """
    
    # 12-Season definitions: (macro_season, sub_season) → season_12
    SEASON_MAP = {
        ("Spring", "light"): "Light Spring",
        ("Spring", "true"): "True Spring",
        ("Spring", "bright"): "Bright Spring",
        ("Summer", "light"): "Light Summer",
        ("Summer", "true"): "True Summer",
        ("Summer", "soft"): "Soft Summer",
        ("Autumn", "soft"): "Soft Autumn",
        ("Autumn", "true"): "True Autumn",
        ("Autumn", "deep"): "Deep Autumn",
        ("Winter", "bright"): "Bright Winter",
        ("Winter", "true"): "True Winter",
        ("Winter", "deep"): "Deep Winter",
    }
    
    def classify(
        self,
        undertone: str,
        value: str,
        chroma: str,
        contrast: str,
        features_dict: Dict[str, Any],
        feature_confidence: float,
    ) -> Season12Result:
        """
        Classify to 12-season using macro and sub-season rules.
        
        Args:
            undertone: "cool", "warm", "neutral"
            value: "light", "medium", "deep"
            chroma: "soft", "true", "bright"
            contrast: "low", "medium", "high"
            features_dict: numeric features for diagnostics
            feature_confidence: from SeasonFeatureExtractor
            
        Returns:
            Season12Result
        """
        
        # === Step 1: Macro season ===
        # Warm undertone: Spring (light/bright) vs Autumn (deep/soft/muted)
        # Cool undertone: Summer (light/soft) vs Winter (deep/bright/high-contrast)
        if undertone == "warm":
            if value == "deep":
                macro_season = "Autumn"
            elif chroma == "soft":
                macro_season = "Autumn"   # Soft Autumn: warm + muted
            elif value == "medium" and contrast != "high":
                # Medium warmth — Autumn unless bright+high-contrast
                macro_season = "Autumn"
            else:
                macro_season = "Spring"   # light/bright warm → Spring
        elif undertone == "cool":
            if value == "deep":
                macro_season = "Winter"
            elif chroma == "bright" or contrast == "high":
                macro_season = "Winter"   # Bright Winter: cool + vivid
            elif value == "medium":
                macro_season = "Summer"   # medium cool → Summer
            else:
                macro_season = "Summer"
        else:
            # Neutral undertone: use chroma and value to decide
            if value == "deep":
                macro_season = "Autumn"
            elif value == "light" and chroma == "bright":
                macro_season = "Spring"
            elif value == "light" and chroma == "soft":
                macro_season = "Summer"
            elif chroma == "soft":
                macro_season = "Summer"
            elif chroma == "bright" and contrast == "high":
                macro_season = "Winter"
            elif value == "light":
                macro_season = "Summer"   # neutral + light → Summer (not Spring)
            else:
                # neutral + medium + true chroma → Autumn (earthy default)
                macro_season = "Autumn"
        
        # === Step 2: Sub-season ===
        sub_season = self._classify_sub_season(macro_season, value, chroma, contrast)
        
        # === Step 3: Look up 12-season ===
        season_key = (macro_season, sub_season)
        season_12 = self.SEASON_MAP.get(season_key, f"{macro_season} (Custom)")
        
        # If key not found, construct manually
        if "Custom" in season_12:
            season_12 = f"{sub_season} {macro_season}"
        
        # === Step 4: Compute confidence ===
        confidence = self._compute_confidence(
            undertone,
            value,
            chroma,
            contrast,
            feature_confidence,
        )
        
        diagnostics = {
            "macro_season_rule": f"{undertone} undertone → {macro_season}",
            "sub_season_rule": f"value={value}, chroma={chroma}, contrast={contrast} → {sub_season}",
            "undertone_type": undertone,
            "value_type": value,
            "chroma_type": chroma,
            "contrast_type": contrast,
            **features_dict,
        }
        
        return Season12Result(
            season_12=season_12,
            macro_season=macro_season,
            sub_season=sub_season,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            diagnostics=diagnostics,
        )
    
    @staticmethod
    def _classify_sub_season(
        macro_season: str,
        value: str,
        chroma: str,
        contrast: str,
    ) -> str:
        """
        Classify sub-season within macro season.
        
        Spring: Light/True/Bright
        Summer: Light/True/Soft
        Autumn: Soft/True/Deep
        Winter: Bright/True/Deep
        """
        
        if macro_season == "Spring":
            if value == "light":
                return "light"
            elif chroma == "bright":
                return "bright"
            else:
                return "true"
        
        elif macro_season == "Summer":
            if value == "light":
                return "light"
            elif chroma == "soft":
                return "soft"
            else:
                return "true"
        
        elif macro_season == "Autumn":
            if value == "deep":
                return "deep"
            elif chroma == "soft":
                return "soft"
            else:
                return "true"
        
        elif macro_season == "Winter":
            if value == "deep":
                return "deep"
            elif chroma == "bright":
                return "bright"
            else:
                return "true"
        
        return "true"
    
    @staticmethod
    def _compute_confidence(
        undertone: str,
        value: str,
        chroma: str,
        contrast: str,
        feature_confidence: float,
    ) -> float:
        """
        Compute final confidence for season classification.

        Uses a weighted average (NOT multiplicative) so that typical
        "medium" features don't crush confidence to near-zero.

        Lower confidence if:
          - Neutral undertone (less decisive)
          - Medium value/chroma/contrast (ambiguous)
          - Low feature_confidence
        """

        # Undertone confidence
        if undertone == "neutral":
            undertone_conf = 0.55
        else:
            undertone_conf = 0.90

        # Feature decisiveness — each distinctive feature boosts score
        decisive_count = 0
        for feat in [value, chroma, contrast]:
            if feat in ["light", "deep", "bright", "soft", "high", "low"]:
                decisive_count += 1
        # 0 decisive → 0.40, 1 → 0.60, 2 → 0.80, 3 → 1.0
        decisiveness = 0.40 + 0.20 * decisive_count

        # Weighted average of three components
        w_ut, w_dec, w_feat = 0.30, 0.30, 0.40
        confidence = (
            w_ut * undertone_conf
            + w_dec * decisiveness
            + w_feat * feature_confidence
        )

        return float(np.clip(confidence, 0.0, 1.0))


class EyeColorClassifier:
    """
    Classify eye color into categories: brown, blue, green, hazel, amber, grey
    
    Based on Lab color values and hue angle.
    """
    
    # Eye color classification thresholds
    EYE_COLOR_RULES = {
        "brown": {
            "hue_range": [-30, 50],  # reddish-yellowish (擴大範圍)
            "chroma_range": [8, 150],  # 限制飽和度範圍，避免太高
            "L_range": [20, 75],
        },
        "hazel": {
            "hue_range": [40, 100],  # yellowish-greenish (擴大黃色範圍)
            "chroma_range": [15, 120],
            "L_range": [40, 75],
        },
        "green": {
            "hue_range": [90, 160],  # greenish (擴大綠色範圍)
            "chroma_range": [15, 120],
            "L_range": [40, 80],
        },
        "amber": {
            "hue_range": [20, 70],  # yellowish-reddish
            "chroma_range": [30, 150],  # 要求更高的飽和度
            "L_range": [50, 85],
        },
        "blue": {
            "hue_range": [200, 320],  # bluish-cyan (擴大藍色範圍 180-280→200-320)
            "chroma_range": [8, 100],  # 允許低飽和度的藍眼（淡藍）
            "L_range": [50, 95],  # 藍眼通常較亮
        },
        "grey": {
            "hue_range": None,  # no hue
            "chroma_range": [0, 15],  # 低飽和度
            "L_range": [40, 85],
        },
    }
    
    @classmethod
    def classify_eye_color(cls, eye_lab: np.ndarray) -> Tuple[str, float]:
        """
        Classify eye color from Lab values.
        
        Args:
            eye_lab: (3,) eye color in Lab
            
        Returns:
            (color_name, confidence)
        """
        L, a_star, b_star = eye_lab
        
        # Compute chroma and hue
        chroma = np.sqrt(a_star**2 + b_star**2)
        hue_angle = np.degrees(np.arctan2(b_star, a_star))
        
        # Normalize hue to [0, 360)
        if hue_angle < 0:
            hue_angle += 360
        
        best_match = None
        best_score = -1
        
        for color_name, rules in cls.EYE_COLOR_RULES.items():
            score = cls._match_score(
                hue_angle, chroma, L,
                rules,
            )
            
            if score > best_score:
                best_score = score
                best_match = color_name
        
        confidence = np.clip(best_score, 0.0, 1.0)
        
        return best_match or "brown", confidence
    
    @staticmethod
    def _match_score(hue: float, chroma: float, L: float, rules: Dict[str, Any]) -> float:
        """Compute match score for eye color rules."""
        score = 1.0
        
        # Hue match
        if rules["hue_range"] is not None:
            hue_low, hue_high = rules["hue_range"]
            if hue_low <= hue < hue_high or (hue_high < hue_low and (hue >= hue_low or hue < hue_high)):
                # Within range
                pass
            else:
                # Outside range: compute penalty
                min_dist = min(abs(hue - hue_low), abs(hue - hue_high))
                score *= max(0, 1.0 - min_dist / 60)
        
        # Chroma match
        chroma_low, chroma_high = rules["chroma_range"]
        if chroma < chroma_low:
            score *= max(0, 1.0 - (chroma_low - chroma) / chroma_low)
        elif chroma_high is not None and chroma > chroma_high:
            score *= max(0, 1.0 - (chroma - chroma_high) / chroma_high)
        
        # L match
        L_low, L_high = rules["L_range"]
        if L < L_low:
            score *= max(0, 1.0 - (L_low - L) / L_low)
        elif L > L_high:
            score *= max(0, 1.0 - (L - L_high) / (100 - L_high))
        
        return score
