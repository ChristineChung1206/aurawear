"""
Integrated Color Analysis Pipeline - Main entry point.

Combines all 8 modules into a single pipeline for personal color diagnosis.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from dataclasses import dataclass

import cv2
import numpy as np

from .image_loader import ImageLoader
from .face_detector import FaceLandmarkDetector
from .segmenter import PersonSegmenter
from .face_parsing_onnx import FaceParsingONNX
from .roi_mask_builder import ROIMaskBuilder
from .pixel_filter import PixelFilterAndNormalize
from .color_estimator import RepresentativeColorEstimator
from .feature_extractor import SeasonFeatureExtractor
from .season_classifier import Season12RuleEngine, EyeColorClassifier


@dataclass
class PersonalColorDiagnosis:
    """Final diagnosis result."""
    season_12: str  # e.g., "Light Spring"
    season_confidence: float
    skin_color_hex: str
    hair_color_hex: str
    eye_color: str
    eye_color_hex: str  # hex code for eye color
    eye_color_confidence: float
    diagnostics: Dict[str, Any]


class ColorAnalysisPipeline:
    """
    Main pipeline for personal color analysis.
    
    Flow:
      image → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → diagnosis
    """
    
    def __init__(self):
        self.loader = ImageLoader(max_side=900)
        self.face_detector = FaceLandmarkDetector()
        self.segmenter = PersonSegmenter()
        self.face_parser = FaceParsingONNX()
        self.roi_builder = ROIMaskBuilder()
        self.pixel_filter = PixelFilterAndNormalize()
        self.color_estimator = RepresentativeColorEstimator(n_min=50, k_cluster=2)  # 調整為更敏感的參數
        self.feature_extractor = SeasonFeatureExtractor()
        self.season_classifier = Season12RuleEngine()
    
    def diagnose(self, image_path_or_bytes: str | bytes) -> Optional[PersonalColorDiagnosis]:
        """
        Run full diagnosis pipeline on selfie image.
        
        Args:
            image_path_or_bytes: path to image or bytes
            
        Returns:
            PersonalColorDiagnosis or None if diagnosis fails
        """
        
        # M1: Load image
        try:
            img_bgr, img_meta = self.loader.load(image_path_or_bytes)
        except Exception as e:
            print(f"[Error] M1 ImageLoader failed: {e}")
            return None
        
        print(f"✓ M1: Image loaded {img_meta['final_size']}")
        
        # M2: Detect face
        face_result = self.face_detector.detect(img_bgr)
        if face_result is None:
            print("[Error] M2: No face detected")
            return None
        
        print(f"✓ M2: Face detected at bbox {face_result.bbox}")
        
        # M3: Segment person
        person_mask = self.segmenter.segment(img_bgr)
        print(f"✓ M3: Person segmented, mask area {person_mask.sum()}")

        # Optional: face parsing for semantic masks (hair/skin/eye)
        parsing_masks = None
        parsing_result = self.face_parser.parse(img_bgr, face_result.bbox)
        if parsing_result is not None:
            parsing_masks = {
                "skin_mask": parsing_result.skin_mask,
                "hair_mask": parsing_result.hair_mask,
                "eye_mask": parsing_result.eye_mask,
            }
            print(
                "✓ M3.5: Face parsing masks available "
                f"(skin={int(parsing_result.skin_mask.sum())}, "
                f"hair={int(parsing_result.hair_mask.sum())}, "
                f"eye={int(parsing_result.eye_mask.sum())})"
            )
        else:
            if self.face_parser.model_path.exists():
                print("✗ M3.5: Face parsing model found but ONNX runtime unavailable or parsing failed")
            else:
                print("(i) M3.5: Face parsing model not configured; using heuristic ROI masks")
        
        # M4: Build ROI masks
        roi_masks = self.roi_builder.build_masks(
            img_bgr.shape[:2],
            face_result.landmarks,
            face_result.bbox,
            person_mask,
            parsing_masks=parsing_masks,
        )
        print(f"✓ M4: ROI masks built (skin/hair/eye)")
        
        # M5 & M6: Extract representative colors
        skin_color_result = self._extract_color(
            img_bgr,
            roi_masks["skin_mask"],
            "skin",
            None,
        )
        if skin_color_result is None:
            print("[Error] M5-M6: Failed to extract skin color")
            return None
        
        print(f"✓ M5-M6: Skin color extracted {skin_color_result.color_hex}")
        
        # Extract hair color
        hair_color_result = self._extract_color(
            img_bgr,
            roi_masks["hair_mask"],
            "hair",
            skin_color_result.color_lab,
        )
        
        hair_lab = hair_color_result.color_lab if hair_color_result else None
        hair_conf = hair_color_result.confidence if hair_color_result else 0.0
        hair_hex = hair_color_result.color_hex if hair_color_result else "#808080"
        print(f"✓ Hair color: {hair_hex} (conf: {hair_conf:.2f})")
        
        # Extract eye color
        eye_color_result = self._extract_color(
            img_bgr,
            roi_masks["eye_mask"],
            "eye",
            skin_color_result.color_lab,
        )
        
        eye_lab = eye_color_result.color_lab if eye_color_result else None
        eye_conf = eye_color_result.confidence if eye_color_result else 0.0
        eye_hex = eye_color_result.color_hex if eye_color_result else "#808080"
        print(f"✓ Eye color detected (conf: {eye_conf:.2f})")
        
        # Classify eye color
        eye_color_name, eye_color_confidence = "brown", 0.5
        if eye_lab is not None:
            eye_color_name, eye_color_confidence = EyeColorClassifier.classify_eye_color(eye_lab)
        
        print(f"  → {eye_color_name} (conf: {eye_color_confidence:.2f}, hex: {eye_hex})")
        
        # M7: Extract features
        features = self.feature_extractor.extract(
            skin_color_result.color_lab,
            skin_color_result.confidence,
            hair_lab,
            hair_conf,
            eye_lab,
            eye_conf,
            face_result.pose,
            lighting_quality=0.8,
        )
        
        print(f"✓ M7: Features extracted")
        print(f"  - Undertone: {features.undertone}")
        print(f"  - Value: {features.value}")
        print(f"  - Chroma: {features.chroma}")
        print(f"  - Contrast: {features.contrast}")
        print(f"  - Confidence: {features.confidence:.2f}")
        
        # M8: Classify season
        season_result = self.season_classifier.classify(
            features.undertone,
            features.value,
            features.chroma,
            features.contrast,
            features.features_dict,
            features.confidence,
        )
        
        print(f"✓ M8: Season classified → {season_result.season_12}")
        print(f"  - Confidence: {season_result.confidence:.2f}")
        
        # Compile diagnosis
        diagnosis = PersonalColorDiagnosis(
            season_12=season_result.season_12,
            season_confidence=season_result.confidence,
            skin_color_hex=skin_color_result.color_hex,
            hair_color_hex=hair_hex,
            eye_color=eye_color_name,
            eye_color_hex=eye_hex,
            eye_color_confidence=eye_color_confidence,
            diagnostics=season_result.diagnostics,
        )
        
        return diagnosis
    
    def _extract_color(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        roi_name: str,
        skin_lab_ref: np.ndarray = None,
    ):
        """Extract representative color from ROI."""
        
        # Filter pixels
        filter_result = self.pixel_filter.filter_and_normalize(
            img_bgr,
            mask,
            roi_name,
            skin_lab_ref,
        )
        
        if filter_result.valid_count == 0:
            return None
        
        # Estimate color
        color_result = self.color_estimator.estimate(
            pixels_lab=filter_result.pixels_lab,
            pixels_bgr=getattr(filter_result, "pixels_bgr", None),
            roi_name=roi_name,
            valid_ratio=filter_result.valid_ratio,
        )
        
        return color_result
