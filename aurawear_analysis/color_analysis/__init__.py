"""
AI Color Analysis Pipeline for Personal Color Diagnosis

12-Season Classification:
  Spring: Light Spring, True Spring, Bright Spring
  Summer: Light Summer, True Summer, Soft Summer
  Autumn: Soft Autumn, True Autumn, Deep Autumn
  Winter: Bright Winter, True Winter, Deep Winter

Pipeline:
  M1. ImageLoader → load & preprocess image
  M2. FaceLandmarkDetector → detect face landmarks
  M3. PersonSegmenter → segment person from background
  M4. ROIMaskBuilder → extract skin/hair/eye ROI masks
  M5. PixelFilterAndNormalize → filter and normalize pixels
  M6. RepresentativeColorEstimator → estimate representative colors
  M7. SeasonFeatureExtractor → extract undertone/value/chroma/contrast
  M8. Season12RuleEngine → classify to 12-season type
"""

from .pipeline import ColorAnalysisPipeline

__all__ = [
    "ColorAnalysisPipeline",
]
