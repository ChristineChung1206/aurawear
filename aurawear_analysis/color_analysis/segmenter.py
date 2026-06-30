"""M3. PersonSegmenter - Segment person from background (simplified)."""

from __future__ import annotations

import cv2
import numpy as np


class PersonSegmenter:
    """
    Segment person from background using skin-color heuristics.
    
    Output:
      - person_mask: binary mask (0/1) same size as input image
    """
    
    def segment(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Segment person from background.
        
        Args:
            img_bgr: BGR image (H, W, 3)
            
        Returns:
            person_mask: binary mask (H, W) of 0/1
        """
        h, w = img_bgr.shape[:2]
        
        # 方法 1：使用 YCrCb 色域進行膚色檢測（比 HSV 更穩定）
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        
        # YCrCb 膚色範圍（較寬鬆）
        lower_skin = np.array([0, 133, 77])
        upper_skin = np.array([255, 173, 127])
        mask1 = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
        
        # 方法 2：使用 Lab 色域作為補充
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        # Lab 膚色範圍：L 40-80, a 5-15, b -5 to 10
        lower_lab = np.array([40, 5, 75])
        upper_lab = np.array([80, 15, 130])
        mask2 = cv2.inRange(img_lab, lower_lab, upper_lab)
        
        # 合併兩種方法
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形態學操作清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 轉換為二進制
        mask = (mask > 127).astype(np.uint8)
        
        # 確保至少有一些前景像素
        if mask.sum() < (h * w * 0.01):  # 少於 1% 像素
            # Fallback：使用中心區域
            mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            mask[max(0, cy-h//3):min(h, cy+h//3), 
                 max(0, cx-w//3):min(w, cx+w//3)] = 1
        
        return mask
