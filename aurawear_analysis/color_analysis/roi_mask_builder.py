"""M4. ROI Mask Builder - 改進版，使用膚色分割而非精確 landmarks."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class ROIMaskBuilder:
    """
    Build ROI masks for skin, hair, and eye regions.
    
    改進方案：用膚色分割 + 簡單形態學操作，而不完全依賴 landmarks
    """

    def build_masks(
        self,
        img_shape: Tuple[int, int],
        landmarks: np.ndarray,
        bbox: Tuple[int, int, int, int],
        person_mask: np.ndarray,
        parsing_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Build skin/hair/eye masks.
        
        Args:
            img_shape: (H, W)
            landmarks: (468, 2) or fake landmarks
            bbox: (x1, y1, x2, y2)
            person_mask: binary person segmentation
            
        Returns:
            {skin_mask, hair_mask, eye_mask}
        """
        h, w = img_shape
        x1, y1, x2, y2 = bbox

        # If semantic parsing masks are provided, prefer them.
        # Expected keys: skin_mask/hair_mask/eye_mask as (H,W) uint8 {0,1}.
        if parsing_masks is not None:
            skin_pm = parsing_masks.get("skin_mask")
            hair_pm = parsing_masks.get("hair_mask")
            eye_pm = parsing_masks.get("eye_mask")

            if skin_pm is not None and hair_pm is not None and eye_pm is not None:
                skin_mask = (skin_pm > 0).astype(np.uint8)
                hair_mask = (hair_pm > 0).astype(np.uint8)
                eye_mask = (eye_pm > 0).astype(np.uint8)

                # Constrain masks similarly to the heuristic path.
                face_bbox_mask = np.zeros((h, w), dtype=np.uint8)
                fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                face_bbox_mask[fy1:fy2, fx1:fx2] = 1

                # Skin/Eye: face bbox ∩ (person mask if reliable)
                skin_mask = (skin_mask * face_bbox_mask).astype(np.uint8)
                eye_mask = (eye_mask * face_bbox_mask).astype(np.uint8)
                if person_mask is not None and person_mask.sum() > (h * w * 0.01):
                    skin_mask = (skin_mask * person_mask).astype(np.uint8)

                # Hair: keep person constraint to avoid background.
                if person_mask is not None:
                    hair_mask = (hair_mask * person_mask).astype(np.uint8)

                # Light morphological cleanup to remove tiny islands.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
                hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
                eye_mask = cv2.morphologyEx(eye_mask, cv2.MORPH_OPEN, kernel)

                return {
                    "skin_mask": (skin_mask > 0).astype(np.uint8),
                    "hair_mask": (hair_mask > 0).astype(np.uint8),
                    "eye_mask": (eye_mask > 0).astype(np.uint8),
                }
        
        # === Heuristic path: construct masks from landmarks + geometry ===
        
        # Skin mask: cheeks + forehead, constrained to face bbox
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        if len(landmarks) >= 350:
            try:
                # 臉頰區域
                cheek_points = landmarks[[116, 123, 147, 340, 349, 371], :2].astype(int)
                cv2.fillPoly(skin_mask, [cheek_points], 1)
                
                # 額頭區域
                forehead_points = landmarks[[10, 109, 338], :2].astype(int)
                cv2.fillPoly(skin_mask, [forehead_points], 1)
            except (IndexError, ValueError):
                # Fallback: center-region mask
                y_idx, x_idx = np.ogrid[:h, :w]
                cy, cx = h // 2, w // 2
                gaussian = np.exp(-((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / (0.2 * max(h, w) ** 2))
                skin_mask = (gaussian > 0.3).astype(np.uint8)
        else:
            # Fallback: center-region mask
            y_idx, x_idx = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            gaussian = np.exp(-((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / (0.2 * max(h, w) ** 2))
            skin_mask = (gaussian > 0.3).astype(np.uint8)
        
        # 排除眼睛和嘴巴
        eye_and_mouth = np.zeros((h, w), dtype=np.uint8)
        try:
            if len(landmarks) >= 350:
                # 左眼
                left_eye = landmarks[33:43, :2].astype(int)
                cv2.fillPoly(eye_and_mouth, [left_eye], 1)
                
                # 右眼
                right_eye = landmarks[263:273, :2].astype(int)
                cv2.fillPoly(eye_and_mouth, [right_eye], 1)
                
                # 嘴巴
                mouth = landmarks[61:91, :2].astype(int)
                cv2.fillPoly(eye_and_mouth, [mouth], 1)
        except (IndexError, ValueError):
            pass
        
        skin_mask = skin_mask * (1 - eye_and_mouth)
        
        # Hair mask: 頭頂區域 ∩ 背景 ∩ ~skin
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        # 優化：擴大髮色區域以捕捉更多髮絲
        hair_y1 = max(0, y1 - int((y2 - y1) * 0.5))  # 向上擴張（0.4→0.5）
        hair_y2 = y1 + int((y2 - y1) * 0.4)  # 向下擴張（0.3→0.4）
        hair_x1 = max(0, x1 - int((x2 - x1) * 0.3))  # 向左擴張（0.2→0.3）
        hair_x2 = min(w, x2 + int((x2 - x1) * 0.3))  # 向右擴張（0.2→0.3）
        
        hair_bbox = np.zeros((h, w), dtype=np.uint8)
        hair_bbox[hair_y1:hair_y2, hair_x1:hair_x2] = 1
        
        # Hair = bbox區域 ∩ 人物 ∩ ~膚色
        hair_mask = hair_bbox * person_mask * (1 - skin_mask)
        
        # 膨脹髮色區域以捕捉更多髮絲
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # 更大的核（7→9）
        hair_mask = cv2.dilate(hair_mask, kernel, iterations=2)  # 更多迭代次數（1→2）
        
        # Eye mask: 眼睛区域
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 首先嘗試使用虹膜 landmarks（若可用）
        iris_used = False
        try:
            if len(landmarks) >= 478:
                # 左眼虹膜
                left_iris = landmarks[468:473, :2].astype(int)
                if len(left_iris) > 0 and np.all(left_iris >= 0):
                    center = left_iris.mean(axis=0)
                    cv2.circle(eye_mask, tuple(center.astype(int)), 8, 1, -1)
                    iris_used = True
                
                # 右眼虹膜
                right_iris = landmarks[473:478, :2].astype(int)
                if len(right_iris) > 0 and np.all(right_iris >= 0):
                    center = right_iris.mean(axis=0)
                    cv2.circle(eye_mask, tuple(center.astype(int)), 8, 1, -1)
                    iris_used = True
        except (IndexError, ValueError, AttributeError):
            pass
        
        # 如果虹膜 landmarks 不可用，使用眼睛輪廓中心點（而不是整個輪廓）
        if not iris_used or eye_mask.sum() == 0:
            try:
                if len(landmarks) >= 350:
                    # 使用眼睛輪廓計算眼睛中心，但只在中心畫小圓
                    # 左眼輪廓 (33-42)
                    left_eye_outline = landmarks[33:42, :2].astype(float)
                    if len(left_eye_outline) > 0:
                        left_center = left_eye_outline.mean(axis=0).astype(int)
                        # 只在中心畫虹膜大小的圓（直徑 8-10 像素）
                        cv2.circle(eye_mask, tuple(left_center), 5, 1, -1)
                    
                    # 右眼輪廓 (263-272)
                    right_eye_outline = landmarks[263:272, :2].astype(float)
                    if len(right_eye_outline) > 0:
                        right_center = right_eye_outline.mean(axis=0).astype(int)
                        cv2.circle(eye_mask, tuple(right_center), 5, 1, -1)
            except (IndexError, ValueError, AttributeError):
                pass
        
        # 最終 fallback：使用近似眼睛位置
        if eye_mask.sum() == 0:
            left_eye_x = int(x1 + (x2 - x1) * 0.35)
            left_eye_y = int(y1 + (y2 - y1) * 0.35)
            right_eye_x = int(x1 + (x2 - x1) * 0.65)
            right_eye_y = int(y1 + (y2 - y1) * 0.35)
            
            cv2.circle(eye_mask, (left_eye_x, left_eye_y), 5, 1, -1)
            cv2.circle(eye_mask, (right_eye_x, right_eye_y), 5, 1, -1)
        
        # 清理mask：確保在有效範圍內
        # Person mask can be noisy with heuristic segmentation.
        # For face-related regions (skin/eye), prefer constraining to face bbox.
        face_bbox_mask = np.zeros((h, w), dtype=np.uint8)
        fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        face_bbox_mask[fy1:fy2, fx1:fx2] = 1

        # Skin: face bbox ∩ (person mask if reliable)
        skin_mask = (skin_mask * face_bbox_mask).astype(np.uint8)
        if person_mask is not None and person_mask.sum() > (h * w * 0.01):
            skin_mask = (skin_mask * person_mask).astype(np.uint8)

        # Hair: still relies on person mask to avoid background
        hair_mask = (hair_mask * person_mask).astype(np.uint8) if person_mask is not None else hair_mask.astype(np.uint8)

        # Eye: face bbox only (avoid empty mask due to person-mask holes)
        eye_mask = (eye_mask * face_bbox_mask).astype(np.uint8)
        
        return {
            'skin_mask': skin_mask,
            'hair_mask': hair_mask,
            'eye_mask': eye_mask,
        }

