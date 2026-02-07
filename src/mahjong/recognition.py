from __future__ import annotations

import logging
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from .constants import TILE_COUNT, TILE_NAME_TO_ID

class TileRecognizer:
    def __init__(self, templates_dir: str) -> None:
        self.templates_dir = templates_dir
        self.templates: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.logger = logging.getLogger("mahjong.recognizer")
        self.template_count = 0
        self.template_total = 0
        self.load_templates()
        self.last_max_score = 0.0
        self.last_mean_score = 0.0
        self.last_threshold = 0.55
        self.enable_debug = False
        self.scales = [0.85, 1.0, 1.15]
        self.equalize_hist = True

    def load_templates(self) -> None:
        self.templates = {}
        self.template_count = 0
        self.template_total = 0
        if not os.path.isdir(self.templates_dir):
            self.logger.warning("Templates dir not found: %s", self.templates_dir)
            return
        for filename in os.listdir(self.templates_dir):
            path = os.path.join(self.templates_dir, filename)
            if not os.path.isfile(path):
                continue
            self.template_total += 1
            name, _ext = os.path.splitext(filename)
            tile_id = None
            if name.isdigit():
                tile_id = int(name)
            elif name in TILE_NAME_TO_ID:
                tile_id = TILE_NAME_TO_ID[name]
            if tile_id is None or not (0 <= tile_id < TILE_COUNT):
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            edge = cv2.Canny(img, 50, 150)
            self.templates.setdefault(tile_id, []).append((img, edge))
            self.template_count += 1
        self.logger.info("Loaded templates: %s/%s from %s", self.template_count, self.template_total, self.templates_dir)

    def recognize_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[int]:
        results: List[int] = []
        if not self.templates:
            return [-1 for _ in regions]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.equalize_hist:
            gray = cv2.equalizeHist(gray)
        scores: List[float] = []
        for x, y, w, h in regions:
            if w <= 0 or h <= 0:
                results.append(-1)
                continue
            roi_gray = gray[y : y + h, x : x + w]
            if roi_gray.size == 0:
                results.append(-1)
                continue
            roi_edge = cv2.Canny(roi_gray, 50, 150)
            tile_id, score = self._match_best(roi_gray, roi_edge)
            scores.append(score)
            if score < self.last_threshold:
                results.append(-1)
            else:
                results.append(tile_id)
        if scores:
            self.last_max_score = max(scores)
            self.last_mean_score = sum(scores) / len(scores)
            if self.enable_debug:
                self.logger.info("Recognizer scores max=%.3f mean=%.3f", self.last_max_score, self.last_mean_score)
        return results

    def recognize_hand(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[int]:
        """
        Recognize tiles in the hand area using global template matching and NMS.
        """
        if not self.templates or not regions:
            return []

        # Calculate bounding box of all hand regions
        xs = [r[0] for r in regions]
        ys = [r[1] for r in regions]
        ws = [r[2] for r in regions]
        hs = [r[3] for r in regions]

        x_min, y_min = min(xs), min(ys)
        x_max = max(x + w for x, w in zip(xs, ws))
        y_max = max(y + h for y, h in zip(ys, hs))

        # Add padding to search area
        pad_x = 20
        pad_y = 20
        img_h, img_w = image.shape[:2]
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(img_w, x_max + pad_x)
        y2 = min(img_h, y_max + pad_y)

        if x2 <= x1 or y2 <= y1:
            return []

        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.equalize_hist:
            gray = cv2.equalizeHist(gray)

        matches = []
        
        # Use a slightly higher threshold for global search to reduce false positives
        threshold = max(0.6, self.last_threshold)

        for tile_id, templates_list in self.templates.items():
            for tmpl_gray, tmpl_edge in templates_list:
                for scale in self.scales:
                    th, tw = tmpl_gray.shape[:2]
                    target_w = int(tw * scale)
                    target_h = int(th * scale)
                    
                    if gray.shape[0] < target_h or gray.shape[1] < target_w:
                        continue
                        
                    resized_tmpl = cv2.resize(tmpl_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(gray, resized_tmpl, cv2.TM_CCOEFF_NORMED)
                    
                    locs = np.where(res >= threshold)
                    for pt in zip(*locs[::-1]):
                        score = res[pt[1], pt[0]]
                        matches.append({
                            "tile_id": tile_id,
                            "score": float(score),
                            "x": pt[0], # Relative to ROI
                            "y": pt[1],
                            "w": target_w,
                            "h": target_h
                        })

        final_matches = self._apply_nms(matches)
        # Sort by X coordinate
        final_matches.sort(key=lambda m: m["x"])
        
        return [m["tile_id"] for m in final_matches]

    def _apply_nms(self, matches: List[Dict], overlap_thresh: float = 0.3) -> List[Dict]:
        if not matches:
            return []
        
        # Sort by score descending
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)
        picked = []
        
        while matches:
            current = matches.pop(0)
            picked.append(current)
            
            cx, cy, cw, ch = current["x"], current["y"], current["w"], current["h"]
            
            filtered_matches = []
            for m in matches:
                mx, my, mw, mh = m["x"], m["y"], m["w"], m["h"]
                
                # Intersection
                ix1 = max(cx, mx)
                iy1 = max(cy, my)
                ix2 = min(cx + cw, mx + mw)
                iy2 = min(cy + ch, my + mh)
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                
                intersection = iw * ih
                if intersection <= 0:
                    filtered_matches.append(m)
                    continue
                
                # Intersection over Min Area
                area_current = cw * ch
                area_m = mw * mh
                min_area = min(area_current, area_m)
                
                overlap = intersection / min_area
                if overlap < overlap_thresh:
                    filtered_matches.append(m)
            
            matches = filtered_matches
            
        return picked

    def _match_best(self, roi_gray: np.ndarray, roi_edge: np.ndarray) -> Tuple[int, float]:
        best_id = -1
        best_score = 0.0
        for tile_id, templates in self.templates.items():
            for tmpl_gray, tmpl_edge in templates:
                for scale in self.scales:
                    h = max(4, int(tmpl_gray.shape[0] * scale))
                    w = max(4, int(tmpl_gray.shape[1] * scale))
                    if roi_gray.shape[0] < h or roi_gray.shape[1] < w:
                        continue
                    resized_gray = cv2.resize(tmpl_gray, (w, h), interpolation=cv2.INTER_AREA)
                    resized_edge = cv2.resize(tmpl_edge, (w, h), interpolation=cv2.INTER_AREA)
                    res_gray = cv2.matchTemplate(roi_gray, resized_gray, cv2.TM_CCOEFF_NORMED)
                    res_edge = cv2.matchTemplate(roi_edge, resized_edge, cv2.TM_CCOEFF_NORMED)
                    score = max(float(res_gray.max()), float(res_edge.max()))
                    if score > best_score:
                        best_score = score
                        best_id = tile_id
        return best_id, best_score
