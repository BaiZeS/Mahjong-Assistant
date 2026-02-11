from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .recognition import TileRecognizer

@dataclass
class Layout:
    hand_regions: List[Tuple[int, int, int, int]]
    discard_regions: Dict[int, List[Tuple[int, int, int, int]]]

    def to_dict(self) -> Dict:
        return {
            "hand_regions": self.hand_regions,
            "discard_regions": self.discard_regions,
        }


class LayoutEstimator:
    def __init__(self) -> None:
        self.hand_bbox: Optional[Tuple[int, int, int, int]] = None

    def estimate(self, image: np.ndarray) -> Layout:
        h, w = image.shape[:2]
        if self.hand_bbox:
            hand_regions = self._segment_hand_tiles(image, self.hand_bbox)
        else:
            hand_regions = self._estimate_hand_regions(w, h)
        discard_regions = self._estimate_discard_regions(w, h)
        return Layout(hand_regions=hand_regions, discard_regions=discard_regions)

    def _segment_hand_tiles(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return []
            
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return []
            
        # 1. 图像预处理：灰度化
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 2. 自适应二值化
        # 使用 cv2.adaptiveThreshold 适应局部光照
        # 注意：通常麻将牌是白色的，背景是深色的。为了提取牌的轮廓，我们需要前景为白色（255）。
        # 因此使用 cv2.THRESH_BINARY。如果使用 INV，则会提取背景或文字。
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 3. 形态学操作（核心）
        # MORPH_OPEN：先腐蚀再膨胀，断开麻将牌之间的连接，去除小噪点
        # 根据麻将间距调整 kernel 大小，这里根据 ROI 高度动态设定，默认 5x5
        k_size = 5 if h > 50 else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 4. 轮廓查找
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_regions = []
        
        # 5. 轮廓过滤（关键参数）
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            
            # 尺寸过滤
            # 排除过小噪点 (10px) 和过大区域
            if rw < 10 or rh < 10:
                continue
            # 上限根据 ROI 动态调整，防止误判
            if rw > w * 0.25: # 单张牌宽度通常小于总宽度的 1/4
                continue
                
            # 宽高比过滤
            # 麻将实际比例约为 1.2~1.5 (竖长方形)
            # 用户建议 1.0 < h/w < 1.6
            ratio = rh / rw
            if not (1.0 < ratio < 1.8): # 稍微放宽上限以适应可能的透视或阴影
                continue
                
            # 高度检查：牌的高度应该占据 ROI 的大部分
            if rh < h * 0.4:
                continue
                
            potential_regions.append((x + rx, y + ry, rw, rh))
            
        # 6. 排序与输出
        potential_regions.sort(key=lambda r: r[0])
        
        # 简单的重叠去除 (NMS-like)
        final_regions = []
        for r in potential_regions:
            if not final_regions:
                final_regions.append(r)
                continue
            
            # 检查与前一个区域的重叠
            lx, ly, lw, lh = final_regions[-1]
            cx, cy, cw, ch = r
            
            # 如果中心点距离太近，视为同一个
            if abs(lx - cx) < 5:
                # 保留较大的那个
                if cw * ch > lw * lh:
                    final_regions[-1] = r
                continue
                
            final_regions.append(r)
            
        # Fallback
        if len(final_regions) < 2:
             return self._estimate_hand_regions_from_bbox(bbox)
             
        return final_regions

    def _estimate_hand_regions(self, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        tile_h = int(h * 0.16)
        tile_w = int(tile_h * 0.75)
        y = int(h * 0.80)
        total_tiles = 14
        spacing = int(tile_w * 0.15)
        total_width = total_tiles * tile_w + (total_tiles - 1) * spacing
        start_x = max(0, (w - total_width) // 2)
        regions = []
        for i in range(total_tiles):
            x = start_x + i * (tile_w + spacing)
            regions.append((x, y, tile_w, tile_h))
        return regions

    def _estimate_hand_regions_from_bbox(self, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        x, y, w, h = bbox
        total_tiles = 14
        tile_w = max(1, w // total_tiles)
        regions = []
        for i in range(total_tiles):
            rx = x + i * tile_w
            regions.append((rx, y, tile_w, h))
        return regions

    def _estimate_discard_regions(self, w: int, h: int) -> Dict[int, List[Tuple[int, int, int, int]]]:
        regions: Dict[int, List[Tuple[int, int, int, int]]] = {}
        
        # Dimensions for upright tiles (P0, P1)
        tile_h_up = int(h * 0.08)
        tile_w_up = int(tile_h_up * 0.75)
        
        # Dimensions for sideways tiles (P2, P3) - simply swapped
        tile_h_side = tile_w_up
        tile_w_side = tile_h_up
        
        center_x = w // 2
        center_y = h // 2
        
        # P0 (Bottom) & P1 (Top)
        # 3 Rows x 6 Cols
        rows_up, cols_up = 3, 6
        
        # P0 Offset
        p0_w = cols_up * tile_w_up
        p0_start_x = center_x - p0_w // 2
        p0_start_y = int(h * 0.58)
        
        regions[0] = []
        for r in range(rows_up):
            for c in range(cols_up):
                regions[0].append((
                    p0_start_x + c * tile_w_up,
                    p0_start_y + r * tile_h_up,
                    tile_w_up, tile_h_up
                ))

        # P1 Offset (Top)
        # Usually mirrored? Or just same grid.
        p1_start_x = p0_start_x
        p1_start_y = int(h * 0.28)
        
        regions[1] = []
        for r in range(rows_up):
            # P1 (Top Player) discards from Their Left (Screen Right) to Their Right (Screen Left)
            # So iterate columns 5 down to 0
            for c in range(cols_up - 1, -1, -1):
                regions[1].append((
                    p1_start_x + c * tile_w_up,
                    p1_start_y + r * tile_h_up,
                    tile_w_up, tile_h_up
                ))

        # P2 (Left) & P3 (Right)
        # 6 Rows x 3 Cols (Visually)
        # The "rows" of the discard pile run along the table edge (Screen Y)
        # The "cols" (layers) run towards the center (Screen X)
        
        side_rows_visual = 6 # Along Y
        side_cols_visual = 3 # Along X
        
        # P2 (Left):
        # Order: Row 1 (Outer/Left) -> Row 3 (Inner/Right)
        # Within Row: Left (Screen Top) -> Right (Screen Bottom)
        # So Col 0..2, Row 0..5
        p2_start_x = int(w * 0.14)
        p2_h = side_rows_visual * tile_h_side # total height
        p2_start_y = center_y - p2_h // 2
        
        regions[2] = []
        for col in range(side_cols_visual): # Layers (X)
            for row in range(side_rows_visual): # Tiles along edge (Y)
                regions[2].append((
                    p2_start_x + col * tile_w_side,
                    p2_start_y + row * tile_h_side,
                    tile_w_side, tile_h_side
                ))
                
        # P3 (Right)
        # Order: Row 1 (Outer/Right) -> Row 3 (Inner/Left)
        # Within Row: Left (Screen Bottom) -> Right (Screen Top)
        
        # P3 visual block starts at p3_start_x (Left/Inner edge of the block)
        p3_inner_x = int(w * 0.78) 
        p3_start_y = p2_start_y
        
        regions[3] = []
        # Cols: Outer (Right) -> Inner (Left). 
        # Screen X indices: 2 -> 0 relative to inner edge?
        # Inner edge is index 0. Outer edge is index 2.
        # We want index 2, 1, 0.
        for col in range(side_cols_visual - 1, -1, -1):
            # Rows: Bottom -> Top
            # Screen Y indices: 5 -> 0
            for row in range(side_rows_visual - 1, -1, -1):
                regions[3].append((
                    p3_inner_x + col * tile_w_side,
                    p3_start_y + row * tile_h_side,
                    tile_w_side, tile_h_side
                ))

        return regions


class GameStateExtractor:
    def __init__(self, recognizer: TileRecognizer, layout_estimator: LayoutEstimator) -> None:
        self.recognizer = recognizer
        self.layout_estimator = layout_estimator
        self.prev_discard_tiles: Dict[int, List[int]] = {i: [] for i in range(4)}
        self.discard_sequence: List[Dict] = []
        self.max_discard_history = 80
        self.last_hand_count = 0
        # Settings
        self.red_zhong_count = 3
        self.min_hu_multiplier = 1
        self.require_que_men = False
        self.self_draw = True
        self.gang_shang_hua = False
        self.gang_shang_pao = False
        self.tian_hu = False
        self.di_hu = False
        self.fan_cap = 40
        self.play_mode = "fast"

    def extract(self, image: np.ndarray) -> Dict:
        layout = self.layout_estimator.estimate(image)
        
        # Use recognize_hand which employs global template matching with NMS
        # This is more robust than recognize_regions which relies on precise segmentation
        if self.layout_estimator.hand_bbox:
            # If we have a calibrated bbox, search the entire area
            # recognize_hand computes the union of regions, so passing the full bbox works perfectly
            search_regions = [self.layout_estimator.hand_bbox]
        else:
            # Otherwise use the estimated grid/regions
            search_regions = layout.hand_regions
            
        hand_tiles = self.recognizer.recognize_hand(image, search_regions)
        
        self.last_hand_count = len(hand_tiles)
        
        # Extract discards directly from layout regions
        # The layout regions are ordered chronologically (Player -> Center, Left -> Right from player perspective)
        # So we can just map regions to tiles.
        discard_sequence: List[Dict] = []
        
        for player_id, regions in layout.discard_regions.items():
            for idx, region in enumerate(regions):
                x, y, w, h = region
                # Boundary check
                if x < 0 or y < 0 or w <= 0 or h <= 0 or \
                   y + h > image.shape[0] or x + w > image.shape[1]:
                    continue
                    
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                if player_id == 2: # Left: Rotate CCW 90
                    roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif player_id == 3: # Right: Rotate CW 90
                    roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
                    
                tile_id = self.recognizer.recognize_roi(roi)
                
                # Only include valid tiles
                if tile_id >= 0:
                    discard_sequence.append({
                        "player": player_id, 
                        "tile": tile_id, 
                        "idx": idx # Relative index for that player
                    })

        current_discard = None
        current_discard_player = None
        
        # Determine "current discard" (the very last one in the sequence?)
        # Or usually the last one added.
        # Since we scan all players, the "last" one is P3's last tile?
        # This is ambiguous without temporal tracking.
        # But usually 'current_discard' is only needed for "Ron" checks.
        # For "Danger Avoidance", we just need the history.
        # We'll just take the last one in the list as a best guess, 
        # or we could try to find the "active" indicator if we were smarter.
        # For now, let's just leave current_discard as None or last detected.
        if discard_sequence:
             # Heuristic: The player with the highest index tile is likely the latest?
             # No, that depends on turn order.
             # Let's just return the full list.
             pass

        return {
            "hand": hand_tiles,
            "discarded_tiles": discard_sequence,
            "discard_sequence": discard_sequence, # Legacy field compatibility
            "current_player": 0,
            "current_discard": current_discard,
            "current_discard_player": current_discard_player,
            "players": [{"style": "balanced"} for _ in range(4)],
            "round": 1,
            "scores": [1000, 1000, 1000, 1000],
            "dealer": 0,
            "red_zhongs_in_wall": self.red_zhong_count,
            "min_hu_multiplier": self.min_hu_multiplier,
            "require_que_men": self.require_que_men,
            "self_draw": self.self_draw,
            "gang_shang_hua": self.gang_shang_hua,
            "gang_shang_pao": self.gang_shang_pao,
            "tian_hu": self.tian_hu,
            "di_hu": self.di_hu,
            "fan_cap": self.fan_cap,
            "play_mode": self.play_mode,
            # Extra info for debug drawing
            "layout": layout.to_dict(),
        }

    def reset(self) -> None:
        """Reset state (if any)."""
        self.last_hand_count = 0

