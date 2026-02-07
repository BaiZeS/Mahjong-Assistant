from __future__ import annotations

import logging
import mss
import numpy as np
import win32gui
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class WindowInfo:
    handle: int
    title: str
    rect: Tuple[int, int, int, int]


class WindowDetector:
    def list_windows(self) -> List[WindowInfo]:
        windows: List[WindowInfo] = []

        def enum_handler(hwnd: int, _ctx: Optional[int]) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return
            rect = win32gui.GetWindowRect(hwnd)
            windows.append(WindowInfo(hwnd, title, rect))

        win32gui.EnumWindows(enum_handler, None)
        return windows


class ScreenCapture:
    def __init__(self) -> None:
        self.sct = mss.mss()
        self.logger = logging.getLogger("mahjong.capture")
        bounds = self.sct.monitors[0]
        self.bounds = (
            bounds["left"],
            bounds["top"],
            bounds["left"] + bounds["width"],
            bounds["top"] + bounds["height"],
        )

    def capture(self, window: WindowInfo) -> Optional[np.ndarray]:
        try:
            left, top, right, bottom = win32gui.GetWindowRect(window.handle)
            min_left, min_top, max_right, max_bottom = self.bounds
            left = max(left, min_left)
            top = max(top, min_top)
            right = min(right, max_right)
            bottom = min(bottom, max_bottom)
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                return None
            grab = {"left": left, "top": top, "width": width, "height": height}
            img = np.array(self.sct.grab(grab))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            self.logger.exception("Capture failed: %s", exc)
            return None
