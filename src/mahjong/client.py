from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Dict

class BackendClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.logger = logging.getLogger("mahjong.backend")

    def predict(self, game_state: Dict) -> Dict:
        url = f"{self.base_url}/predict"
        payload = json.dumps(game_state).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            self.logger.exception("Backend predict failed: %s", exc)
            return {"predictions": [], "recommendations": [], "confidence": 0.0}
