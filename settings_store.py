"""Small JSON-backed operational settings, editable from the Flask console."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Any, Dict
from face_service import DATABASE_DIR

DEFAULTS: Dict[str, Any] = {"threshold": 0.50, "face_interval": 4, "inference_max_width": 960, "alert_cooldown": 30, "tracker": "bytetrack.yaml", "cameras": [{"id": "laptop", "name": "Laptop camera", "source": "0", "enabled": True}]}

class SettingsStore:
    def __init__(self, path: Path = DATABASE_DIR / "settings.json") -> None:
        self.path, self.lock = path, threading.RLock(); self.data = DEFAULTS.copy(); self.load()
    def load(self) -> None:
        with self.lock:
            if self.path.exists():
                self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
    def get(self) -> Dict[str, Any]:
        with self.lock: return json.loads(json.dumps(self.data))
    def save(self, values: Dict[str, Any]) -> None:
        with self.lock:
            self.data.update(values); self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
