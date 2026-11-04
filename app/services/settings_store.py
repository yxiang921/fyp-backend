"""Small JSON-backed operational settings, editable from the Flask console."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from app.services.face_service import DATABASE_DIR

DEFAULTS: Dict[str, Any] = {
    # ── Recognition ─────────────────────────────────────────────────
    "threshold": 0.50,              # ArcFace cosine-similarity threshold
    "face_interval": 4,             # legacy: frames between face recognition per track
    "inference_max_width": 960,     # max width for video-file inference
    "alert_cooldown": 30,           # seconds between repeat alerts for the same person
    # ── Tracker ─────────────────────────────────────────────────────
    "tracker": "bytetrack.yaml",    # tracker config name (display only; standalone ByteTrack used internally)
    # ── Cameras ─────────────────────────────────────────────────────
    "cameras": [
        {
            "id": "laptop",
            "name": "Laptop camera",
            "source": "0",
            "enabled": True,
        }
    ],
    # ── New optimised parameters ────────────────────────────────────
    "yolo_interval": 3,             # run YOLO detection every N frames (1 = every frame)
    "recognition_interval": 15,     # frames between ArcFace re-recognition per tracked person
    "unknown_recheck_interval": 30, # frames before re-checking an "Unknown" person
    "min_face_size": 60,            # px – minimum face width/height for ArcFace
    "face_conf_threshold": 0.45,    # minimum SCRFD detection confidence for recognition
    "ai_width": 640,                # AI inference width (height auto-scaled from frame)
    "max_track_age": 60,            # frames before a stale track is purged from cache
}


class SettingsStore:
    def __init__(self, path: Path = DATABASE_DIR / "settings.json") -> None:
        self.path = path
        self.lock = threading.RLock()
        self.data = DEFAULTS.copy()
        self.load()

    def load(self) -> None:
        with self.lock:
            if self.path.exists():
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                # Merge stored values on top of defaults (so new keys always exist)
                self.data.update(stored)

    def get(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.data))

    def save(self, values: Dict[str, Any]) -> None:
        with self.lock:
            self.data.update(values)
            self.path.write_text(
                json.dumps(self.data, indent=2), encoding="utf-8"
            )
