"""Optimized face detection, tracking, and recognition engine.

Architecture
------------
RTSP Camera → Frame Capture → (Display Pipeline | AI Pipeline)
  Display Pipeline: raw frame + cached AI overlays → smooth 20-30 FPS stream
  AI Pipeline:    YOLO periodically → ByteTrack continuously → Selective SCRFD → Selective ArcFace → Alert

Key optimisations:
- YOLO interval       – run person detection every N frames, ByteTrack bridges the gap
- Selective SCRFD     – only detect faces when a track is new / unknown / expired
- Selective ArcFace   – skip recognition when a confident match already exists
- Face quality filter – reject tiny / low-confidence faces before expensive embedding
- Recognition cache   – per-track cache indexed by ByteTrack ID
- Latest-frame buffer – always process the newest frame; drop stale ones
- Two-thread design   – capture+display thread runs at camera FPS; AI thread runs async
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import pickle
import shutil
import subprocess
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover
    FaceAnalysis = None

# ---------------------------------------------------------------------------
# Configurable defaults (overridable via settings.json or constructor kwargs)
# ---------------------------------------------------------------------------
DEFAULT_YOLO_INTERVAL = 3            # run YOLO every N frames (1 = every frame)
DEFAULT_RECOGNITION_INTERVAL = 15    # frames between ArcFace re-recognition per track
DEFAULT_UNKNOWN_RECHECK_INTERVAL = 30  # frames before re-checking an "Unknown" person
DEFAULT_MIN_FACE_SIZE = 60           # px – minimum face width/height for ArcFace
DEFAULT_FACE_CONF_THRESHOLD = 0.45   # minimum SCRFD detection confidence for ArcFace
DEFAULT_AI_WIDTH = 640               # YOLO inference width (height auto-scaled)
DEFAULT_MAX_TRACK_AGE = 60           # frames before a stale track is purged from cache

# ---------------------------------------------------------------------------
# Paths & helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = ROOT / "database"
OUTPUT_DIR = ROOT / "outputs"
FACE_DB_FILE = DATABASE_DIR / "face_db.pkl"
METADATA_FILE = DATABASE_DIR / "face_metadata.json"
ALERT_URL = "http://localhost:8000/sendAlert"
MALAYSIA_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def ensure_directories() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def slugify(value: str) -> str:
    allowed = []
    for ch in value.strip().lower():
        if ch.isalnum():
            allowed.append(ch)
        elif ch in {" ", "-", "_"}:
            allowed.append("_")
    slug = "".join(allowed).strip("_")
    return slug or "person"


def timestamp_slug() -> str:
    return datetime.datetime.now(MALAYSIA_TZ).strftime("%Y%m%d_%H%M%S")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_y = max(0, y - text_h - baseline - 8)
    cv2.rectangle(
        frame,
        (x, label_y),
        (x + text_w + 10, label_y + text_h + baseline + 8),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 5, label_y + text_h + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Enhanced per-track recognition cache
# ---------------------------------------------------------------------------
@dataclass
class TrackCacheEntry:
    """Holds the latest recognition result for one ByteTrack ID."""
    name: str = "Unknown"
    confidence: float = 0.0
    last_recognition_frame: int = 0
    last_seen_frame: int = 0
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # relative to person crop
    face_quality_score: float = 0.0
    is_unknown: bool = True
    unknown_since_frame: int = 0
    recognition_count: int = 0
    # Cached face boxes relative to person-crop for drawing between recogs
    local_face_boxes: List[Tuple[Tuple[int, int, int, int], "MatchResult", float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    name: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class AIResults:
    """Thread-safe container for AI overlays shared with the display pipeline."""
    detections: List[Dict[str, Any]] = field(default_factory=list)
    frame_number: int = 0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Face database (local file-based)
# ---------------------------------------------------------------------------
class FaceDatabase:
    def __init__(self, db_file: Path = FACE_DB_FILE, metadata_file: Path = METADATA_FILE) -> None:
        ensure_directories()
        self.db_file = db_file
        self.metadata_file = metadata_file
        self.lock = threading.RLock()
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        with self.lock:
            if self.db_file.exists():
                with self.db_file.open("rb") as f:
                    raw = pickle.load(f)
                parsed: Dict[str, np.ndarray] = {}
                for name, value in raw.items():
                    if isinstance(value, dict) and "embedding" in value:
                        value = value["embedding"]
                    parsed[name] = np.asarray(value, dtype=np.float32).reshape(-1)
                self.embeddings = parsed
            else:
                self.embeddings = {}

            if self.metadata_file.exists():
                with self.metadata_file.open("r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}

    def save(self) -> None:
        with self.lock:
            with self.db_file.open("wb") as f:
                pickle.dump(self.embeddings, f)
            with self.metadata_file.open("w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def upsert(
        self,
        name: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self.lock:
            self.embeddings[name] = np.asarray(embedding, dtype=np.float32).reshape(-1)
            current = self.metadata.get(name, {})
            current.update(metadata or {})
            self.metadata[name] = current
            self.save()

    def list_people(self) -> List[Dict[str, Any]]:
        with self.lock:
            rows: List[Dict[str, Any]] = []
            for name in sorted(self.embeddings.keys()):
                row = {"name": name, "metadata": self.metadata.get(name, {})}
                rows.append(row)
            return rows

    def clear(self) -> None:
        with self.lock:
            self.embeddings = {}
            self.metadata = {}
            self.save()

    def delete_case(self, case_number: str) -> bool:
        """Remove the face profile associated with a Laravel case."""
        with self.lock:
            names = {
                name
                for name, metadata in self.metadata.items()
                if metadata.get("case_number") == case_number
                or metadata.get("person_id") == case_number
                or name == case_number
            }
            for name in names:
                self.embeddings.pop(name, None)
                self.metadata.pop(name, None)
            if names:
                self.save()
            return bool(names)

    def best_match(self, embedding: np.ndarray, threshold: float = 0.5) -> MatchResult:
        with self.lock:
            best_name = "Unknown"
            best_score = -1.0
            best_meta: Dict[str, Any] = {}

            for name, db_embedding in self.embeddings.items():
                score = cosine_similarity(embedding, db_embedding)
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_meta = self.metadata.get(name, {})

            if best_score < threshold:
                return MatchResult("Unknown", best_score, {})
            return MatchResult(best_name, best_score, best_meta)

# ---------------------------------------------------------------------------
# Optimised Vision Engine
# ---------------------------------------------------------------------------
class VisionEngine:
    def __init__(
        self,
        face_db: FaceDatabase,
        yolo_model_path: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",         # kept for backward-compat; standalone ByteTrack used internally
        recognition_threshold: float = 0.5,
        alert_cooldown_seconds: int = 30,
        face_recognition_interval_frames: int = 4,  # legacy, mapped to recognition_interval
        inference_max_width: int = 960,
        camera_database_id: Optional[int] = None,
        detection_callback: Optional[Callable[[int, Dict[str, Any], float, List[np.ndarray]], None]] = None,
        # ── New optimised parameters ──────────────────────────────────────
        yolo_interval: int = DEFAULT_YOLO_INTERVAL,
        recognition_interval: int = DEFAULT_RECOGNITION_INTERVAL,
        unknown_recheck_interval: int = DEFAULT_UNKNOWN_RECHECK_INTERVAL,
        min_face_size: int = DEFAULT_MIN_FACE_SIZE,
        face_conf_threshold: float = DEFAULT_FACE_CONF_THRESHOLD,
        ai_width: int = DEFAULT_AI_WIDTH,
        max_track_age: int = DEFAULT_MAX_TRACK_AGE,
    ) -> None:
        ensure_directories()
        self.face_db = face_db
        self.yolo_model_path = yolo_model_path
        self.tracker = tracker  # kept for settings display only

        # Recognition
        self.recognition_threshold = recognition_threshold
        self.face_recognition_interval_frames = max(1, face_recognition_interval_frames)
        self.inference_max_width = max(320, inference_max_width)
        self.camera_database_id = camera_database_id
        self.detection_callback = detection_callback
        self._detection_frames = deque(maxlen=8)
        self._detection_frames_lock = threading.Lock()
        self.detection_burst_seconds = max(0.0, float(os.environ.get("DETECTION_BURST_SECONDS", "1.0")))
        self.detection_burst_images = max(1, min(8, int(os.environ.get("DETECTION_BURST_IMAGES", "5"))))

        # ── New configurable parameters ───────────────────────────────────
        self.yolo_interval = max(1, yolo_interval)
        self.recognition_interval = max(1, recognition_interval)
        self.unknown_recheck_interval = max(1, unknown_recheck_interval)
        self.min_face_size = max(20, min_face_size)
        self.face_conf_threshold = max(0.1, min(1.0, face_conf_threshold))
        self.ai_width = max(160, ai_width)
        self.max_track_age = max(10, max_track_age)

        # Alert cooldown
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.alert_lock = threading.Lock()
        self.last_alert_by_name: Dict[str, float] = {}
        self.last_alert_by_track: Dict[str, float] = {}

        # Frame counter & tracking state
        self.frame_number = 0
        self.process_lock = threading.RLock()

        # ── Enhanced face cache  ──────────────────────────────────────────
        self.face_cache: Dict[str, TrackCacheEntry] = {}

        # Uploaded image/video tests operate directly on faces rather than
        # requiring a YOLO person track.  The lightweight cache lets video
        # tests redraw boxes every frame while avoiding a database lookup on
        # every frame for the same face.
        self._test_face_cache: List[Tuple[float, float, MatchResult]] = []

        # ── AI results (thread-safe, shared with display pipeline) ────────
        self.ai_results = AIResults()
        self.ai_results_lock = threading.Lock()

        # ── Per-track motion history (smooth overlay extrapolation) ───────
        self._motion_lock = threading.Lock()
        self._track_history: Dict[str, Dict[str, Any]] = {}

        # ── Standalone ByteTrack ──────────────────────────────────────────
        self._byte_tracker: Any = None
        self._byte_tracker_ready = False

        # ── FPS / latency monitoring ──────────────────────────────────────
        self._fps_lock = threading.Lock()
        self._fps_data: Dict[str, float] = {
            "camera_fps": 0.0,
            "processing_fps": 0.0,
            "yolo_fps": 0.0,
            "recognition_fps": 0.0,
            "latency_ms": 0.0,
            "tracked_persons": 0,
        }
        self._last_camera_time = 0.0
        self._last_process_time = 0.0
        self._last_yolo_time = 0.0
        self._last_recog_time = 0.0
        self._process_start_time = 0.0
        self._camera_frame_count = 0
        self._process_frame_count = 0
        self._yolo_frame_count = 0
        self._recog_count = 0
        self._fps_update_time = time.time()

        # Model checks
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed in this environment.")
        if FaceAnalysis is None:
            raise RuntimeError("insightface is not installed in this environment.")

        # ── Load models ONCE ──────────────────────────────────────────────
        self.person_model = YOLO(yolo_model_path)
        self.face_app = FaceAnalysis(name="buffalo_s")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

    # ------------------------------------------------------------------
    # Standalone ByteTrack (not coupled to YOLO's model.track())
    # ------------------------------------------------------------------
    def _init_byte_tracker(self, frame_rate: float = 30.0) -> None:
        """Create a standalone ByteTrack instance for YOLO-interval support."""
        if self._byte_tracker_ready:
            return
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            class _Args:
                track_high_thresh = 0.5
                track_low_thresh = 0.1
                new_track_thresh = 0.6
                track_buffer = 30
                match_thresh = 0.8
                fuse_score = True

            self._byte_tracker = BYTETracker(_Args(), frame_rate=frame_rate)
            self._byte_tracker_ready = True
        except Exception:
            # Fallback: keep using model.track() for every frame
            self._byte_tracker = None
            self._byte_tracker_ready = False

    # ------------------------------------------------------------------
    # Alert helpers (unchanged from original)
    # ------------------------------------------------------------------
    def can_alert(self, name: str, track_key: str) -> bool:
        now = time.time()
        with self.alert_lock:
            last_name = self.last_alert_by_name.get(name, 0.0)
            last_track = self.last_alert_by_track.get(track_key, 0.0)
            if now - last_name < self.alert_cooldown_seconds:
                return False
            if now - last_track < self.alert_cooldown_seconds:
                return False
            self.last_alert_by_name[name] = now
            self.last_alert_by_track[track_key] = now
            return True

    def send_alert(self, name: str, similarity: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        payload = b""
        alert_url = os.environ.get("LARAVEL_ALERT_URL", ALERT_URL)
        request = urllib.request.Request(alert_url, data=payload, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                _ = response.read()
            info = metadata or {}
            print(
                f"[ALERT] Sent to Laravel API ({alert_url}) for {name} "
                f"(similarity={similarity:.3f}, metadata={info})"
            )
            return True
        except Exception as exc:
            print(f"[ALERT] Failed to send alert for {name}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Person detection with YOLO interval support
    # ------------------------------------------------------------------
    def _extract_detections_from_predict(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO predict and return detections as (N, 6) array [x1,y1,x2,y2,conf,cls]."""
        h, w = frame.shape[:2]
        results = self.person_model.predict(
            frame,
            imgsz=self.ai_width,
            classes=[0],
            conf=0.35,
            verbose=False,
            device=self.person_model.device,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 6))
        boxes = results[0].boxes.data.cpu().numpy()  # (N, 6): x1,y1,x2,y2,conf,cls
        return boxes

    def _person_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Get person detections for the current frame.

        On YOLO frames: run full detection + feed to ByteTrack.
        On skip frames: advance ByteTrack with empty detections (Kalman predict only).
        """
        h, w = frame.shape[:2]

        # Lazy-init ByteTrack on first call
        if not self._byte_tracker_ready:
            self._init_byte_tracker()

        is_yolo_frame = (
            self.yolo_interval <= 1
            or self.frame_number <= 1  # always detect on the first AI frame
            or self.frame_number % self.yolo_interval == 0
            or self._byte_tracker is None  # fallback
        )

        if is_yolo_frame:
            # Run YOLO detection
            t0 = time.time()
            dets = self._extract_detections_from_predict(frame)
            self._record_yolo_fps(t0)

            if self._byte_tracker is not None:
                # Feed to standalone ByteTrack
                tracks = self._byte_tracker.update(dets, (h, w), (h, w))
                detections = self._tracks_to_detections(tracks)
            else:
                # Fallback: use model.track() (original behaviour)
                detections = self._fallback_track(frame)
        else:
            # Skip frame – advance ByteTrack with empty detections
            if self._byte_tracker is not None:
                tracks = self._byte_tracker.update(np.empty((0, 6)), (h, w), (h, w))
                detections = self._tracks_to_detections(tracks)
            else:
                detections = self._fallback_track(frame)

        return detections

    def _tracks_to_detections(self, tracks: List[Any]) -> List[Dict[str, Any]]:
        """Convert ByteTrack STrack objects to detection dicts."""
        detections: List[Dict[str, Any]] = []
        for track in tracks:
            if not getattr(track, "is_activated", False):
                continue
            try:
                tlbr = track.tlbr  # [x1, y1, x2, y2]
            except Exception:
                continue
            x1, y1, x2, y2 = map(int, tlbr)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "track_id": track.track_id,
            })
        return detections

    def _fallback_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Original model.track() behaviour for environments where BYTETracker
        cannot be imported from ultralytics."""
        results = self.person_model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],
            conf=0.35,
            verbose=False,
        )
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            track_id = None
            if getattr(box, "id", None) is not None:
                try:
                    track_id = int(box.id[0].cpu().numpy())
                except Exception:
                    track_id = int(box.id[0]) if box.id is not None else None
            detections.append({"bbox": (x1, y1, x2, y2), "track_id": track_id})
        return detections

    # ------------------------------------------------------------------
    # Face quality check
    # ------------------------------------------------------------------
    def _face_is_usable(self, face, person_crop: np.ndarray) -> Tuple[bool, float]:
        """Return (usable, quality_score) for a detected face."""
        fx1, fy1, fx2, fy2 = face.bbox.astype(int).tolist()
        fw = fx2 - fx1
        fh = fy2 - fy1

        if fw < self.min_face_size or fh < self.min_face_size:
            return False, 0.0

        det_score = float(getattr(face, "det_score", 0.5))
        if det_score < self.face_conf_threshold:
            return False, det_score

        # Quality score combines size ratio and detection confidence
        crop_area = person_crop.shape[0] * person_crop.shape[1]
        face_area = fw * fh
        area_ratio = min(face_area / max(crop_area, 1), 1.0) * 5.0  # scale up
        area_ratio = min(area_ratio, 1.0)
        quality = 0.4 * area_ratio + 0.6 * det_score

        return True, quality

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        *,
        send_alerts: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        with self.process_lock:
            return self._process_frame(frame, send_alerts=send_alerts)

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        send_alerts: bool,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process one frame through the AI pipeline.

        Returns (annotated_frame, detections_list).
        This method is designed to be called from the AI thread only.
        """
        t_start = time.time()
        self.frame_number += 1
        self._process_start_time = t_start

        annotated = frame.copy()
        height, width = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        # ── Step 1: Person detection (YOLO periodic + ByteTrack continuous) ──
        persons = self._person_detections(frame)

        # Track seen IDs this frame for cache cleanup
        seen_track_ids: set = set()

        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            x1 = clamp(x1, 0, width - 1)
            y1 = clamp(y1, 0, height - 1)
            x2 = clamp(x2, 0, width - 1)
            y2 = clamp(y2, 0, height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            track_key = str(
                person["track_id"]
                if person["track_id"] is not None
                else f"{x1}-{y1}-{x2}-{y2}"
            )
            seen_track_ids.add(track_key)

            # ── Step 2: Cache lookup ──────────────────────────────────────
            cached = self.face_cache.get(track_key)

            # Decide whether to run SCRFD + ArcFace
            should_recognize = self._should_recognize(track_key, cached)

            match_for_person: Optional[MatchResult] = None
            local_face_boxes: List[Tuple[Tuple[int, int, int, int], MatchResult, float]] = []

            if should_recognize:
                # ── Step 3: Selective SCRFD (face detection) ──────────
                faces = self.face_app.get(person_crop)
                for face in faces:
                    # ── Step 4: Face quality filter ───────────────────
                    usable, quality = self._face_is_usable(face, person_crop)
                    if not usable:
                        continue

                    # ── Step 5: Selective ArcFace (recognition) ───────
                    t_recog = time.time()
                    match = self.face_db.best_match(
                        face.embedding, threshold=self.recognition_threshold
                    )
                    self._record_recognition_fps(t_recog)

                    fx1, fy1, fx2, fy2 = face.bbox.astype(int).tolist()
                    local_face_boxes.append(((fx1, fy1, fx2, fy2), match, quality))

                # ── Step 6: Update cache ──────────────────────────────
                best_match_for_cache = None
                best_score_for_cache = -1.0
                for _, m, _ in local_face_boxes:
                    if m.score > best_score_for_cache:
                        best_score_for_cache = m.score
                        best_match_for_cache = m

                entry = TrackCacheEntry()
                entry.last_recognition_frame = self.frame_number
                entry.last_seen_frame = self.frame_number
                entry.recognition_count = (cached.recognition_count + 1) if cached else 1

                if best_match_for_cache is not None:
                    entry.name = best_match_for_cache.name
                    entry.confidence = best_match_for_cache.score
                    entry.is_unknown = (best_match_for_cache.name == "Unknown")
                    entry.face_quality_score = (
                        max((q for _, _, q in local_face_boxes), default=0.5)
                    )
                else:
                    entry.name = "Unknown"
                    entry.confidence = 0.0
                    entry.is_unknown = True
                    entry.face_quality_score = 0.0

                if entry.is_unknown:
                    entry.unknown_since_frame = (
                        cached.unknown_since_frame
                        if cached and cached.is_unknown
                        else self.frame_number
                    )
                else:
                    entry.unknown_since_frame = 0

                entry.local_face_boxes = local_face_boxes
                self.face_cache[track_key] = entry
            else:
                # Reuse cached result
                local_face_boxes = cached.local_face_boxes if cached else []
                cached.last_seen_frame = self.frame_number

            # Use the best face quality from detected faces for cache decisions
            best_face_quality = max((q for _, _, q in local_face_boxes), default=0.0)

            # ── Step 7: Convert local face boxes to global coords ─────
            face_boxes = []
            for (fx1, fy1, fx2, fy2), match, _ in local_face_boxes:
                global_bbox = (
                    clamp(x1 + fx1, 0, width - 1),
                    clamp(y1 + fy1, 0, height - 1),
                    clamp(x1 + fx2, 0, width - 1),
                    clamp(y1 + fy2, 0, height - 1),
                )
                face_boxes.append((global_bbox, match))
                if match_for_person is None or match.score > match_for_person.score:
                    match_for_person = match

            # ── Step 8: Draw face boxes and labels ────────────────────
            for (fx1, fy1, fx2, fy2), match in face_boxes:
                face_color = (
                    (0, 200, 255) if match.name != "Unknown" else (0, 255, 0)
                )
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), face_color, 2)
                label = (
                    f"{match.name} {match.score:.2f}"
                    if match.name != "Unknown"
                    else f"Unknown {match.score:.2f}"
                )
                draw_label(annotated, label, fx1, fy1, face_color)
                detections.append({
                    "person_bbox": (x1, y1, x2, y2),
                    "face_bbox": (fx1, fy1, fx2, fy2),
                    "name": match.name,
                    "score": float(match.score),
                    "metadata": match.metadata,
                    "track_id": person["track_id"],
                })

            # ── Step 9: Alert for matched persons ─────────────────────
            if (
                match_for_person
                and match_for_person.name != "Unknown"
                and send_alerts
                and should_recognize
                and self.can_alert(match_for_person.name, track_key)
            ):
                print(
                    f"[DETECTED] {match_for_person.name} "
                    f"similarity={match_for_person.score:.3f}"
                )
                if self.detection_callback and self.camera_database_id is not None:
                    # Offload blocking DB write to a background thread
                    threading.Thread(
                        target=self._safe_detection_callback,
                        args=(
                            self.camera_database_id,
                            match_for_person.metadata,
                            match_for_person.score,
                            annotated.copy(),  # copy for thread safety (now includes face boxes)
                        ),
                        daemon=True,
                    ).start()

        self._remember_detection_frame(annotated)

        # ── Step 10: Cleanup stale cache entries ──────────────────────
        self._cleanup_cache(seen_track_ids)

        # ── Step 11: Update shared AI results for display pipeline ────
        now = time.time()
        with self.ai_results_lock:
            self.ai_results.detections = detections
            self.ai_results.frame_number = self.frame_number
            self.ai_results.timestamp = now
        self._update_motion_history(detections, now)

        # ── Step 12: Update processing FPS ────────────────────────────
        self._record_process_fps(t_start)
        self._update_fps_metrics(len(persons))

        return annotated, detections

    # ------------------------------------------------------------------
    # Recognition decision logic
    # ------------------------------------------------------------------
    def _should_recognize(
        self, track_key: str, cached: Optional[TrackCacheEntry]
    ) -> bool:
        """Decide whether to run SCRFD+ArcFace for this track on this frame."""
        # Always recognize new tracks
        if cached is None:
            return True

        frames_since_recog = self.frame_number - cached.last_recognition_frame

        # Unknown persons: re-check after unknown_recheck_interval
        if cached.is_unknown:
            return frames_since_recog >= self.unknown_recheck_interval

        # Known (matched) persons: re-check after recognition_interval
        if frames_since_recog >= self.recognition_interval:
            return True

        # Low-quality previous recognition – re-check sooner
        if cached.face_quality_score < 0.3 and frames_since_recog >= self.recognition_interval // 2:
            return True

        return False

    # ------------------------------------------------------------------
    # Cache cleanup
    # ------------------------------------------------------------------
    def _cleanup_cache(self, seen_track_ids: set) -> None:
        """Remove cache entries for tracks that haven't been seen recently."""
        stale = []
        for track_key, entry in self.face_cache.items():
            frames_since_seen = self.frame_number - entry.last_seen_frame
            if frames_since_seen > self.max_track_age:
                stale.append(track_key)
        for key in stale:
            del self.face_cache[key]
        if stale and len(stale) > 0:
            # Only log when cleaning substantial entries
            pass

    # ------------------------------------------------------------------
    # Safe async detection callback
    # ------------------------------------------------------------------
    def _safe_detection_callback(
        self, camera_id: int, metadata: Dict[str, Any], score: float, frame: np.ndarray
    ) -> None:
        """Collect a short pre/post-match burst before persisting the detection."""
        try:
            if self.detection_callback:
                time.sleep(self.detection_burst_seconds)
                with self._detection_frames_lock:
                    candidates = [frame.copy(), *[item.copy() for item in self._detection_frames]]
                selected: List[np.ndarray] = []
                for candidate in reversed(candidates):
                    if selected and candidate.shape == selected[-1].shape:
                        if float(np.mean(cv2.absdiff(candidate, selected[-1]))) < 1.0:
                            continue
                    selected.append(candidate)
                    if len(selected) >= self.detection_burst_images:
                        break
                self.detection_callback(camera_id, metadata, score, list(reversed(selected)))
        except Exception as exc:
            print(f"[DETECTION] Database record failed: {exc}")

    def _remember_detection_frame(self, frame: np.ndarray) -> None:
        with self._detection_frames_lock:
            self._detection_frames.append(frame.copy())

    # ------------------------------------------------------------------
    # FPS / latency monitoring
    # ------------------------------------------------------------------
    def _record_yolo_fps(self, t_start: float) -> None:
        elapsed = time.time() - t_start
        with self._fps_lock:
            self._yolo_frame_count += 1
            if self._last_yolo_time > 0:
                alpha = 0.1  # EMA smoothing
                instant = 1.0 / max(elapsed, 0.001)
                self._fps_data["yolo_fps"] = (
                    alpha * instant + (1 - alpha) * self._fps_data["yolo_fps"]
                )
            self._last_yolo_time = time.time()

    def _record_recognition_fps(self, t_start: float) -> None:
        elapsed = time.time() - t_start
        with self._fps_lock:
            self._recog_count += 1
            if self._last_recog_time > 0:
                alpha = 0.1
                instant = 1.0 / max(elapsed, 0.001)
                self._fps_data["recognition_fps"] = (
                    alpha * instant + (1 - alpha) * self._fps_data["recognition_fps"]
                )
            self._last_recog_time = time.time()

    def _record_process_fps(self, t_start: float) -> None:
        elapsed = time.time() - t_start
        with self._fps_lock:
            self._process_frame_count += 1
            if self._last_process_time > 0:
                alpha = 0.1
                instant = 1.0 / max(elapsed, 0.001)
                self._fps_data["processing_fps"] = (
                    alpha * instant + (1 - alpha) * self._fps_data["processing_fps"]
                )
            self._last_process_time = time.time()
            self._fps_data["latency_ms"] = elapsed * 1000.0

    def _update_fps_metrics(self, tracked_count: int) -> None:
        with self._fps_lock:
            self._fps_data["tracked_persons"] = tracked_count

    def record_camera_fps(self) -> None:
        """Called by the capture thread each time a new frame is grabbed."""
        now = time.time()
        with self._fps_lock:
            self._camera_frame_count += 1
            if self._last_camera_time > 0:
                alpha = 0.1
                dt = now - self._last_camera_time
                if dt > 0:
                    instant = 1.0 / dt
                    self._fps_data["camera_fps"] = (
                        alpha * instant + (1 - alpha) * self._fps_data["camera_fps"]
                    )
            self._last_camera_time = now

    def get_fps_metrics(self) -> Dict[str, float]:
        """Return current FPS/latency snapshot for display/logging."""
        with self._fps_lock:
            return dict(self._fps_data)

    def get_fps_text(self) -> str:
        """Return a compact FPS string for overlay on the video feed."""
        m = self.get_fps_metrics()
        return (
            f"Cam:{m['camera_fps']:.0f} "
            f"Proc:{m['processing_fps']:.0f} "
            f"YOLO:{m['yolo_fps']:.0f} "
            f"Recog:{m['recognition_fps']:.0f} "
            f"Lat:{m['latency_ms']:.0f}ms "
            f"Trk:{int(m['tracked_persons'])}"
        )

    # ------------------------------------------------------------------
    # Motion history (smooth overlay extrapolation)
    # ------------------------------------------------------------------
    def _update_motion_history(
        self, detections: List[Dict[str, Any]], now: float
    ) -> None:
        """Record each track's velocity so draw_overlays can carry its box
        forward to the current frame, hiding the AI processing latency."""
        with self._motion_lock:
            seen: set = set()
            for d in detections:
                track_id = d.get("track_id")
                if track_id is None:
                    continue
                key = str(track_id)
                seen.add(key)
                face = d.get("face_bbox")
                prev = self._track_history.get(key)
                if face is not None and prev is not None and prev.get("face_bbox"):
                    dt = now - prev["t"]
                    if dt > 1e-3:
                        vx = ((face[0] + face[2]) * 0.5 - prev["center_x"]) / dt
                        vy = ((face[1] + face[3]) * 0.5 - prev["center_y"]) / dt
                        ox, oy = prev.get("vel", (0.0, 0.0))
                        vel = (0.6 * vx + 0.4 * ox, 0.6 * vy + 0.4 * oy)
                    else:
                        vel = prev.get("vel", (0.0, 0.0))
                else:
                    vel = (0.0, 0.0)
                self._track_history[key] = {
                    "t": now,
                    "face_bbox": face,
                    "center_x": (face[0] + face[2]) * 0.5 if face else 0.0,
                    "center_y": (face[1] + face[3]) * 0.5 if face else 0.0,
                    "vel": vel,
                }
            for key in list(self._track_history.keys()):
                if key not in seen:
                    del self._track_history[key]

    # ------------------------------------------------------------------
    # Draw AI overlays onto a display frame (called by capture thread)
    # ------------------------------------------------------------------
    def draw_overlays(self, display_frame: np.ndarray) -> np.ndarray:
        """Draw cached AI results onto a display frame. Fast – no AI inference.

        Boxes are extrapolated forward by each track's velocity so they follow a
        moving face instead of lagging behind it between recognition updates.
        """
        with self.ai_results_lock:
            dets = list(self.ai_results.detections)
            last_ts = self.ai_results.timestamp

        if not dets:
            return display_frame

        height, width = display_frame.shape[:2]
        # Extrapolation horizon: carry each box forward by up to 250ms. Capped so
        # a stalled AI thread can't throw boxes wildly off the frame.
        dt = min(max(time.time() - last_ts, 0.0), 0.25)

        for d in dets:
            fx1, fy1, fx2, fy2 = d.get("face_bbox", (0, 0, 0, 0))
            name = d.get("name", "Unknown")
            score = d.get("score", 0.0)

            track_id = d.get("track_id")
            if track_id is not None and dt > 0.0:
                with self._motion_lock:
                    motion = self._track_history.get(str(track_id))
                if motion and motion.get("vel"):
                    vx, vy = motion["vel"]
                    dx = int(round(vx * dt))
                    dy = int(round(vy * dt))
                    fx1 += dx
                    fy1 += dy
                    fx2 += dx
                    fy2 += dy
                    fx1 = clamp(fx1, 0, width - 1)
                    fy1 = clamp(fy1, 0, height - 1)
                    fx2 = clamp(fx2, 0, width - 1)
                    fy2 = clamp(fy2, 0, height - 1)

            face_color = (0, 200, 255) if name != "Unknown" else (0, 255, 0)
            cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), face_color, 2)
            label = (
                f"{name} {score:.2f}" if name != "Unknown" else f"Unknown {score:.2f}"
            )
            draw_label(display_frame, label, fx1, fy1, face_color)

        # Draw FPS overlay in top-left corner
        fps_text = self.get_fps_text()
        cv2.putText(
            display_frame,
            fps_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return display_frame

    # ------------------------------------------------------------------
    # Register person from image (unchanged)
    # ------------------------------------------------------------------
    def register_person_from_image(
        self,
        image: np.ndarray,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        faces = self.face_app.get(image)
        if len(faces) != 1:
            raise ValueError(
                f"Expected exactly 1 face in the image, found {len(faces)}."
            )
        embedding = faces[0].embedding
        self.face_db.upsert(name, embedding, metadata or {})
        x1, y1, x2, y2 = faces[0].bbox.astype(int).tolist()
        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 255), 2)
        draw_label(preview, f"Registered: {name}", x1, y1, (0, 200, 255))
        return preview, {"name": name, "metadata": metadata or {}}

    # ------------------------------------------------------------------
    # Image / video testing
    # ------------------------------------------------------------------
    def _process_test_frame(
        self,
        frame: np.ndarray,
        *,
        send_alerts: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Recognize faces directly in uploaded test media.

        The live-camera pipeline intentionally runs YOLO before SCRFD for
        throughput.  Requiring a person track for an uploaded portrait is too
        restrictive: cropped and close-up faces frequently have no COCO
        ``person`` detection, which previously produced an unchanged image and
        an empty detection list.  Test media therefore uses SCRFD on the whole
        frame and ArcFace on every new/periodic face.
        """
        with self.process_lock:
            t_start = time.time()
            self.frame_number += 1
            annotated = frame.copy()
            height, width = frame.shape[:2]
            detections: List[Dict[str, Any]] = []
            faces = self.face_app.get(frame)

            refresh_matches = (
                self.frame_number <= 1
                or self.frame_number % self.recognition_interval == 0
            )
            previous_cache = list(self._test_face_cache)
            used_cache_indexes: set[int] = set()
            next_cache: List[Tuple[float, float, MatchResult]] = []

            for face_index, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox.astype(int).tolist()
                x1 = clamp(x1, 0, width - 1)
                y1 = clamp(y1, 0, height - 1)
                x2 = clamp(x2, 0, width - 1)
                y2 = clamp(y2, 0, height - 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                match: Optional[MatchResult] = None

                if not refresh_matches and previous_cache:
                    nearest_index = None
                    nearest_distance = float("inf")
                    for cache_index, (cached_x, cached_y, _) in enumerate(previous_cache):
                        if cache_index in used_cache_indexes:
                            continue
                        distance = (center_x - cached_x) ** 2 + (center_y - cached_y) ** 2
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_index = cache_index

                    max_distance = float(max(x2 - x1, y2 - y1) * 2) ** 2
                    if nearest_index is not None and nearest_distance <= max_distance:
                        used_cache_indexes.add(nearest_index)
                        match = previous_cache[nearest_index][2]

                if match is None:
                    t_recog = time.time()
                    match = self.face_db.best_match(
                        face.embedding, threshold=self.recognition_threshold
                    )
                    self._record_recognition_fps(t_recog)

                next_cache.append((center_x, center_y, match))
                display_score = max(float(match.score), 0.0)
                face_color = (
                    (0, 200, 255) if match.name != "Unknown" else (0, 255, 0)
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), face_color, 2)
                label = (
                    f"{match.name} {display_score:.2f}"
                    if match.name != "Unknown"
                    else f"Unknown {display_score:.2f}"
                )
                draw_label(annotated, label, x1, y1, face_color)

                detections.append({
                    "person_bbox": None,
                    "face_bbox": (x1, y1, x2, y2),
                    "name": match.name,
                    "score": display_score,
                    "face_detection_score": float(getattr(face, "det_score", 0.0)),
                    "metadata": match.metadata,
                    "track_id": f"test-face-{face_index}",
                })

                if (
                    match.name != "Unknown"
                    and send_alerts
                    and refresh_matches
                    and self.can_alert(match.name, f"test-face-{face_index}")
                ):
                    print(
                        f"[DETECTED] {match.name} "
                        f"similarity={display_score:.3f} (uploaded test media)"
                    )
                    if self.detection_callback and self.camera_database_id is not None:
                        threading.Thread(
                            target=self._safe_detection_callback,
                            args=(
                                self.camera_database_id,
                                match.metadata,
                                display_score,
                                annotated.copy(),
                            ),
                            daemon=True,
                        ).start()

            self._test_face_cache = next_cache
            self._remember_detection_frame(annotated)
            now = time.time()
            with self.ai_results_lock:
                self.ai_results.detections = detections
                self.ai_results.frame_number = self.frame_number
                self.ai_results.timestamp = now
            self._record_process_fps(t_start)
            self._update_fps_metrics(len(detections))

            return annotated, detections

    def process_image(
        self,
        image: np.ndarray,
        *,
        send_alerts: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        self.reset_tracking()
        return self._process_test_frame(image, send_alerts=send_alerts)

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        *,
        send_alerts: bool = False,
    ) -> Tuple[Path, Dict[str, Any]]:
        self.reset_tracking()
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read video size.")

        intermediate_path = output_path.with_name(
            f"{output_path.stem}_intermediate.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(intermediate_path), fourcc, fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create the intermediate annotated video.")
        frame_count = 0
        detection_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            original_height, original_width = frame.shape[:2]
            inference_frame = frame
            if original_width > self.inference_max_width:
                scale = self.inference_max_width / original_width
                inference_frame = cv2.resize(
                    frame,
                    (self.inference_max_width, int(original_height * scale)),
                )
            annotated, detections = self._process_test_frame(
                inference_frame, send_alerts=send_alerts
            )
            detection_count += len(detections)
            if annotated.shape[:2] != (original_height, original_width):
                annotated = cv2.resize(
                    annotated, (original_width, original_height)
                )
            writer.write(annotated)

        cap.release()
        writer.release()
        try:
            self._encode_browser_video(intermediate_path, output_path)
        finally:
            intermediate_path.unlink(missing_ok=True)
        return output_path, {
            "frames": frame_count,
            "detections": detection_count,
            "codec": "H.264",
        }

    @staticmethod
    def _ffmpeg_executable() -> Optional[str]:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return shutil.which("ffmpeg")

    def _encode_browser_video(self, source: Path, destination: Path) -> None:
        ffmpeg = self._ffmpeg_executable()
        if not ffmpeg:
            raise RuntimeError(
                "Cannot create a browser-playable MP4. Install imageio-ffmpeg "
                "(included in requirements.txt) or install FFmpeg on PATH."
            )
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=300
        )
        if (
            completed.returncode != 0
            or not destination.exists()
            or destination.stat().st_size == 0
        ):
            message = (completed.stderr or "FFmpeg failed without an error message.")[
                -500:
            ]
            raise RuntimeError(f"H.264 video conversion failed: {message}")

    # ------------------------------------------------------------------
    # Reset tracking state
    # ------------------------------------------------------------------
    def reset_tracking(self) -> None:
        """Start a fresh ByteTrack session for a new image/video source."""
        # Reset ultralytics internal tracker if it exists (for fallback mode)
        predictor = getattr(self.person_model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

        # Reset standalone ByteTrack
        self._byte_tracker = None
        self._byte_tracker_ready = False

        # Clear enhanced face cache
        self.face_cache.clear()
        self._test_face_cache.clear()

        # Reset shared AI results
        with self.ai_results_lock:
            self.ai_results = AIResults()

        self.frame_number = 0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def encode_image_to_data_uri(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
