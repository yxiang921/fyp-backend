from __future__ import annotations

import json
import os
import pickle
import threading
import time
import urllib.request
import base64
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime
    YOLO = None

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - handled at runtime
    FaceAnalysis = None


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
DATABASE_DIR = ROOT / "database"
OUTPUT_DIR = ROOT / "outputs"
FACE_DB_FILE = DATABASE_DIR / "face_db.pkl"
METADATA_FILE = DATABASE_DIR / "face_metadata.json"
ALERT_URL = "http://localhost:8000/sendAlert"  # Legacy fallback; use LARAVEL_ALERT_URL.


def ensure_directories() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
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
    return time.strftime("%Y%m%d_%H%M%S")


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


@dataclass
class MatchResult:
    name: str
    score: float
    metadata: Dict[str, Any]


class FaceDatabase:
    def __init__(self, db_file: Path = FACE_DB_FILE, metadata_file: Path = METADATA_FILE) -> None:
        ensure_directories()
        self.db_file = db_file
        self.metadata_file = metadata_file
        # save() is also called from upsert(), so this must be re-entrant.
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
        """Remove all local face profiles and embeddings."""
        with self.lock:
            self.embeddings = {}
            self.metadata = {}
            self.save()

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

    def build_from_images_folder(self, images_dir: Path = IMAGES_DIR) -> List[str]:
        ensure_directories()
        if FaceAnalysis is None:
            raise RuntimeError("insightface is not installed in this environment.")

        face_app = FaceAnalysis(name="buffalo_s")
        face_app.prepare(ctx_id=-1, det_size=(640, 640))

        registered: List[str] = []
        for path in sorted(images_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            image = cv2.imread(str(path))
            if image is None:
                continue

            faces = face_app.get(image)
            if len(faces) != 1:
                continue

            name = path.stem
            self.upsert(name, faces[0].embedding, {"source_image": str(path.name)})
            registered.append(name)

        return registered


class VisionEngine:
    def __init__(
        self,
        face_db: FaceDatabase,
        yolo_model_path: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",
        recognition_threshold: float = 0.5,
        alert_cooldown_seconds: int = 30,
        face_recognition_interval_frames: int = 4,
        inference_max_width: int = 960,
        camera_database_id: Optional[int] = None,
        detection_callback: Optional[Callable[[int, Dict[str, Any], float, np.ndarray], None]] = None,
    ) -> None:
        ensure_directories()
        self.face_db = face_db
        self.yolo_model_path = yolo_model_path
        self.tracker = tracker
        self.recognition_threshold = recognition_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds
        # YOLO tracking still runs each frame for smooth boxes, but face detection
        # and ArcFace are reused per track between these intervals.
        self.face_recognition_interval_frames = max(1, face_recognition_interval_frames)
        self.inference_max_width = max(320, inference_max_width)
        self.camera_database_id = camera_database_id
        self.detection_callback = detection_callback
        self.alert_lock = threading.Lock()
        self.process_lock = threading.RLock()
        self.last_alert_by_name: Dict[str, float] = {}
        self.last_alert_by_track: Dict[str, float] = {}
        self.frame_number = 0
        self.face_cache: Dict[str, Tuple[int, List[Tuple[Tuple[int, int, int, int], MatchResult]]]] = {}

        if YOLO is None:
            raise RuntimeError("ultralytics is not installed in this environment.")
        if FaceAnalysis is None:
            raise RuntimeError("insightface is not installed in this environment.")

        self.person_model = YOLO(yolo_model_path)
        self.face_app = FaceAnalysis(name="buffalo_s")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

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
        request = urllib.request.Request(
            alert_url,
            data=payload,
            method="POST",
        )
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

    def _person_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
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
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "track_id": track_id,
                }
            )
        return detections

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        send_alerts: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        # Ultralytics trackers are stateful; protect them if camera and an upload
        # are requested at the same time.
        with self.process_lock:
            return self._process_frame(frame, send_alerts=send_alerts)

    def _process_frame(
        self,
        frame: np.ndarray,
        *,
        send_alerts: bool,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        self.frame_number += 1
        annotated = frame.copy()
        height, width = frame.shape[:2]
        detections: List[Dict[str, Any]] = []

        persons = self._person_detections(frame)
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

            track_key = str(person["track_id"] if person["track_id"] is not None else f"{x1}-{y1}-{x2}-{y2}")
            cached = self.face_cache.get(track_key)
            should_recognize = (
                cached is None
                or self.frame_number - cached[0] >= self.face_recognition_interval_frames
            )
            match_for_person: Optional[MatchResult] = None
            # Cache face boxes relative to the tracked person's crop, so they can
            # be rendered correctly while ArcFace is intentionally skipped.
            local_face_boxes: List[Tuple[Tuple[int, int, int, int], MatchResult]] = []
            if should_recognize:
                faces = self.face_app.get(person_crop)
                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int).tolist()
                    match = self.face_db.best_match(face.embedding, threshold=self.recognition_threshold)
                    local_face_boxes.append(((fx1, fy1, fx2, fy2), match))
                self.face_cache[track_key] = (self.frame_number, local_face_boxes)
            else:
                local_face_boxes = cached[1]

            face_boxes = []
            for (fx1, fy1, fx2, fy2), match in local_face_boxes:
                global_bbox = (
                    clamp(x1 + fx1, 0, width - 1),
                    clamp(y1 + fy1, 0, height - 1),
                    clamp(x1 + fx2, 0, width - 1),
                    clamp(y1 + fy2, 0, height - 1),
                )
                face_boxes.append((global_bbox, match))
                if match_for_person is None or match.score > match_for_person.score:
                    match_for_person = match

            if match_for_person and match_for_person.name != "Unknown":
                if send_alerts and should_recognize and self.can_alert(match_for_person.name, track_key):
                    print(f"[DETECTED] {match_for_person.name} similarity={match_for_person.score:.3f}")
                    if self.detection_callback and self.camera_database_id is not None:
                        try:
                            self.detection_callback(self.camera_database_id, match_for_person.metadata, match_for_person.score, annotated)
                        except Exception as exc:
                            print(f"[DETECTION] Database record failed: {exc}")
                    # No outbound alert is sent. The database callback above is
                    # the complete detection workflow: screenshot + detection +
                    # pending verification request.

            # Person boxes are only used internally for YOLO/ByteTrack. Rendering
            # them would duplicate the yellow face-recognition overlay.
            for (fx1, fy1, fx2, fy2), match in face_boxes:
                face_color = (0, 200, 255) if match.name != "Unknown" else (0, 255, 0)
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), face_color, 2)
                label = f"{match.name} {match.score:.2f}" if match.name != "Unknown" else f"Unknown {match.score:.2f}"
                draw_label(annotated, label, fx1, fy1, face_color)
                detections.append(
                    {
                        "person_bbox": (x1, y1, x2, y2),
                        "face_bbox": (fx1, fy1, fx2, fy2),
                        "name": match.name,
                        "score": float(match.score),
                        "metadata": match.metadata,
                        "track_id": person["track_id"],
                    }
                )

        return annotated, detections

    def register_person_from_image(
        self,
        image: np.ndarray,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        faces = self.face_app.get(image)
        if len(faces) != 1:
            raise ValueError(f"Expected exactly 1 face in the image, found {len(faces)}.")

        embedding = faces[0].embedding
        self.face_db.upsert(name, embedding, metadata or {})

        x1, y1, x2, y2 = faces[0].bbox.astype(int).tolist()
        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 255), 2)
        draw_label(preview, f"Registered: {name}", x1, y1, (0, 200, 255))

        return preview, {"name": name, "metadata": metadata or {}}

    def process_image(
        self,
        image: np.ndarray,
        *,
        send_alerts: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return self.process_frame(image, send_alerts=send_alerts)

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

        # MP4V files produced by OpenCV often do not play in browsers. Convert
        # the intermediate result to a standards-compatible H.264 MP4 afterwards.
        intermediate_path = output_path.with_name(f"{output_path.stem}_intermediate.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(intermediate_path), fourcc, fps, (width, height))
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
            # Large source videos are processed at a practical resolution, then
            # restored to their original size for an output video of the same shape.
            original_height, original_width = frame.shape[:2]
            inference_frame = frame
            if original_width > self.inference_max_width:
                scale = self.inference_max_width / original_width
                inference_frame = cv2.resize(frame, (self.inference_max_width, int(original_height * scale)))
            annotated, detections = self.process_frame(inference_frame, send_alerts=send_alerts)
            detection_count += len(detections)
            if annotated.shape[:2] != (original_height, original_width):
                annotated = cv2.resize(annotated, (original_width, original_height))
            writer.write(annotated)

        cap.release()
        writer.release()
        try:
            self._encode_browser_video(intermediate_path, output_path)
        finally:
            intermediate_path.unlink(missing_ok=True)
        return output_path, {"frames": frame_count, "detections": detection_count, "codec": "H.264"}

    @staticmethod
    def _ffmpeg_executable() -> Optional[str]:
        """Find a bundled imageio FFmpeg binary before looking on PATH."""
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
        command = [ffmpeg, "-y", "-i", str(source), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
            message = (completed.stderr or "FFmpeg failed without an error message.")[-500:]
            raise RuntimeError(f"H.264 video conversion failed: {message}")

    def reset_tracking(self) -> None:
        """Start a fresh ByteTrack session for a new image/video source."""
        predictor = getattr(self.person_model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()
        self.face_cache.clear()
        self.frame_number = 0


def encode_image_to_data_uri(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
