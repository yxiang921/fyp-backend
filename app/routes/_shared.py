"""Shared helpers and state accessors used across route modules."""
from __future__ import annotations

from typing import Optional

from flask import abort, current_app

from app.camera_stream import CameraStream
from app.services.face_service import VisionEngine

ALLOWED_IMAGES = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def settings_store():
    return current_app.extensions["settings"]


def face_db():
    return current_app.extensions["face_db"]


def uses_postgres() -> bool:
    return current_app.extensions["uses_postgres"]


def default_engine() -> VisionEngine:
    return current_app.extensions["vision_engine"]


def camera_streams() -> dict:
    return current_app.extensions["camera_streams"]


def new_engine(camera_database_id: Optional[int] = None) -> VisionEngine:
    values = settings_store().get()
    db = face_db()
    return VisionEngine(
        db,
        tracker=values["tracker"],
        recognition_threshold=float(values["threshold"]),
        alert_cooldown_seconds=int(values["alert_cooldown"]),
        face_recognition_interval_frames=int(values["face_interval"]),
        inference_max_width=int(values["inference_max_width"]),
        camera_database_id=camera_database_id,
        detection_callback=db.record_detection if uses_postgres() else None,
        yolo_interval=int(values.get("yolo_interval", 3)),
        recognition_interval=int(values.get("recognition_interval", 15)),
        unknown_recheck_interval=int(values.get("unknown_recheck_interval", 30)),
        min_face_size=int(values.get("min_face_size", 60)),
        face_conf_threshold=float(values.get("face_conf_threshold", 0.45)),
        ai_width=int(values.get("ai_width", 640)),
        max_track_age=int(values.get("max_track_age", 60)),
    )


def cameras():
    if uses_postgres():
        return face_db().list_cameras()
    return settings_store().get()["cameras"]


def camera_stream(camera_id: str) -> CameraStream:
    streams = camera_streams()
    for camera in cameras():
        if camera["id"] == camera_id and camera.get("enabled", True):
            if camera_id not in streams:
                streams[camera_id] = CameraStream(
                    new_engine(camera.get("database_id")), camera["source"]
                )
            return streams[camera_id]
    abort(404, "Camera not found or disabled")


def apply_settings_to_engine(engine: VisionEngine, values: dict) -> None:
    engine.recognition_threshold = values["threshold"]
    engine.face_recognition_interval_frames = values["face_interval"]
    engine.inference_max_width = values["inference_max_width"]
    engine.alert_cooldown_seconds = values["alert_cooldown"]
    engine.yolo_interval = int(values.get("yolo_interval", 3))
    engine.recognition_interval = int(values.get("recognition_interval", 15))
    engine.unknown_recheck_interval = int(values.get("unknown_recheck_interval", 30))
    engine.min_face_size = int(values.get("min_face_size", 60))
    engine.face_conf_threshold = float(values.get("face_conf_threshold", 0.45))
    engine.ai_width = int(values.get("ai_width", 640))
    engine.max_track_age = int(values.get("max_track_age", 60))
