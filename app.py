"""Flask web application for the missing-person recognition demonstration."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional
from uuid import uuid4

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from face_service import (
    OUTPUT_DIR,
    FaceDatabase,
    VisionEngine,
    encode_image_to_data_uri,
    ensure_directories,
    slugify,
    timestamp_slug,
)
from postgres_face_database import PostgresFaceDatabase
from settings_store import SettingsStore

ALLOWED_IMAGES = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv", "webm"}


def allowed_file(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


class CameraStream:
    """Owns the camera and keeps the latest annotated JPEG frame for all viewers."""

    def __init__(self, engine: VisionEngine, camera_index: str | int = 0) -> None:
        self.engine = engine
        self.camera_index = camera_index
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._started = False
        self._stop_requested = threading.Event()
        self._camera: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True
            self._stop_requested.clear()
            self.engine.reset_tracking()
            threading.Thread(target=self._run, daemon=True, name="camera-recognition").start()

    def stop(self) -> None:
        """Stop inference and free the camera/RTSP connection."""
        self._stop_requested.set()
        with self._lock:
            self._frame = None
        with self._state_lock:
            camera = self._camera
        if camera is not None:
            camera.release()

    def _run(self) -> None:
        source = int(self.camera_index) if str(self.camera_index).isdigit() else self.camera_index
        camera = cv2.VideoCapture(source)
        with self._state_lock:
            self._camera = camera
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not camera.isOpened():
            print("[CAMERA] Unable to open camera 0. Check camera permissions or index.")
            with self._state_lock:
                self._started = False
                self._camera = None
            return
        try:
            while not self._stop_requested.is_set():
                ok, frame = camera.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                annotated, _ = self.engine.process_frame(frame, send_alerts=True)
                ok, encoded = cv2.imencode(".jpg", annotated)
                if ok:
                    with self._lock:
                        self._frame = encoded.tobytes()
        finally:
            camera.release()
            with self._state_lock:
                self._started = False
                self._camera = None

    def frames(self) -> Iterator[bytes]:
        while True:
            with self._lock:
                frame = self._frame
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)


def create_app() -> Flask:
    load_dotenv()
    ensure_directories()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "change-this-for-production"),
        MAX_CONTENT_LENGTH=250 * 1024 * 1024,
        UPLOAD_FOLDER=str(OUTPUT_DIR / "uploads"),
    )
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    settings = SettingsStore()
    config = settings.get()
    uses_postgres = bool(os.environ.get("DATABASE_URL"))
    face_db = PostgresFaceDatabase() if uses_postgres else FaceDatabase()
    def new_engine(camera_database_id: Optional[int] = None) -> VisionEngine:
        values = settings.get()
        return VisionEngine(face_db, tracker=values["tracker"], recognition_threshold=float(values["threshold"]),
            alert_cooldown_seconds=int(values["alert_cooldown"]), face_recognition_interval_frames=int(values["face_interval"]),
            inference_max_width=int(values["inference_max_width"]), camera_database_id=camera_database_id,
            detection_callback=face_db.record_detection if uses_postgres else None)
    engine = new_engine()
    camera_streams: dict[str, CameraStream] = {}
    def cameras():
        return face_db.list_cameras() if uses_postgres else settings.get()["cameras"]
    def camera_stream(camera_id: str) -> CameraStream:
        for camera in cameras():
            if camera["id"] == camera_id and camera.get("enabled", True):
                if camera_id not in camera_streams:
                    camera_streams[camera_id] = CameraStream(new_engine(camera.get("database_id")), camera["source"])
                return camera_streams[camera_id]
        abort(404, "Camera not found or disabled")
    app.extensions["face_db"] = face_db
    app.extensions["vision_engine"] = engine
    app.extensions["settings"] = settings

    @app.context_processor
    def navigation_context():
        return {"people_count": len(face_db.list_people())}

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html", people=face_db.list_people(), cameras=cameras())

    @app.route("/settings", methods=["GET", "POST"])
    def settings_page():
        if request.method == "POST":
            try:
                values = {"threshold": float(request.form["threshold"]), "face_interval": int(request.form["face_interval"]), "inference_max_width": int(request.form["inference_max_width"]), "alert_cooldown": int(request.form["alert_cooldown"]), "tracker": request.form["tracker"]}
                if not 0 <= values["threshold"] <= 1 or values["face_interval"] < 1 or values["inference_max_width"] < 320: raise ValueError
                settings.save(values)
                for stream in camera_streams.values():
                    stream.engine.recognition_threshold = values["threshold"]
                    stream.engine.face_recognition_interval_frames = values["face_interval"]
                    stream.engine.inference_max_width = values["inference_max_width"]
                    stream.engine.alert_cooldown_seconds = values["alert_cooldown"]
                engine.recognition_threshold = values["threshold"]
                engine.face_recognition_interval_frames = values["face_interval"]
                engine.inference_max_width = values["inference_max_width"]
                engine.alert_cooldown_seconds = values["alert_cooldown"]
                flash("Settings saved. Recognition settings have been applied to active cameras (restart for tracker changes).", "success")
            except (ValueError, KeyError): flash("Please enter valid setting values.", "error")
            return redirect(url_for("settings_page"))
        return render_template("settings.html", settings=settings.get(), cameras=cameras(), laravel_cameras=uses_postgres)

    @app.post("/face-database/reset")
    def reset_face_database():
        if request.form.get("confirmation") != "RESET":
            flash("Face database was not reset. Type RESET exactly to confirm.", "error")
            return redirect(url_for("settings_page"))
        face_db.clear()
        for stream in camera_streams.values():
            stream.engine.face_cache.clear()
        flash("Face database reset successfully. Camera, detection, and verification records were not changed.", "success")
        return redirect(url_for("settings_page"))

    @app.post("/cameras")
    def add_camera():
        name = request.form.get("name", "").strip()
        source = request.form.get("source", "").strip()
        camera_id = slugify(request.form.get("camera_id", "") or name)
        if not name or not source:
            flash("Camera name and source are required.", "error")
            return redirect(url_for("settings_page"))
        current = settings.get()
        if any(camera["id"] == camera_id for camera in cameras()):
            flash("That camera ID already exists. Choose a different name or ID.", "error")
            return redirect(url_for("settings_page"))
        # Allow a local camera index, RTSP, HTTP/MJPEG, or another OpenCV-supported URL.
        if not (source.isdigit() or source.startswith(("rtsp://", "http://", "https://"))):
            flash("Source must be a camera index, RTSP URL, or HTTP/HTTPS stream URL.", "error")
            return redirect(url_for("settings_page"))
        if uses_postgres:
            registered_by = request.form.get("registered_by", "").strip()
            if not registered_by.isdigit():
                flash("Registered by must be the numeric Laravel user ID.", "error")
                return redirect(url_for("settings_page"))
            try:
                face_db.create_camera({"camera_id": camera_id, "name": name, "stream_url": source,
                    "stream_type": request.form.get("stream_type", "mjpeg"), "location": request.form.get("location", "").strip(),
                    "description": request.form.get("description", "").strip(), "registered_by": int(registered_by),
                    "ip_address": request.form.get("ip_address", "").strip(), "port": request.form.get("port", type=int),
                    "username": request.form.get("username", "").strip(), "password": request.form.get("password", "")})
            except Exception as exc:
                flash(f"Camera could not be saved to PostgreSQL: {exc}", "error")
                return redirect(url_for("settings_page"))
        else:
            current["cameras"].append({"id": camera_id, "name": name, "source": source, "enabled": True})
            settings.save({"cameras": current["cameras"]})
        flash(f"Camera '{name}' added. Start recognition from the dashboard.", "success")
        return redirect(url_for("settings_page"))

    @app.post("/cameras/<camera_id>/delete")
    def delete_camera(camera_id: str):
        if uses_postgres:
            abort(403, "Cameras are managed by Laravel.")
        if camera_id == "laptop":
            flash("The default laptop camera cannot be removed; disable it in settings.json if needed.", "error")
            return redirect(url_for("settings_page"))
        current = settings.get()
        cameras = [camera for camera in current["cameras"] if camera["id"] != camera_id]
        if len(cameras) == len(current["cameras"]): abort(404)
        stream = camera_streams.pop(camera_id, None)
        if stream: stream.stop()
        settings.save({"cameras": cameras})
        flash(f"Camera '{camera_id}' removed.", "success")
        return redirect(url_for("settings_page"))

    @app.post("/api/missing-persons")
    def api_register_person():
        api_key = os.environ.get("LARAVEL_API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key: abort(401)
        image_file, name = request.files.get("photo"), request.form.get("fullname", "").strip()
        case_number = request.form.get("case_number", "").strip()
        if not name or not case_number or not image_file or not allowed_file(image_file.filename, ALLOWED_IMAGES):
            return jsonify(error="multipart fields fullname, case_number, and photo are required"), 400
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None: return jsonify(error="Invalid image"), 400
        metadata = {"person_id": case_number, "case_number": case_number}
        try:
            _, record = engine.register_person_from_image(image, name, {k:v for k,v in metadata.items() if v})
        except ValueError as exc: return jsonify(error=str(exc)), 422
        except Exception as exc:
            app.logger.exception("Failed to store face profile from Laravel API")
            return jsonify(error=f"Face profile database save failed: {exc}"), 500
        return jsonify(status="registered", **record), 201

    @app.route("/people/register", methods=["GET", "POST"])
    def register_person():
        if request.method == "GET":
            return render_template("register_person.html")
        file = request.files.get("image")
        name = request.form.get("name", "").strip()
        if not name or file is None or not file.filename or not allowed_file(file.filename, ALLOWED_IMAGES):
            flash("Provide a name and a JPG, PNG, BMP, or WEBP image.", "error")
            return redirect(url_for("register_person"))
        image_bytes = file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            flash("The uploaded file could not be read as an image.", "error")
            return redirect(url_for("register_person"))
        metadata = {key: request.form.get(key, "").strip() for key in ("age", "gender", "notes")}
        metadata = {key: value for key, value in metadata.items() if value}
        try:
            preview, _ = engine.register_person_from_image(image, name, metadata)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("register_person"))
        image_name = f"registered_{slugify(name)}_{timestamp_slug()}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / image_name), preview)
        flash(f"{name} was registered successfully.", "success")
        return render_template("register_person.html", result_url=url_for("output_file", filename=image_name))

    @app.route("/test/image", methods=["GET", "POST"])
    def test_image():
        if request.method == "GET":
            return render_template("test_image.html")
        file = request.files.get("image")
        if file is None or not file.filename or not allowed_file(file.filename, ALLOWED_IMAGES):
            flash("Upload a supported image file.", "error")
            return redirect(url_for("test_image"))
        raw = file.read()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            flash("The uploaded file could not be read as an image.", "error")
            return redirect(url_for("test_image"))
        engine.reset_tracking()
        annotated, detections = engine.process_image(image, send_alerts=request.form.get("send_alerts") == "on")
        return render_template("test_image.html", result_data=encode_image_to_data_uri(annotated), detections=detections)

    @app.route("/test/video", methods=["GET", "POST"])
    def test_video():
        if request.method == "GET":
            return render_template("test_video.html")
        file = request.files.get("video")
        if file is None or not file.filename or not allowed_file(file.filename, ALLOWED_VIDEOS):
            flash("Upload a supported video file.", "error")
            return redirect(url_for("test_video"))
        extension = Path(secure_filename(file.filename)).suffix.lower()
        token = f"{timestamp_slug()}_{uuid4().hex[:8]}"
        input_path = Path(app.config["UPLOAD_FOLDER"]) / f"input_{token}{extension}"
        output_name = f"annotated_{token}.mp4"
        output_path = OUTPUT_DIR / output_name
        file.save(input_path)
        try:
            _, stats = engine.process_video(input_path, output_path, send_alerts=request.form.get("send_alerts") == "on")
        except Exception as exc:
            flash(f"Video processing failed: {exc}", "error")
            return redirect(url_for("test_video"))
        return render_template("test_video.html", result_url=url_for("output_file", filename=output_name), stats=stats)

    @app.get("/video_feed")
    def video_feed():
        available = cameras()
        if not available:
            abort(404, "No online cameras are available")
        return Response(camera_stream(available[0]["id"]).frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/video_feed/<camera_id>")
    def named_video_feed(camera_id: str):
        return Response(camera_stream(camera_id).frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/cameras/<camera_id>/recognition")
    def camera_recognition(camera_id: str):
        stream = camera_stream(camera_id)
        action = request.form.get("action")
        if action == "start":
            stream.start()
            flash(f"Recognition started for {camera_id}.", "success")
        elif action == "stop":
            stream.stop()
            flash(f"Recognition stopped for {camera_id}.", "success")
        else:
            abort(400, "Action must be start or stop")
        return redirect(request.referrer or url_for("dashboard"))

    @app.get("/outputs/<path:filename>")
    def output_file(filename: str):
        return send_from_directory(OUTPUT_DIR, filename)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), threaded=True, debug=False)
