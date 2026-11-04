"""Settings, face-database reset, and camera management routes."""
from __future__ import annotations

from flask import Flask, abort, flash, redirect, render_template, request, url_for

from app.routes._shared import (
    apply_settings_to_engine,
    camera_streams,
    cameras,
    default_engine,
    face_db,
    settings_store,
    uses_postgres,
)
from app.services.face_service import slugify


def register(app: Flask) -> None:
    @app.route("/settings", methods=["GET", "POST"])
    def settings_page():
        if request.method == "POST":
            try:
                values = {
                    "threshold": float(request.form["threshold"]),
                    "face_interval": int(request.form["face_interval"]),
                    "inference_max_width": int(request.form["inference_max_width"]),
                    "alert_cooldown": int(request.form["alert_cooldown"]),
                    "tracker": request.form["tracker"],
                    "yolo_interval": int(request.form.get("yolo_interval", 3)),
                    "recognition_interval": int(
                        request.form.get("recognition_interval", 15)
                    ),
                    "unknown_recheck_interval": int(
                        request.form.get("unknown_recheck_interval", 30)
                    ),
                    "min_face_size": int(request.form.get("min_face_size", 60)),
                    "face_conf_threshold": float(
                        request.form.get("face_conf_threshold", 0.45)
                    ),
                    "ai_width": int(request.form.get("ai_width", 640)),
                    "max_track_age": int(request.form.get("max_track_age", 60)),
                }
                if (
                    not 0 <= values["threshold"] <= 1
                    or values["face_interval"] < 1
                    or values["inference_max_width"] < 320
                ):
                    raise ValueError
                settings_store().save(values)

                apply_settings_to_engine(default_engine(), values)
                for stream in camera_streams().values():
                    apply_settings_to_engine(stream.engine, values)

                flash(
                    "Settings saved. Recognition settings have been applied to "
                    "active cameras (restart for tracker changes).",
                    "success",
                )
            except (ValueError, KeyError):
                flash("Please enter valid setting values.", "error")
            return redirect(url_for("settings_page"))
        return render_template(
            "settings.html",
            settings=settings_store().get(),
            cameras=cameras(),
            laravel_cameras=uses_postgres(),
        )

    @app.post("/face-database/reset")
    def reset_face_database():
        if request.form.get("confirmation") != "RESET":
            flash(
                "Face database was not reset. Type RESET exactly to confirm.",
                "error",
            )
            return redirect(url_for("settings_page"))
        face_db().clear()
        for stream in camera_streams().values():
            stream.engine.face_cache.clear()
        flash(
            "Face database reset successfully. Camera, detection, and "
            "verification records were not changed.",
            "success",
        )
        return redirect(url_for("settings_page"))

    @app.post("/cameras")
    def add_camera():
        name = request.form.get("name", "").strip()
        source = request.form.get("source", "").strip()
        camera_id = slugify(request.form.get("camera_id", "") or name)
        if not name or not source:
            flash("Camera name and source are required.", "error")
            return redirect(url_for("settings_page"))
        if any(camera["id"] == camera_id for camera in cameras()):
            flash(
                "That camera ID already exists. Choose a different name or ID.",
                "error",
            )
            return redirect(url_for("settings_page"))
        if not (
            source.isdigit()
            or source.startswith(("rtsp://", "http://", "https://"))
        ):
            flash(
                "Source must be a camera index, RTSP URL, or HTTP/HTTPS stream URL.",
                "error",
            )
            return redirect(url_for("settings_page"))
        if uses_postgres():
            registered_by = request.form.get("registered_by", "").strip()
            if not registered_by.isdigit():
                flash(
                    "Registered by must be the numeric Laravel user ID.", "error"
                )
                return redirect(url_for("settings_page"))
            try:
                face_db().create_camera(
                    {
                        "camera_id": camera_id,
                        "name": name,
                        "stream_url": source,
                        "stream_type": request.form.get("stream_type", "mjpeg"),
                        "location": request.form.get("location", "").strip(),
                        "description": request.form.get("description", "").strip(),
                        "registered_by": int(registered_by),
                        "ip_address": request.form.get("ip_address", "").strip(),
                        "port": request.form.get("port", type=int),
                        "username": request.form.get("username", "").strip(),
                        "password": request.form.get("password", ""),
                    }
                )
            except Exception as exc:
                flash(f"Camera could not be saved to PostgreSQL: {exc}", "error")
                return redirect(url_for("settings_page"))
        else:
            current = settings_store().get()
            current["cameras"].append(
                {
                    "id": camera_id,
                    "name": name,
                    "source": source,
                    "enabled": True,
                }
            )
            settings_store().save({"cameras": current["cameras"]})
        flash(
            f"Camera '{name}' added. Start recognition from the dashboard.",
            "success",
        )
        return redirect(url_for("settings_page"))

    @app.post("/cameras/<camera_id>/delete")
    def delete_camera(camera_id: str):
        if uses_postgres():
            abort(403, "Cameras are managed by Laravel.")
        if camera_id == "laptop":
            flash(
                "The default laptop camera cannot be removed; disable it in "
                "settings.json if needed.",
                "error",
            )
            return redirect(url_for("settings_page"))
        current = settings_store().get()
        cameras_list = [
            camera for camera in current["cameras"] if camera["id"] != camera_id
        ]
        if len(cameras_list) == len(current["cameras"]):
            abort(404)
        stream = camera_streams().pop(camera_id, None)
        if stream:
            stream.stop()
        settings_store().save({"cameras": cameras_list})
        flash(f"Camera '{camera_id}' removed.", "success")
        return redirect(url_for("settings_page"))
