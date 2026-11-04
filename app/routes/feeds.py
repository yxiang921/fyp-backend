"""Live camera feed, recognition control, and output file serving."""
from __future__ import annotations

from flask import Flask, Response, abort, flash, jsonify, redirect, request, send_from_directory, url_for

from app.routes._shared import camera_stream, camera_streams, cameras
from app.services.face_service import OUTPUT_DIR


def register(app: Flask) -> None:
    @app.get("/video_feed")
    def video_feed():
        available = cameras()
        if not available:
            abort(404, "No online cameras are available")
        return Response(
            camera_stream(available[0]["id"]).frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/video_feed/<camera_id>")
    def named_video_feed(camera_id: str):
        return Response(
            camera_stream(camera_id).frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

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

    @app.route("/api/cameras/<camera_id>/recognition", methods=["GET", "POST"])
    def camera_recognition_api(camera_id: str):
        available = next((camera for camera in cameras() if camera["id"] == camera_id), None)
        if not available:
            return jsonify({"success": False, "message": "Camera not found or disabled"}), 404

        if request.method == "POST":
            action = (request.get_json(silent=True) or {}).get("action")
            stream = camera_stream(camera_id)
            if action == "start":
                stream.start()
            elif action == "stop":
                stream.stop()
            else:
                return jsonify({"success": False, "message": "Action must be start or stop"}), 422

        stream = camera_streams().get(camera_id)
        running = bool(stream and stream.is_running)
        return jsonify({
            "success": True,
            "camera_id": camera_id,
            "running": running,
            "status": "running" if running else "stopped",
        })

    @app.get("/outputs/<path:filename>")
    def output_file(filename: str):
        return send_from_directory(OUTPUT_DIR, filename)
