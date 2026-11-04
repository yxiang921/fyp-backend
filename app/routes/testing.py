"""Image and video testing routes."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.routes._shared import (
    ALLOWED_IMAGES,
    ALLOWED_VIDEOS,
    allowed_file,
    default_engine,
)
from app.services.face_service import (
    OUTPUT_DIR,
    encode_image_to_data_uri,
    timestamp_slug,
)


def register(app: Flask) -> None:
    @app.route("/test/image", methods=["GET", "POST"])
    def test_image():
        if request.method == "GET":
            return render_template("test_image.html")
        file = request.files.get("image")
        if (
            file is None
            or not file.filename
            or not allowed_file(file.filename, ALLOWED_IMAGES)
        ):
            flash("Upload a supported image file.", "error")
            return redirect(url_for("test_image"))
        raw = file.read()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            flash("The uploaded file could not be read as an image.", "error")
            return redirect(url_for("test_image"))
        default_engine().reset_tracking()
        annotated, detections = default_engine().process_image(
            image, send_alerts=request.form.get("send_alerts") == "on"
        )
        return render_template(
            "test_image.html",
            result_data=encode_image_to_data_uri(annotated),
            detections=detections,
        )

    @app.route("/test/video", methods=["GET", "POST"])
    def test_video():
        if request.method == "GET":
            return render_template("test_video.html")
        file = request.files.get("video")
        if (
            file is None
            or not file.filename
            or not allowed_file(file.filename, ALLOWED_VIDEOS)
        ):
            flash("Upload a supported video file.", "error")
            return redirect(url_for("test_video"))
        extension = Path(secure_filename(file.filename)).suffix.lower()
        token = f"{timestamp_slug()}_{uuid4().hex[:8]}"
        input_path = Path(app.config["UPLOAD_FOLDER"]) / f"input_{token}{extension}"
        output_name = f"annotated_{token}.mp4"
        output_path = OUTPUT_DIR / output_name
        file.save(input_path)
        try:
            _, stats = default_engine().process_video(
                input_path,
                output_path,
                send_alerts=request.form.get("send_alerts") == "on",
            )
        except Exception as exc:
            flash(f"Video processing failed: {exc}", "error")
            return redirect(url_for("test_video"))
        return render_template(
            "test_video.html",
            result_url=url_for("output_file", filename=output_name),
            stats=stats,
        )
