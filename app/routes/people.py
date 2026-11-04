"""Person registration routes (web form and Laravel ingestion API)."""
from __future__ import annotations

import os
import hmac

import cv2
import numpy as np
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, url_for

from app.routes._shared import ALLOWED_IMAGES, allowed_file, camera_streams, default_engine, face_db
from app.services.face_service import OUTPUT_DIR, slugify, timestamp_slug


def register(app: Flask) -> None:
    @app.post("/api/duplicate-check")
    def duplicate_check():
        api_key = os.environ.get("LARAVEL_API_KEY", "")
        if not api_key or not hmac.compare_digest(request.headers.get("X-API-Key", ""), api_key):
            abort(401)
        photo = request.files.get("photo")
        if not photo:
            return jsonify(error="photo is required"), 422
        raw = photo.read(10 * 1024 * 1024 + 1)
        if len(raw) > 10 * 1024 * 1024:
            abort(413)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(error="Invalid image"), 422
        engine = default_engine()
        with engine.process_lock:
            faces = engine.face_app.get(frame)
        if len(faces) != 1:
            return jsonify(face_status="no_face" if not faces else "multiple_faces", matches=[])
        try:
            threshold = max(0.5, min(0.95, float(request.form.get("threshold", 0.65))))
        except ValueError:
            return jsonify(error="Invalid threshold"), 422
        matches = face_db().duplicate_matches(faces[0].normed_embedding, threshold)
        return jsonify(face_status="checked", matches=matches)

    @app.post("/api/missing-persons")
    def api_register_person():
        api_key = os.environ.get("LARAVEL_API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key:
            abort(401)
        image_file = request.files.get("photo")
        name = request.form.get("fullname", "").strip()
        case_number = request.form.get("case_number", "").strip()
        if (
            not name
            or not case_number
            or not image_file
            or not allowed_file(image_file.filename, ALLOWED_IMAGES)
        ):
            return (
                jsonify(error="multipart fields fullname, case_number, and photo are required"),
                400,
            )
        image = cv2.imdecode(
            np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            return jsonify(error="Invalid image"), 400
        metadata = {"person_id": case_number, "case_number": case_number}
        try:
            _, record = default_engine().register_person_from_image(
                image, name, {k: v for k, v in metadata.items() if v}
            )
        except ValueError as exc:
            return jsonify(error=str(exc)), 422
        except Exception as exc:
            app.logger.exception("Failed to store face profile from Laravel API")
            return (
                jsonify(error=f"Face profile database save failed: {exc}"),
                500,
            )
        return jsonify(status="registered", **record), 201

    @app.delete("/api/missing-persons/<case_number>")
    def api_delete_person(case_number: str):
        api_key = os.environ.get("LARAVEL_API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key:
            abort(401)

        removed = face_db().delete_case(case_number)
        # Prevent an already-tracked face from continuing to be displayed or
        # reported after its Laravel case has been deleted.
        engines = [default_engine()]
        engines.extend(stream.engine for stream in camera_streams().values())
        for engine in engines:
            engine.reset_tracking()

        return jsonify(status="deleted", case_number=case_number, removed=removed)

    @app.route("/people/register", methods=["GET", "POST"])
    def register_person():
        if request.method == "GET":
            return render_template("register_person.html")
        file = request.files.get("image")
        name = request.form.get("name", "").strip()
        if (
            not name
            or file is None
            or not file.filename
            or not allowed_file(file.filename, ALLOWED_IMAGES)
        ):
            flash(
                "Provide a name and a JPG, PNG, BMP, or WEBP image.", "error"
            )
            return redirect(url_for("register_person"))
        image_bytes = file.read()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            flash("The uploaded file could not be read as an image.", "error")
            return redirect(url_for("register_person"))
        metadata = {
            key: request.form.get(key, "").strip()
            for key in ("age", "gender", "notes")
        }
        metadata = {key: value for key, value in metadata.items() if value}
        try:
            preview, _ = default_engine().register_person_from_image(image, name, metadata)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("register_person"))
        image_name = f"registered_{slugify(name)}_{timestamp_slug()}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / image_name), preview)
        flash(f"{name} was registered successfully.", "success")
        return render_template(
            "register_person.html",
            result_url=url_for("output_file", filename=image_name),
        )
