"""Flask application factory for the missing-person recognition system.

Layout
------
services/         – recognition engine, face database, and settings store
routes/           – HTTP handlers, each registered via ``register(app)``
camera_stream.py  – the two-thread camera capture / AI stream
"""
from __future__ import annotations

import os
import base64
import hmac
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

from app.services.face_service import (
    OUTPUT_DIR,
    FaceDatabase,
    ensure_directories,
)
from app.services.postgres_face_database import PostgresFaceDatabase
from app.services.settings_store import SettingsStore


def create_app() -> Flask:
    load_dotenv()
    ensure_directories()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "change-this-for-production"),
        MAX_CONTENT_LENGTH=250 * 1024 * 1024,
        UPLOAD_FOLDER=str(OUTPUT_DIR / "uploads"),
    )

    @app.before_request
    def protect_ui_when_configured():
        """Optional HTTP Basic protection for the Flask operator UI.

        Development remains backwards-compatible when credentials are not
        configured. Production deployments should set both variables or put
        the service behind an authenticated reverse proxy.
        """
        if request.endpoint == "health":
            return None
        if request.path.startswith("/api/"):
            expected = os.environ.get("PYTHON_API_KEY", "")
            supplied = request.headers.get("X-API-Key", "")
            if expected and not hmac.compare_digest(supplied, expected):
                return jsonify({"success": False, "message": "Invalid API key"}), 401
            return None
        if not (
            os.environ.get("FLASK_UI_USERNAME") and os.environ.get("FLASK_UI_PASSWORD")
        ):
            return None
        header = request.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="FindSight"'})
        try:
            decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError):
            username = password = ""
        valid = hmac.compare_digest(username, os.environ.get("FLASK_UI_USERNAME", "")) and hmac.compare_digest(
            password, os.environ.get("FLASK_UI_PASSWORD", "")
        )
        if not valid:
            return Response("Invalid credentials", 401, {"WWW-Authenticate": 'Basic realm="FindSight"'})
        return None
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    settings = SettingsStore()
    uses_postgres = bool(os.environ.get("DATABASE_URL"))
    face_db = PostgresFaceDatabase() if uses_postgres else FaceDatabase()

    app.extensions["settings"] = settings
    app.extensions["face_db"] = face_db
    app.extensions["uses_postgres"] = uses_postgres
    app.extensions["camera_streams"] = {}
    app.extensions["vision_engine"] = None

    # Build the default engine inside an app context so the shared helpers
    # (which read current_app.extensions) can be used.
    with app.app_context():
        from app.routes._shared import new_engine

        app.extensions["vision_engine"] = new_engine()

    from app.routes import register_routes

    register_routes(app)

    @app.context_processor
    def navigation_context():
        return {"people_count": len(face_db.list_people())}

    @app.get("/health")
    def health():
        return {"status": "ok", "postgres": uses_postgres}

    return app
