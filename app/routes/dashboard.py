"""Landing page route."""
from __future__ import annotations

from flask import Flask, render_template

from app.routes._shared import cameras, face_db


def register(app: Flask) -> None:
    @app.get("/")
    def dashboard():
        return render_template(
            "dashboard.html", people=face_db().list_people(), cameras=cameras()
        )
