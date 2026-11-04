"""Monitoring / performance API."""
from __future__ import annotations

from flask import Flask, jsonify

from app.routes._shared import camera_streams, default_engine


def register(app: Flask) -> None:
    @app.get("/api/performance")
    def performance_api():
        """Return live FPS and latency metrics for all active camera streams."""
        result = {}
        for cam_id, stream in camera_streams().items():
            result[cam_id] = stream.engine.get_fps_metrics()
        if not result:
            result["default"] = default_engine().get_fps_metrics()
        return jsonify(result)
