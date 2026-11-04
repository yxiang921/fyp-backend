"""HTTP route registration. Each module exposes a ``register(app)`` function."""
from __future__ import annotations

from flask import Flask

from app.routes import api, dashboard, feeds, people, settings, testing


def register_routes(app: Flask) -> None:
    dashboard.register(app)
    settings.register(app)
    people.register(app)
    testing.register(app)
    feeds.register(app)
    api.register(app)
