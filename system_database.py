"""PostgreSQL access layer for the Python missing-person web system."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from werkzeug.security import check_password_hash, generate_password_hash


class SystemDatabase:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.environ["DATABASE_URL"]

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                yield cur

    def authenticate(self, email: str, password: str) -> dict | None:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE lower(email)=lower(%s) AND is_active=TRUE", (email,))
            user = cur.fetchone()
        if not user:
            return None
        # Laravel's bcrypt hashes and Werkzeug hashes are both accepted while migrating.
        stored = user["password"]
        if stored.startswith("$2"):
            import bcrypt
            valid = bcrypt.checkpw(password.encode(), stored.encode())
        else:
            valid = check_password_hash(stored, password)
        return user if valid else None

    def bootstrap_admin(self) -> None:
        """Create a safe initial administrator from environment variables once."""
        email = os.environ.get("INITIAL_ADMIN_EMAIL")
        password = os.environ.get("INITIAL_ADMIN_PASSWORD")
        if not email or not password:
            return
        with self.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE lower(email)=lower(%s)", (email,))
            if not cur.fetchone():
                cur.execute("INSERT INTO users (name,email,password,role,created_at,updated_at) VALUES (%s,%s,%s,'super_admin',now(),now())",
                            (os.environ.get("INITIAL_ADMIN_NAME", "System Administrator"), email, generate_password_hash(password)))

    def dashboard_counts(self, role: str, user_id: int) -> dict:
        with self.cursor() as cur:
            cur.execute("SELECT count(*) AS cases FROM missing_persons WHERE status IN ('pending','active','reopened')")
            result = {"open_cases": cur.fetchone()["cases"]}
            cur.execute("SELECT count(*) AS alerts FROM alerts WHERE status IN ('under_review','escalated')")
            result["confirmed_alerts"] = cur.fetchone()["alerts"]
            if role == "patrol_unit":
                cur.execute("SELECT count(*) AS jobs FROM alerts WHERE assigned_to=%s AND status IN ('under_review','escalated')", (user_id,))
                result["active_jobs"] = cur.fetchone()["jobs"]
            else:
                cur.execute("SELECT count(*) AS detections FROM detections WHERE status='new'")
                result["pending_detections"] = cur.fetchone()["detections"]
            return result
