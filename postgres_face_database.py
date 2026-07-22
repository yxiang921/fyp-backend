"""PostgreSQL-backed face database. Embeddings are stored as float32 BYTEA."""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2

from face_service import MatchResult, cosine_similarity


class PostgresFaceDatabase:
    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or os.environ["DATABASE_URL"]
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] to use PostgreSQL.") from exc

    def _connect(self):
        import psycopg
        return psycopg.connect(self.database_url)

    def upsert(self, name: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        metadata = metadata or {}
        person_id = str(metadata.get("person_id") or name)
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO face_profiles (external_id, full_name, age, gender, notes, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                   ON CONFLICT (external_id) DO UPDATE SET full_name=EXCLUDED.full_name,
                   age=EXCLUDED.age, gender=EXCLUDED.gender, notes=EXCLUDED.notes,
                   metadata=EXCLUDED.metadata, updated_at=NOW()""",
                (person_id, name, metadata.get("age") or None, metadata.get("gender") or None,
                 metadata.get("notes") or None, json.dumps(metadata)),
            )
            cur.execute(
                """INSERT INTO face_embeddings (external_id, embedding, dimensions)
                   VALUES (%s, %s, %s) ON CONFLICT (external_id) DO UPDATE SET
                   embedding=EXCLUDED.embedding, dimensions=EXCLUDED.dimensions, updated_at=NOW()""",
                (person_id, vector.tobytes(), vector.size),
            )

    def list_people(self) -> List[Dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT external_id, full_name, metadata FROM face_profiles ORDER BY full_name")
            return [{"name": row[1], "metadata": {**(row[2] or {}), "person_id": row[0]}} for row in cur.fetchall()]

    def clear(self) -> None:
        """Remove Python face profiles only; Laravel cases/detections are retained."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM face_embeddings")
            cur.execute("DELETE FROM face_profiles")

    def best_match(self, embedding: np.ndarray, threshold: float = 0.5) -> MatchResult:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT p.external_id, p.full_name, p.metadata, f.embedding, f.dimensions
                           FROM face_profiles p JOIN face_embeddings f USING (external_id)""")
            best = MatchResult("Unknown", -1.0, {})
            for person_id, name, metadata, raw, dimensions in cur.fetchall():
                score = cosine_similarity(embedding, np.frombuffer(raw, dtype=np.float32, count=dimensions))
                if score > best.score:
                    best = MatchResult(name, score, {**(metadata or {}), "person_id": person_id})
        return best if best.score >= threshold else MatchResult("Unknown", best.score, {})

    def list_cameras(self) -> List[Dict[str, Any]]:
        """Read Laravel-managed cameras; Flask never writes the cameras table."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id, camera_id, name, location, stream_url, stream_type,
                           ip_address, port, username, password, status, description
                           FROM cameras WHERE status = 'online' ORDER BY name""")
            cameras = []
            for database_id, camera_id, name, location, url, stream_type, ip, port, username, password, status, description in cur.fetchall():
                source = url
                if not source and ip:
                    source = f"http://{ip}:{port or 80}"
                if source and username and password and "://" in source and "@" not in source:
                    scheme, rest = source.split("://", 1)
                    source = f"{scheme}://{username}:{password}@{rest}"
                if source:
                    cameras.append({"id": str(camera_id), "database_id": database_id, "name": name, "location": location,
                                    "source": source, "stream_type": stream_type, "status": status,
                                    "description": description, "enabled": True})
            return cameras

    def create_camera(self, camera: Dict[str, Any]) -> None:
        """Create a camera in the shared Laravel cameras table."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO cameras (camera_id, name, location, status, description,
                   registered_by, ip_address, port, stream_url, stream_type, username, password,
                   created_at, updated_at)
                   VALUES (%s, %s, %s, 'online', %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())""",
                (camera["camera_id"], camera["name"], camera.get("location") or None,
                 camera.get("description") or None, camera["registered_by"],
                 camera.get("ip_address") or None, camera.get("port") or None,
                 camera["stream_url"], camera.get("stream_type") or "mjpeg",
                 camera.get("username") or None, camera.get("password") or None),
            )

    def record_detection(self, camera_id: int, metadata: Dict[str, Any], similarity: float, frame: np.ndarray) -> None:
        """Create the detection and its pending manual-verification request."""
        case_number = str(metadata.get("case_number") or "")
        if not case_number:
            print("[DETECTION] Skipped: matched profile has no case number.")
            return
        table = os.environ.get("MISSING_PERSON_TABLE", "missing_persons")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise RuntimeError("MISSING_PERSON_TABLE contains an invalid table name.")
        requested_by = int(os.environ.get("PYTHON_SYSTEM_USER_ID", "0"))
        if requested_by <= 0:
            raise RuntimeError("Set PYTHON_SYSTEM_USER_ID to a Laravel user ID.")
        from face_service import OUTPUT_DIR, timestamp_slug
        screenshots = OUTPUT_DIR / "detections"; screenshots.mkdir(parents=True, exist_ok=True)
        filename = f"detection_{timestamp_slug()}_{uuid.uuid4().hex[:8]}.jpg"
        relative_path = f"detections/{filename}"
        if not cv2.imwrite(str(screenshots / filename), frame):
            raise RuntimeError("Could not save detection screenshot.")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {table} WHERE case_number = %s LIMIT 1", (case_number,))
            person = cur.fetchone()
            if not person:
                raise RuntimeError(f"No Laravel missing-person record found for case {case_number}.")
            cur.execute(
                """INSERT INTO detections (detection_id, camera_id, captured_image, reference_type, reference_id,
                   similarity_score, status, detected_at, created_at, updated_at)
                   VALUES (%s, %s, %s, 'missing_person', %s, %s, 'new', NOW(), NOW(), NOW()) RETURNING id""",
                (f"DET-{uuid.uuid4().hex[:12].upper()}", camera_id, relative_path, person[0], round(similarity * 100, 2)),
            )
            detection_id = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO verification_requests (request_id, detection_id, requested_by, status, created_at, updated_at)
                   VALUES (%s, %s, %s, 'pending', NOW(), NOW())""",
                (f"VR-{uuid.uuid4().hex[:12].upper()}", detection_id, requested_by),
            )
