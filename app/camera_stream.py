"""Two-thread camera stream: a capture/display thread plus an asynchronous AI thread."""
from __future__ import annotations

import threading
import time
from typing import Iterator, Optional

import cv2
import numpy as np

from app.services.face_service import VisionEngine


class CameraStream:
    """Captures camera frames at full FPS and runs AI asynchronously.

    Thread 1 (display):  camera.read() → draw AI overlays → encode JPEG → publish
    Thread 2 (AI):       latest raw frame → VisionEngine.process_frame()
    """

    def __init__(self, engine: VisionEngine, camera_index: str | int = 0) -> None:
        self.engine = engine
        self.camera_index = camera_index

        # ── Display frame (JPEG bytes) – consumed by Flask generator ──
        self._display_frame: Optional[bytes] = None
        self._display_lock = threading.Lock()

        # ── Raw frame buffer – single slot, always the latest frame ──
        self._raw_frame: Optional[np.ndarray] = None
        self._raw_lock = threading.Lock()
        self._raw_frame_ready = threading.Event()

        # ── State ──────────────────────────────────────────────────────
        self._state_lock = threading.Lock()
        self._started = False
        self._stop_requested = threading.Event()
        self._camera: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True
            self._stop_requested.clear()
            self.engine.reset_tracking()

        threading.Thread(
            target=self._display_loop, daemon=True, name="camera-display"
        ).start()

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._started and not self._stop_requested.is_set()
        threading.Thread(
            target=self._ai_loop, daemon=True, name="camera-ai"
        ).start()

    def stop(self) -> None:
        """Stop inference and free the camera/RTSP connection."""
        self._stop_requested.set()
        self._raw_frame_ready.set()  # unblock AI thread if waiting
        with self._display_lock:
            self._display_frame = None
        with self._state_lock:
            camera = self._camera
        if camera is not None:
            camera.release()

    def frames(self) -> Iterator[bytes]:
        """Generator for Flask multi-part streaming. Yields JPEG frames."""
        while True:
            with self._display_lock:
                frame = self._display_frame
            if frame is not None:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            # Adaptive sleep: fast enough for smooth display, slow enough to not
            # burn CPU when frames arrive slower than expected.
            time.sleep(0.005)  # up to ~200 Hz check rate

    # ------------------------------------------------------------------
    # Thread 1 – Display loop (camera FPS)
    # ------------------------------------------------------------------
    def _display_loop(self) -> None:
        source = (
            int(self.camera_index)
            if str(self.camera_index).isdigit()
            else self.camera_index
        )
        camera = cv2.VideoCapture(source)
        with self._state_lock:
            self._camera = camera

        # Set camera resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not camera.isOpened():
            print(
                "[CAMERA] Unable to open camera. Check camera index or permissions."
            )
            with self._state_lock:
                self._started = False
                self._camera = None
            return

        try:
            while not self._stop_requested.is_set():
                ok, frame = camera.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                # Record camera FPS for monitoring
                self.engine.record_camera_fps()

                # ── Publish raw frame for AI thread ──────────────────
                # Only keep the latest frame; if AI hasn't consumed the
                # previous one, replace it (latest-frame strategy).
                with self._raw_lock:
                    self._raw_frame = frame.copy()
                self._raw_frame_ready.set()

                # ── Draw AI overlays on display copy ─────────────────
                display = frame.copy()
                self.engine.draw_overlays(display)

                # ── Encode and publish ───────────────────────────────
                ok, encoded = cv2.imencode(".jpg", display)
                if ok:
                    with self._display_lock:
                        self._display_frame = encoded.tobytes()
        finally:
            camera.release()
            with self._state_lock:
                self._started = False
                self._camera = None

    # ------------------------------------------------------------------
    # Thread 2 – AI loop (asynchronous, at its own pace)
    # ------------------------------------------------------------------
    def _ai_loop(self) -> None:
        while not self._stop_requested.is_set():
            # Wait for the next raw frame (with timeout so we can check stop)
            signaled = self._raw_frame_ready.wait(timeout=0.5)
            if not signaled:
                continue

            with self._raw_lock:
                frame = self._raw_frame
                self._raw_frame = None
            self._raw_frame_ready.clear()

            if frame is None:
                continue

            try:
                # This is the expensive call – YOLO + ByteTrack + SCRFD + ArcFace
                self.engine.process_frame(frame, send_alerts=True)
            except Exception as exc:
                print(f"[AI] Processing error: {exc}")
