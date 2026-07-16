"""Perception fan-out publisher for the CrowdCount worker (hot path).

Emits per-frame detection boxes + track IDs to the upstream-streaming-server
perception-fanout sidecar so they render as a live overlay on the Upstream
dashboard — the same envelope DeepStream workers publish (see
upstream-processor app/perception/fanout_publisher.py), so the dashboard's one
renderer works unchanged.

The worker's frame loop is synchronous, so this runs a background daemon thread
that POSTs the latest snapshot on a subscriber-aware cadence (fast when someone
is watching, idle otherwise). The per-frame hook is cheap and never blocks the
counting loop; publishing is entirely best-effort and never raises into it.
"""

import logging
import threading
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger("CounterWorker.perception")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PerceptionPublisher:
    """Best-effort, non-blocking publisher of detection snapshots.

    Use as the processor's ``perception_hook``: it is called
    ``publisher(frame, detections, frame_number)`` once per processed frame.
    """

    def __init__(
        self,
        *,
        camera_id,
        session_id,
        url,
        secret,
        processor_id=None,
        package_id=None,
        fps=15.0,
        active_hz=12.0,
        idle_hz=0.5,
        timeout=2.0,
    ):
        self._camera_id = camera_id
        self._session_id = session_id or None
        self._url = url
        self._secret = secret
        self._processor_id = processor_id or None
        self._package_id = package_id or None
        self._fps = float(fps) if fps else None
        self._active_hz = active_hz
        self._idle_hz = idle_hz
        self._timeout = timeout

        self._latest = None  # (w, h, frame_num, objects) — set by the frame loop
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._subscribers = 0
        self.published = 0
        self.errors = 0

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="perception-publisher"
        )
        self._thread.start()
        logger.info("perception publisher started (camera=%s)", camera_id)

    # -- frame-loop hook (cheap, never blocks) -----------------------------

    def __call__(self, frame, detections, frame_number):
        try:
            h, w = frame.shape[:2]
            objects = []
            for d in detections:
                x, y, bw, bh = d.bbox
                objects.append(
                    {
                        "track_id": str(d.track_id) if d.track_id is not None else None,
                        "class_id": d.class_id,
                        "label": d.class_name,
                        "confidence": round(float(d.confidence), 3),
                        "bbox": {
                            "x": float(x),
                            "y": float(y),
                            "w": float(bw),
                            "h": float(bh),
                        },
                        "roi_labels": [],
                    }
                )
            with self._lock:
                self._latest = (int(w), int(h), int(frame_number), objects)
        except Exception:
            # The overlay must never be able to break the counting loop.
            pass

    # -- background publisher ----------------------------------------------

    def _loop(self):
        last_frame_num = None
        while not self._stop.is_set():
            hz = self._active_hz if self._subscribers > 0 else self._idle_hz
            self._stop.wait(1.0 / max(hz, 0.001))
            if self._stop.is_set():
                break
            with self._lock:
                latest = self._latest
            if latest is None:
                continue
            w, h, frame_num, objects = latest
            if frame_num == last_frame_num:
                continue  # nothing new since the last publish
            last_frame_num = frame_num
            self._publish(w, h, frame_num, objects)

    def _publish(self, w, h, frame_num, objects):
        envelope = {
            "type": "snapshot",
            "camera_id": self._camera_id,
            "stream_session_id": self._session_id,
            "processor_id": self._processor_id,
            "package_id": self._package_id,
            "observed_at": _iso_now(),
            "snapshot": {
                "frame": {
                    "width": w,
                    "height": h,
                    "frame_num": frame_num,
                    "fps": self._fps,
                },
                "objects": objects,
                "geometry": {"rois": [], "lines": []},
            },
        }
        try:
            r = requests.post(
                self._url,
                json=envelope,
                headers={"X-Upstream-Webhook-Secret": self._secret},
                timeout=self._timeout,
            )
            if r.status_code == 202:
                self.published += 1
                try:
                    subs = r.json().get("subscribers")
                    self._subscribers = subs if isinstance(subs, int) and subs >= 0 else 0
                except Exception:
                    pass
            else:
                self.errors += 1
                self._subscribers = 0  # unknown watchers on error -> idle cadence
                if self.errors <= 3 or self.errors % 100 == 0:
                    logger.warning("fanout publish HTTP %s (errors=%d)", r.status_code, self.errors)
        except Exception as e:
            self.errors += 1
            self._subscribers = 0
            if self.errors <= 3 or self.errors % 100 == 0:
                logger.warning("fanout publish failed (%s; errors=%d)", e, self.errors)

    def stop(self):
        self._stop.set()
