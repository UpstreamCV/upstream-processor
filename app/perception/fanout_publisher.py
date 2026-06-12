"""P5-F — best-effort publisher for the perception-fanout hot path.

POSTs one fanout envelope per call to the perception-fanout sidecar
(``upstream-streaming-server/perception-fanout``, ``POST /v1/publish`` —
the sidecar's sanctioned stdlib/HTTP integration mode). The wire contract
is the sidecar's: auth via the dedicated ``X-Upstream-Webhook-Secret``
publish header (NOT the api-server webhook secret), a ``202`` whose body
carries the current ``subscribers`` count, and an opaque ``snapshot``
that must be byte-shape-identical to the cold path's
``latest_perception_snapshots.snapshot`` (we reuse the SAME
``build_perception_snapshot`` output, so it is).

Behavioral contract (mirrors docs/perception-fanout.md in the
streaming-server repo): publish at 10-15 Hz while ``subscribers > 0``,
throttle to ~idle Hz when nobody is watching. The POST response is the
only feedback channel in HTTP mode, so the idle cadence doubles as the
subscriber-count poll and keeps the sidecar's last-frame cache warm.

Strictly best-effort: ``publish`` NEVER raises (errors are counted,
logged bounded, and collapse the cadence to idle so a dead sidecar is
never hammered at 12 Hz). A fanout failure must never disturb the frame
loop, the observation paths, or the cold-path snapshot POSTs.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

_log = logging.getLogger("upstream_processor.perception.fanout")

# Log the first few failures loudly, then sample so a dead sidecar over a
# long session can't flood the worker log.
_ERROR_LOG_HEAD = 3
_ERROR_LOG_SAMPLE = 50


class FanoutPublisher:
    """One per analytics worker. Owns its httpx client (closed via
    ``aclose``); tests may inject a client double."""

    def __init__(
        self,
        *,
        url: str,
        secret: str,
        hot_interval_seconds: float,
        idle_interval_seconds: float,
        timeout_seconds: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not url or not secret:
            raise ValueError("fanout publisher requires both url and secret")
        self._url = url
        self._secret = secret
        self._hot = max(hot_interval_seconds, 0.01)
        self._idle = max(idle_interval_seconds, self._hot)
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._subscribers = 0
        self._consecutive_errors = 0
        self.published = 0
        self.errors = 0

    @property
    def subscribers(self) -> int:
        """Last subscriber count the sidecar reported (0 after any error)."""
        return self._subscribers

    @property
    def interval_seconds(self) -> float:
        """The CURRENT publish interval: hot while the sidecar reports
        watchers and the last publish succeeded, idle otherwise."""
        if self._consecutive_errors > 0 or self._subscribers <= 0:
            return self._idle
        return self._hot

    async def publish(self, envelope: dict[str, Any]) -> bool:
        """POST one envelope. Returns True on 202. NEVER raises."""
        try:
            response = await self._client.post(
                self._url,
                json=envelope,
                headers={"X-Upstream-Webhook-Secret": self._secret},
            )
        except httpx.HTTPError as exc:
            self._record_error(exc.__class__.__name__)
            return False
        except Exception as exc:  # defensive: best-effort path, never raise
            self._record_error(f"unexpected {exc.__class__.__name__}")
            return False
        if response.status_code != 202:
            self._record_error(f"http_{response.status_code}")
            return False
        self._consecutive_errors = 0
        self.published += 1
        try:
            body = response.json()
            subscribers = body.get("subscribers") if isinstance(body, dict) else None
            if isinstance(subscribers, int) and subscribers >= 0:
                self._subscribers = subscribers
        except Exception:  # body parse is best-effort; the 202 already counted
            pass
        return True

    def _record_error(self, reason: str) -> None:
        self.errors += 1
        self._consecutive_errors += 1
        # Unknown watcher state on error — fall back to the idle cadence.
        self._subscribers = 0
        if self.errors <= _ERROR_LOG_HEAD or self.errors % _ERROR_LOG_SAMPLE == 0:
            _log.warning(
                "fanout publish failed (%s); errors=%d (cadence dropped to idle)",
                reason,
                self.errors,
            )

    async def aclose(self) -> None:
        await self._client.aclose()


def build_fanout_envelope(
    *,
    camera_id: str,
    stream_session_id: str | None,
    processor_id: str | None,
    package_id: str | None,
    observed_at_iso: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """The sidecar's publish envelope. ``snapshot`` is the exact
    ``build_perception_snapshot`` dict — the same bytes the cold path
    stores — so the dashboard's single renderer works for both paths."""
    return {
        "type": "snapshot",
        "camera_id": camera_id,
        "stream_session_id": stream_session_id,
        "processor_id": processor_id,
        "package_id": package_id,
        "observed_at": observed_at_iso,
        "snapshot": snapshot,
    }
