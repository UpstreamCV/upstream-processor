"""Outbound poll loop that provisions MediaMTX paths for pending sessions.

Runs alongside the worker supervisor. Where ``WorkerManager`` polls for
sessions to *process*, this polls for sessions to *provision* — the same
outbound direction, so the api-server never needs a route into this network.

Latency matters here in a way it doesn't for worker reconcile: an agent's
``POST /v1/stream-sessions/start`` is blocked waiting for our ack, so the
poll interval is deliberately much tighter (1s vs 5s) and the work per tick
is one loopback HTTP call per pending session.

Failure posture: every error is reported to the api-server as
``ok=false`` rather than being retried locally. The api-server owns the
retry policy (the agent re-calls ``/start``), and a silent local retry would
just burn the agent's wait budget with no visibility.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.api_client.upstream_api import UpstreamApiClient
from app.backoff import Backoff, wait_or_event
from app.config import Settings
from app.errors import ApiAuthError, ApiError, ApiValidationError
from app.provisioning.mediamtx_local import (
    LocalProvisioningError,
    provision_local_path,
)
from app.provisioning.schemas import PendingProvision, ProvisionAckPayload

_log = logging.getLogger("upstream_processor.provisioner")


class Provisioner:
    """Polls pending-provisions and creates each path on the local MediaMTX."""

    def __init__(
        self,
        *,
        api: UpstreamApiClient,
        settings: Settings,
        shutdown: asyncio.Event,
    ) -> None:
        self._api = api
        self._settings = settings
        self._shutdown = shutdown
        # Dedicated client: loopback only, short timeouts. Kept separate from
        # the api client's pool so a stalled cloud call can never queue behind
        # a local one (or vice versa).
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.mediamtx_connect_timeout_seconds,
                read=settings.mediamtx_read_timeout_seconds,
                write=settings.mediamtx_read_timeout_seconds,
                pool=settings.mediamtx_connect_timeout_seconds,
            )
        )
        self._backoff = Backoff(
            initial=settings.backoff_initial_seconds,
            factor=settings.backoff_factor,
            maximum=settings.backoff_max_seconds,
            jitter=settings.backoff_jitter,
        )
        # Sessions already handled. Bounds duplicate work if an ack response
        # is lost and the row is briefly re-offered before the write lands.
        self._done: set[str] = set()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def run(self) -> None:
        if not self._settings.provisioning_enabled:
            _log.info("pull provisioning disabled; provisioner idle")
            return

        _log.info(
            "provisioner started: mediamtx=%s poll=%.1fs",
            self._settings.mediamtx_api_base_url,
            self._settings.provision_poll_interval_seconds,
        )
        while not self._shutdown.is_set():
            try:
                resp = await self._api.get_pending_provisions(limit=self._settings.provision_limit)
            except ApiAuthError:
                raise  # fatal config — bubble to main
            except ApiValidationError:
                # An older api-server that lacks the route returns 404, which
                # surfaces as ApiError, not this. A 422 means our client is
                # wrong — but unlike the worker loop we degrade rather than
                # kill the process, since provisioning is additive.
                _log.error("pending-provisions contract mismatch; pausing provisioner")
                return
            except ApiError as exc:
                _log.warning("pending-provisions fetch failed (transient): %s", exc)
                if await wait_or_event(self._shutdown, self._backoff.next_delay()):
                    break
                continue

            self._backoff.reset()
            for item in resp.items:
                if self._shutdown.is_set():
                    break
                if str(item.session_id) in self._done:
                    continue
                await self._handle(item)

            # Keep the dedupe set from growing without bound over a long
            # uptime; the api-server's pending window is the real guard.
            if len(self._done) > 512:
                self._done.clear()

            if await wait_or_event(self._shutdown, self._settings.provision_poll_interval_seconds):
                break

        await self.aclose()

    async def _handle(self, item: PendingProvision) -> None:
        session_id = str(item.session_id)
        try:
            await provision_local_path(
                self._http,
                base_url=self._settings.mediamtx_api_base_url,
                path=item.stream_key,
                passphrase=item.srt_publish_passphrase.get_secret_value(),
                runon_ready=self._settings.mediamtx_runon_ready or None,
                runon_not_ready=self._settings.mediamtx_runon_not_ready or None,
            )
        except LocalProvisioningError as exc:
            _log.error("provision failed for %s: %s", session_id, exc)
            await self._ack(item, ok=False, code=exc.code, message=exc.message)
            return

        await self._ack(item, ok=True)

    async def _ack(
        self,
        item: PendingProvision,
        *,
        ok: bool,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        """Report the outcome. A lost ack is recoverable; a lost path is not.

        On ack failure we deliberately do NOT mark the session done, so the
        next poll retries. Re-provisioning an existing path is idempotent
        (POST→PATCH), so a duplicate attempt is harmless.
        """
        try:
            await self._api.ack_provision(
                item.session_id,
                ProvisionAckPayload(ok=ok, error_code=code, error_message=message),
            )
        except ApiError as exc:
            _log.warning("ack failed for %s (will retry): %s", item.session_id, exc)
            return
        self._done.add(str(item.session_id))
