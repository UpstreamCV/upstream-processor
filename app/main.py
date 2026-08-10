"""Entry point.

`upstream-processor` console script and `python -m app.main`. Owns the event
loop, signal handling, and the exit-code contract:
  0  clean shutdown (SIGINT/SIGTERM)
  2  fatal auth/config (wrong UPSTREAM_PROCESSOR_WEBHOOK_SECRET)
  3  fatal API contract mismatch on eligible-sessions
  1  unexpected error
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from app.api_client.upstream_api import UpstreamApiClient
from app.config import get_settings
from app.errors import ApiAuthError, ApiValidationError
from app.logging_setup import setup_logging
from app.provisioning.provisioner import Provisioner
from app.supervisor.session_poller import SessionPoller
from app.supervisor.worker_manager import WorkerManager

_log = logging.getLogger("upstream_processor.main")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown: asyncio.Event) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:  # pragma: no cover - non-Unix fallback
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(shutdown.set))


async def _run() -> int:
    settings = get_settings()
    setup_logging(settings.log_level, settings.processor_webhook_secret.get_secret_value())
    _log.info(
        "starting upstream-processor: api=%s processor_id=%s poll=%.0fs obs=%.0fs limit=%d",
        settings.api_base_url,
        settings.processor_id,
        settings.poll_interval_seconds,
        settings.mock_observation_interval_seconds,
        settings.eligible_session_limit,
    )

    shutdown = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), shutdown)

    api = UpstreamApiClient(settings)
    poller = SessionPoller(api, settings.eligible_session_limit)
    manager = WorkerManager(poller=poller, api=api, settings=settings, shutdown=shutdown)

    # Pull-mode MediaMTX provisioning runs beside the worker supervisor on
    # the same outbound-only posture. Off by default; when disabled its run()
    # returns immediately, so this costs one finished task.
    provisioner = Provisioner(api=api, settings=settings, shutdown=shutdown)
    provisioner_task = asyncio.create_task(provisioner.run(), name="provisioner")

    try:
        await manager.run()
        _log.info("clean shutdown")
        return 0
    except ApiAuthError:
        _log.error("fatal: processor auth rejected — check UPSTREAM_PROCESSOR_WEBHOOK_SECRET")
        return 2
    except ApiValidationError:
        _log.error("fatal: API contract mismatch on eligible-sessions")
        return 3
    except Exception:
        _log.exception("fatal: unexpected error")
        return 1
    finally:
        # The supervisor owns the shutdown event; setting it here covers the
        # paths where manager.run() raised before signalling.
        shutdown.set()
        provisioner_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await provisioner_task
        await provisioner.aclose()
        await api.aclose()


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
