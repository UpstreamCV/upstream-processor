"""Create a MediaMTX path on the box's OWN MediaMTX, over loopback.

The mirror image of the api-server's ``services/mediamtx_provisioning.py``.
Same two-step POST→PATCH dance against the same MediaMTX 1.9.3 API — the only
difference is who dials whom. Here the call is `127.0.0.1`, so it cannot be
broken by DNS, a router port-forward, or a changed WAN IP.

Keeping the request body byte-identical to the api-server's matters: both
modes must produce the same path config, or a fleet mid-rollout would behave
differently depending on which server provisioned the session.
"""

from __future__ import annotations

import logging

import httpx

_log = logging.getLogger("upstream_processor.provisioning.mediamtx")


class LocalProvisioningError(Exception):
    """Path creation failed. ``code`` is reported verbatim in the ack."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _hook_paths(runon_ready: str | None, runon_not_ready: str | None) -> dict[str, str]:
    """MediaMTX lifecycle hooks, omitted entirely when not configured.

    Sending an empty string would CLEAR a hook that the static mediamtx.yml
    had configured, silently disabling stream-start/stream-ended reporting —
    so an unset value must mean "don't touch", not "set to blank".
    """
    hooks: dict[str, str] = {}
    if runon_ready:
        hooks["runOnReady"] = runon_ready
    if runon_not_ready:
        hooks["runOnNotReady"] = runon_not_ready
    return hooks


async def provision_local_path(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    path: str,
    passphrase: str,
    runon_ready: str | None = None,
    runon_not_ready: str | None = None,
) -> None:
    """Create (or update) ``path`` with a session-specific SRT passphrase.

    MediaMTX splits path management: ``POST add/{name}`` only creates (400 if
    it exists) and ``PATCH patch/{name}`` only updates (404 if it doesn't).
    Neither is idempotent alone, so we POST first — each ``sess_<hex>`` is
    unique, so creation is the overwhelmingly common case — and fall back to
    PATCH on a 400 conflict, which happens when a previous deprovision left
    an orphan behind.

    Raises:
        LocalProvisioningError: transport failure or a non-2xx that survived
            the POST→PATCH fallback.
    """
    base = base_url.rstrip("/")
    body: dict[str, object] = {
        "source": "publisher",
        "srtPublishPassphrase": passphrase,
        **_hook_paths(runon_ready, runon_not_ready),
    }

    try:
        response = await client.post(f"{base}/v3/config/paths/add/{path}", json=body)
        if response.status_code == 400:
            # Already exists -> update it in place so the new session's
            # passphrase takes effect rather than the stale one.
            response = await client.patch(f"{base}/v3/config/paths/patch/{path}", json=body)
    except httpx.HTTPError as exc:
        # Loopback transport failure means MediaMTX is down or not listening
        # on the configured port — a local problem, reported honestly.
        raise LocalProvisioningError("transport", str(exc)[:200]) from exc

    if response.status_code >= 300:
        raise LocalProvisioningError(
            "http_error",
            f"status={response.status_code} body={response.text[:160]}",
        )

    _log.info("provisioned local mediamtx path %s", path)


async def deprovision_local_path(client: httpx.AsyncClient, *, base_url: str, path: str) -> None:
    """Best-effort path removal. Never raises — cleanup must not break a loop.

    A 404 is success (already gone). Anything else is logged and swallowed;
    an orphan path is untidy but harmless, whereas an exception here would
    take down the caller's reconcile cycle.
    """
    base = base_url.rstrip("/")
    try:
        response = await client.delete(f"{base}/v3/config/paths/delete/{path}")
    except httpx.HTTPError as exc:
        _log.warning("deprovision %s failed (transport): %s", path, exc)
        return
    if response.status_code not in (200, 204, 404):
        _log.warning("deprovision %s failed: status=%s", path, response.status_code)
