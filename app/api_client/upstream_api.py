"""Async client for the M8.0 processor contract.

Discipline (ported from upstream-agent's api client):
- Auth header built in ONE place (`_auth_header`); only
  ``X-Upstream-Webhook-Secret`` — never ``Authorization``/Bearer.
- The secret and request/response bodies are NEVER logged.
- Raises only the typed errors in ``app.errors`` (never a raw httpx error).
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.errors import (
    ApiAuthError,
    ApiConnectionError,
    ApiContractError,
    ApiError,
    ApiRateLimitError,
    ApiServerError,
    ApiSessionGoneError,
    ApiValidationError,
)
from app.observations.schemas import (
    EligibleSessionsResponse,
    ObservationPayload,
    ObservationResponse,
    ProcessorStatusPayload,
    ProcessorStatusResponse,
)

_log = logging.getLogger("upstream_processor.api")

_BASE_PATH = "/v1/internal/processor"
_ELIGIBLE = f"{_BASE_PATH}/eligible-sessions"
_STATUS = f"{_BASE_PATH}/status"
_OBSERVATIONS = f"{_BASE_PATH}/observations"

_AUTH_HEADER = "X-Upstream-Webhook-Secret"

_M = TypeVar("_M", bound=BaseModel)


class UpstreamApiClient:
    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._secret = settings.processor_webhook_secret.get_secret_value()
        self._client = client or httpx.AsyncClient(
            base_url=settings.api_base_url,
            timeout=httpx.Timeout(
                connect=settings.http_connect_timeout_seconds,
                read=settings.http_read_timeout_seconds,
                write=settings.http_write_timeout_seconds,
                pool=settings.http_pool_timeout_seconds,
            ),
        )

    def _auth_header(self) -> dict[str, str]:
        # Single source of truth for the auth scheme. No Bearer/Authorization.
        return {_AUTH_HEADER: self._secret}

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- low level --------------------------------------------------------

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            resp = await self._client.request(
                method,
                path,
                params=params,
                json=json,
                headers=self._auth_header(),
            )
        except httpx.TimeoutException as exc:
            raise ApiConnectionError(f"request timed out: {method} {path}") from exc
        except httpx.TransportError as exc:
            # ConnectError / ReadError / NetworkError / ProtocolError, etc.
            raise ApiConnectionError(f"transport error: {method} {path}") from exc

        _log.debug("api %s %s -> %s", method, path, resp.status_code)
        if resp.is_success:
            return resp
        raise self._error_for(resp)

    @staticmethod
    def _envelope(resp: httpx.Response) -> tuple[str | None, str | None]:
        try:
            body = resp.json()
            err = body.get("error", {}) if isinstance(body, dict) else {}
            code = err.get("code") if isinstance(err, dict) else None
            rid = err.get("request_id") if isinstance(err, dict) else None
            return (
                code if isinstance(code, str) else None,
                rid if isinstance(rid, str) else None,
            )
        except (ValueError, AttributeError):
            return None, None

    def _error_for(self, resp: httpx.Response) -> ApiError:
        code, rid = self._envelope(resp)
        status = resp.status_code
        msg = f"API error on {resp.request.method} {resp.request.url.path}"
        if status == 401:
            return ApiAuthError(msg, status=status, code=code, request_id=rid)
        if status == 404:
            return ApiSessionGoneError(msg, status=status, code=code, request_id=rid)
        if status == 422:
            return ApiValidationError(msg, status=status, code=code, request_id=rid)
        if status == 429:
            return ApiRateLimitError(msg, status=status, code=code, request_id=rid)
        if status >= 500:
            return ApiServerError(msg, status=status, code=code, request_id=rid)
        return ApiError(msg, status=status, code=code, request_id=rid)

    @staticmethod
    def _parse(resp: httpx.Response, model: type[_M]) -> _M:
        try:
            return model.model_validate(resp.json())
        except (ValueError, ValidationError) as exc:
            raise ApiContractError(
                f"unparsable response from {resp.request.url.path}",
                status=resp.status_code,
            ) from exc

    # --- contract methods -------------------------------------------------

    async def get_eligible_sessions(self, limit: int = 50) -> EligibleSessionsResponse:
        resp = await self._send("GET", _ELIGIBLE, params={"limit": limit})
        return self._parse(resp, EligibleSessionsResponse)

    async def post_processor_status(
        self, payload: ProcessorStatusPayload
    ) -> ProcessorStatusResponse:
        resp = await self._send(
            "POST", _STATUS, json=payload.model_dump(mode="json", exclude_none=True)
        )
        return self._parse(resp, ProcessorStatusResponse)

    async def post_observation(self, observation: ObservationPayload) -> ObservationResponse:
        resp = await self._send(
            "POST",
            _OBSERVATIONS,
            json=observation.model_dump(mode="json", exclude_none=True),
        )
        return self._parse(resp, ObservationResponse)
