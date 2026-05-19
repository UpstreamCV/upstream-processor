"""Typed error hierarchy.

Mirrors the discipline of ``upstream-agent/upstream_agent/api/client.py``:
the API client raises ONLY these typed errors, never a raw httpx exception,
and error messages never carry the secret or a response body.
"""

from __future__ import annotations


class ProcessorError(Exception):
    """Base for every error this service raises deliberately."""


class ConfigError(ProcessorError):
    """Invalid/missing configuration."""


class PackageResolutionError(ProcessorError):
    """package_id unknown, or runtime not supported in this milestone."""


class ApiError(ProcessorError):
    """An API call failed. ``code``/``request_id`` come from the server's
    error envelope when present; ``message`` is safe to log (no body)."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.request_id = request_id

    def __str__(self) -> str:
        bits = [self.message]
        if self.status is not None:
            bits.append(f"status={self.status}")
        if self.code is not None:
            bits.append(f"code={self.code}")
        if self.request_id is not None:
            bits.append(f"request_id={self.request_id}")
        return " ".join(bits)


class ApiAuthError(ApiError):
    """401 invalid_webhook_secret — fatal config error, do NOT retry."""


class ApiSessionGoneError(ApiError):
    """404 session_not_found — the session disappeared; stop quietly."""


class ApiValidationError(ApiError):
    """422 validation_error — a contract/programming bug, do NOT retry."""


class ApiContractError(ApiError):
    """2xx but the body did not match the expected schema."""


class ApiRateLimitError(ApiError):
    """429 — transient, back off and retry."""


class ApiServerError(ApiError):
    """5xx — transient, back off and retry."""


class ApiConnectionError(ApiError):
    """Timeout / DNS / refused / reset — transient, back off and retry."""


# Errors a caller may retry through (transient). Auth/validation/session-gone
# are deliberately excluded.
TRANSIENT_API_ERRORS: tuple[type[ApiError], ...] = (
    ApiConnectionError,
    ApiServerError,
    ApiRateLimitError,
    ApiContractError,
)
