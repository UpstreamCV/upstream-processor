"""Poll the eligible-sessions endpoint.

Thin: unwrap the ``{"sessions":[...]}`` envelope to a list; let typed errors
propagate so the worker manager owns the retry/backoff policy.
"""

from __future__ import annotations

from app.api_client.base import ProcessorApi
from app.observations.schemas import EligibleSession


class SessionPoller:
    def __init__(self, api_client: ProcessorApi, limit: int) -> None:
        self._api = api_client
        self._limit = limit

    async def fetch_eligible(self) -> list[EligibleSession]:
        resp = await self._api.get_eligible_sessions(limit=self._limit)
        return resp.sessions
