"""Runtime interface shared by every processor runtime.

M8.1 ships only ``MockRuntime``. This ABC exists so M8.2's DeepStream runtime
can drop in behind the same supervisor without changes there.

Lifecycle (status values are the M8.0 contract's):

    starting -> attached -> processing (repeated) -> stopped  (graceful)
                                                  -> failed    (error)

``lagging`` is a contract-legal status but the mock never emits it (there is
no real frame pipeline to fall behind); a DeepStream runtime will.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from uuid import UUID


class WorkerState(enum.StrEnum):
    INIT = "init"
    STARTING = "starting"
    ATTACHED = "attached"
    PROCESSING = "processing"
    STOPPED = "stopped"
    FAILED = "failed"


class Runtime(ABC):
    """One instance processes exactly one stream session."""

    @property
    @abstractmethod
    def session_id(self) -> UUID: ...

    @property
    @abstractmethod
    def state(self) -> WorkerState: ...

    @abstractmethod
    async def run(self) -> None:
        """Drive the full lifecycle until graceful stop, error, or shutdown.
        Must not raise on ordinary API trouble — it owns its own status
        posts (terminal ``stopped`` or ``failed``)."""

    @abstractmethod
    def request_stop(self) -> None:
        """Ask the runtime to wind down at the next opportunity (idempotent)."""
