"""P5-Gap: analytics lifecycle policy.

Controls how the analytics adapters treat a tracked object that DISAPPEARS
(its track id vanishes) while it was inside a zone / dwelling. Pre-P5-Gap the
adapters never acted on disappearance (a dropped frame couldn't be told apart
from a real exit), so a person who walked in and then was lost never produced a
``zone_exited`` and never closed their dwell. This policy makes that an
explicit, conservative behavior: after a grace window with the track absent,
infer a ``zone_exited`` (which also closes the dwell).

Conservative and ON BY DEFAULT; both fields are env-overridable via
``Settings.deepstream_lifecycle_*`` so it can be tuned/disabled in production.
Built once per worker and passed to the zone adapter.

Note: there is intentionally no "minimum presence" knob — a track only reaches
this path after it has COMMITTED inside (i.e. it was observed inside for
``membership_stability_frames`` frames), which is already a confirmed presence.
An extra presence gate was both redundant and leak-prone (it could suppress the
exit while leaving the dwell session open), so the single grace window is the
guard against acting on a brief dropped frame.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LifecyclePolicy:
    """Disappearance-based inferred-exit configuration."""

    # Seconds a committed-inside track must be ABSENT (its id not present in the
    # frame) before we infer a ``zone_exited`` and close its dwell. The single
    # guard against acting on a brief tracker drop.
    track_missing_grace_seconds: float = 2.0
    # Master switch. When false, restores the pre-P5-Gap behavior
    # (disappearance never emits an exit / never closes a dwell).
    infer_zone_exit_on_disappearance: bool = True
