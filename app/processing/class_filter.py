"""M7D — pure post-inference per-object class filter.

The operator-saved ``selected_classes`` is applied by FILTERING the per-object
output paths *after* inference, not by reconfiguring the detector. Inference
still detects every class (``nvinfer``/PGIE class-id config is untouched); this
helper drops the objects whose resolved class is not in the operator's
allow-list before they reach the USER-FACING per-object surfaces (the
perception snapshot and the zone/dwell observation paths).

Deliberately out of scope (and therefore NOT filtered by this helper):

- the nvdsanalytics AGGREGATE counts (line-crossing / ROI-occupancy) — those
  are per-line / per-ROI totals across ALL classes, computed in-container, and
  are not per-class filterable post-inference;
- the RAW status debug counters (class_counts / objects_seen) — those report
  what the detector actually produced.

Pure / side-effect free: no I/O, no logging, no mutation of the passed objects.
Deterministic and order-preserving.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

# TrafficCamNet label order. Documented fallback used to resolve an object's
# class when the producer did not attach a textual ``label``. NOT a class
# filter itself — purely the int->label map for ``canonical_class``.
CLASS_ID_TO_LABEL: dict[int, str] = {
    0: "car",
    1: "bicycle",
    2: "person",
    3: "roadsign",
}


class _ClassifiedObject(Protocol):
    """Structural type for anything carrying a class: matches
    ``NeutralObjectMetadata`` (``class_id: int`` + ``label: str | None``)
    without importing it, so this helper stays dependency-light and reusable."""

    @property
    def class_id(self) -> int: ...

    @property
    def label(self) -> str | None: ...


_ObjectT = TypeVar("_ObjectT", bound=_ClassifiedObject)


def canonical_class(label: str | None, class_id: int | None) -> str | None:
    """Resolve an object's canonical class name for matching.

    Prefers the producer's textual ``label`` (lower-cased, stripped) when it is
    a non-empty string; otherwise falls back to the TrafficCamNet
    ``CLASS_ID_TO_LABEL`` map keyed by ``class_id``. Returns ``None`` when the
    class cannot be resolved (no usable label and no mapped class id)."""
    if isinstance(label, str):
        stripped = label.strip().lower()
        if stripped:
            return stripped
    if class_id is None:
        return None
    return CLASS_ID_TO_LABEL.get(class_id)


def filter_objects_by_class(
    objects: Sequence[_ObjectT],
    selected_classes: Sequence[str] | None,
) -> Sequence[_ObjectT]:
    """Return only the objects whose canonical class is in ``selected_classes``.

    When ``selected_classes`` is ``None`` or empty the input is returned
    UNCHANGED (same object, no copying, no filtering) — this is the byte/
    behavior-identical "no filter" path. Otherwise the allow-list is normalized
    (each entry stripped + lower-cased) and only objects whose
    ``canonical_class`` is in that set are kept. Objects with no resolvable
    class are EXCLUDED from a non-empty filter.

    Order-preserving and non-mutating: the original object instances are
    returned as-is (label casing is never touched)."""
    if not selected_classes:
        return objects
    wanted = {c.strip().lower() for c in selected_classes}
    return [o for o in objects if canonical_class(o.label, o.class_id) in wanted]
