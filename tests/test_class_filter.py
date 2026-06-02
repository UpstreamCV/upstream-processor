"""M7D — pure per-object class-filter helper.

Covers ``canonical_class`` resolution (label-preferred, class-id fallback) and
``filter_objects_by_class`` semantics: the empty/None list returns the input
UNCHANGED (same object, no copy), a non-empty list keeps only matching objects,
unresolvable-class objects are excluded from a non-empty filter, normalization
is case-/whitespace-insensitive, order is preserved, and the original instances
are never mutated.
"""

from __future__ import annotations

from app.observations.metadata_models import NeutralObjectMetadata
from app.processing.class_filter import (
    CLASS_ID_TO_LABEL,
    canonical_class,
    filter_objects_by_class,
)


def _obj(class_id: int, label: str | None = None) -> NeutralObjectMetadata:
    return NeutralObjectMetadata(class_id=class_id, label=label)


# --- canonical_class ---------------------------------------------------


def test_canonical_class_prefers_label_normalized() -> None:
    # Label wins over class_id and is stripped + lower-cased.
    assert canonical_class("  Person ", 0) == "person"
    assert canonical_class("CAR", 2) == "car"


def test_canonical_class_falls_back_to_class_id_map() -> None:
    # Empty / whitespace / None label -> TrafficCamNet id map.
    assert canonical_class(None, 0) == "car"
    assert canonical_class("", 2) == "person"
    assert canonical_class("   ", 1) == "bicycle"
    assert canonical_class(None, 3) == "roadsign"


def test_canonical_class_unresolvable_is_none() -> None:
    assert canonical_class(None, None) is None
    assert canonical_class("", None) is None
    # class_id not in the TrafficCamNet map and no usable label.
    assert canonical_class(None, 99) is None


def test_class_id_to_label_is_trafficcamnet_order() -> None:
    assert CLASS_ID_TO_LABEL == {0: "car", 1: "bicycle", 2: "person", 3: "roadsign"}


# --- filter_objects_by_class : no-filter pass-through ------------------


def test_empty_selected_classes_returns_input_unchanged() -> None:
    objs = [_obj(0), _obj(2, "person")]
    # Empty list -> SAME object back (identity), byte-identical to "no filter".
    assert filter_objects_by_class(objs, []) is objs


def test_none_selected_classes_returns_input_unchanged() -> None:
    objs = [_obj(0), _obj(2, "person")]
    assert filter_objects_by_class(objs, None) is objs


# --- filter_objects_by_class : non-empty filter ------------------------


def test_filter_keeps_only_matching_by_label() -> None:
    objs = [_obj(2, "person"), _obj(0, "car"), _obj(2, "person")]
    out = filter_objects_by_class(objs, ["person"])
    assert out == [objs[0], objs[2]]


def test_filter_matches_via_class_id_fallback_when_label_missing() -> None:
    # No label -> resolved through the class-id map; "car" matches class_id 0.
    objs = [_obj(0), _obj(2)]
    out = filter_objects_by_class(objs, ["car"])
    assert out == [objs[0]]


def test_filter_normalizes_the_allow_list() -> None:
    objs = [_obj(2, "person"), _obj(0, "car")]
    out = filter_objects_by_class(objs, ["  PERSON ", "Car"])
    assert out == objs  # both kept, normalization is case/space-insensitive


def test_filter_excludes_unresolvable_class_objects() -> None:
    keep = _obj(2, "person")
    drop = _obj(99)  # no label, class_id not in map -> unresolvable
    out = filter_objects_by_class([keep, drop], ["person"])
    assert out == [keep]


def test_filter_preserves_order_and_instances() -> None:
    a, b, c = _obj(2, "person"), _obj(0, "car"), _obj(2, "person")
    out = filter_objects_by_class([a, b, c], ["person", "car"])
    # Order preserved; the EXACT original instances are returned.
    assert out == [a, b, c]
    assert out[0] is a and out[1] is b and out[2] is c


def test_filter_does_not_mutate_label_casing() -> None:
    obj = _obj(2, "Person")
    out = filter_objects_by_class([obj], ["person"])
    assert out == [obj]
    # Original casing untouched.
    assert obj.label == "Person"


def test_filter_empty_result_when_nothing_matches() -> None:
    objs = [_obj(0, "car"), _obj(1, "bicycle")]
    assert filter_objects_by_class(objs, ["person"]) == []


# --- label-vocabulary drift: model "road_sign" vs registry "roadsign" --


def test_canonical_class_strips_separators() -> None:
    # The DeepStream model emits "road_sign"; the registry says "roadsign".
    # Both must canonicalize identically (case-folded, separators removed).
    assert canonical_class("road_sign", 3) == "roadsign"
    assert canonical_class("Road Sign", 3) == "roadsign"
    assert canonical_class("road-sign", None) == "roadsign"
    assert canonical_class("ROAD_SIGN", None) == "roadsign"


def test_filter_matches_across_separator_drift() -> None:
    # Selecting the registry label "roadsign" keeps the model's "road_sign"
    # objects (and the other separator/case variants) — the underscore fix.
    objs = [
        _obj(3, "road_sign"),
        _obj(3, "Road Sign"),
        _obj(3, "road-sign"),
        _obj(2, "person"),
    ]
    out = filter_objects_by_class(objs, ["roadsign"])
    assert out == [objs[0], objs[1], objs[2]]


def test_filter_separator_insensitive_on_both_sides() -> None:
    # A selected entry with a separator also canonicalizes — symmetry.
    objs = [_obj(3, "road_sign"), _obj(2, "person")]
    assert filter_objects_by_class(objs, ["road sign"]) == [objs[0]]
    assert filter_objects_by_class(objs, ["road_sign"]) == [objs[0]]


def test_clean_classes_unaffected_by_canonicalization() -> None:
    # Regression: the already-correct labels are unchanged by _canon.
    objs = [_obj(0, "car"), _obj(2, "person"), _obj(1, "bicycle")]
    assert filter_objects_by_class(objs, ["car"]) == [objs[0]]
    assert filter_objects_by_class(objs, ["person", "bicycle"]) == [objs[1], objs[2]]
