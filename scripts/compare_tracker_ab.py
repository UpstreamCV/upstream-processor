#!/usr/bin/env python3
"""P5-E — IOU vs NvDCF A/B comparison (pure Python, no GPU/Docker/pyds).

Reads two capture files (JSON-lines or a JSON array of status / observation /
snapshot records — see ``scripts/_tracker_ab_helpers.py``) and prints a redacted
side-by-side comparison: unique tracker ids, longest track persistence, an
id-switch proxy, per-event counts (zone_exited / line_crossing /
line_crossing_object / dwell_exceeded / ...), and FPS. Used by
``scripts/probe_tracker_ab.sh`` and the P5-E runbook in docs/local-dev.md.

Usage:
  scripts/compare_tracker_ab.py LABEL_A CAPTURE_A LABEL_B CAPTURE_B
Exit codes: 0 ok; 2 bad usage / unreadable capture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _tracker_ab_helpers import compare, parse_capture, scrub_rtsp, summarize


def _load(path: str) -> list[dict[str, object]]:
    return parse_capture(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: compare_tracker_ab.py LABEL_A CAPTURE_A LABEL_B CAPTURE_B",
            file=sys.stderr,
        )
        return 2
    label_a, capture_a, label_b, capture_b = argv
    try:
        summary_a = summarize(_load(capture_a), label_a)
        summary_b = summarize(_load(capture_b), label_b)
    except OSError as exc:
        print(f"FAIL unreadable capture: {exc}", file=sys.stderr)
        return 2
    report = {
        "summary_a": summary_a,
        "summary_b": summary_b,
        "comparison": compare(summary_a, summary_b),
    }
    # Belt-and-braces: never let a raw stream key survive into the printed report.
    print(scrub_rtsp(json.dumps(report, indent=2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
