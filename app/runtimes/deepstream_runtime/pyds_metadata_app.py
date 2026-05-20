#!/usr/bin/env python3
"""P2 / P3 — supervisor-managed in-container pyds metadata app (DeepStream 8.0).

**Runtime-only.** This module is intentionally NOT imported by any other
``app/`` code. It is bind-mounted into the DeepStream 8.0 + pyds container
by :class:`app.runtimes.deepstream_runtime.metadata_probe.DeepStreamMetadataProcessManager`
and executed with ``python3 -B``. Mypy/ruff treat it as a runtime probe:
the gi/pyds imports only resolve inside the container at run time.

The pipeline mirrors the live-verified manual P2 pyds probe; in **P3
tracker mode** (CLI ``--tracker-config-path`` set) an ``nvtracker`` element
is spliced in between ``nvinfer`` and ``nvvideoconvert`` and the pad probe
moves from ``nvinfer.src`` to ``tracker.src`` so ``NvDsObjectMeta.object_id``
is a real stable tracker id across frames:

    rtspsrc(tcp, latency=200) -> rtph264depay -> h264parse ->
    nvv4l2decoder -> nvstreammux(batch=1, 1280x720, live-source=1) ->
    nvinfer(config=<pgie.txt>) -> [nvtracker(config=<tracker.yml>) ->]
    nvvideoconvert -> fakesink

A pad probe walks
``pyds.gst_buffer_get_nvds_batch_meta -> NvDsFrameMeta -> NvDsObjectMeta``
and emits one structured line **per frame** on stdout:

    UPSTREAM_METADATA=<json>

…matching ``NeutralFrameMetadata`` (validated outside the container).
A single final summary line is emitted on shutdown:

    UPSTREAM_METADATA_SUMMARY=<json>

The pad probe itself does near-zero work: no HTTP, no async, no
``ObservationWriter`` calls. The container never calls UpStream APIs;
status posting stays with the outer supervisor. The raw RTSP URL goes
only into the (already redacted) per-run pgie config + argv; if anything
ever echoes ``rtsp://...``, it is scrubbed before printing.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import Counter
from typing import Any

import gi  # type: ignore[import-not-found]

gi.require_version("Gst", "1.0")
# isort: split  (gi.require_version must precede the gi.repository import)
from gi.repository import GLib, Gst  # type: ignore[import-not-found]

# isort: split
import pyds  # type: ignore[import-not-found]

_RTSP_PATH_RE = re.compile(r"^(rtsp://[^/]+/)(.+)$", re.IGNORECASE)
_RTSP_INLINE_RE = re.compile(r"rtsp://[^\s\"']+", re.IGNORECASE)
# pyds UNTRACKED_OBJECT_ID (max u64). Mapped to ``tracker_id=None`` so the
# outer supervisor's NeutralObjectMetadata validation works.
_UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF

# Structured stdout markers consumed by the outer worker. Keep them
# distinctive so they cannot be confused with normal GST/nvinfer log lines.
_META_PREFIX = "UPSTREAM_METADATA="
_SUMMARY_PREFIX = "UPSTREAM_METADATA_SUMMARY="


def _redact(url: str) -> str:
    m = _RTSP_PATH_RE.match(url.strip())
    return f"{m.group(1)}sess_<redacted>" if m else "<redacted-rtsp>"


def _scrub(text: str) -> str:
    return _RTSP_INLINE_RE.sub(lambda m: _redact(m.group(0)), text)


def _norm(v: float, denom: int) -> float | None:
    return round(v / denom, 4) if denom > 0 else None


class _State:
    """Aggregate counters. Updated only on the GLib mainloop thread."""

    def __init__(self, sample_limit: int) -> None:
        self.frames_seen = 0
        self.frames_with_objects = 0
        self.objects_seen = 0
        self.class_counts: Counter[int] = Counter()
        self.label_counts: Counter[str] = Counter()
        self.has_tracker_id = False
        self.has_bbox = False
        self.has_confidence = False
        self.samples_emitted = 0
        self.metadata_emitted = 0
        self.sample_limit = sample_limit
        self.stream_session_id: str | None = None


def _build_object_dict(obj_meta: Any, frame_w: int, frame_h: int) -> dict[str, Any]:
    label = obj_meta.obj_label or ""
    if isinstance(label, bytes):
        label = label.decode("utf-8", "replace")
    conf = float(obj_meta.confidence)
    rp = obj_meta.rect_params
    left, top, w, h = float(rp.left), float(rp.top), float(rp.width), float(rp.height)
    tracker_raw = int(obj_meta.object_id)
    tracker_id = None if tracker_raw == _UNTRACKED_OBJECT_ID else str(tracker_raw)
    return {
        "class_id": int(obj_meta.class_id),
        "label": label,
        "confidence": round(conf, 4),
        "bbox_pixels": {
            "left": round(left, 2),
            "top": round(top, 2),
            "width": round(w, 2),
            "height": round(h, 2),
        },
        "bbox_normalized": {
            "x": _norm(left, frame_w),
            "y": _norm(top, frame_h),
            "w": _norm(w, frame_w),
            "h": _norm(h, frame_h),
        },
        "tracker_id": tracker_id,
    }


def _make_pad_probe(state: _State, width: int, height: int) -> Any:
    """Pad probe callback. Near-zero work, no I/O beyond a single stdout
    line per frame (Python's stdout is line-buffered when piped, so the
    outer asyncio reader drains it without blocking us)."""

    def _probe(_pad: Any, info: Any, _user_data: Any) -> Any:
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        try:
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        except Exception:
            return Gst.PadProbeReturn.OK
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            state.frames_seen += 1
            objects: list[dict[str, Any]] = []
            l_obj = frame_meta.obj_meta_list
            num_obj = 0
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                num_obj += 1
                state.objects_seen += 1
                cls_id = int(obj_meta.class_id)
                state.class_counts[cls_id] += 1
                obj_dict = _build_object_dict(obj_meta, width, height)
                if obj_dict["label"]:
                    state.label_counts[obj_dict["label"]] += 1
                if obj_dict["confidence"] > 0.0:
                    state.has_confidence = True
                bbox = obj_dict["bbox_pixels"]
                if bbox["width"] > 0 and bbox["height"] > 0:
                    state.has_bbox = True
                if obj_dict["tracker_id"] is not None:
                    state.has_tracker_id = True
                if len(objects) < state.sample_limit:
                    objects.append(obj_dict)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            if num_obj > 0:
                state.frames_with_objects += 1
                state.samples_emitted += 1 if objects else 0

            frame_payload: dict[str, Any] = {
                "runtime": "deepstream",
                "metadata_source": "pyds_pad_probe",
                "stream_session_id": state.stream_session_id,
                "frame": {
                    "frame_num": int(frame_meta.frame_num),
                    "width": width,
                    "height": height,
                },
                "objects": objects,
            }
            sys.stdout.write(_META_PREFIX + json.dumps(frame_payload) + "\n")
            sys.stdout.flush()
            state.metadata_emitted += 1

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    return _probe


def _on_rtspsrc_pad_added(_src: Any, src_pad: Any, depay: Any) -> None:
    caps = src_pad.get_current_caps() or src_pad.query_caps(None)
    name = caps.get_structure(0).get_name() if caps and caps.get_size() else ""
    if name.startswith("application/x-rtp"):
        sink = depay.get_static_pad("sink")
        if sink is not None and not sink.is_linked():
            src_pad.link(sink)


def _bus_call(_bus: Any, message: Any, loop: Any, _state: _State) -> bool:
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        # Errors might contain the RTSP URL; scrub before printing.
        print(f"GST_ERROR: {_scrub(f'{err}: {dbg}')}", file=sys.stderr, flush=True)
        loop.quit()
    return True


def _build_pipeline(
    rtsp_url: str,
    pgie_config: str,
    width: int,
    height: int,
    *,
    tracker_config: str | None = None,
    tracker_lib: str | None = None,
    tracker_width: int = 960,
    tracker_height: int = 544,
) -> tuple[Any, Any]:
    """Build the GStreamer pipeline. When ``tracker_config`` is given, an
    ``nvtracker`` element is added after ``nvinfer`` and the function returns
    its src pad's owner so the caller attaches the pad probe there (so the
    probe sees populated ``NvDsObjectMeta.object_id``)."""
    Gst.init(None)
    pipeline = Gst.Pipeline.new("upstream-pyds-metadata")

    base_elems: list[tuple[str, str]] = [
        ("rtspsrc", "rtspsrc"),
        ("depay", "rtph264depay"),
        ("parse", "h264parse"),
        ("dec", "nvv4l2decoder"),
        ("mux", "nvstreammux"),
        ("pgie", "nvinfer"),
    ]
    if tracker_config:
        base_elems.append(("tracker", "nvtracker"))
    base_elems.extend([("conv", "nvvideoconvert"), ("sink", "fakesink")])

    elems: dict[str, Any] = {}
    for name, factory in base_elems:
        el = Gst.ElementFactory.make(factory, name)
        if el is None:
            raise RuntimeError(f"missing GStreamer element: {factory}")
        elems[name] = el

    elems["rtspsrc"].set_property("location", rtsp_url)
    elems["rtspsrc"].set_property("protocols", "tcp")
    elems["rtspsrc"].set_property("latency", 200)
    elems["mux"].set_property("batch-size", 1)
    elems["mux"].set_property("width", width)
    elems["mux"].set_property("height", height)
    elems["mux"].set_property("live-source", 1)
    elems["mux"].set_property("batched-push-timeout", 40000)
    elems["pgie"].set_property("config-file-path", pgie_config)
    elems["pgie"].set_property("batch-size", 1)
    if tracker_config:
        elems["tracker"].set_property("tracker-width", int(tracker_width))
        elems["tracker"].set_property("tracker-height", int(tracker_height))
        elems["tracker"].set_property("gpu-id", 0)
        if tracker_lib:
            elems["tracker"].set_property("ll-lib-file", tracker_lib)
        elems["tracker"].set_property("ll-config-file", tracker_config)
    elems["sink"].set_property("sync", False)
    elems["sink"].set_property("async", False)

    for el in elems.values():
        pipeline.add(el)

    elems["rtspsrc"].connect("pad-added", _on_rtspsrc_pad_added, elems["depay"])
    elems["depay"].link(elems["parse"])
    elems["parse"].link(elems["dec"])

    request = getattr(elems["mux"], "request_pad_simple", None) or elems["mux"].get_request_pad
    sinkpad = request("sink_0")
    if sinkpad is None:
        raise RuntimeError("nvstreammux sink_0 request failed")
    if elems["dec"].get_static_pad("src").link(sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("decoder -> streammux link failed")
    elems["mux"].link(elems["pgie"])
    if tracker_config:
        elems["pgie"].link(elems["tracker"])
        elems["tracker"].link(elems["conv"])
        probe_element = elems["tracker"]
    else:
        elems["pgie"].link(elems["conv"])
        probe_element = elems["pgie"]
    elems["conv"].link(elems["sink"])

    return pipeline, probe_element


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", default=os.environ.get("RTSP_URL", ""))
    ap.add_argument("--pgie-config", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=int, default=int(os.environ.get("PROBE_SECONDS", "45")))
    ap.add_argument("--sample-limit", type=int, default=3)
    ap.add_argument(
        "--stream-session-id",
        default=os.environ.get("STREAM_SESSION_ID", ""),
        help="Optional UpStream session id, echoed in NeutralFrameMetadata.",
    )
    # P3: optional tracker. When --tracker-config-path is set, nvtracker is
    # inserted after nvinfer and the pad probe attaches to its src pad.
    ap.add_argument(
        "--tracker-config-path",
        default="",
        help="Path to nvtracker low-level config YAML; if set, nvtracker is enabled.",
    )
    ap.add_argument(
        "--tracker-lib-path",
        default="",
        help="Path to nvtracker ll-lib-file (libnvds_nvmultiobjecttracker.so).",
    )
    ap.add_argument("--tracker-width", type=int, default=960)
    ap.add_argument("--tracker-height", type=int, default=544)
    args = ap.parse_args()

    if not args.rtsp:
        print("ERROR: --rtsp (or RTSP_URL env) required", file=sys.stderr)
        return 2
    if not os.path.isfile(args.pgie_config):
        print(f"ERROR: pgie config not found: {args.pgie_config}", file=sys.stderr)
        return 2
    tracker_config = args.tracker_config_path or None
    if tracker_config and not os.path.isfile(tracker_config):
        print(f"ERROR: tracker config not found: {tracker_config}", file=sys.stderr)
        return 2
    tracker_lib = args.tracker_lib_path or None
    if tracker_config and tracker_lib and not os.path.isfile(tracker_lib):
        print(f"ERROR: tracker lib not found: {tracker_lib}", file=sys.stderr)
        return 2

    print(
        f"pyds metadata app starting url={_redact(args.rtsp)} "
        f"pgie={args.pgie_config} seconds={args.seconds} "
        f"tracker={'on' if tracker_config else 'off'}",
        flush=True,
    )

    state = _State(args.sample_limit)
    state.stream_session_id = args.stream_session_id or None

    try:
        pipeline, probe_element = _build_pipeline(
            args.rtsp,
            args.pgie_config,
            args.width,
            args.height,
            tracker_config=tracker_config,
            tracker_lib=tracker_lib,
            tracker_width=args.tracker_width,
            tracker_height=args.tracker_height,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    probe = _make_pad_probe(state, args.width, args.height)
    probe_element.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, None)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", _bus_call, loop, state)

    pipeline.set_state(Gst.State.PLAYING)
    print(f"playing; bounded {args.seconds}s", flush=True)
    deadline = time.monotonic() + args.seconds

    def _tick() -> bool:
        if time.monotonic() >= deadline:
            loop.quit()
            return False
        return True

    GLib.timeout_add(500, _tick)

    def _stop_handler(_sig: int, _frm: Any) -> None:
        loop.quit()

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    summary: dict[str, Any] = {
        "runtime": "deepstream",
        "metadata_source": "pyds_pad_probe",
        "duration_seconds": args.seconds,
        "frames_seen": state.frames_seen,
        "frames_with_objects": state.frames_with_objects,
        "objects_seen": state.objects_seen,
        "has_bbox": state.has_bbox,
        "has_confidence": state.has_confidence,
        "has_tracker_id": state.has_tracker_id,
        "class_counts": {str(k): v for k, v in state.class_counts.items()},
        "label_counts": dict(state.label_counts),
        "metadata_emitted": state.metadata_emitted,
        "samples_emitted": state.samples_emitted,
    }
    print(_SUMMARY_PREFIX + json.dumps(summary), flush=True)
    return 0 if state.frames_seen > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
