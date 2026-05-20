#!/usr/bin/env python3
"""P2 — Manual pyds metadata probe (DeepStream 8.0).

Builds, inside the DeepStream 8.0 container, an RTSP -> NVDEC -> nvstreammux
-> nvinfer (TrafficCamNet sample) -> fakesink pipeline, attaches a pad probe
on the nvinfer src pad, and walks
``pyds.gst_buffer_get_nvds_batch_meta`` ->
``NvDsBatchMeta`` -> ``NvDsFrameMeta`` -> ``NvDsObjectMeta`` to extract
``class_id`` / ``obj_label`` / ``confidence`` / ``rect_params`` / ``object_id``.

Status: **manual probe only.** This is NOT the MetadataSource, NOT the
metadata adapter, NOT the ObservationWriter. The pad probe does near-zero
work and only updates counters + a tiny bounded sample. **No HTTP, no API
writes, no observations.** The RTSP URL is consumed via CLI/env and never
echoed; all logs use a redacted form.

Run inside the pyds-enabled DeepStream 8.0 image (built by
``docker/Dockerfile.pyds`` and driven by
``scripts/probe_deepstream_pyds_metadata.sh``), not on the Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from typing import Any

# Helpers live in a sibling module so they import without gi/pyds (testable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gi
from _pyds_probe_helpers import (
    UNTRACKED_OBJECT_ID,
    build_neutral_object_sample,
    redact_rtsp,
    scrub_rtsp_inline,
)

gi.require_version("Gst", "1.0")
# isort: split  (gi.require_version must precede the gi.repository import)
from gi.repository import GLib, Gst  # noqa: E402  type: ignore[import-not-found]

# isort: split
import pyds  # noqa: E402  type: ignore[import-not-found]


class _State:
    """Counters + tiny bounded sample. Touched only by the pad probe; one
    GLib thread, so plain attribute writes are fine."""

    def __init__(self, sample_limit: int) -> None:
        self.frames_seen = 0
        self.frames_with_objects = 0
        self.objects_seen = 0
        self.class_counts: Counter[int] = Counter()
        self.label_counts: Counter[str] = Counter()
        self.has_tracker_id = False
        self.has_bbox = False
        self.has_confidence = False
        self.sample: list[dict[str, Any]] = []
        self.sample_limit = sample_limit


def _ensure_str(b: Any) -> str:
    if isinstance(b, bytes):
        return b.decode("utf-8", "replace")
    return "" if b is None else str(b)


def _make_pad_probe(state: _State, width: int, height: int, max_objs_per_frame: int) -> Any:
    """Return a buffer probe func. Near-zero work, no I/O, no async."""

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
            sample_objs: list[dict[str, Any]] = []
            frame_objs_logged = 0
            if int(frame_meta.num_obj_meta) > 0:
                state.frames_with_objects += 1

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                state.objects_seen += 1
                cls_id = int(obj_meta.class_id)
                state.class_counts[cls_id] += 1
                label = _ensure_str(obj_meta.obj_label)
                if label:
                    state.label_counts[label] += 1
                conf = float(obj_meta.confidence)
                rp = obj_meta.rect_params
                left = float(rp.left)
                top = float(rp.top)
                w = float(rp.width)
                h = float(rp.height)
                state.has_bbox = state.has_bbox or (w > 0 and h > 0)
                state.has_confidence = state.has_confidence or (conf > 0.0)
                tracker_id_raw = int(obj_meta.object_id)
                if tracker_id_raw != UNTRACKED_OBJECT_ID:
                    state.has_tracker_id = True

                if (
                    len(state.sample) < state.sample_limit
                    and frame_objs_logged < max_objs_per_frame
                ):
                    sample_objs.append(
                        build_neutral_object_sample(
                            class_id=cls_id,
                            label=label,
                            confidence=conf,
                            left=left,
                            top=top,
                            width=w,
                            height=h,
                            frame_w=width,
                            frame_h=height,
                            tracker_id_raw=tracker_id_raw,
                        )
                    )
                    frame_objs_logged += 1

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            if sample_objs and len(state.sample) < state.sample_limit:
                state.sample.append(
                    {
                        "runtime": "deepstream",
                        "metadata_source": "pyds_pad_probe",
                        "frame": {
                            "frame_num": int(frame_meta.frame_num),
                            "width": width,
                            "height": height,
                        },
                        "objects": sample_objs,
                    }
                )

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
        print(f"GST_ERROR: {scrub_rtsp_inline(f'{err}: {dbg}')}", file=sys.stderr, flush=True)
        loop.quit()
    return True


def _build_pipeline(rtsp_url: str, pgie_config: str, width: int, height: int) -> tuple[Any, Any]:
    Gst.init(None)
    pipeline = Gst.Pipeline.new("pyds-probe")

    elems: dict[str, Any] = {}
    for name, factory in (
        ("rtspsrc", "rtspsrc"),
        ("depay", "rtph264depay"),
        ("parse", "h264parse"),
        ("dec", "nvv4l2decoder"),
        ("mux", "nvstreammux"),
        ("pgie", "nvinfer"),
        ("conv", "nvvideoconvert"),
        ("sink", "fakesink"),
    ):
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
    elems["pgie"].link(elems["conv"])
    elems["conv"].link(elems["sink"])

    return pipeline, elems["pgie"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", default=os.environ.get("RTSP_URL", ""))
    ap.add_argument("--pgie-config", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=int, default=int(os.environ.get("PROBE_SECONDS", "45")))
    ap.add_argument("--sample-limit", type=int, default=3)
    ap.add_argument("--max-objs-per-frame", type=int, default=4)
    args = ap.parse_args()

    if not args.rtsp:
        print("ERROR: --rtsp (or RTSP_URL env) required", file=sys.stderr)
        return 2
    if not os.path.isfile(args.pgie_config):
        print(f"ERROR: pgie config not found: {args.pgie_config}", file=sys.stderr)
        return 2

    print(
        f"pyds probe starting url={redact_rtsp(args.rtsp)} "
        f"pgie={args.pgie_config} seconds={args.seconds}",
        flush=True,
    )

    try:
        pipeline, pgie = _build_pipeline(args.rtsp, args.pgie_config, args.width, args.height)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    state = _State(args.sample_limit)
    probe = _make_pad_probe(state, args.width, args.height, args.max_objs_per_frame)
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, None)

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
        "sample": state.sample,
    }
    print("PYDS_PROBE_SUMMARY=" + json.dumps(summary), flush=True)

    return 0 if state.frames_seen > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
