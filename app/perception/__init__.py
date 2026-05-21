"""P5-C latest perception snapshot.

A perception snapshot is the processor's current visual state — detected
objects (bbox, tracker id, class, confidence, ROI labels) plus the configured
ROI/line geometry. It is a dedicated **latest-state** path, deliberately
separate from ``app/observations/`` (an append-only event log): the processor
posts a low-rate snapshot to a distinct API endpoint that upserts one row per
camera. A future dashboard overlay (P5-D) consumes it.
"""
