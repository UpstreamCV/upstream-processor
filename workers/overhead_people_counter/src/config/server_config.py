"""
Worker Configuration
Tuning constants for the overhead people-counter worker image.

Ported from the CrowdCount orchestrator's server_config.py. The orchestrator-
era settings (MediaMTX/RTSP ports, HLS/WebRTC, SSH device control, the device
sqlite DB, CORS, the processor slot pool and the GPU slot allocator) are gone:
this image runs ONE processor per container against CC_RTSP_URL, and Upstream
owns the stream, the GPU assignment (docker run --gpus) and the lifecycle.
Session configuration arrives as CC_* env vars — see worker/counter_worker.py.
"""

import os

# =============================================================================
# VIDEO SETTINGS - Change these values to adjust FPS and resolution globally
# =============================================================================
VIDEO_FPS = 15              # Frames per second (was 30)
VIDEO_WIDTH = 640           # Video width in pixels
VIDEO_HEIGHT = 480          # Video height in pixels

# ByteTrack timing (track_buffer should be ~2 seconds worth of frames)
# At 15fps: 30 frames = 2 seconds | At 30fps: 60 frames = 2 seconds
TRACK_BUFFER_SECONDS = 2.0    # How many seconds to keep lost tracks
TRACK_BUFFER_FRAMES = int(VIDEO_FPS * TRACK_BUFFER_SECONDS)

# =============================================================================
# FRAME BUFFER SETTINGS (Jitter smoothing for network streams)
# =============================================================================
# Enable frame buffering to smooth out network jitter from remote devices
# When enabled, frames are buffered for 2-3 seconds before processing to ensure
# steady FPS output regardless of network timing variations
FRAME_BUFFER_ENABLED = os.getenv("FRAME_BUFFER_ENABLED", "false").lower() == "true"
FRAME_BUFFER_SECONDS = float(os.getenv("FRAME_BUFFER_SECONDS", "2.5"))  # Buffer duration

# =============================================================================
# INFERENCE SETTINGS
# =============================================================================
# Read by the centralized-inference path (PeopleCounterProcessor's
# _on_start_centralized). The single-session worker passes no inference queues,
# so it runs the legacy per-process path and these are inert — kept because the
# shared per-GPU inference sidecar is a live migration decision, not dead code.
INFERENCE_QUEUE_MAXSIZE = int(os.getenv("INFERENCE_QUEUE_MAXSIZE", "100"))

# Batched inference settings (collect multiple frames into one GPU call)
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
INFERENCE_BATCH_WAIT_MS = int(os.getenv("INFERENCE_BATCH_WAIT_MS", "30"))

# YOLO inference image size for centralized inference (smaller = faster)
INFERENCE_IMGSZ = int(os.getenv("INFERENCE_IMGSZ", "480"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
