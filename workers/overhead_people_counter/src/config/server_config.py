"""
Server Configuration
Central configuration for the CrowdCount Counter Server

All port settings can be overridden via environment variables for Docker deployment.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
STREAMS_DIR = DATA_DIR / "streams"
RECORDINGS_DIR = DATA_DIR / "recordings"

# =============================================================================
# PORT CONFIGURATION (overridable via environment variables)
# =============================================================================
RTSP_PORT = int(os.getenv("RTSP_PORT", "8554"))
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "5000"))
HLS_PORT = int(os.getenv("HLS_PORT", "8888"))
WEBRTC_PORT = int(os.getenv("WEBRTC_PORT", "8889"))
MEDIAMTX_API_PORT = int(os.getenv("MEDIAMTX_API_PORT", "9997"))
MEDIAMTX_METRICS_PORT = int(os.getenv("MEDIAMTX_METRICS_PORT", "9998"))

# MediaMTX settings
MEDIAMTX_BINARY = BASE_DIR / "bin" / "mediamtx"
MEDIAMTX_CONFIG = SRC_DIR / "config" / "mediamtx.yml"
MEDIAMTX_API_URL = f"http://127.0.0.1:{MEDIAMTX_API_PORT}"
MEDIAMTX_METRICS_URL = f"http://127.0.0.1:{MEDIAMTX_METRICS_PORT}/metrics"

# RTSP settings
RTSP_HOST = "127.0.0.1"

# Orchestrator settings
ORCHESTRATOR_HOST = "0.0.0.0"

# Stream processor settings
PROCESSOR_TYPE = "subprocess"  # Options: "subprocess", "docker" (docker not implemented yet)
MAX_PROCESSORS = 35  # Maximum number of concurrent stream processors (GPU 1: 25 + GPU 0: 10)
PROCESSOR_RESTART_DELAY = 5  # Seconds to wait before restarting a failed processor

# Stream settings
STREAM_RECONNECT_TIMEOUT = 15  # Seconds to wait for stream reconnection before marking offline
STREAM_HEALTH_CHECK_INTERVAL = 5  # Seconds between stream health checks

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "server.log"

# Database settings (for device management)
DB_PATH = DATA_DIR / "devices.db"

# SSH settings for remote device control
SSH_KEY_PATH = os.path.expanduser('~/.ssh/id_rsa')
SSH_USER = 'pi'  # Default Raspberry Pi username

# Device management API key (required for /api/device/ endpoints)
DEVICE_API_KEY = os.getenv("DEVICE_API_KEY", "")

# CORS allowed origins (comma-separated, empty = allow all for backwards compat)
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "https://api.crowdcount.tech,https://crowdcount.tech").split(",") if o.strip()]

# Web dashboard
STATIC_DIR = BASE_DIR / "static"

# =============================================================================
# GPU / INFERENCE SERVER SETTINGS
# =============================================================================
# Number of GPUs available for centralized inference servers (one server per GPU)
# Default is 1 because each Docker container typically sees only its assigned GPU.
# Set to 2+ if running outside Docker with multiple GPUs visible.
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))

# Per-GPU slot limits: {gpu_id: max_connections}
# GPU 1 is primary (25 slots), GPU 0 is overflow (10 slots)
# This allows prioritizing the more powerful GPU while keeping a backup
GPU_SLOT_LIMITS = {
    0: int(os.getenv("GPU_0_SLOTS", "10")),   # Overflow GPU
    1: int(os.getenv("GPU_1_SLOTS", "25")),   # Primary GPU
}

# GPU priority order: fill these GPUs in order (GPU 1 first, then overflow to GPU 0)
GPU_PRIORITY_ORDER = [int(x) for x in os.getenv("GPU_PRIORITY_ORDER", "1,0").split(",")]

# Inference queue settings
INFERENCE_QUEUE_MAXSIZE = int(os.getenv("INFERENCE_QUEUE_MAXSIZE", "100"))

# Batched inference settings (collect multiple frames into one GPU call)
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
INFERENCE_BATCH_WAIT_MS = int(os.getenv("INFERENCE_BATCH_WAIT_MS", "30"))

# YOLO inference image size for centralized inference (smaller = faster)
INFERENCE_IMGSZ = int(os.getenv("INFERENCE_IMGSZ", "480"))

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

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
STREAMS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
