#!/usr/bin/env python3
"""
Launcher for the CrowdCount single-session worker.

Runs ONE camera's counting loop, configured entirely by CC_* environment
variables — no orchestrator, no MediaMTX, no Flask. This is the container
entrypoint for running the counter as a container-per-session under the
Upstream processor supervisor, and works standalone for testing.

Usage:
  CC_RTSP_URL=rtsp://host:8554/stream python3 scripts/run_counter_worker.py

Run with --help for the full environment contract.
"""

import sys
import argparse
from pathlib import Path

# Add src to path (same convention as run_server.py)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from worker.counter_worker import main  # noqa: E402


ENV_HELP = """
Environment variables:
  CC_RTSP_URL          (required) stream to count on; Upstream passes the
                       session's internal RTSP URL; a video file path also
                       works for smoke tests
  CC_DEVICE_ID         device id; empty together with the password => PROBE
                       MODE (counting runs, nothing posts)
  CC_DEVICE_PASSWORD   device password; must be set together with the id
  CC_API_BASE_URL      default https://api.crowdcount.tech
  CC_STREAM_NAME       log prefix; defaults to the device id or "camera0"
  CC_MODEL_PATH        optional model weights override (testing)
  CC_FRAME_WIDTH/HEIGHT/FPS  informational only

Exit codes: 0 requested stop or ran-and-ended; 1 never reached ready;
2 missing CC_RTSP_URL; 3 credential misconfiguration/rejection.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CrowdCount single-session counting worker (env-driven)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=ENV_HELP,
    )
    parser.parse_args()  # no arguments — configuration is env-only
    main()
