"""
Base Stream Processor
Abstract base class for all stream processors
"""

import cv2
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

# Import video settings from central config
try:
    from config.server_config import VIDEO_FPS, FRAME_BUFFER_ENABLED, FRAME_BUFFER_SECONDS
except ImportError:
    VIDEO_FPS = 15  # Fallback default
    FRAME_BUFFER_ENABLED = False
    FRAME_BUFFER_SECONDS = 2.5

_STREAM_URL_RE = re.compile(
    r"^([a-z][a-z0-9+.-]*://)(?:[^/@]*@)?([^/@]+)/(?:.+)$", re.IGNORECASE
)


def redact_stream_url(url: str) -> str:
    """'rtsp://host:8554/sess_abc123' -> 'rtsp://host:8554/<redacted>'.

    Upstream's internal RTSP URL embeds the per-session stream key, which is
    the secret the --env-file delivery exists to protect. Container stdout
    lands in docker logs/journald, so the path must never be logged. Keeps
    scheme://host:port (reachability is what you debug from a connect log)
    and drops both the path and any user:password@ authority. Mirrors
    upstream-processor's app/redact.py; unparseable input is fully masked.
    """
    m = _STREAM_URL_RE.match((url or "").strip())
    if not m:
        return "<redacted-url>"
    return f"{m.group(1)}{m.group(2)}/<redacted>"

class BaseStreamProcessor(ABC):
    """
    Base class for stream processors.
    Each processor handles one video stream and runs its own processing logic.
    """

    def __init__(self, stream_name: str, rtsp_url: str, device_info: Optional[dict] = None,
                 frame_queue=None, inference_request_queue=None, inference_result_queue=None,
                 processor_heartbeat=None, inference_slot_name=None):
        """
        Initialize the stream processor

        Args:
            stream_name: Unique identifier for this stream (e.g., 'camera1')
            rtsp_url: Full RTSP URL to pull frames from (e.g., 'rtsp://127.0.0.1:8554/camera1')
            device_info: Optional dictionary containing device information (device_id, password, etc.)
            frame_queue: Optional multiprocessing.Queue for sending processed frames for live viewing
            inference_request_queue: Optional queue for sending inference requests to centralized GPU server
            inference_result_queue: Optional queue for receiving inference results from GPU server
            inference_slot_name: Optional slot name for self-healing re-registration with inference server
        """
        self.stream_name = stream_name
        self.rtsp_url = rtsp_url
        self.device_info = device_info or {}
        self.device_id = self.device_info.get('device_id', stream_name)
        self.password = self.device_info.get('password', '')
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = frame_queue  # Queue for live viewing

        # Centralized inference queues (None = legacy per-process GPU mode)
        self.inference_request_queue = inference_request_queue
        self.inference_result_queue = inference_result_queue

        # Slot name for self-healing re-registration with inference server
        self.inference_slot_name = inference_slot_name

        # Heartbeat for stuck-processor detection by orchestrator
        self.processor_heartbeat = processor_heartbeat

        # Setup logging (only add handler if not already configured)
        self.logger = logging.getLogger(f"Processor-{stream_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {stream_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Stats
        self.frames_processed = 0
        self.start_time = None
        self.last_frame_time = time.time()
        self.last_log_time = None
        self.frames_since_log = 0
        self._current_interval_fps = 0.0  # Updated every 5s, published to frame_data

        # Frame validation and recovery (optimized for minimal overhead)
        self._last_valid_frame_ref: Optional[np.ndarray] = None  # Reference only, no copy
        self._validation_frame_counter = 0
        self._validation_interval = 5  # Only validate every N frames
        self.corrupted_frame_count = 0
        self.max_corrupted_before_warn = 10

        # Frame buffer settings (jitter smoothing for network streams)
        self.use_frame_buffer = FRAME_BUFFER_ENABLED
        self.frame_buffer_seconds = FRAME_BUFFER_SECONDS
        self._jitter_buffer = None  # FrameJitterBuffer instance when enabled

    def connect_stream(self, max_retries: int = 10, initial_delay: float = 0.5) -> bool:
        """
        Connect to the RTSP stream with retry logic

        MediaMTX may take a moment to fully establish the stream after the webhook fires,
        so we retry with exponential backoff.

        Args:
            max_retries: Maximum number of connection attempts (increased from 10 to 15)
            initial_delay: Initial delay between retries in seconds (increased from 0.5 to 1.0)

        Returns:
            True if connection successful, False otherwise
        """
        delay = initial_delay

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Connecting to stream (attempt {attempt}/{max_retries}): {redact_stream_url(self.rtsp_url)}")

                # Set FFmpeg options for robust, low-latency RTSP streaming
                # Critical for stable 15fps at 640p over Tailscale
                import os
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join([
                    'rtsp_transport;tcp',         # Use TCP instead of UDP (more reliable)
                    'stimeout;10000000',          # Socket timeout 10 seconds (in microseconds) - increased for stability
                    'max_delay;100000',           # Max demux delay 100ms (reduced for lower latency)
                    'reorder_queue_size;0',       # Disable packet reordering (lower latency)
                    'fflags;nobuffer+discardcorrupt+igndts+flush_packets',  # No buffering + flush immediately
                    'flags;low_delay',            # Enable low delay mode
                    'err_detect;ignore_err',      # Ignore decoding errors (macroblock errors)
                    'analyzeduration;500000',     # Reduce stream analysis time to 0.5s
                    'probesize;500000',           # Reduce probe size to 0.5MB
                    'buffer_size;0',              # Disable internal buffer
                    'fifo_size;0',                # Disable FIFO buffer
                ])

                # Create VideoCapture with FFmpeg backend
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

                # Set small buffer for jitter compensation (was 1, too aggressive)
                # 3 frames = 200ms at 15fps - enough to handle network jitter without
                # causing significant latency or frame drops during processing spikes
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

                # Request specific FPS (helps OpenCV's frame timing)
                self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

                if not self.cap.isOpened():
                    self.logger.warning(f"Failed to open stream (attempt {attempt}/{max_retries})")
                    if self.cap:
                        self.cap.release()
                        self.cap = None

                    if attempt < max_retries:
                        self.logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= 1.5  # Exponential backoff
                    continue

                # Test read a frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.logger.warning(f"Failed to read initial frame (attempt {attempt}/{max_retries})")
                    self.cap.release()
                    self.cap = None

                    if attempt < max_retries:
                        self.logger.info(f"Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= 1.5  # Exponential backoff
                    continue

                # Success! Get stream properties
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.logger.info(f"Successfully connected to stream")
                self.logger.info(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                self.logger.info(f"  Reported FPS: {actual_fps:.1f}")
                self.logger.info(f"  Buffer size: {self.cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

                # Stabilization: drain any buffered frames and let stream settle
                self.logger.info(f"  Stabilizing stream (draining buffers)...")
                drain_start = time.time()
                frames_drained = 0
                while time.time() - drain_start < 1.0:  # Drain for up to 1 second
                    if self.cap.grab():  # grab() is faster than read() - doesn't decode
                        frames_drained += 1
                    else:
                        break
                    time.sleep(0.01)
                self.logger.info(f"  Drained {frames_drained} buffered frames, stream ready")

                # Initialize jitter buffer if enabled (smooths network timing variations)
                if self.use_frame_buffer:
                    from .frame_buffer import FrameJitterBuffer

                    self._jitter_buffer = FrameJitterBuffer(
                        cap=self.cap,
                        target_fps=VIDEO_FPS,
                        buffer_seconds=self.frame_buffer_seconds,
                        logger=self.logger
                    )

                    if not self._jitter_buffer.start(prefill_timeout=15.0):
                        self.logger.error("Failed to prefill jitter buffer")
                        self._jitter_buffer = None
                        return False

                    self.logger.info(f"  Jitter buffer enabled: {self.frame_buffer_seconds}s buffer")

                return True

            except Exception as e:
                self.logger.warning(f"Error connecting to stream (attempt {attempt}/{max_retries}): {e}")
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None

                if attempt < max_retries:
                    self.logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 1.5  # Exponential backoff

        self.logger.error(f"Failed to connect to stream after {max_retries} attempts")
        return False

    def disconnect_stream(self):
        """Disconnect from the RTSP stream"""
        # Stop jitter buffer first if enabled
        if self._jitter_buffer is not None:
            self._jitter_buffer.stop()
            self._jitter_buffer = None

        if self.cap:
            self.cap.release()
            self.cap = None
            self.logger.info("Disconnected from stream")

    def _validate_frame_fast(self, frame: np.ndarray) -> tuple[bool, str]:
        """
        Fast frame validation with minimal overhead.

        Uses min/max instead of std for speed, samples center region only.

        Args:
            frame: The video frame to validate

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Basic null/empty check (very fast)
        if frame is None:
            return False, "frame is None"

        if frame.size == 0:
            return False, "frame is empty"

        # Shape validation (very fast)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False, f"invalid shape: {frame.shape}"

        # Sample center 50% of frame, every 20th pixel (much smaller than before)
        h, w = frame.shape[:2]
        sample = frame[h//4:3*h//4:20, w//4:3*w//4:20]

        # Use min/max instead of std (faster) for corruption detection
        sample_min = int(sample.min())
        sample_max = int(sample.max())

        # All-zero or uniform frame check
        if sample_max - sample_min < 2:
            return False, "uniform/black frame"

        # Green frame check (common FFmpeg decode error)
        # Only do channel analysis if max is suspiciously high
        if sample_max > 200:
            green_mean = sample[:, :, 1].mean()
            if green_mean > 200:
                red_mean = sample[:, :, 2].mean()
                blue_mean = sample[:, :, 0].mean()
                if red_mean < 50 and blue_mean < 50:
                    return False, "green frame corruption"

        return True, "valid"

    def _get_validated_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Validate frame periodically and recover from corruption.

        Optimized to:
        - Only validate every N frames (not every frame)
        - Store reference instead of copy for valid frames
        - Only copy when actually recovering from corruption

        Args:
            frame: The frame to validate

        Returns:
            Valid frame (original or cached), or None if no valid frame available
        """
        self._validation_frame_counter += 1

        # Only validate every N frames for performance
        # But always validate if we're recovering from corrupted frames
        should_validate = (
            self._validation_frame_counter >= self._validation_interval or
            self.corrupted_frame_count > 0
        )

        if not should_validate:
            # Trust the frame, store reference (no copy!)
            self._last_valid_frame_ref = frame
            return frame

        # Reset counter and validate
        self._validation_frame_counter = 0
        is_valid, reason = self._validate_frame_fast(frame)

        if is_valid:
            # Store reference only (no copy!) for potential future recovery
            self._last_valid_frame_ref = frame
            self.corrupted_frame_count = 0
            return frame

        # Frame is corrupted
        self.corrupted_frame_count += 1

        if self.corrupted_frame_count <= self.max_corrupted_before_warn:
            self.logger.debug(f"Corrupted frame detected: {reason}")
        elif self.corrupted_frame_count == self.max_corrupted_before_warn + 1:
            self.logger.warning(f"Many corrupted frames detected (>{self.max_corrupted_before_warn}), suppressing further debug logs")

        # Return cached frame if available - NOW we copy (only on actual corruption)
        if self._last_valid_frame_ref is not None:
            return self._last_valid_frame_ref.copy()

        # No cached frame available - return original
        return frame

    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Optional[np.ndarray]:
        """
        Process a single frame. Override this method with your custom logic.

        Args:
            frame: The video frame as a numpy array (BGR format)
            frame_number: Sequential frame number
            timestamp: Current timestamp

        Returns:
            Processed frame (optional, can return None if no visualization needed)
        """
        pass

    @abstractmethod
    def on_start(self):
        """
        Called when the processor starts.
        Override to perform initialization tasks.
        """
        pass

    @abstractmethod
    def on_stop(self):
        """
        Called when the processor stops.
        Override to perform cleanup tasks.
        """
        pass

    def run(self):
        """
        Main processing loop.
        Continuously reads frames from the stream and processes them.
        """
        print(f"[{self.stream_name}] Starting processor", flush=True)
        self.logger.info("Starting processor")
        self.running = True
        self.start_time = time.time()

        # Connect to stream
        print(f"[{self.stream_name}] Connecting to RTSP stream...", flush=True)
        if not self.connect_stream():
            print(f"[{self.stream_name}] ERROR: Failed to connect to stream", flush=True)
            self.logger.error("Failed to connect to stream, exiting")
            self.running = False
            return

        print(f"[{self.stream_name}] Stream connected! Initializing processor...", flush=True)
        # Call custom start logic
        try:
            self.on_start()
        except Exception as e:
            print(f"[{self.stream_name}] ERROR in on_start: {e}", flush=True)
            self.logger.error(f"Error in on_start: {e}")

        # Log buffer status clearly
        if self._jitter_buffer is not None:
            print(f"[{self.stream_name}] FRAME BUFFER ACTIVE: {self.frame_buffer_seconds}s buffer, targeting {VIDEO_FPS} FPS", flush=True)
        else:
            print(f"[{self.stream_name}] Frame buffer DISABLED (direct stream mode)", flush=True)

        print(f"[{self.stream_name}] Processor ready! Starting frame processing loop...", flush=True)
        # Main processing loop
        frame_number = 0
        consecutive_failures = 0
        max_consecutive_failures = 10  # Exit quickly (~1s) — let orchestrator handle reconnection

        try:
            while self.running:
                # Update heartbeat so orchestrator knows we're alive
                if self.processor_heartbeat is not None:
                    self.processor_heartbeat.value = time.time()

                # Read frame from jitter buffer or directly from capture
                try:
                    if self._jitter_buffer is not None:
                        # Use jitter buffer for steady FPS emission
                        ret, frame = self._jitter_buffer.read()
                    else:
                        # Direct read from VideoCapture
                        ret, frame = self.cap.read()
                except Exception as e:
                    # FFmpeg decode error (e.g., macroblock corruption)
                    consecutive_failures += 1
                    if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                        self.logger.warning(f"Decode error ({consecutive_failures}/{max_consecutive_failures}): {e}")
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error("Too many decode errors, exiting")
                        break
                    time.sleep(0.05)  # Brief pause before retry
                    continue

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                        self.logger.warning(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error("Stream failed, exiting (orchestrator will handle reconnection)")
                        break

                    time.sleep(0.1)
                    continue

                # Reset failure counter on successful read
                consecutive_failures = 0
                frame_number += 1
                self.frames_processed += 1
                timestamp = time.time()
                self.last_frame_time = timestamp

                # Validate frame and recover from corruption if needed
                # This is especially important for night mode where dark frames
                # can become corrupted due to codec/compression issues
                validated_frame = self._get_validated_frame(frame)
                if validated_frame is None:
                    # Extremely rare - no valid frame and no cache
                    self.logger.warning("No valid frame available, skipping")
                    continue
                frame = validated_frame

                # Process the frame (custom logic)
                try:
                    # Save clean frame before processing (in case process_frame modifies in-place)
                    clean_frame = frame.copy() if self.frame_queue is not None else None

                    processed_frame = self.process_frame(frame, frame_number, timestamp)

                    # Send both clean and processed frames to queue for live viewing (non-blocking)
                    if self.frame_queue is not None and processed_frame is not None:
                        try:
                            # input_fps = raw camera rate (from jitter buffer if active,
                            # otherwise processor loop rate which equals camera rate in direct mode)
                            if self._jitter_buffer is not None:
                                input_fps = self._jitter_buffer.stats.get('input_fps', 0.0)
                            else:
                                input_fps = self._current_interval_fps
                            frame_data = {
                                'clean': clean_frame,
                                'annotated': processed_frame,
                                'processor_fps': input_fps,
                            }
                            self.frame_queue.put_nowait(frame_data)
                        except Exception:
                            # Queue full, skip this frame
                            pass

                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_number}: {e}")
                    continue

                # Initialize log timing on first frame
                if self.last_log_time is None:
                    self.last_log_time = timestamp

                self.frames_since_log += 1

                # Log stats periodically (every 5 seconds)
                if timestamp - self.last_log_time >= 5.0:
                    elapsed = timestamp - self.start_time
                    overall_fps = self.frames_processed / elapsed if elapsed > 0 else 0
                    interval_time = timestamp - self.last_log_time
                    interval_fps = self.frames_since_log / interval_time if interval_time > 0 else 0
                    self._current_interval_fps = interval_fps

                    # Build log message
                    log_msg = (f"Processed {self.frames_processed} frames | "
                               f"Overall: {overall_fps:.1f} fps | "
                               f"Last 5s: {interval_fps:.1f} fps")

                    # Add jitter buffer stats if enabled
                    if self._jitter_buffer is not None:
                        level, max_level = self._jitter_buffer.get_buffer_level()
                        buf_stats = self._jitter_buffer.stats
                        log_msg += (f" | Buffer: {level}/{max_level} "
                                   f"(under:{buf_stats['underruns']} over:{buf_stats['overflows']} "
                                   f"dups:{buf_stats['duplicates_emitted']} read:{buf_stats['frames_read']})")

                    self.logger.info(log_msg)

                    self.last_log_time = timestamp
                    self.frames_since_log = 0

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}")
        finally:
            # Cleanup
            self.running = False
            elapsed = time.time() - self.start_time if self.start_time else 0

            try:
                self.on_stop()
            except Exception as e:
                self.logger.error(f"Error in on_stop: {e}")

            self.disconnect_stream()
            self.logger.info(f"Processor stopped. Processed {self.frames_processed} frames in {elapsed:.1f}s")

    def stop(self):
        """Stop the processor"""
        self.logger.info("Stop requested")
        self.running = False

    def get_stats(self) -> dict:
        """
        Get current processor statistics

        Returns:
            Dictionary with processor stats
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frames_processed / elapsed if elapsed > 0 else 0

        return {
            'stream_name': self.stream_name,
            'device_id': self.device_id,
            'rtsp_url': self.rtsp_url,
            'running': self.running,
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed,
            'fps': fps,
            'last_frame_time': self.last_frame_time,
            'has_password': bool(self.password)
        }
