"""
Frame Jitter Buffer
Smooths out network jitter by buffering video frames and emitting them at a steady frame rate.

This module provides a FrameJitterBuffer class that wraps cv2.VideoCapture and absorbs
timing variations from network streams, outputting frames at a consistent FPS.
"""

import cv2
import time
import threading
import logging
import numpy as np
from collections import deque
from typing import Optional, Tuple
from dataclasses import dataclass

# Import video settings from central config
try:
    from config.server_config import VIDEO_FPS
except ImportError:
    VIDEO_FPS = 15  # Fallback default


@dataclass
class BufferedFrame:
    """Container for a buffered frame with metadata"""
    frame: np.ndarray
    capture_time: float
    sequence_num: int


class FrameJitterBuffer:
    """
    Jitter buffer that absorbs network timing variations and emits frames at a steady rate.

    Architecture:
    - Reader thread: Continuously reads from VideoCapture as fast as the source provides
    - Ring buffer: Stores frames with timestamps (deque with maxlen for automatic overflow)
    - Consumer: read() method blocks until next scheduled emission time

    Handles:
    - Buffer underrun: Duplicates last valid frame
    - Buffer overflow: Automatically drops oldest frames (deque behavior)
    - Pre-fill: Waits for buffer to reach minimum level before starting emission

    Usage:
        cap = cv2.VideoCapture(rtsp_url)
        buffer = FrameJitterBuffer(cap, target_fps=15, buffer_seconds=2.5)
        buffer.start()

        while running:
            ret, frame = buffer.read()  # Returns at steady 15 FPS
            if ret:
                process(frame)

        buffer.stop()
    """

    DEFAULT_BUFFER_SECONDS = 2.5
    DEFAULT_TARGET_FPS = VIDEO_FPS
    MIN_PREFILL_RATIO = 0.5  # Wait for 50% buffer fill before starting emission
    MAX_DUPLICATE_SECONDS = 5.0  # Signal stream failure after 5s of frame duplication (allow network hiccups to recover)

    def __init__(
        self,
        cap: cv2.VideoCapture,
        target_fps: float = None,
        buffer_seconds: float = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the jitter buffer.

        Args:
            cap: OpenCV VideoCapture object connected to the stream
            target_fps: Target output frame rate (default: VIDEO_FPS from config)
            buffer_seconds: How many seconds of frames to buffer (default: 2.5)
            logger: Optional logger instance
        """
        self.cap = cap
        self.target_fps = target_fps or self.DEFAULT_TARGET_FPS
        self.buffer_seconds = buffer_seconds or self.DEFAULT_BUFFER_SECONDS
        self.logger = logger or logging.getLogger(__name__)

        # Calculate buffer size in frames
        self.buffer_size = int(self.target_fps * self.buffer_seconds)
        self.min_prefill = int(self.buffer_size * self.MIN_PREFILL_RATIO)

        # Thread-safe ring buffer
        self._buffer: deque[BufferedFrame] = deque(maxlen=self.buffer_size)
        self._buffer_lock = threading.Lock()

        # Frame emission timing
        self._frame_interval = 1.0 / self.target_fps
        self._last_emit_time: Optional[float] = None
        self._next_emit_time: Optional[float] = None

        # Last valid frame for underrun handling
        self._last_valid_frame: Optional[np.ndarray] = None

        # Track when duplication started (for stale stream detection)
        self._duplicate_start_time: Optional[float] = None
        self._staleness_logged = False  # Only log staleness error once per episode

        # Reader thread state
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._sequence_counter = 0

        # Statistics
        self.stats = {
            'frames_read': 0,
            'frames_emitted': 0,
            'frames_skipped': 0,  # Frames grabbed but not decoded (saves CPU)
            'underruns': 0,
            'overflows': 0,
            'duplicates_emitted': 0,
            'input_fps': 0.0,  # Actual camera frame rate (computed in reader thread)
        }

        # Input FPS tracking (reader thread)
        self._input_fps_last_count = 0
        self._input_fps_last_time = None

        # Events
        self._prefill_complete = threading.Event()
        self._stop_event = threading.Event()

        self.logger.info(
            f"FrameJitterBuffer initialized: {self.buffer_seconds}s buffer "
            f"({self.buffer_size} frames), target {self.target_fps} FPS"
        )

    def _reader_loop(self):
        """
        Background thread that continuously reads frames from VideoCapture.

        Optimized to only decode frames we actually need:
        - If buffer is getting full, use grab() to skip frames without decoding
        - Only decode (retrieve()) when buffer has room
        - This significantly reduces CPU usage when source FPS > target FPS
        """
        consecutive_failures = 0
        max_failures = 30  # ~1 second of failures at 30fps before warning
        frames_skipped = 0

        while not self._stop_event.is_set():
            try:
                # Check buffer level to decide if we should skip (grab) or decode (read)
                with self._buffer_lock:
                    buffer_level = len(self._buffer)
                    buffer_nearly_full = buffer_level >= (self.buffer_size * 0.8)

                if buffer_nearly_full:
                    # Buffer is nearly full - skip this frame without decoding
                    # grab() advances the stream but doesn't decode (much faster)
                    if self.cap.grab():
                        frames_skipped += 1
                        self.stats['frames_skipped'] = frames_skipped
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            if consecutive_failures == max_failures:
                                self.logger.warning(
                                    f"Buffer reader: {max_failures} consecutive read failures"
                                )
                        time.sleep(0.01)
                    continue

                # Buffer has room - decode and store this frame
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        if consecutive_failures == max_failures:
                            self.logger.warning(
                                f"Buffer reader: {max_failures} consecutive read failures"
                            )
                    # Brief pause on read failure, then retry
                    time.sleep(0.01)
                    continue

                # Reset failure counter on success
                consecutive_failures = 0

                capture_time = time.time()
                self._sequence_counter += 1

                buffered = BufferedFrame(
                    frame=frame,
                    capture_time=capture_time,
                    sequence_num=self._sequence_counter
                )

                with self._buffer_lock:
                    was_full = len(self._buffer) >= self.buffer_size
                    self._buffer.append(buffered)

                    if was_full:
                        self.stats['overflows'] += 1

                    self.stats['frames_read'] += 1

                    # Signal prefill complete when we have enough frames
                    if not self._prefill_complete.is_set() and len(self._buffer) >= self.min_prefill:
                        self._prefill_complete.set()

                # Compute actual camera input FPS (every 2 seconds)
                if self._input_fps_last_time is None:
                    self._input_fps_last_time = capture_time
                    self._input_fps_last_count = self.stats['frames_read']
                else:
                    fps_elapsed = capture_time - self._input_fps_last_time
                    if fps_elapsed >= 2.0:
                        frames_delta = self.stats['frames_read'] - self._input_fps_last_count
                        self.stats['input_fps'] = frames_delta / fps_elapsed
                        self._input_fps_last_count = self.stats['frames_read']
                        self._input_fps_last_time = capture_time

                # Store for underrun recovery (no lock needed, atomic reference assignment)
                self._last_valid_frame = frame

            except Exception as e:
                self.logger.warning(f"Buffer reader error: {e}")
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame, blocking until the scheduled emission time.

        This method:
        1. Calculates when the next frame should be emitted
        2. Sleeps until that time (ensuring steady FPS)
        3. Returns a frame from the buffer (or duplicate on underrun)

        Returns:
            Tuple of (success: bool, frame: np.ndarray or None)
        """
        if not self._running:
            return False, None

        # Calculate next emission time
        now = time.time()
        if self._next_emit_time is None:
            # First frame - start emission timing
            self._next_emit_time = now

        # Sleep until emission time (with small tolerance for precision)
        sleep_time = self._next_emit_time - now
        if sleep_time > 0.001:  # Only sleep if > 1ms
            time.sleep(sleep_time)

        # Schedule next emission - increment from scheduled time to prevent drift
        # (not from actual time, which would accumulate processing delays)
        self._last_emit_time = time.time()
        self._next_emit_time += self._frame_interval

        # If we've fallen behind (next_emit_time is in the past), catch up
        # This can happen if processing takes longer than frame interval
        now = time.time()
        if self._next_emit_time < now - self._frame_interval:
            # We're more than 1 frame behind, skip ahead to catch up
            frames_behind = int((now - self._next_emit_time) / self._frame_interval)
            self._next_emit_time += frames_behind * self._frame_interval
            self.logger.debug(f"Timing catchup: skipped {frames_behind} frame slots")

        # Get frame from buffer
        with self._buffer_lock:
            if len(self._buffer) > 0:
                buffered = self._buffer.popleft()
                self.stats['frames_emitted'] += 1
                self._duplicate_start_time = None  # Reset duplication timer on real frame
                self._staleness_logged = False  # Reset so we log again if staleness recurs
                return True, buffered.frame
            else:
                # Underrun - buffer empty, must duplicate last frame
                self.stats['underruns'] += 1
                self.stats['duplicates_emitted'] += 1

                # Track how long we've been duplicating
                now = time.time()
                if self._duplicate_start_time is None:
                    self._duplicate_start_time = now

                duplicate_duration = now - self._duplicate_start_time

                # Signal stream failure after prolonged duplication
                if duplicate_duration >= self.MAX_DUPLICATE_SECONDS:
                    if not self._staleness_logged:
                        self.logger.error(
                            f"Stream stale - duplicating frames for {duplicate_duration:.1f}s"
                        )
                        self._staleness_logged = True
                    return False, None  # Signal failure to processor

                # Warn if underruns are frequent (source FPS may be lower than target)
                if self.stats['underruns'] == 50:
                    self.logger.warning(
                        f"Frequent buffer underruns ({self.stats['underruns']}). "
                        f"Source may be slower than {self.target_fps} FPS target."
                    )

                if self._last_valid_frame is not None:
                    # Return copy to prevent mutation issues
                    return True, self._last_valid_frame.copy()
                else:
                    return False, None

    def start(self, prefill_timeout: float = 15.0) -> bool:
        """
        Start the buffer and wait for prefill.

        Args:
            prefill_timeout: Maximum seconds to wait for buffer prefill

        Returns:
            True if started and prefilled successfully
        """
        if self._running:
            return True

        self._running = True
        self._stop_event.clear()
        self._prefill_complete.clear()

        # Reset timing
        self._next_emit_time = None
        self._last_emit_time = None

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="FrameBufferReader",
            daemon=True
        )
        self._reader_thread.start()

        # Wait for prefill
        self.logger.info(
            f"Waiting for buffer prefill ({self.min_prefill}/{self.buffer_size} frames)..."
        )
        prefilled = self._prefill_complete.wait(timeout=prefill_timeout)

        if prefilled:
            with self._buffer_lock:
                buffer_level = len(self._buffer)
            self.logger.info(
                f"Buffer prefilled with {buffer_level} frames "
                f"({buffer_level / self.target_fps:.1f}s), starting emission at {self.target_fps} FPS"
            )
            return True
        else:
            with self._buffer_lock:
                buffer_level = len(self._buffer)
            self.logger.warning(
                f"Buffer prefill timeout after {prefill_timeout}s "
                f"(got {buffer_level}/{self.min_prefill} frames), starting anyway"
            )
            return buffer_level > 0

    def stop(self):
        """Stop the buffer and reader thread"""
        self._running = False
        self._stop_event.set()

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
            if self._reader_thread.is_alive():
                self.logger.warning("Reader thread did not stop cleanly")

        self.logger.info(
            f"Buffer stopped. Stats: read={self.stats['frames_read']}, "
            f"skipped={self.stats['frames_skipped']}, "
            f"emitted={self.stats['frames_emitted']}, "
            f"underruns={self.stats['underruns']}, "
            f"overflows={self.stats['overflows']}, "
            f"duplicates={self.stats['duplicates_emitted']}"
        )

    def reset(self):
        """Reset buffer state for reconnection attempt."""
        with self._buffer_lock:
            self._buffer.clear()
        self._duplicate_start_time = None
        self._staleness_logged = False
        self._last_valid_frame = None
        self._prefill_complete.clear()
        self._next_emit_time = None
        self.stats = {
            'frames_read': 0,
            'frames_emitted': 0,
            'frames_skipped': 0,
            'underruns': 0,
            'overflows': 0,
            'duplicates_emitted': 0,
            'input_fps': 0.0,
        }
        self._input_fps_last_count = 0
        self._input_fps_last_time = None
        self.logger.info("Buffer state reset for reconnection")

    def get_buffer_level(self) -> Tuple[int, int]:
        """
        Get current buffer fill level.

        Returns:
            Tuple of (current_frames, max_frames)
        """
        with self._buffer_lock:
            return len(self._buffer), self.buffer_size

    def get_stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        current, maximum = self.get_buffer_level()
        return {
            **self.stats,
            'buffer_level': current,
            'buffer_max': maximum,
            'buffer_percent': (current / maximum * 100) if maximum > 0 else 0,
        }

    def isOpened(self) -> bool:
        """
        Check if buffer is running and underlying capture is open.
        Compatibility with cv2.VideoCapture interface.

        Returns:
            True if buffer is operational
        """
        return self._running and self.cap.isOpened()

    def set(self, prop_id: int, value: float) -> bool:
        """
        Pass-through to underlying VideoCapture.set().
        Compatibility with cv2.VideoCapture interface.
        """
        return self.cap.set(prop_id, value)

    def get(self, prop_id: int) -> float:
        """
        Pass-through to underlying VideoCapture.get().
        Compatibility with cv2.VideoCapture interface.
        """
        return self.cap.get(prop_id)

    def release(self):
        """
        Stop buffer and release underlying VideoCapture.
        Compatibility with cv2.VideoCapture interface.
        """
        self.stop()
        if self.cap:
            self.cap.release()
