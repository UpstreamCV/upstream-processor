"""
People Counter Processor - Method 5.4: Occlusion + ReID Relink + Reflection Prevention
Enhanced overhead doorway person counter using YOLO with ByteTrack tracking

METHOD 5: Explicit Occlusion Handling with ReID Relink
- Proactive ReID checkpointing when tracks enter danger zone
- Active search for lost tracks on expected exit side
- Maintains count accuracy even with poor tracking through doorway
- Best for overhead cameras where occlusion is the #1 failure mode

METHOD 5.4: Velocity-Gated Crossing Validation (Reflection Prevention)
- Rejects crossings where track velocity was not toward the crossing line
- Prevents false counts from reflections in glass doors (reflections move away from line)
- Requires tracks to transit through danger zone (anti-teleport check)
- Configurable thresholds for approach speed, frame counts, and consistency ratios
"""

import cv2
import numpy as np
import time
import math
import threading
import requests
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
import os
import queue as queue_module

from .base_processor import BaseStreamProcessor

# Import API client
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from api.crowdcount_api_client import CrowdCountAPIClient

# Get project root directory (crowdcount-counter-server/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Try to import YOLO, provide helpful error if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

# Try to import torch for ReID feature extraction
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not installed. ReID features disabled.")

# Import deep ReID extractor
from .deep_reid import create_reid_extractor, TORCHREID_AVAILABLE


class CrossingDirection(Enum):
    """Direction of crossing the doorway line"""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    NONE = "none"


class DangerZone(Enum):
    """METHOD 5: Zone definitions for occlusion-aware tracking"""
    LEFT_ZONE = "left_zone"           # Safe zone on left side
    DANGER_ZONE = "danger_zone"       # High occlusion risk zone (near/on the line)
    RIGHT_ZONE = "right_zone"         # Safe zone on right side


@dataclass
class Detection:
    """Represents a single person detection with optional ReID features"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    timestamp: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None  # ByteTrack assigned ID
    appearance_features: Optional[np.ndarray] = None  # ReID embedding

    def __hash__(self):
        """Make Detection hashable for set operations"""
        return hash((self.center, self.timestamp))


@dataclass
class ReIDCheckpoint:
    """METHOD 5: Checkpoint data for ReID-based track recovery"""
    embedding: np.ndarray          # ReID features when entering danger zone
    entry_side: str                # 'left' or 'right'
    entry_time: float              # Timestamp when checkpoint created
    entry_position: Tuple[int, int]  # Position when entering danger zone
    expected_exit_side: str        # Expected destination based on velocity
    confidence: float              # Detection confidence at checkpoint
    velocity_x: float              # X velocity at checkpoint time


class PersonTracker:
    """
    METHOD 5: Explicit Occlusion Handling with ReID Relink (RECOMMENDED)

    Key features:
    - Proactive ReID checkpointing when entering danger zone
    - Active search for lost tracks using appearance matching
    - Tracks entering danger zone are "checkpointed" with ReID embedding
    - If track lost in zone, search for it on expected exit side
    """

    # Method 5 configuration constants
    DANGER_ZONE_BUFFER = 0.10       # Width of danger zone (10% of frame)
    CHECKPOINT_TIMEOUT = 5.0        # Max seconds to search for lost track
    RELINK_SIMILARITY_THRESHOLD = 0.55  # ReID match threshold for relinking
    LINGERING_TIMEOUT = 10.0        # Extended timeout for slow-moving people

    def __init__(self, track_id: int, initial_detection: Detection):
        self.track_id = track_id
        self.detections = deque(maxlen=60)  # Increased for better trajectory analysis
        self.positions = deque(maxlen=60)
        self.velocities = deque(maxlen=15)
        self.creation_time = time.time()
        self.last_update_time = time.time()

        # Multiple crossing state tracking
        self.total_crossings = 0
        self.crossing_events = []  # List of (timestamp, direction) tuples
        self.last_side = None  # Track which side of center line person is on
        self.current_side = None
        self.last_crossing_time = 0
        self.min_crossing_interval = 0.5  # Increased: minimum seconds between crossings

        # Overall tracking stats
        self.peak_confidence = 0.0

        # Trajectory validation for counting
        self.side_history = deque(maxlen=10)  # Track recent side positions
        self.detection_count = 0  # Total detections for this track
        self.min_detections_to_count = 2  # TUNED: Lowered for faster crossing detection
        self.consecutive_side_detections = 0  # Consecutive detections on same side
        self.min_consecutive_for_crossing = 1  # Reduced from 2 - trajectory validation handles consistency

        # ReID appearance features
        self.appearance_history = deque(maxlen=10)  # Store recent appearance embeddings
        self.mean_appearance = None  # Running mean of appearance features

        # Pending crossing validation
        self.pending_crossing = None  # (direction, timestamp) if crossing needs validation
        self.crossing_start_position = None  # Position when crossing started

        # METHOD 5: Checkpoint and zone tracking
        self.checkpoint: Optional[ReIDCheckpoint] = None  # Current checkpoint if in danger zone
        self.current_zone: Optional[DangerZone] = None     # Current zone
        self.previous_zone: Optional[DangerZone] = None    # Previous zone
        self.in_danger_zone = False                        # Currently in danger zone
        self.search_active = False                         # Searching for lost track
        self.relinked_from: Optional[int] = None           # Track ID we were relinked from
        self.needs_checkpoint = False                      # Flag for processor to create checkpoint

        # METHOD 5.4: Velocity-gated crossing validation (reflection prevention)
        self.danger_zone_frame_count = 0                   # Frames spent in danger zone during current transit
        self.approach_velocities: deque = deque(maxlen=30) # X-velocities while approaching

        # Add initial detection
        self.update(initial_detection)

    def update(self, detection: Detection):
        """Update tracker with new detection and ReID features"""
        self.detections.append(detection)
        self.positions.append(detection.center)
        self.last_update_time = time.time()
        self.peak_confidence = max(self.peak_confidence, detection.confidence)
        self.detection_count += 1

        # Calculate velocity
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            dt = detection.timestamp - self.detections[-2].timestamp if len(self.detections) >= 2 else 0.033
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                self.velocities.append((vx, vy))

        # Update ReID appearance features
        if detection.appearance_features is not None:
            self.appearance_history.append(detection.appearance_features)
            # Update running mean of appearance (re-normalize to maintain L2 norm)
            if len(self.appearance_history) > 0:
                self.mean_appearance = np.mean(list(self.appearance_history), axis=0)
                norm = np.linalg.norm(self.mean_appearance)
                if norm > 0:
                    self.mean_appearance = self.mean_appearance / norm

    def update_appearance(self, features: np.ndarray):
        """Update appearance features separately"""
        if features is not None:
            self.appearance_history.append(features)
            # Update running mean of appearance (re-normalize to maintain L2 norm)
            if len(self.appearance_history) > 0:
                self.mean_appearance = np.mean(list(self.appearance_history), axis=0)
                norm = np.linalg.norm(self.mean_appearance)
                if norm > 0:
                    self.mean_appearance = self.mean_appearance / norm

    def get_appearance_similarity(self, features: np.ndarray) -> float:
        """Calculate cosine similarity with this tracker's appearance.

        Note: Features are already L2-normalized, so cosine similarity = dot product.
        """
        if self.mean_appearance is None or features is None:
            return 0.0
        # For L2-normalized vectors, cosine similarity is just the dot product
        similarity = np.dot(self.mean_appearance, features)
        # Clamp to [0, 1] range (can be slightly out of range due to floating point)
        return max(0.0, min(1.0, float(similarity)))

    def _determine_zone(self, x: int, frame_width: int, doorway_center: float,
                        danger_buffer: float) -> DangerZone:
        """METHOD 5: Determine which zone the x-coordinate is in."""
        center_x = frame_width * doorway_center
        buffer_px = frame_width * danger_buffer / 2

        if x < center_x - buffer_px:
            return DangerZone.LEFT_ZONE
        elif x > center_x + buffer_px:
            return DangerZone.RIGHT_ZONE
        else:
            return DangerZone.DANGER_ZONE

    def create_checkpoint(self, embedding: np.ndarray, position: Tuple[int, int],
                          confidence: float) -> None:
        """METHOD 5: Create a checkpoint when entering danger zone."""
        vx, _ = self.get_smooth_velocity()

        # Determine entry side based on previous zone
        entry_side = 'left' if self.previous_zone == DangerZone.LEFT_ZONE else 'right'

        # Predict expected exit side based on velocity
        if vx > 0:
            expected_exit = 'right'
        elif vx < 0:
            expected_exit = 'left'
        else:
            # No clear velocity - predict based on entry side (likely continues same direction)
            expected_exit = 'right' if entry_side == 'left' else 'left'

        self.checkpoint = ReIDCheckpoint(
            embedding=embedding,
            entry_side=entry_side,
            entry_time=time.time(),
            entry_position=position,
            expected_exit_side=expected_exit,
            confidence=confidence,
            velocity_x=vx
        )
        self.search_active = True

    def clear_checkpoint(self) -> None:
        """METHOD 5: Clear the checkpoint after successful crossing or timeout."""
        self.checkpoint = None
        self.search_active = False

    def check_crossing(self, frame_width: int, doorway_center: float, doorway_width: float,
                       danger_buffer: float = None,
                       velocity_config: Optional[dict] = None) -> Optional[CrossingDirection]:
        """
        METHOD 5: Explicit Occlusion Handling with ReID Relink
        METHOD 5.4: Added velocity-gated crossing validation for reflection prevention

        Zones:
        - LEFT_ZONE: Safe zone on left side
        - DANGER_ZONE: High occlusion risk (around the line)
        - RIGHT_ZONE: Safe zone on right side

        Logic:
        1. When entering DANGER_ZONE, create checkpoint with ReID embedding
        2. Track normal crossing if track survives through zone
        3. If track lost, checkpoint enables ReID-based relinking
        4. Clear checkpoint on successful exit or timeout
        5. (5.4) Validate velocity was toward crossing line before allowing crossing
        """
        if len(self.positions) < 2:
            return None

        current_x = self.positions[-1][0]
        current_time = time.time()

        # Use configured buffer or default
        buffer = danger_buffer if danger_buffer is not None else self.DANGER_ZONE_BUFFER

        # Determine current zone
        new_zone = self._determine_zone(current_x, frame_width, doorway_center, buffer)

        # Track zone transitions
        zone_changed = (new_zone != self.current_zone)
        self.previous_zone = self.current_zone
        self.current_zone = new_zone

        # Update in_danger_zone flag
        self.in_danger_zone = (new_zone == DangerZone.DANGER_ZONE)

        # METHOD 5.4: Track velocity and danger zone frames for validation
        if self.velocities:
            vx = self.velocities[-1][0]  # Most recent X velocity
            # Record approach velocity when in safe zones, but only if actually moving
            # Idle/near-zero velocities dilute the approach ratio and cause valid crossings to be rejected
            if new_zone in (DangerZone.LEFT_ZONE, DangerZone.RIGHT_ZONE):
                if abs(vx) > 3.0:  # Ignore idle frames (noise threshold ~3 px/s)
                    self.approach_velocities.append(vx)

        # Track frames in danger zone
        if new_zone == DangerZone.DANGER_ZONE:
            self.danger_zone_frame_count += 1
        elif zone_changed and new_zone != DangerZone.DANGER_ZONE:
            # Entering a safe zone - if coming from danger zone, keep count for validation
            # If entering danger zone fresh, reset will happen below
            pass

        # Reset counters when entering danger zone from a safe zone
        if zone_changed and new_zone == DangerZone.DANGER_ZONE:
            self.danger_zone_frame_count = 1  # Start counting (we're in it now)
            # Don't clear approach_velocities - we need the history from before entering

        # Update legacy side tracking
        if new_zone == DangerZone.LEFT_ZONE:
            self.current_side = 'left'
        elif new_zone == DangerZone.RIGHT_ZONE:
            self.current_side = 'right'
        # Don't update side while in danger zone

        # Initialize last_side if needed
        if self.last_side is None and self.current_side:
            self.last_side = self.current_side

        # Track side history
        if self.current_side:
            self.side_history.append(self.current_side)

        # METHOD 5: Checkpoint management
        if zone_changed and new_zone == DangerZone.DANGER_ZONE:
            # Signal processor to create checkpoint (needs ReID extractor + frame)
            self.needs_checkpoint = True

        # Check for checkpoint timeout
        if self.checkpoint and (current_time - self.checkpoint.entry_time > self.CHECKPOINT_TIMEOUT):
            self.clear_checkpoint()

        # Track consecutive side detections
        if self.current_side:
            if self.side_history and len(self.side_history) >= 2:
                if self.side_history[-1] == self.side_history[-2]:
                    self.consecutive_side_detections += 1
                else:
                    self.consecutive_side_detections = 1

        # Check for crossing
        crossing_to_emit = None

        # Don't count crossings until we have enough detections
        if self.detection_count < self.min_detections_to_count:
            return None

        # METHOD 5: Zone-based crossing detection
        if zone_changed and self.previous_zone and self.current_zone:
            # Exited danger zone to a safe zone
            if self.previous_zone == DangerZone.DANGER_ZONE:
                if self.checkpoint:
                    entry_side = self.checkpoint.entry_side
                    exit_side = 'left' if self.current_zone == DangerZone.LEFT_ZONE else 'right'

                    if entry_side != exit_side:
                        # METHOD 5.4: Validate velocity before counting crossing
                        velocity_valid = True
                        if velocity_config and velocity_config.get('velocity_validation_enabled', False):
                            velocity_valid = self.validate_velocity_for_crossing(
                                entry_side=entry_side,
                                min_approach_speed=velocity_config.get('velocity_min_approach_speed', 0.0234) * frame_width,
                                min_approach_frames=velocity_config.get('velocity_min_approach_frames', 5),
                                approach_ratio_threshold=velocity_config.get('velocity_approach_ratio_threshold', 0.5),
                                min_transit_frames=velocity_config.get('velocity_min_transit_frames', 2),
                                validation_window=velocity_config.get('velocity_validation_window', 20)
                            )

                        if velocity_valid:
                            # Successful crossing through danger zone
                            if entry_side == 'left' and exit_side == 'right':
                                if current_time - self.last_crossing_time >= self.min_crossing_interval:
                                    crossing_to_emit = CrossingDirection.LEFT_TO_RIGHT
                                    self.total_crossings += 1
                                    self.crossing_events.append((current_time, crossing_to_emit))
                                    self.last_crossing_time = current_time
                                    self.last_side = 'right'
                            elif entry_side == 'right' and exit_side == 'left':
                                if current_time - self.last_crossing_time >= self.min_crossing_interval:
                                    crossing_to_emit = CrossingDirection.RIGHT_TO_LEFT
                                    self.total_crossings += 1
                                    self.crossing_events.append((current_time, crossing_to_emit))
                                    self.last_crossing_time = current_time
                                    self.last_side = 'left'

                    # Clear checkpoint after processing
                    self.clear_checkpoint()

                else:
                    # No checkpoint - use simple side change detection
                    if self.last_side and self.current_side and self.last_side != self.current_side:
                        # METHOD 5.4: Validate velocity for non-checkpoint crossings too
                        velocity_valid = True
                        if velocity_config and velocity_config.get('velocity_validation_enabled', False):
                            velocity_valid = self.validate_velocity_for_crossing(
                                entry_side=self.last_side,
                                min_approach_speed=velocity_config.get('velocity_min_approach_speed', 0.0234) * frame_width,
                                min_approach_frames=velocity_config.get('velocity_min_approach_frames', 5),
                                approach_ratio_threshold=velocity_config.get('velocity_approach_ratio_threshold', 0.5),
                                min_transit_frames=velocity_config.get('velocity_min_transit_frames', 2),
                                validation_window=velocity_config.get('velocity_validation_window', 20)
                            )

                        if velocity_valid:
                            if self.last_side == 'left' and self.current_side == 'right':
                                if current_time - self.last_crossing_time >= self.min_crossing_interval:
                                    crossing_to_emit = CrossingDirection.LEFT_TO_RIGHT
                                    self.total_crossings += 1
                                    self.crossing_events.append((current_time, crossing_to_emit))
                                    self.last_crossing_time = current_time
                                    self.last_side = self.current_side
                            elif self.last_side == 'right' and self.current_side == 'left':
                                if current_time - self.last_crossing_time >= self.min_crossing_interval:
                                    crossing_to_emit = CrossingDirection.RIGHT_TO_LEFT
                                    self.total_crossings += 1
                                    self.crossing_events.append((current_time, crossing_to_emit))
                                    self.last_crossing_time = current_time
                                    self.last_side = self.current_side

                # METHOD 5.4: Reset velocity tracking after crossing attempt (valid or not)
                self.danger_zone_frame_count = 0
                self.approach_velocities.clear()

            # Direct zone skip: LEFT_ZONE <-> RIGHT_ZONE without passing through DANGER_ZONE
            # This happens when the danger zone is narrow and the person's center jumps over it
            elif (self.previous_zone in (DangerZone.LEFT_ZONE, DangerZone.RIGHT_ZONE) and
                  self.current_zone in (DangerZone.LEFT_ZONE, DangerZone.RIGHT_ZONE) and
                  self.previous_zone != self.current_zone):
                if self.last_side and self.current_side and self.last_side != self.current_side:
                    if current_time - self.last_crossing_time >= self.min_crossing_interval:
                        if self.last_side == 'left' and self.current_side == 'right':
                            crossing_to_emit = CrossingDirection.LEFT_TO_RIGHT
                        elif self.last_side == 'right' and self.current_side == 'left':
                            crossing_to_emit = CrossingDirection.RIGHT_TO_LEFT

                        if crossing_to_emit:
                            self.total_crossings += 1
                            self.crossing_events.append((current_time, crossing_to_emit))
                            self.last_crossing_time = current_time
                            self.last_side = self.current_side

                    self.approach_velocities.clear()

        return crossing_to_emit

    def _validate_crossing(self, frame_width: int, doorway_center: float) -> bool:
        """Legacy validation method - always returns True for Method 5."""
        return True

    def get_smooth_velocity(self) -> Tuple[float, float]:
        """Get smoothed velocity estimate"""
        if not self.velocities:
            return (0.0, 0.0)

        # Weighted average with recent velocities having more weight
        weights = np.linspace(0.5, 1.0, len(self.velocities))
        vx_avg = np.average([v[0] for v in self.velocities], weights=weights)
        vy_avg = np.average([v[1] for v in self.velocities], weights=weights)

        return (vx_avg, vy_avg)

    def validate_velocity_for_crossing(
        self,
        entry_side: str,
        min_approach_speed: float = 15.0,
        min_approach_frames: int = 5,
        approach_ratio_threshold: float = 0.5,
        min_transit_frames: int = 2,
        validation_window: int = 20
    ) -> bool:
        """
        METHOD 5.4: Validate that velocity history is consistent with a real crossing.

        Reflections on glass doors have velocity AWAY from the crossing line,
        while real people have velocity TOWARD the line before crossing.

        Args:
            entry_side: 'left' or 'right' - which side the track entered from
            min_approach_speed: Minimum speed (pixels/sec) toward line to count as approaching
            min_approach_frames: Minimum frames with valid approach velocity
            approach_ratio_threshold: Minimum ratio of frames moving toward line
            min_transit_frames: Minimum frames in danger zone (anti-teleport check)
            validation_window: Number of recent velocities to analyze

        Returns:
            True if velocity pattern is consistent with real crossing, False otherwise
        """
        # Check 1: Did the track actually transit through the danger zone?
        # Reflections or track swaps may "teleport" across without being in the zone
        if self.danger_zone_frame_count < min_transit_frames:
            return False

        # Check 2: Analyze approach velocity history
        # Need enough velocity samples to make a determination
        if len(self.approach_velocities) < min_approach_frames:
            # Not enough history - be permissive for tracks that started near the line
            # but still require they spent time in the danger zone
            return self.danger_zone_frame_count >= min_transit_frames

        # Get recent approach velocities (X component only - horizontal movement)
        recent_velocities = list(self.approach_velocities)[-validation_window:]

        # Determine expected velocity direction based on entry side
        # Left side entry: should be moving RIGHT (positive vx) toward center line
        # Right side entry: should be moving LEFT (negative vx) toward center line
        if entry_side == 'left':
            # Count frames moving toward line (positive X velocity above speed threshold)
            frames_toward = sum(1 for vx in recent_velocities if vx > min_approach_speed)
            # Also count frames moving in the correct direction at any speed
            frames_correct_direction = sum(1 for vx in recent_velocities if vx > 0)
        else:  # entry_side == 'right'
            # Count frames moving toward line (negative X velocity above speed threshold)
            frames_toward = sum(1 for vx in recent_velocities if vx < -min_approach_speed)
            frames_correct_direction = sum(1 for vx in recent_velocities if vx < 0)

        # Check 3: Was the track consistently moving toward the line?
        # Primary: strict check with speed threshold
        approach_ratio = frames_toward / len(recent_velocities)
        if approach_ratio >= approach_ratio_threshold:
            return True

        # Secondary: if most frames move in the correct direction (even slowly),
        # accept the crossing - this prevents rejecting slow but legitimate crossings
        direction_ratio = frames_correct_direction / len(recent_velocities)
        if direction_ratio >= 0.6:
            return True

        return False

    def time_since_update(self) -> float:
        """Time since last update"""
        return time.time() - self.last_update_time

    def get_predicted_position(self, current_time: float = None) -> Tuple[int, int]:
        """
        Get predicted position based on velocity interpolation.
        Used to maintain tracking through dropped frames.

        Args:
            current_time: Current timestamp. If None, uses time.time()

        Returns:
            Predicted (x, y) position tuple
        """
        if not self.positions:
            return (0, 0)

        last_pos = self.positions[-1]

        if current_time is None:
            current_time = time.time()

        time_delta = current_time - self.last_update_time

        # Don't extrapolate too far into the future (max 1 second)
        time_delta = min(time_delta, 1.0)

        # Get velocity and predict new position
        vx, vy = self.get_smooth_velocity()

        predicted_x = last_pos[0] + vx * time_delta
        predicted_y = last_pos[1] + vy * time_delta

        return (int(predicted_x), int(predicted_y))

    def interpolate_positions(self, current_time: float, num_steps: int = 3) -> List[Tuple[int, int]]:
        """
        Generate interpolated positions between last known position and predicted current position.
        Useful for checking crossings that may have occurred during dropped frames.

        Args:
            current_time: Current timestamp
            num_steps: Number of intermediate positions to generate

        Returns:
            List of interpolated (x, y) positions
        """
        if not self.positions:
            return []

        last_pos = self.positions[-1]
        predicted_pos = self.get_predicted_position(current_time)

        # Generate intermediate positions
        positions = []
        for i in range(1, num_steps + 1):
            t = i / num_steps
            interp_x = int(last_pos[0] + (predicted_pos[0] - last_pos[0]) * t)
            interp_y = int(last_pos[1] + (predicted_pos[1] - last_pos[1]) * t)
            positions.append((interp_x, interp_y))

        return positions


class PeopleCounterProcessor(BaseStreamProcessor):
    """
    Overhead doorway person counter processor using YOLO
    Tracks people crossing a virtual line and maintains counts
    """

    def __init__(self, stream_name: str, rtsp_url: str, device_info: Optional[dict] = None,
                 frame_queue=None, inference_request_queue=None, inference_result_queue=None,
                 processor_heartbeat=None, inference_slot_name=None):
        super().__init__(stream_name, rtsp_url, device_info, frame_queue,
                         inference_request_queue, inference_result_queue,
                         processor_heartbeat=processor_heartbeat,
                         inference_slot_name=inference_slot_name)

        # YOLO model setup
        self.model = None
        self.model_path = str(MODELS_DIR / "crowdcount-v6.pt")  # V6 model trained on V6_data dataset (mAP50=0.944)
        self.tracker_config = str(MODELS_DIR / "botsort.yaml")  # BoT-SORT config (XYWH Kalman, no ReID)
        self.class_names = {0: 'person'}  # Default

        # Doorway configuration - TUNED FOR GROUP DETECTION
        self.doorway_config = {
            'doorway_center': 0.5,      # 50% of frame width (center line)
            'doorway_width': 0.02,      # 2% of frame width (narrow line)
            'approach_zone': 0.15,      # 15% approach zone on each side
            'min_path_length': 50,      # Minimum pixels traveled
            'max_tracking_distance': 0.156,  # Proportion of frame width (~100px at 640w)
            'tracker_timeout': 1.5,     # TUNED: Reduced from 3.0 to let false tracks expire faster
            'yolo_conf_threshold': 0.31,  # F1-optimal threshold from V6 PR curve analysis
            'yolo_iou_threshold': 0.3,   # TUNED: Lowered from 0.45 to separate overlapping people
            'min_crossing_interval': 0.5,  # TUNED: Increased from 0.3 to prevent double counts
            'min_peak_confidence': 0.35,  # Aligned with YOLO conf — final quality gate before counting a crossing
            'reid_similarity_threshold': 0.6,  # TUNED: Increased for deep ReID (more discriminative)
            'use_bytetrack': True,       # NEW: Enable ByteTrack
            'use_reid': True,            # NEW: Enable ReID features
            'reid_model': 'osnet_x0_25',  # Deep ReID model: osnet_x0_25 (fastest ~3ms GPU) - was osnet_x1_0
            # METHOD 5: Occlusion + ReID Relink parameters (RECOMMENDED)
            'danger_zone_buffer': 0.10,         # Width of danger zone (10% of frame)
            'checkpoint_timeout': 5.0,          # Max seconds to search for lost track
            'relink_similarity_threshold': 0.55,  # ReID match threshold for relinking (lower OK for occlusion)
            'lingering_timeout': 10.0,          # Extended timeout for slow-moving people
            # METHOD 5.4: Velocity-gated crossing validation (reflection prevention)
            'velocity_validation_enabled': True,        # Enable velocity-based crossing validation
            'velocity_min_approach_speed': 0.0234,       # Proportion of frame width/sec (~15px/s at 640w)
            'velocity_min_approach_frames': 3,          # Min frames with approach velocity before crossing (was 5 at 30fps)
            'velocity_approach_ratio_threshold': 0.5,   # Min ratio of frames moving toward line (0.0-1.0)
            'velocity_min_transit_frames': 1,           # Min frames detected in danger zone (anti-teleport, was 2 at 30fps)
            'velocity_validation_window': 10,           # Frames to analyze for velocity validation (was 20 at 30fps)
            # Night mode: CLAHE preprocessing for low-light enhancement
            'use_clahe': True,                  # Enable CLAHE preprocessing for night/low-light
            'clahe_clip_limit': 2.0,            # Contrast limit (1.5-3.0, higher = more contrast but more noise)
            'clahe_tile_size': 8,               # Tile grid size (8 is standard)
            'clahe_engage_threshold': 60,       # Engage CLAHE when brightness drops below this (0-255)
            'clahe_disengage_threshold': 80,    # Disengage CLAHE when brightness rises above this (0-255)
        }

        # METHOD 5: Active checkpoints for lost track recovery
        self.active_checkpoints: Dict[int, ReIDCheckpoint] = {}

        # Deep ReID feature extractor (initialized in on_start)
        self.reid_extractor = None

        # CLAHE preprocessor for night mode (initialized in on_start)
        self.clahe = None
        self.clahe_engaged = False  # Hysteresis state: whether CLAHE is currently active

        # Tracking
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_track_id = 1
        self.track_id_pool = []

        # Crossing counts
        self.crossings = {
            CrossingDirection.LEFT_TO_RIGHT: 0,
            CrossingDirection.RIGHT_TO_LEFT: 0
        }
        self.crossing_log = []

        # Initialize API client with device credentials.
        # Enabled by either a per-device password or a shared service secret
        # (the Upstream worker path — posts on behalf of device_id without a
        # per-device password).
        device_id = self.device_info.get('device_id', self.stream_name)
        password = self.device_info.get('password', '')
        service_secret = self.device_info.get('service_secret')
        base_url = self.device_info.get('api_base_url') or 'https://api.crowdcount.tech'
        self.api_client = CrowdCountAPIClient(
            device_id=device_id,
            password=password,
            base_url=base_url,
            timeout=5,
            service_secret=service_secret,
        )

        # Fetch direction flip setting and set up periodic refresh
        self.direction_flipped = self.api_client.get_direction_flipped()
        self.direction_refresh_timer = None
        self.logger.info(f"Direction flipped setting initialized: {self.direction_flipped}")

        # Optional perception hook: a callable(frame, detections, frame_number)
        # invoked once per frame with the tracked detections, used to publish
        # live boxes to the dashboard overlay. Never counting-critical — the
        # process_frame call site swallows any error it raises.
        self.perception_hook = None

    def on_start(self):
        """Initialize models or connect to centralized inference server when processor starts"""
        # Start periodic refresh of direction_flipped setting (every 60 seconds)
        self._schedule_direction_refresh()

        # Determine mode: centralized inference vs legacy per-process GPU
        self.use_centralized_inference = (
            self.inference_request_queue is not None and
            self.inference_result_queue is not None
        )

        if self.use_centralized_inference:
            self._on_start_centralized()
        else:
            self._on_start_legacy()

    def _on_start_centralized(self):
        """Initialize for centralized inference mode (CPU-only worker)"""
        from .inference_server import RequestType, InferenceRequest
        # Store these for use in extract_detections / ReID calls
        self._RequestType = RequestType
        self._InferenceRequest = InferenceRequest

        # Consecutive inference failure counter for fast backoff
        self._centralized_inference_failures = 0
        self._last_inference_retry_time = 0.0
        self._last_queue_full_log = 0  # Rate-limit "queue full" warnings
        self._last_inference_error_log = 0  # Rate-limit inference timeout errors

        print(f"[{self.stream_name}] Centralized inference mode (CPU-only worker)", flush=True)
        self.logger.info("Using centralized inference server (no local GPU)")

        # Initialize standalone tracker for local tracking
        # We use model.predict() on the server side + local tracker
        self._init_standalone_tracker()

        # Initialize CLAHE for night mode preprocessing (runs on CPU)
        self._init_clahe()

        # Log velocity validation settings
        self._log_velocity_settings()

        print(f"[{self.stream_name}] Worker ready (tracker + crossing logic on CPU)", flush=True)

    def _on_start_legacy(self):
        """Initialize for legacy per-process GPU mode (original behavior)"""
        if not YOLO_AVAILABLE:
            print(f"ERROR: YOLO not available - cannot start people counter", flush=True)
            self.logger.error("YOLO not available - cannot start people counter")
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")

        try:
            import torch

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            self.device = 'cuda' if cuda_available else 'cpu'

            print(f"Loading YOLO model: {self.model_path}", flush=True)
            print(f"Using device: {self.device}", flush=True)
            print(f"Tracker config: {self.tracker_config}", flush=True)
            if cuda_available:
                print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

            self.logger.info(f"Loading YOLO model: {self.model_path}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"ByteTrack enabled: {self.doorway_config['use_bytetrack']}")
            self.logger.info(f"ReID enabled: {self.doorway_config['use_reid']}")

            # Load model with explicit CUDA device
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            # Verify device
            model_device = next(self.model.model.parameters()).device
            print(f"Model loaded on device: {model_device}", flush=True)
            self.logger.info(f"Model loaded on device: {model_device}")

            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"Model classes: {', '.join([f'{k}:{v}' for k, v in list(self.class_names.items())[:5]])}...", flush=True)
                self.logger.info(f"Model classes: {', '.join([f'{k}:{v}' for k, v in self.class_names.items()])}")

            # Log tuned parameters
            print(f"Tuned parameters for group detection:", flush=True)
            print(f"  - Confidence threshold: {self.doorway_config['yolo_conf_threshold']}", flush=True)
            print(f"  - IOU threshold: {self.doorway_config['yolo_iou_threshold']}", flush=True)
            print(f"  - Min crossing interval: {self.doorway_config['min_crossing_interval']}s", flush=True)
            print(f"  - Tracker timeout: {self.doorway_config['tracker_timeout']}s", flush=True)

            print("YOLO + ByteTrack loaded successfully with CUDA!" if cuda_available else "YOLO + ByteTrack loaded (CPU mode)", flush=True)
            self.logger.info("YOLO model with ByteTrack loaded successfully")

            # Initialize Deep ReID extractor
            if self.doorway_config['use_reid']:
                print(f"Loading Deep ReID model: {self.doorway_config['reid_model']}...", flush=True)
                self.reid_extractor = create_reid_extractor(
                    model_name=self.doorway_config['reid_model'],
                    device=self.device,
                    verbose=True
                )
                print(f"Deep ReID loaded (feature dim: {self.reid_extractor.feature_dim})", flush=True)
                self.logger.info(f"Deep ReID initialized: {self.doorway_config['reid_model']}")

            # Initialize CLAHE for night mode preprocessing
            self._init_clahe()

            # Log velocity validation settings
            self._log_velocity_settings()

        except Exception as e:
            print(f"ERROR: Failed to load YOLO model: {e}", flush=True)
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _init_standalone_tracker(self):
        """Initialize standalone tracker (BoT-SORT or ByteTrack) for centralized inference mode."""
        try:
            import yaml
            import os

            # Load tracker config from yaml
            with open(self.tracker_config, 'r') as f:
                bt_config = yaml.safe_load(f)

            tracker_type = bt_config.get('tracker_type', 'bytetrack')

            # Create args object with shared parameters
            class TrackerArgs:
                pass

            args = TrackerArgs()
            args.track_high_thresh = bt_config.get('track_high_thresh', 0.25)
            args.track_low_thresh = bt_config.get('track_low_thresh', 0.15)
            args.new_track_thresh = bt_config.get('new_track_thresh', 0.3)
            args.track_buffer = bt_config.get('track_buffer', 30)
            args.match_thresh = bt_config.get('match_thresh', 0.75)
            args.fuse_score = bt_config.get('fuse_score', True)

            if tracker_type == 'botsort':
                from ultralytics.trackers.bot_sort import BOTSORT

                # BoT-SORT specific args
                args.gmc_method = bt_config.get('gmc_method', 'none')
                args.proximity_thresh = bt_config.get('proximity_thresh', 0.5)
                args.appearance_thresh = bt_config.get('appearance_thresh', 0.25)
                args.with_reid = bt_config.get('with_reid', False)
                # Resolve model path relative to MODELS_DIR for reliability
                model_name = bt_config.get('model', 'yolo11n-cls.pt')
                model_path = MODELS_DIR / model_name
                args.model = str(model_path) if model_path.exists() else model_name

                # Force CPU for ReID model — processors are CPU-only workers
                old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''

                try:
                    self.byte_tracker = BOTSORT(args, frame_rate=8)
                finally:
                    # Restore original CUDA_VISIBLE_DEVICES
                    if old_cuda is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
                    else:
                        del os.environ['CUDA_VISIBLE_DEVICES']

                reid_status = "with ReID" if args.with_reid else "without ReID"
                print(f"[{self.stream_name}] Standalone BoT-SORT initialized {reid_status}", flush=True)
                self.logger.info(f"Standalone BoT-SORT tracker initialized {reid_status} (model={args.model})")
            else:
                from ultralytics.trackers.byte_tracker import BYTETracker

                self.byte_tracker = BYTETracker(args, frame_rate=8)
                print(f"[{self.stream_name}] Standalone ByteTrack initialized", flush=True)
                self.logger.info("Standalone ByteTrack tracker initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize standalone tracker: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.byte_tracker = None

    def _init_clahe(self):
        """Initialize CLAHE for night mode preprocessing."""
        if self.doorway_config['use_clahe']:
            clip_limit = self.doorway_config['clahe_clip_limit']
            tile_size = self.doorway_config['clahe_tile_size']
            engage_threshold = self.doorway_config['clahe_engage_threshold']
            disengage_threshold = self.doorway_config['clahe_disengage_threshold']
            self.clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_size, tile_size)
            )
            print(f"CLAHE night mode enabled (clipLimit={clip_limit}, tileSize={tile_size}x{tile_size}, engage<{engage_threshold}, disengage>{disengage_threshold})", flush=True)
            self.logger.info(f"CLAHE initialized: clipLimit={clip_limit}, tileGridSize={tile_size}, engageThreshold={engage_threshold}, disengageThreshold={disengage_threshold}")

    def _log_velocity_settings(self):
        """Log METHOD 5.4 velocity validation settings."""
        if self.doorway_config.get('velocity_validation_enabled', False):
            print(f"Velocity-gated crossing validation enabled (reflection prevention)", flush=True)
            print(f"  - Min approach speed: {self.doorway_config['velocity_min_approach_speed']} (proportion of frame width/s)", flush=True)
            print(f"  - Min approach frames: {self.doorway_config['velocity_min_approach_frames']}", flush=True)
            print(f"  - Approach ratio threshold: {self.doorway_config['velocity_approach_ratio_threshold']}", flush=True)
            print(f"  - Min transit frames: {self.doorway_config['velocity_min_transit_frames']}", flush=True)
            self.logger.info(f"Velocity validation enabled: min_speed={self.doorway_config['velocity_min_approach_speed']}, "
                           f"min_frames={self.doorway_config['velocity_min_approach_frames']}, "
                           f"ratio={self.doorway_config['velocity_approach_ratio_threshold']}")

    def on_stop(self):
        """Cleanup when processor stops"""
        # Cancel the direction refresh timer
        if self.direction_refresh_timer:
            self.direction_refresh_timer.cancel()

        # Print summary
        total = sum(self.crossings.values())
        lr_count = self.crossings[CrossingDirection.LEFT_TO_RIGHT]
        rl_count = self.crossings[CrossingDirection.RIGHT_TO_LEFT]
        net_flow = lr_count - rl_count

        self.logger.info("="*60)
        self.logger.info("PEOPLE COUNTER - Summary")
        self.logger.info("="*60)
        self.logger.info(f"Total Crossings: {total}")
        self.logger.info(f"  Left → Right: {lr_count}")
        self.logger.info(f"  Right → Left: {rl_count}")
        self.logger.info(f"  Net Flow (L→R): {net_flow:+d}")
        self.logger.info("="*60)

    def _schedule_direction_refresh(self):
        """Schedule the next direction_flipped setting refresh"""
        self.direction_refresh_timer = threading.Timer(60.0, self._refresh_direction_setting)
        self.direction_refresh_timer.daemon = True
        self.direction_refresh_timer.start()

    def _refresh_direction_setting(self):
        """Fetch the latest direction_flipped setting from API"""
        try:
            new_value = self.api_client.get_direction_flipped()
            if new_value != self.direction_flipped:
                self.logger.info(f"Direction flipped setting changed: {self.direction_flipped} → {new_value}")
                self.direction_flipped = new_value
        except Exception as e:
            self.logger.warning(f"Failed to refresh direction_flipped setting: {e}")

        # Schedule next refresh
        self._schedule_direction_refresh()

    def _drain_frame_buffer(self, current_frame: np.ndarray) -> np.ndarray:
        """Drain any buffered frames and return the most recent one.

        This prevents buffer buildup when processing is slower than frame rate
        (e.g., during night mode with CLAHE). By always processing the latest
        frame, we avoid decode errors while maintaining tracking accuracy.

        Args:
            current_frame: The frame we already have from the main loop

        Returns:
            The most recent frame available (may be the same as current_frame)
        """
        if self.cap is None:
            return current_frame

        latest_frame = current_frame
        frames_drained = 0
        max_drain = 5  # Limit to prevent infinite loop on live streams

        # Drain up to max_drain buffered frames
        # Note: On live RTSP streams, grab() blocks if buffer is empty,
        # so we limit iterations and use the frame we already have as fallback
        while frames_drained < max_drain:
            # Quick non-blocking check using read with minimal timeout isn't possible,
            # so we just do a limited number of grabs
            ret, frame = self.cap.read()
            if ret and frame is not None:
                latest_frame = frame
                frames_drained += 1
            else:
                break

        if frames_drained > 0:
            self.logger.debug(f"Drained {frames_drained} buffered frames")

        return latest_frame

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE preprocessing to enhance visibility in low-light conditions.

        Uses hysteresis to avoid flickering:
        - Engages when brightness drops below engage_threshold (default 60)
        - Stays engaged until brightness rises above disengage_threshold (default 80)

        Applies contrast enhancement to the luminance channel only,
        preserving color information while improving feature visibility.
        """
        try:
            if self.clahe is None:
                return frame

            # Validate frame
            if frame is None or frame.size == 0:
                return frame

            # Ensure frame is 3-channel BGR
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return frame

            # Check frame brightness with hysteresis logic
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)

            engage_threshold = self.doorway_config.get('clahe_engage_threshold', 60)
            disengage_threshold = self.doorway_config.get('clahe_disengage_threshold', 80)

            # Hysteresis: engage below 60, disengage above 80
            if not self.clahe_engaged:
                # Not currently engaged - only engage if brightness drops below engage threshold
                if mean_brightness >= engage_threshold:
                    return frame
                # Brightness is below engage threshold - engage CLAHE
                self.clahe_engaged = True
            else:
                # Currently engaged - only disengage if brightness rises above disengage threshold
                if mean_brightness > disengage_threshold:
                    self.clahe_engaged = False
                    return frame
                # Still below disengage threshold - keep CLAHE engaged

            # Convert BGR to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE to luminance channel only
            l_enhanced = self.clahe.apply(l_channel)

            # Merge channels and convert back to BGR
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            return enhanced_frame

        except Exception as e:
            # If CLAHE fails for any reason, return original frame
            self.logger.warning(f"CLAHE preprocessing failed: {e}")
            return frame

    def extract_detections(self, frame) -> List[Detection]:
        """Extract person detections from frame using YOLO with ByteTrack tracking.

        Supports two modes:
        - Centralized: sends frame to inference server, runs ByteTrack locally
        - Legacy: runs YOLO model.track() locally with per-process GPU
        """
        # Apply CLAHE preprocessing for night mode if enabled
        if self.doorway_config['use_clahe'] and self.clahe is not None:
            frame = self._apply_clahe(frame)

        if self.use_centralized_inference:
            return self._extract_detections_centralized(frame)
        else:
            return self._extract_detections_legacy(frame)

    def _wait_for_response(self, request_id, timeout=5.0):
        """Wait for a response matching a specific request_id, discarding mismatched ones.

        Eliminates the drain-before-request race condition by using correlation IDs.
        Stale or mismatched responses are silently discarded.
        """
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise queue_module.Empty()
            response = self.inference_result_queue.get(timeout=remaining)
            resp_id = getattr(response, 'request_id', None)
            if resp_id is None or resp_id == request_id:
                return response
            # Stale/mismatched — discard and keep waiting

    def _get_inference_suppression_interval(self) -> int:
        """Get log suppression interval based on consecutive failure count.
        Scales up to reduce log spam during sustained outages."""
        failures = self._centralized_inference_failures
        if failures < 50:
            return 30
        elif failures < 200:
            return 60
        elif failures < 500:
            return 120
        else:
            return 300

    def _get_inference_retry_interval(self) -> float:
        """Get backoff retry interval based on consecutive failure count.
        Scales up to reduce REGISTER spam during sustained outages."""
        failures = self._centralized_inference_failures
        if failures < 30:
            return 10.0
        elif failures < 100:
            return 30.0
        else:
            return 60.0

    def _extract_detections_centralized(self, frame) -> List[Detection]:
        """Extract detections via centralized inference server + local ByteTrack."""
        detections = []

        # Fast backoff: after 5 consecutive failures, only retry every 10s
        # to avoid wasting time on get() timeout while allowing recovery
        if self._centralized_inference_failures >= 5:
            now = time.time()
            # Always drain during backoff to prevent result queue buildup
            # from orphaned responses the server is still sending
            try:
                while True:
                    self.inference_result_queue.get_nowait()
            except queue_module.Empty:
                pass
            retry_interval = self._get_inference_retry_interval()
            if now - self._last_inference_retry_time < retry_interval:
                return detections  # Still in backoff window
            self._last_inference_retry_time = now
            self.logger.info(f"Inference backoff: attempting periodic retry (interval: {retry_interval:.0f}s)")
            # Self-healing: re-send REGISTER in case inference server lost our mapping
            if self.inference_slot_name:
                try:
                    self.inference_request_queue.put(self._InferenceRequest(
                        type=self._RequestType.REGISTER,
                        camera_id=self.stream_name,
                        slot_name=self.inference_slot_name,
                    ), timeout=1.0)
                    self.logger.info("Re-sent REGISTER to inference server (self-healing)")
                except Exception:
                    pass

        try:
            # Send detection request to inference server
            request = self._InferenceRequest(
                type=self._RequestType.DETECT,
                camera_id=self.stream_name,
                frame=frame,
                conf=self.doorway_config['yolo_conf_threshold'],
                iou=self.doorway_config['yolo_iou_threshold'],
                slot_name=self.inference_slot_name,
            )
            try:
                self.inference_request_queue.put(request, timeout=2.0)
            except queue_module.Full:
                now = time.time()
                if now - self._last_queue_full_log > 30:
                    self.logger.warning("Inference request queue full - skipping detection (suppressing for 30s)")
                    self._last_queue_full_log = now
                self._centralized_inference_failures += 1
                return detections

            # Wait for correlated response (discards stale/mismatched responses)
            response = self._wait_for_response(request.request_id, timeout=5.0)

            # Only reset failure counter on genuine success (not error/stale responses)
            if not response.error:
                if self._centralized_inference_failures > 0:
                    self.logger.info(
                        f"Inference recovered after {self._centralized_inference_failures} consecutive failures"
                    )
                self._centralized_inference_failures = 0

            if response.error:
                self.logger.warning(f"Inference server error: {response.error}")

            if response.detections:
                # Run standalone ByteTrack on the detection results
                det_results = response.detections
                track_ids = self._run_standalone_bytetrack(det_results, frame)

                for i, det_result in enumerate(det_results):
                    track_id = track_ids[i] if i < len(track_ids) else None

                    detection = Detection(
                        center=det_result.center,
                        bbox=det_result.bbox,
                        confidence=det_result.confidence,
                        timestamp=time.time(),
                        class_id=det_result.class_id,
                        class_name=det_result.class_name,
                        track_id=track_id,
                        appearance_features=None  # Extracted on-demand in update_trackers
                    )
                    detections.append(detection)

        except queue_module.Empty:
            # Inference server not responding — expected during restarts/hangs
            self._centralized_inference_failures += 1
            now = time.time()
            suppression_interval = self._get_inference_suppression_interval()
            if now - self._last_inference_error_log > suppression_interval:
                self.logger.warning(
                    f"Inference response timeout ({self._centralized_inference_failures} consecutive failures, suppressing for {suppression_interval}s)"
                )
                self._last_inference_error_log = now
        except Exception as e:
            self._centralized_inference_failures += 1
            self.logger.error(f"Centralized detection error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return detections

    def _run_standalone_bytetrack(self, det_results, frame) -> List[Optional[int]]:
        """Run standalone ByteTrack tracker on detection results.

        Args:
            det_results: List of DetectionResult from inference server
            frame: Current frame (for dimensions)

        Returns:
            List of track IDs corresponding to each detection (None if untracked)
        """
        if self.byte_tracker is None or not det_results:
            return [None] * len(det_results)

        try:
            import numpy as np

            # Build arrays in the format ultralytics BYTETracker expects
            xyxy_list = []
            xywh_list = []
            conf_list = []
            cls_list = []
            for det in det_results:
                x1, y1, w, h = det.bbox
                x2 = x1 + w
                y2 = y1 + h
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                xyxy_list.append([x1, y1, x2, y2])
                xywh_list.append([cx, cy, w, h])
                conf_list.append(det.confidence)
                cls_list.append(det.class_id)

            if not xyxy_list:
                return [None] * len(det_results)

            # Ultralytics BYTETracker.update() expects a Results-like object with
            # .conf, .cls, .xywh, .xyxy attributes and boolean/integer indexing
            class DetectionWrapper:
                """Lightweight wrapper to satisfy ultralytics BYTETracker API."""
                def __init__(self, xyxy, xywh, conf, cls):
                    self.xyxy = np.array(xyxy, dtype=np.float32)
                    self.xywh = np.array(xywh, dtype=np.float32)
                    self.conf = np.array(conf, dtype=np.float32)
                    self.cls = np.array(cls, dtype=np.float32)
                def __len__(self):
                    return len(self.conf)
                def __getitem__(self, idx):
                    return DetectionWrapper(
                        self.xyxy[idx], self.xywh[idx],
                        self.conf[idx], self.cls[idx],
                    )

            wrapper = DetectionWrapper(xyxy_list, xywh_list, conf_list, cls_list)

            # Update ByteTrack with detections
            # Returns numpy array of shape (N, 8): [x1, y1, x2, y2, track_id, score, cls, det_idx]
            online_targets = self.byte_tracker.update(wrapper, frame)

            # Match ByteTrack outputs back to input detections by center distance
            track_ids = [None] * len(det_results)

            if len(online_targets) == 0:
                return track_ids

            for target in online_targets:
                t_x1, t_y1, t_x2, t_y2 = target[0], target[1], target[2], target[3]
                t_track_id = int(target[4])
                t_center = ((t_x1 + t_x2) / 2, (t_y1 + t_y2) / 2)

                # Find closest input detection by center distance
                best_idx = None
                best_dist = float('inf')
                for i, det in enumerate(det_results):
                    if track_ids[i] is not None:
                        continue  # Already matched
                    dx = det.center[0] - t_center[0]
                    dy = det.center[1] - t_center[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                if best_idx is not None and best_dist < 50:  # Within 50px
                    track_ids[best_idx] = t_track_id

            return track_ids

        except Exception as e:
            self.logger.error(f"ByteTrack error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [None] * len(det_results)

    def _extract_detections_legacy(self, frame) -> List[Detection]:
        """Extract detections using local YOLO model (legacy per-process GPU mode)."""
        detections = []

        try:
            if self.doorway_config['use_bytetrack']:
                # Use ByteTrack for improved tracking (especially for groups)
                results = self.model.track(
                    frame,
                    conf=self.doorway_config['yolo_conf_threshold'],
                    iou=self.doorway_config['yolo_iou_threshold'],
                    classes=[0],  # Only detect person class
                    persist=True,  # Persist tracks across frames
                    tracker=self.tracker_config,  # Use our custom tracker config
                    verbose=False
                )
            else:
                # Fallback to standard detection
                results = self.model(
                    frame,
                    conf=self.doorway_config['yolo_conf_threshold'],
                    iou=self.doorway_config['yolo_iou_threshold'],
                    classes=[0],
                    verbose=False
                )

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'unknown')

                        # ONLY process person detections (class_id == 0)
                        if class_id != 0:
                            continue

                        # Get ByteTrack ID if available
                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0])

                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Calculate center and dimensions
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width = int(x2 - x1)
                        height = int(y2 - y1)

                        # Create detection WITHOUT ReID features
                        # ReID is extracted later only for unmatched detections (performance optimization)
                        detection = Detection(
                            center=(center_x, center_y),
                            bbox=(int(x1), int(y1), width, height),
                            confidence=confidence,
                            timestamp=time.time(),
                            class_id=class_id,
                            class_name=class_name,
                            track_id=track_id,
                            appearance_features=None  # Extracted on-demand in update_trackers
                        )
                        detections.append(detection)

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return detections

    def _extract_appearance_features(self, frame, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """
        Extract deep ReID appearance features from a person bounding box.
        Uses OSNet pretrained on person re-identification datasets.
        Returns 512-dimensional L2-normalized feature vector.

        In centralized mode, sends a ReID request to the inference server.
        In legacy mode, runs ReID locally on GPU.
        """
        if self.use_centralized_inference:
            return self._extract_appearance_centralized(frame, [(x1, y1, x2, y2)])[0]
        else:
            if self.reid_extractor is None:
                return None
            return self.reid_extractor.extract(frame, x1, y1, x2, y2)

    def _extract_appearance_batch_centralized(self, frame, boxes: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        """Extract ReID features for multiple boxes via centralized inference server."""
        return self._extract_appearance_centralized(frame, boxes)

    def _extract_appearance_centralized(self, frame, boxes: List[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        """Send ReID extraction request to inference server and wait for response."""
        if not boxes:
            return []

        # Share failure counter with detection — both use the same inference server
        if self._centralized_inference_failures >= 5:
            now = time.time()
            # Always drain during backoff to prevent result queue buildup
            try:
                while True:
                    self.inference_result_queue.get_nowait()
            except queue_module.Empty:
                pass
            retry_interval = self._get_inference_retry_interval()
            if now - self._last_inference_retry_time < retry_interval:
                return [None] * len(boxes)
            self._last_inference_retry_time = now

        try:
            request = self._InferenceRequest(
                type=self._RequestType.REID,
                camera_id=self.stream_name,
                frame=frame,
                boxes=boxes,
            )
            try:
                self.inference_request_queue.put(request, timeout=2.0)
            except queue_module.Full:
                now = time.time()
                if now - self._last_queue_full_log > 30:
                    self.logger.warning("Inference request queue full - skipping ReID (suppressing for 30s)")
                    self._last_queue_full_log = now
                self._centralized_inference_failures += 1
                return [None] * len(boxes)

            # Wait for correlated response (discards stale/mismatched responses)
            response = self._wait_for_response(request.request_id, timeout=5.0)

            # Only reset failure counter on genuine success (not error/stale responses)
            if not response.error:
                if self._centralized_inference_failures > 0:
                    self.logger.info(
                        f"Inference recovered after {self._centralized_inference_failures} consecutive failures"
                    )
                self._centralized_inference_failures = 0

            if response.error:
                self.logger.warning(f"ReID server error: {response.error}")

            return response.features if response.features else [None] * len(boxes)

        except queue_module.Empty:
            self._centralized_inference_failures += 1
            now = time.time()
            suppression_interval = self._get_inference_suppression_interval()
            if now - self._last_inference_error_log > suppression_interval:
                self.logger.warning(
                    f"Inference ReID timeout ({self._centralized_inference_failures} consecutive failures, suppressing for {suppression_interval}s)"
                )
                self._last_inference_error_log = now
            return [None] * len(boxes)
        except Exception as e:
            self._centralized_inference_failures += 1
            self.logger.error(f"Centralized ReID error: {e}")
            return [None] * len(boxes)

    def _check_and_handle_crossing(self, track_id: int, detection: Detection, frame_width: int, current_time: float, frame=None):
        """Check for crossing and handle API calls if crossing detected."""
        tracker = self.trackers[track_id]

        # Skip crossing check if track hasn't met minimum confidence threshold
        # This filters out shadows, reflections, and other low-confidence artifacts
        min_conf = self.doorway_config.get('min_peak_confidence', 0.40)
        if tracker.peak_confidence < min_conf:
            return  # Track hasn't proven to be a real person yet

        crossing_dir = tracker.check_crossing(
            frame_width,
            self.doorway_config['doorway_center'],
            self.doorway_config['doorway_width'],
            self.doorway_config['danger_zone_buffer'],  # METHOD 5: pass danger zone buffer
            self.doorway_config  # METHOD 5.4: pass velocity config for reflection prevention
        )

        # METHOD 5: Create checkpoint when tracker enters danger zone
        if tracker.needs_checkpoint and frame is not None:
            tracker.needs_checkpoint = False
            if self.doorway_config['use_reid'] and self.reid_extractor:
                x, y, w, h = detection.bbox
                embedding = self._extract_appearance_features(frame, x, y, x + w, y + h)
                if embedding is not None:
                    tracker.create_checkpoint(embedding, detection.center, detection.confidence)

        if crossing_dir and crossing_dir != CrossingDirection.NONE:
            # A validated crossing occurred!
            self.crossings[crossing_dir] += 1

            # Make API call with device credentials
            # Apply direction flip logic if enabled
            if self.direction_flipped:
                if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                    threading.Thread(target=self.api_client.subtract_from_count, args=(1,), daemon=True).start()
                elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                    threading.Thread(target=self.api_client.add_to_count, args=(1,), daemon=True).start()
            else:
                if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                    threading.Thread(target=self.api_client.add_to_count, args=(1,), daemon=True).start()
                elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                    threading.Thread(target=self.api_client.subtract_from_count, args=(1,), daemon=True).start()

            # Log the crossing
            crossing_info = {
                'timestamp': current_time,
                'track_id': track_id,
                'direction': crossing_dir.value,
                'confidence': self.trackers[track_id].peak_confidence,
                'total_crossings_by_person': self.trackers[track_id].total_crossings,
                'position': detection.center,
                'detection_count': self.trackers[track_id].detection_count
            }
            self.crossing_log.append(crossing_info)

            total = sum(self.crossings.values())
            self.logger.info(f"CROSSING #{total}: {crossing_dir.value} | "
                           f"Track #{track_id} (detections: {self.trackers[track_id].detection_count}) | "
                           f"L->R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
                           f"R->L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")

    def _match_by_position_and_reid(self, detection: Detection, frame_width: int = 640) -> Optional[int]:
        """
        Match detection to tracker using position distance and ReID similarity.
        Used for detections that ByteTrack couldn't match directly.
        """
        if not self.trackers:
            return None

        best_match_id = None
        best_score = float('inf')

        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) == 0:
                continue

            # Calculate distance to last known position
            last_pos = tracker.positions[-1]
            dist = math.sqrt(
                (detection.center[0] - last_pos[0])**2 +
                (detection.center[1] - last_pos[1])**2
            )

            # Consider velocity for prediction
            vx, vy = tracker.get_smooth_velocity()
            time_diff = detection.timestamp - tracker.last_update_time

            predicted_x = last_pos[0] + vx * time_diff
            predicted_y = last_pos[1] + vy * time_diff

            pred_dist = math.sqrt(
                (detection.center[0] - predicted_x)**2 +
                (detection.center[1] - predicted_y)**2
            )

            position_dist = min(dist, pred_dist)

            if position_dist > self.doorway_config['max_tracking_distance'] * frame_width:
                continue

            # Calculate ReID similarity score (if available)
            reid_score = 0.0
            if (self.doorway_config['use_reid'] and
                detection.appearance_features is not None and
                tracker.mean_appearance is not None):
                reid_score = tracker.get_appearance_similarity(detection.appearance_features)

            # Combined score: weight position and appearance
            if reid_score > self.doorway_config['reid_similarity_threshold']:
                final_score = position_dist * (1.0 - reid_score * 0.5)
            else:
                final_score = position_dist

            if final_score < best_score:
                best_score = final_score
                best_match_id = track_id

        return best_match_id

    def match_detection_to_tracker(self, detection: Detection, frame_width: int = 640) -> Optional[int]:
        """
        Match a detection to existing tracker using ByteTrack ID, ReID, and distance.
        Priority: ByteTrack ID > ReID similarity > Position distance
        """
        if not self.trackers:
            return None

        # Priority 1: If ByteTrack provides a track ID, try to match it first
        if detection.track_id is not None:
            # Check if we have a tracker with this ByteTrack ID
            if detection.track_id in self.trackers:
                return detection.track_id
            # ByteTrack assigned a new ID - this might be a new person or re-ID needed

        best_match_id = None
        best_score = float('inf')  # Lower is better for distance-based scoring

        # Priority 2 & 3: Combined ReID + distance matching
        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) == 0:
                continue

            # Calculate distance to last known position
            last_pos = tracker.positions[-1]
            dist = math.sqrt(
                (detection.center[0] - last_pos[0])**2 +
                (detection.center[1] - last_pos[1])**2
            )

            # Consider velocity for prediction
            vx, vy = tracker.get_smooth_velocity()
            time_diff = detection.timestamp - tracker.last_update_time

            # Predict where tracker should be
            predicted_x = last_pos[0] + vx * time_diff
            predicted_y = last_pos[1] + vy * time_diff

            # Distance to predicted position
            pred_dist = math.sqrt(
                (detection.center[0] - predicted_x)**2 +
                (detection.center[1] - predicted_y)**2
            )

            # Use minimum of actual and predicted distance
            position_dist = min(dist, pred_dist)

            # Skip if too far away
            if position_dist > self.doorway_config['max_tracking_distance'] * frame_width:
                continue

            # Calculate ReID similarity score (if available)
            reid_score = 0.0
            if (self.doorway_config['use_reid'] and
                detection.appearance_features is not None and
                tracker.mean_appearance is not None):
                reid_score = tracker.get_appearance_similarity(detection.appearance_features)

            # Combined score: weight position and appearance
            # Lower score = better match
            # If ReID similarity is high (>0.5), reduce the effective distance
            if reid_score > self.doorway_config['reid_similarity_threshold']:
                # High appearance similarity - give significant boost
                final_score = position_dist * (1.0 - reid_score * 0.5)
            else:
                final_score = position_dist

            if final_score < best_score:
                best_score = final_score
                best_match_id = track_id

        return best_match_id

    def _find_best_reid_match(self, detection: Detection, max_distance: float = 200) -> Optional[int]:
        """
        Find the best ReID match for a detection among lost/stale trackers.
        Used when ByteTrack loses a track (e.g., during occlusion in groups).
        """
        if detection.appearance_features is None:
            return None

        best_match_id = None
        best_similarity = self.doorway_config['reid_similarity_threshold']

        for track_id, tracker in self.trackers.items():
            # Only consider trackers that haven't been updated recently (lost tracks)
            if tracker.time_since_update() < 0.5:
                continue  # Still being tracked, skip

            if tracker.mean_appearance is None:
                continue

            # Check distance isn't too far (sanity check)
            if len(tracker.positions) > 0:
                last_pos = tracker.positions[-1]
                dist = math.sqrt(
                    (detection.center[0] - last_pos[0])**2 +
                    (detection.center[1] - last_pos[1])**2
                )
                if dist > max_distance:
                    continue

            # Calculate appearance similarity
            similarity = tracker.get_appearance_similarity(detection.appearance_features)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = track_id

        return best_match_id

    def update_trackers(self, frame, detections: List[Detection]):
        """
        Update trackers with new detections and check for crossings.
        Uses ByteTrack IDs when available, falls back to ReID and position matching.

        Performance optimization: ReID features are only extracted for unmatched detections,
        not for every detection on every frame.
        """
        height, width = frame.shape[:2]
        current_time = time.time()

        # PHASE 1: Fast matching using ByteTrack IDs only (no ReID yet)
        matched_trackers = set()
        unmatched_detections = []

        for detection in detections:
            # Quick check: does ByteTrack ID match an existing tracker?
            if detection.track_id is not None and detection.track_id in self.trackers:
                track_id = detection.track_id
                # Update existing tracker
                self.trackers[track_id].update(detection)
                matched_trackers.add(track_id)

                # Check for crossing with trajectory validation
                self._check_and_handle_crossing(track_id, detection, width, current_time, frame)
            else:
                # No ByteTrack match - will need ReID
                unmatched_detections.append(detection)

        # PHASE 2: Extract ReID features ONLY for unmatched detections (performance optimization)
        reid_available = (
            self.doorway_config['use_reid'] and
            (self.reid_extractor is not None or self.use_centralized_inference)
        )
        if unmatched_detections and reid_available:
            # Collect bounding boxes for batch extraction
            reid_boxes = []
            for det in unmatched_detections:
                x, y, w, h = det.bbox
                reid_boxes.append((x, y, x + w, y + h))

            # Extract features via centralized server or local GPU
            if self.use_centralized_inference:
                features_list = self._extract_appearance_batch_centralized(frame, reid_boxes)
            else:
                features_list = self.reid_extractor.extract_batch(frame, reid_boxes)

            # Assign features to detections
            for i, det in enumerate(unmatched_detections):
                det.appearance_features = features_list[i]

        # PHASE 3: Try to match unmatched detections using ReID + position
        still_unmatched = []
        for detection in unmatched_detections:
            # Try position + ReID matching
            track_id = self._match_by_position_and_reid(detection, width)

            if track_id is not None:
                self.trackers[track_id].update(detection)
                matched_trackers.add(track_id)
                self._check_and_handle_crossing(track_id, detection, width, current_time, frame)
            else:
                # Try ReID recovery for lost tracks
                if self.doorway_config['use_reid']:
                    reid_match = self._find_best_reid_match(detection)
                    if reid_match is not None:
                        self.trackers[reid_match].update(detection)
                        matched_trackers.add(reid_match)
                        self.logger.debug(f"ReID recovered track #{reid_match}")
                        self._check_and_handle_crossing(reid_match, detection, width, current_time, frame)
                        continue
                still_unmatched.append(detection)

        # Create new trackers for truly unmatched detections
        for detection in still_unmatched:
            # Use ByteTrack ID if available, otherwise generate our own
            if detection.track_id is not None:
                new_id = detection.track_id
            elif self.track_id_pool:
                new_id = self.track_id_pool.pop(0)
            else:
                new_id = self.next_track_id
                self.next_track_id += 1

            # Create new tracker (ensure we don't overwrite existing)
            if new_id not in self.trackers:
                self.trackers[new_id] = PersonTracker(new_id, detection)

                # Check active_checkpoints for a ReID match from an expired tracker
                if detection.appearance_features is not None and self.active_checkpoints:
                    best_cp_id = None
                    best_cp_sim = self.doorway_config.get('relink_similarity_threshold', 0.55)

                    for cp_id, cp_data in self.active_checkpoints.items():
                        sim = float(np.dot(cp_data['embedding'], detection.appearance_features))
                        sim = max(0.0, min(1.0, sim))
                        if sim > best_cp_sim:
                            best_cp_sim = sim
                            best_cp_id = cp_id

                    if best_cp_id is not None:
                        recovered_cp = self.active_checkpoints.pop(best_cp_id)
                        checkpoint = recovered_cp['checkpoint']
                        tracker = self.trackers[new_id]

                        # Determine which zone the person is currently in
                        det_x = detection.center[0]
                        current_zone = tracker._determine_zone(
                            det_x, width,
                            self.doorway_config['doorway_center'],
                            self.doorway_config.get('danger_zone_buffer', 0.10))

                        if current_zone in (DangerZone.LEFT_ZONE, DangerZone.RIGHT_ZONE):
                            exit_side = 'left' if current_zone == DangerZone.LEFT_ZONE else 'right'

                            if checkpoint.entry_side != exit_side:
                                # Person entered from one side, now on the other — count crossing directly
                                crossing_dir = (CrossingDirection.LEFT_TO_RIGHT
                                               if checkpoint.entry_side == 'left'
                                               else CrossingDirection.RIGHT_TO_LEFT)
                                self.crossings[crossing_dir] += 1

                                if self.direction_flipped:
                                    api_method = (self.api_client.subtract_from_count
                                                 if crossing_dir == CrossingDirection.LEFT_TO_RIGHT
                                                 else self.api_client.add_to_count)
                                else:
                                    api_method = (self.api_client.add_to_count
                                                 if crossing_dir == CrossingDirection.LEFT_TO_RIGHT
                                                 else self.api_client.subtract_from_count)
                                threading.Thread(target=api_method, args=(1,), daemon=True).start()

                                tracker.last_crossing_time = current_time
                                tracker.last_side = exit_side
                                tracker.total_crossings += 1

                                total = sum(self.crossings.values())
                                self.logger.info(
                                    f"CHECKPOINT CROSSING #{total}: {crossing_dir.value} | "
                                    f"Track #{new_id} recovered from #{best_cp_id} "
                                    f"(similarity={best_cp_sim:.2f}, entry_side={checkpoint.entry_side})")
                            else:
                                # Same side as entry — no crossing occurred
                                tracker.last_side = exit_side
                                self.logger.info(
                                    f"Checkpoint recovered (same side): track #{new_id} from #{best_cp_id} "
                                    f"(similarity={best_cp_sim:.2f}, side={exit_side})")
                        else:
                            # Still in danger zone — transfer checkpoint for zone-transition detection
                            tracker.checkpoint = checkpoint
                            tracker.search_active = True
                            tracker.checkpoint_is_recovered = True
                            self.logger.info(
                                f"Checkpoint transferred: track #{new_id} from #{best_cp_id} "
                                f"(similarity={best_cp_sim:.2f}, entry_side={checkpoint.entry_side})")

        # Check for crossings on unmatched trackers using interpolation
        # This catches crossings that may have occurred during dropped frames
        interp_crossing_this_frame = False
        for track_id, tracker in self.trackers.items():
            if track_id in matched_trackers:
                continue  # Already processed

            # Only allow one interpolated crossing per frame to prevent
            # multiple fragmented trackers from each counting the same person
            if interp_crossing_this_frame:
                continue

            time_gap = tracker.time_since_update()

            # Only interpolate if there's a significant time gap (dropped frames)
            # but not too stale (likely lost track)
            # Also require minimum peak confidence to avoid counting shadows/reflections
            # Require 5+ positions to filter out short-lived fragmented tracks
            min_conf = self.doorway_config.get('min_peak_confidence', 0.40)
            if 0.1 < time_gap < 1.0 and len(tracker.positions) >= 5 and tracker.peak_confidence >= min_conf:
                # Get interpolated positions through the gap
                interpolated = tracker.interpolate_positions(current_time, num_steps=5)

                # Save all mutable state that check_crossing() modifies
                saved_state = {
                    'current_zone': tracker.current_zone,
                    'previous_zone': tracker.previous_zone,
                    'current_side': tracker.current_side,
                    'last_side': tracker.last_side,
                    'in_danger_zone': tracker.in_danger_zone,
                    'danger_zone_frame_count': tracker.danger_zone_frame_count,
                    'approach_velocities': deque(tracker.approach_velocities, maxlen=30),
                    'needs_checkpoint': tracker.needs_checkpoint,
                    'side_history': deque(tracker.side_history, maxlen=10),
                    'consecutive_side_detections': tracker.consecutive_side_detections,
                    'checkpoint': tracker.checkpoint,
                    'search_active': tracker.search_active,
                    'total_crossings': tracker.total_crossings,
                    'crossing_events': list(tracker.crossing_events),
                    'last_crossing_time': tracker.last_crossing_time,
                }

                interp_crossing_fired = False
                for interp_pos in interpolated:
                    # Temporarily add interpolated position to check crossing
                    original_positions = list(tracker.positions)
                    tracker.positions.append(interp_pos)

                    crossing_dir = tracker.check_crossing(
                        width,
                        self.doorway_config['doorway_center'],
                        self.doorway_config['doorway_width'],
                        self.doorway_config['danger_zone_buffer'],  # METHOD 5: pass danger zone buffer
                        self.doorway_config  # METHOD 5.4: pass velocity config for reflection prevention
                    )

                    # Restore original positions
                    tracker.positions = deque(original_positions, maxlen=60)

                    if crossing_dir and crossing_dir != CrossingDirection.NONE:
                        # Interpolation detected a crossing during frame gap
                        interp_crossing_fired = True
                        interp_crossing_this_frame = True
                        self.crossings[crossing_dir] += 1

                        # Make API call
                        if self.direction_flipped:
                            if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                                threading.Thread(
                                    target=self.api_client.subtract_from_count,
                                    args=(1,),
                                    daemon=True
                                ).start()
                            elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                                threading.Thread(
                                    target=self.api_client.add_to_count,
                                    args=(1,),
                                    daemon=True
                                ).start()
                        else:
                            if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                                threading.Thread(
                                    target=self.api_client.add_to_count,
                                    args=(1,),
                                    daemon=True
                                ).start()
                            elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                                threading.Thread(
                                    target=self.api_client.subtract_from_count,
                                    args=(1,),
                                    daemon=True
                                ).start()

                        total = sum(self.crossings.values())
                        self.logger.info(f"INTERPOLATED CROSSING #{total}: {crossing_dir.value} | "
                                       f"Track #{track_id} (gap: {time_gap:.2f}s) | "
                                       f"L->R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
                                       f"R->L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")
                        break  # Only count one crossing per interpolation check

                # Restore zone/velocity state so real detections start fresh
                tracker.current_zone = saved_state['current_zone']
                tracker.previous_zone = saved_state['previous_zone']
                tracker.current_side = saved_state['current_side']
                tracker.in_danger_zone = saved_state['in_danger_zone']
                tracker.danger_zone_frame_count = saved_state['danger_zone_frame_count']
                tracker.approach_velocities = saved_state['approach_velocities']
                tracker.needs_checkpoint = saved_state['needs_checkpoint']
                tracker.side_history = saved_state['side_history']
                tracker.consecutive_side_detections = saved_state['consecutive_side_detections']
                tracker.checkpoint = saved_state['checkpoint']
                tracker.search_active = saved_state['search_active']

                if interp_crossing_fired:
                    # Keep crossing-related state so tracker "remembers" the crossing.
                    # This prevents the same tracker from re-firing every frame
                    # (the per-tracker cooldown now applies to future interpolation attempts).
                    pass
                else:
                    tracker.last_side = saved_state['last_side']
                    tracker.total_crossings = saved_state['total_crossings']
                    tracker.crossing_events = saved_state['crossing_events']
                    tracker.last_crossing_time = saved_state['last_crossing_time']

        # Remove stale trackers
        trackers_to_remove = []
        for track_id, tracker in self.trackers.items():
            if tracker.time_since_update() > self.doorway_config['tracker_timeout']:
                trackers_to_remove.append(track_id)

                # Save checkpoint from expiring tracker for cross-track recovery
                # Don't save recovered checkpoints back — prevents recycling loop
                # where fragmented tracks endlessly pass the same checkpoint around
                if (tracker.checkpoint and tracker.mean_appearance is not None
                        and not getattr(tracker, 'checkpoint_is_recovered', False)):
                    self.active_checkpoints[track_id] = {
                        'checkpoint': tracker.checkpoint,
                        'embedding': tracker.mean_appearance.copy(),
                        'saved_time': current_time,
                    }
                    self.logger.debug(f"Saved checkpoint from expiring track #{track_id} "
                                    f"(entry_side={tracker.checkpoint.entry_side})")

                # Return ID to pool for reuse (only for non-ByteTrack IDs)
                if track_id not in self.track_id_pool and track_id < 1000:
                    self.track_id_pool.append(track_id)

        for track_id in trackers_to_remove:
            del self.trackers[track_id]

        # Clean up expired active checkpoints
        checkpoint_timeout = self.doorway_config.get('checkpoint_timeout', 5.0)
        expired_checkpoints = [
            cp_id for cp_id, cp_data in self.active_checkpoints.items()
            if current_time - cp_data['saved_time'] > checkpoint_timeout
        ]
        for cp_id in expired_checkpoints:
            del self.active_checkpoints[cp_id]

    def draw_visualization(self, frame, detections: List[Detection]) -> np.ndarray:
        """Draw visualization overlay on frame"""
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]

        # Calculate doorway line position
        center_x = int(width * self.doorway_config['doorway_center'])
        line_half_width = int(width * self.doorway_config['doorway_width'] / 2)

        # Draw approach zones
        approach_width = int(width * self.doorway_config['approach_zone'])

        # Left approach zone
        cv2.rectangle(vis_frame,
                     (center_x - approach_width - line_half_width, 0),
                     (center_x - line_half_width, height),
                     (100, 100, 200), 1)

        # Right approach zone
        cv2.rectangle(vis_frame,
                     (center_x + line_half_width, 0),
                     (center_x + approach_width + line_half_width, height),
                     (100, 100, 200), 1)

        # Draw doorway center line (bright green)
        cv2.line(vis_frame,
                (center_x, 0),
                (center_x, height),
                (0, 255, 0), 2)

        # Draw thin lines on either side to show doorway width
        cv2.line(vis_frame,
                (center_x - line_half_width, 0),
                (center_x - line_half_width, height),
                (0, 180, 0), 1)
        cv2.line(vis_frame,
                (center_x + line_half_width, 0),
                (center_x + line_half_width, height),
                (0, 180, 0), 1)

        # Draw current detections
        for detection in detections:
            x, y, w, h = detection.bbox

            # Color based on whether ByteTrack assigned an ID
            box_color = (0, 255, 255) if detection.track_id is not None else (255, 255, 0)

            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), box_color, 2)

            # Draw center point
            cv2.circle(vis_frame, detection.center, 5, (0, 255, 255), -1)

            # Draw confidence and ByteTrack ID if available
            if detection.track_id is not None:
                conf_text = f"BT#{detection.track_id} {detection.confidence:.2f}"
            else:
                conf_text = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_frame, conf_text,
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Draw trackers
        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) < 2:
                continue

            # Determine color based on side
            if tracker.current_side == 'left':
                track_color = (255, 100, 100)  # Blue-ish
            elif tracker.current_side == 'right':
                track_color = (100, 100, 255)  # Red-ish
            else:
                track_color = (0, 255, 0)  # Green

            # Draw path
            positions = list(tracker.positions)
            for i in range(1, len(positions)):
                pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                pt2 = (int(positions[i][0]), int(positions[i][1]))
                thickness = max(1, int(3 * (i / len(positions))))
                cv2.line(vis_frame, pt1, pt2, track_color, thickness)

            # Draw current position
            if positions:
                current_pos = positions[-1]
                cv2.circle(vis_frame,
                          (int(current_pos[0]), int(current_pos[1])),
                          8, track_color, -1)

                # Draw track ID, detection count, and crossing count
                info_text = f"#{track_id}"
                # Show detection count (important for trajectory validation)
                info_text += f" d:{tracker.detection_count}"
                if tracker.total_crossings > 0:
                    info_text += f" x{tracker.total_crossings}"

                cv2.putText(vis_frame, info_text,
                           (int(current_pos[0]) + 10, int(current_pos[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)

                # Draw velocity arrow
                vx, vy = tracker.get_smooth_velocity()
                if abs(vx) > 5 or abs(vy) > 5:
                    arrow_end = (int(current_pos[0] + vx * 0.5),
                               int(current_pos[1] + vy * 0.5))
                    cv2.arrowedLine(vis_frame,
                                  (int(current_pos[0]), int(current_pos[1])),
                                  arrow_end,
                                  (0, 255, 0), 2, tipLength=0.3)

        # Draw statistics overlay
        stats_bg_height = 160
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (380, stats_bg_height), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)

        # Statistics text
        total_crossings = sum(self.crossings.values())
        lr_count = self.crossings[CrossingDirection.LEFT_TO_RIGHT]
        rl_count = self.crossings[CrossingDirection.RIGHT_TO_LEFT]
        net_flow = lr_count - rl_count

        # Mode indicator
        mode_text = "ByteTrack" if self.doorway_config['use_bytetrack'] else "Basic"
        if self.doorway_config['use_reid']:
            mode_text += "+ReID"

        cv2.putText(vis_frame, f"PEOPLE COUNTER [{mode_text}]",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Total Crossings: {total_crossings}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Left -> Right: {lr_count}",
                   (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(vis_frame, f"Right -> Left: {rl_count}",
                   (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        cv2.putText(vis_frame, f"Net Flow (L->R): {net_flow:+d}",
                   (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Active Tracks: {len(self.trackers)} | Detections: {len(detections)}",
                   (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return vis_frame

    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Optional[np.ndarray]:
        """Process a single frame - main processing logic"""
        # Note: Buffer draining disabled - OpenCV read() blocks on live RTSP streams
        # The base processor already sets CAP_PROP_BUFFERSIZE=1 for low latency
        # frame = self._drain_frame_buffer(frame)

        # Extract detections using YOLO
        detections = self.extract_detections(frame)

        # Publish live boxes to the perception overlay (best-effort; must never
        # break counting).
        if self.perception_hook is not None:
            try:
                self.perception_hook(frame, detections, frame_number)
            except Exception:
                pass

        # Update trackers and check for crossings
        self.update_trackers(frame, detections)

        # Create visualization
        vis_frame = self.draw_visualization(frame, detections)

        return vis_frame

