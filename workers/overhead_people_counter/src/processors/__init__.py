"""CrowdCount overhead people-counter processors (worker subset).

Only the modules the single-session counting worker needs. The legacy
example/display processors and the shared InferenceServer stayed behind in
crowdcount-counter-server — this worker runs inference in-process.
"""

from .base_processor import BaseStreamProcessor
from .people_counter_processor import PeopleCounterProcessor
from .frame_buffer import FrameJitterBuffer

__all__ = [
    "BaseStreamProcessor",
    "PeopleCounterProcessor",
    "FrameJitterBuffer",
]
