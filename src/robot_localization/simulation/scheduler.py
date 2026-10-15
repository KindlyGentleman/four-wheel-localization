"""
robot_localization.simulation.scheduler

Timestamp-driven asynchronous simulation event scheduler.
Ensures strict chronological ordering of sensor observations, prevents future-data leakage,
and supports sensor-specific sampling frequencies and optional latency modeling.
"""

import heapq
from dataclasses import dataclass, field
from typing import Any, List, Optional, Callable


@dataclass(order=True)
class SensorEvent:
    """A scheduled discrete sensor event in the chronological queue."""
    timestamp: float
    sensor_type: str = field(compare=False)    # 'encoder', 'imu', 'compass', 'gnss', 'lidar', 'ultrasonic'
    event_id: int = field(compare=True)        # Tie-breaker for identical timestamps
    data: Any = field(default=None, compare=False)


class EventScheduler:
    """
    Priority-queue event scheduler managing asynchronous multi-rate sensor sampling.
    """

    def __init__(self):
        self._queue: List[SensorEvent] = []
        self._counter: int = 0
        self.current_time: float = 0.0

    def schedule(self, timestamp: float, sensor_type: str, data: Any = None):
        """Insert a future event into the chronological queue."""
        event = SensorEvent(
            timestamp=timestamp,
            sensor_type=sensor_type,
            event_id=self._counter,
            data=data
        )
        self._counter += 1
        heapq.heappush(self._queue, event)

    def has_next(self) -> bool:
        """Check if any events remain in the queue."""
        return len(self._queue) > 0

    def peek_next_time(self) -> Optional[float]:
        """Return timestamp of the earliest pending event without popping."""
        if not self._queue:
            return None
        return self._queue[0].timestamp

    def pop_next(self) -> Optional[SensorEvent]:
        """Pop the chronologically earliest event."""
        if not self._queue:
            return None
        event = heapq.heappop(self._queue)
        assert event.timestamp >= self.current_time, (
            f"Causal violation: event timestamp {event.timestamp} < current_time {self.current_time}"
        )
        self.current_time = event.timestamp
        return event

    def clear(self):
        """Reset the scheduler queue."""
        self._queue.clear()
        self._counter = 0
        self.current_time = 0.0
