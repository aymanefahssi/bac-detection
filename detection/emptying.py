"""
Emptying event state machine.
Detects normal -> emptying -> normal transitions with noise filtering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import cfg


@dataclass
class EmptyingStateMachine:
    current_state: str = "unknown"
    normal_frames: List[Dict] = field(default_factory=list)
    completed_events: List[Dict] = field(default_factory=list)
    event_id: int = 0
    last_event_time: Optional[float] = None
    current_emptying_start: Optional[float] = None
    last_bins_timestamp: Optional[float] = None
    current_event_index: Optional[int] = None
    consecutive_normal_count: int = 0
    consecutive_emptying_count: int = 0
    confirmed_normal_state: bool = False

    def _has_active_event(self) -> bool:
        return (
            self.current_event_index is not None
            and self.current_event_index < len(self.completed_events)
            and self.current_emptying_start is not None
        )

    def check_forced_end(self, timestamp: float, num_bins: int,
                         current_bbox=None) -> tuple:
        if num_bins > 0:
            self.last_bins_timestamp = timestamp

        if self.current_state != "emptying" or not self._has_active_event():
            return False, "", False

        if self.current_emptying_start is not None:
            if timestamp - self.current_emptying_start >= cfg.MAX_EVENT_DURATION:
                return True, f"max duration {cfg.MAX_EVENT_DURATION}s", False

        if num_bins == 0 and self.last_bins_timestamp is not None:
            if timestamp - self.last_bins_timestamp >= cfg.NO_BINS_TIMEOUT:
                return True, f"no bins for {cfg.NO_BINS_TIMEOUT}s", False

        return False, "", False

    def force_end_event(self, timestamp: float, reason: str):
        if self._has_active_event():
            event = self.completed_events[self.current_event_index]
            event["end_time"] = timestamp
            duration = timestamp - self.current_emptying_start
            event["duration"] = duration
            event["end_reason"] = reason
            print(f"  [S2] Emptying FORCE-ENDED at t={timestamp:.1f}s ({reason})")

        self.current_state = "normal"
        self.confirmed_normal_state = False
        self.current_emptying_start = None
        self.current_event_index = None
        self.normal_frames = []
        self.consecutive_normal_count = 0

    def force_end_if_active(self, timestamp: float):
        if self.current_state == "emptying" and self._has_active_event():
            self.force_end_event(timestamp, "cycle_ended")

    def process_frame(self, state: str, timestamp: float, crop=None,
                      bbox=None, num_bins: int = -1, bin_bbox=None,
                      is_model_detection: bool = True):
        current_bbox = bin_bbox or bbox

        if num_bins >= 0:
            should_end, reason, should_start_new = self.check_forced_end(
                timestamp, num_bins, current_bbox)
            if should_end:
                self.force_end_event(timestamp, reason)
                return

        if state == "normal":
            self.consecutive_emptying_count = 0

            # "No detection" (is_model_detection=False) counts toward initial
            # normal-state confirmation AND resetting emptying counters, but
            # ending an active emptying event requires model-confirmed normal.
            if is_model_detection:
                self.consecutive_normal_count += 1
            else:
                # Implicit normal (no detection) — count for initial state
                # but not for ending an active event
                self.consecutive_normal_count += 1

            if self.current_state == "emptying":
                if (is_model_detection
                        and self.consecutive_normal_count >= cfg.CONSECUTIVE_NORMAL_FRAMES):
                    if self._has_active_event():
                        event = self.completed_events[self.current_event_index]
                        event["end_time"] = timestamp
                        event["duration"] = timestamp - self.current_emptying_start
                        event["end_reason"] = "natural"
                        print(f"  [S2] Emptying ended at t={timestamp:.1f}s [natural]")
                    self.current_state = "normal"
                    self.confirmed_normal_state = True
                    self.current_emptying_start = None
                    self.current_event_index = None
                    self.normal_frames = []

            elif self.current_state == "unknown":
                if self.consecutive_normal_count >= cfg.CONSECUTIVE_NORMAL_FRAMES_TO_START:
                    self.current_state = "normal"
                    self.confirmed_normal_state = True
            else:
                self.confirmed_normal_state = True

            if (self.confirmed_normal_state and crop is not None
                    and len(self.normal_frames) < cfg.EMPTYING_FRAMES_TO_CAPTURE):
                self.normal_frames.append({
                    "crop": crop, "timestamp": timestamp, "bbox": current_bbox,
                })

        elif state == "emptying":
            self.consecutive_normal_count = 0
            self.consecutive_emptying_count += 1

            if self.current_state == "normal" and self.confirmed_normal_state:
                if self.consecutive_emptying_count < cfg.CONSECUTIVE_EMPTYING_FRAMES:
                    return

                if self.last_event_time is not None:
                    if timestamp - self.last_event_time < cfg.EMPTYING_COOLDOWN:
                        self.current_state = "emptying"
                        self.confirmed_normal_state = False
                        return

                print(f"  [S2] Emptying START at t={timestamp:.1f}s (event #{self.event_id})")
                self.current_emptying_start = timestamp
                self.last_bins_timestamp = timestamp
                self.confirmed_normal_state = False

                event = {
                    "event_id": self.event_id,
                    "start_time": timestamp,
                    "end_time": None,
                    "duration": None,
                    "timestamp": timestamp,
                    "frames": self.normal_frames,
                    "bbox": (self.normal_frames[0]["bbox"]
                             if self.normal_frames else current_bbox),
                    "end_reason": None,
                }
                self.completed_events.append(event)
                self.current_event_index = len(self.completed_events) - 1
                self.event_id += 1
                self.last_event_time = timestamp
                self.normal_frames = []
                self.current_state = "emptying"
