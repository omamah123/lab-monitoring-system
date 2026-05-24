"""
State tracking and management for lab monitoring.
Handles seat assignment, phone detection, and focus tracking.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from config import (
    BoundingBox, 
    ActivityStatus,
    SeatConfig, 
    LabConfig,
    DetectionClass,
    get_config
)


logger = logging.getLogger(__name__)


@dataclass
class PersonState:
    """State of an individual person being tracked."""
    person_id: int
    seat_name: Optional[str] = None
    status: ActivityStatus = ActivityStatus.SEATED
    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_phone_time: Optional[float] = None
    consecutive_frames_with_phone: int = 0
    consecutive_frames_focused: int = 0
    
    # Track when distraction ended to enforce "earning" focus back
    last_distraction_end_time: float = 0.0
    
    # Gamification
    focus_score: float = 0.0
    
    @property
    def current_seat_duration(self) -> float:
        """How long the person has been in their current seat."""
        return time.time() - self.start_time
    
    @property
    def seconds_since_seen(self) -> float:
        """How long since this person was last detected."""
        return time.time() - self.last_seen
    
    def update_seen(self) -> None:
        """Update the last seen timestamp."""
        self.last_seen = time.time()
    
    def update_status(self, new_status: ActivityStatus) -> None:
        """Update status and reset timers if needed."""
        if new_status != self.status:
            logger.debug(f"Person {self.person_id} status changed: {self.status} -> {new_status}")
            
            # If leaving DISTRACTED, record the time
            if self.status == ActivityStatus.DISTRACTED:
                self.last_distraction_end_time = time.time()
                
            self.status = new_status
    
    def record_phone_detection(self) -> None:
        """Record that a phone was detected with this person."""
        current_time = time.time()
        # If it's been a while since last phone, reset counter
        if self.last_phone_time is None or (current_time - self.last_phone_time) > 1.0:
            self.consecutive_frames_with_phone = 1
        else:
            self.consecutive_frames_with_phone += 1
        
        self.last_phone_time = current_time
        self.consecutive_frames_focused = 0  # Reset focus counter
    
    def record_focused_frame(self) -> None:
        """Record a frame where person was focused (no phone)."""
        self.consecutive_frames_with_phone = 0
        self.consecutive_frames_focused += 1


class LabStateManager:
    """
    Manages the state of all persons in the lab.
    Thread-safe for concurrent updates.
    """
    
    def __init__(self, config: Optional[LabConfig] = None):
        self.config = config or get_config()
        self.person_states: Dict[int, PersonState] = {}
        self.seats: Dict[str, SeatConfig] = self.config.seats.copy()
        self.lock = threading.RLock()
        
        # Statistics
        self.total_persons_seen = 0
        self.frame_count = 0
        self.focus_count = 0
        self.distraction_count = 0
        
        # Time tracking for scoring
        self.last_update_time = time.time()
    
    def update_seats(self, new_seats: Dict[str, SeatConfig]) -> None:
        """Update seat configuration."""
        with self.lock:
            self.seats = new_seats.copy()
            logger.info(f"Updated {len(new_seats)} seats")
    
    def assign_seat(self, person_box: BoundingBox) -> Optional[str]:
        """Assign a seat to a person based on their position."""
        if not self.seats:
            return None
        
        for seat_name, seat in self.seats.items():
            if seat.contains_person(person_box):
                return seat_name
        
        return None
    
    def find_nearby_phones(self, person_box: BoundingBox, phones: List[BoundingBox]) -> bool:
        """
        Check if any phone is near the person.
        Uses expanded bounding box for proximity check.
        """
        if not phones:
            return False
        
        # Expanded person box for proximity check
        expanded_person = person_box.expanded(self.config.phone_proximity_threshold)
        
        for phone_box in phones:
            phone_center = phone_box.center
            if expanded_person.contains_point(*phone_center):
                return True
        
        return False
    
    def update_state(self, detections: List[Any]) -> Dict[int, PersonState]:
        """
        Update the state based on new detections.
        """
        with self.lock:
            self.frame_count += 1
            current_time = time.time()
            time_elapsed = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Separate detections
            persons: List[Tuple[int, BoundingBox]] = []
            phones: List[BoundingBox] = []
            
            for det in detections:
                if det.is_person and det.track_id is not None:
                    persons.append((det.track_id, det.box))
                elif det.is_phone:
                    phones.append(det.box)
            
            # Track active person IDs
            active_person_ids = set()
            
            # Update existing persons and create new ones
            for person_id, person_box in persons:
                active_person_ids.add(person_id)
                
                if person_id not in self.person_states:
                    # New person
                    self.total_persons_seen += 1
                    assigned_seat = self.assign_seat(person_box)
                    
                    self.person_states[person_id] = PersonState(
                        person_id=person_id,
                        seat_name=assigned_seat,
                        status=ActivityStatus.SEATED if assigned_seat else ActivityStatus.SEAT_EMPTY,
                        start_time=current_time,
                        last_seen=current_time
                    )
                    logger.info(f"New person detected: ID={person_id}, Seat={assigned_seat}")
                else:
                    # Update existing person
                    state = self.person_states[person_id]
                    state.update_seen()
                    
                    # Check for seat change
                    new_seat = self.assign_seat(person_box)
                    if new_seat != state.seat_name:
                        state.seat_name = new_seat
                        state.start_time = current_time
                        state.consecutive_frames_focused = 0
                        state.status = ActivityStatus.SEATED if new_seat else ActivityStatus.SEAT_EMPTY
                        logger.debug(f"Person {person_id} seat: {state.seat_name} -> {new_seat}")
                
                # Phone detection and status update
                state = self.person_states[person_id]
                has_phone = self.find_nearby_phones(person_box, phones)
                
                if has_phone:
                    state.record_phone_detection()
                    # Mark as distracted if phone detected in consecutive frames
                    if state.consecutive_frames_with_phone >= 3:
                        if state.status != ActivityStatus.DISTRACTED:
                            self.distraction_count += 1
                        state.update_status(ActivityStatus.DISTRACTED)
                else:
                    state.record_focused_frame()
                    
                    # LOGIC: Reset to Seated after Distracted
                    # If person was distracted recently, they must wait before being focused again
                    time_since_distraction = current_time - state.last_distraction_end_time
                    
                    if state.seat_name is None:
                        state.update_status(ActivityStatus.SEAT_EMPTY)
                    elif state.status == ActivityStatus.DISTRACTED:
                         # Leaving distracted state -> Seated
                        state.update_status(ActivityStatus.SEATED)
                    elif time_since_distraction < self.config.focus_time_threshold:
                        # If recently distracted, stay Seated (re-earning focus)
                        state.update_status(ActivityStatus.SEATED)
                    elif state.current_seat_duration > self.config.focus_time_threshold:
                        # User has been seated long enough AND no recent distraction
                        if state.status != ActivityStatus.FOCUSED:
                            self.focus_count += 1
                        state.update_status(ActivityStatus.FOCUSED)
                    else:
                        state.update_status(ActivityStatus.SEATED)
                
                # Update Focus Score
                if state.status == ActivityStatus.FOCUSED:
                    state.focus_score += self.config.focus_points_per_second * time_elapsed
            
            # Clean up old persons
            self._cleanup_old_persons(current_time)
            
            # Update current person count
            self.current_person_count = len(active_person_ids)
            
            return self.person_states.copy()
    
    def _cleanup_old_persons(self, current_time: float) -> None:
        """Remove persons that haven't been seen recently."""
        to_remove = []
        timeout = self.config.person_timeout_seconds
        
        for person_id, state in self.person_states.items():
            if state.seconds_since_seen > timeout:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_states[person_id]
            logger.debug(f"Removed inactive person: ID={person_id}")
    
    def get_seat_occupancy(self) -> Dict[str, str]:
        """Get current seat occupancy status."""
        with self.lock:
            occupancy = {seat_name: ActivityStatus.SEAT_EMPTY.value 
                        for seat_name in self.seats}
            
            for state in self.person_states.values():
                if (state.seat_name and 
                    state.seat_name in occupancy and
                    state.seconds_since_seen < 1.0):
                    occupancy[state.seat_name] = state.status.value
            
            return occupancy
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Get the current leaderboard.
        Returns list of dicts with 'id', 'seat', 'score', sorted by score desc.
        Only includes active persons with score > 0.
        """
        with self.lock:
            leaderboard = []
            for state in self.person_states.values():
                if state.focus_score > 0 and state.seconds_since_seen < 5.0:
                    leaderboard.append({
                        "id": state.person_id,
                        "seat": state.seat_name or "Unknown",
                        "score": int(state.focus_score)
                    })
            
            # Sort by score descending
            leaderboard.sort(key=lambda x: x["score"], reverse=True)
            return leaderboard

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        with self.lock:
            occupied_seats = 0
            focused_count = 0
            distracted_count = 0
            
            for state in self.person_states.values():
                if state.seat_name and state.seconds_since_seen < 1.0:
                    occupied_seats += 1
                    if state.status == ActivityStatus.FOCUSED:
                        focused_count += 1
                    elif state.status == ActivityStatus.DISTRACTED:
                        distracted_count += 1
            
            return {
                "current_persons": self.current_person_count,
                "total_persons_seen": self.total_persons_seen,
                "total_seats": len(self.seats),
                "occupied_seats": occupied_seats,
                "focused_persons": focused_count,
                "distracted_persons": distracted_count,
                "total_focused": self.focus_count,
                "total_distracted": self.distraction_count,
                "frame_count": self.frame_count,
                "leaderboard": self.get_leaderboard(),
            }


class SeatCalibrator:
    """Handles automatic seat calibration without file saving."""
    
    def __init__(self, config: Optional[LabConfig] = None):
        self.config = config or get_config()
        self.samples: Dict[int, List[BoundingBox]] = defaultdict(list)
        self.calibration_start_time: Optional[float] = None
        self.is_calibrating = False
    
    def start_calibration(self) -> None:
        """Start a new calibration session."""
        self.samples.clear()
        self.calibration_start_time = time.time()
        self.is_calibrating = True
        logger.info("Calibration started - Please sit in the seats")
    
    def add_sample(self, detections: List[Any]) -> None:
        """Add detection samples for calibration."""
        if not self.is_calibrating:
            return
        
        for det in detections:
            if det.is_person and det.track_id is not None:
                self.samples[det.track_id].append(det.box)
    
    def calculate_seats(self) -> Dict[str, SeatConfig]:
        """
        Calculate seat positions from collected samples.
        Returns seat configurations in memory only.
        """
        seats = {}
        
        for seat_idx, (person_id, boxes) in enumerate(self.samples.items(), 1):
            if not boxes or len(boxes) < 5:  # Require minimum samples
                continue
            
            # Calculate robust bounding box (median instead of mean)
            x1_values = [b.x1 for b in boxes]
            y1_values = [b.y1 for b in boxes]
            x2_values = [b.x2 for b in boxes]
            y2_values = [b.y2 for b in boxes]
            
            def median(values):
                sorted_vals = sorted(values)
                mid = len(sorted_vals) // 2
                if len(sorted_vals) % 2 == 0:
                    return (sorted_vals[mid-1] + sorted_vals[mid]) / 2
                return sorted_vals[mid]
            
            # Use median for robustness against outliers
            median_x1 = median(x1_values)
            median_y1 = median(y1_values)
            median_x2 = median(x2_values)
            median_y2 = median(y2_values)
            
            # Add padding for comfortable seat area
            padding = 30
            bounds = BoundingBox(
                x1=median_x1 - padding,
                y1=median_y1 - padding,
                x2=median_x2 + padding,
                y2=median_y2 + padding
            )
            
            seat_name = f"Seat {seat_idx}"
            seats[seat_name] = SeatConfig(name=seat_name, bounds=bounds)
        
        self.is_calibrating = False
        
        if seats:
            logger.info(f"Calibration complete: {len(seats)} seats defined")
        else:
            logger.warning("Calibration failed: No valid seats detected")
        
        return seats
    
    def get_progress(self) -> float:
        """Get calibration progress (0.0 to 1.0)."""
        if not self.calibration_start_time or not self.is_calibrating:
            return 0.0
        
        elapsed = time.time() - self.calibration_start_time
        return min(elapsed / self.config.calibration_duration, 1.0)
    
    def get_current_samples(self) -> int:
        """Get number of unique persons sampled."""
        return len(self.samples)