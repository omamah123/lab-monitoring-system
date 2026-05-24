"""
Configuration management for the Lab Monitor System.
Uses Pydantic for validation and type safety.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DetectionClass(Enum):
    """YOLO detection class IDs."""
    PERSON = 0
    CELL_PHONE = 67


class ActivityStatus(str, Enum):
    """Improved names for person activity states."""
    SEAT_EMPTY = "Seat Empty"
    SEATED = "Seated"
    FOCUSED = "Focused"
    DISTRACTED = "Distracted"


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box representation."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def expanded(self, pixels: int) -> "BoundingBox":
        """Return a new bounding box expanded by given pixels."""
        return BoundingBox(
            x1=self.x1 - pixels,
            y1=self.y1 - pixels,
            x2=self.x2 + pixels,
            y2=self.y2 + pixels
        )
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class SeatConfig(BaseModel):
    """Configuration for a single seat."""
    name: str
    bounds: BoundingBox
    is_active: bool = True
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def contains_person(self, person_box: BoundingBox) -> bool:
        """Check if a person's bottom center is within this seat."""
        person_bottom_center = person_box.bottom_center
        return self.bounds.contains_point(*person_bottom_center)


class LabConfig(BaseModel):
    """Main configuration for the lab monitoring system."""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    
    # Detection settings
    model_path: str = "yolov8m.pt"
    confidence_threshold: float = 0.55
    iou_threshold: float = 0.45
    min_person_area: int = 8000  # pixels
    
    # Tracking settings
    focus_time_threshold: float = 10.0  # seconds
    phone_proximity_threshold: int = 50  # pixels
    person_timeout_seconds: float = 1.0  # cleanup old persons
    calibration_duration: float = 5.0  # seconds
    focus_points_per_second: float = 1.0  # points earned per second of focus
    
    # Seat configuration (stored in memory only)
    seats: Dict[str, SeatConfig] = Field(default_factory=dict)
    
    # Visualization settings with improved color scheme
    colors: Dict[ActivityStatus, Tuple[int, int, int]] = Field(
        default_factory=lambda: {
            ActivityStatus.SEAT_EMPTY: (150, 150, 150),    # Gray for empty seats
            ActivityStatus.SEATED: (255, 165, 0),          # Orange for seated
            ActivityStatus.FOCUSED: (0, 255, 0),           # Green for focused
            ActivityStatus.DISTRACTED: (0, 0, 255),        # Red for distracted
        }
    )
    
    def add_seat(self, name: str, bounds: BoundingBox) -> None:
        """Add a new seat configuration."""
        self.seats[name] = SeatConfig(name=name, bounds=bounds)
    
    def remove_seat(self, name: str) -> bool:
        """Remove a seat configuration."""
        if name in self.seats:
            del self.seats[name]
            return True
        return False
    
    def clear_seats(self) -> None:
        """Clear all seat configurations."""
        self.seats.clear()


# Global configuration instance
_config_instance: Optional[LabConfig] = None


def get_config() -> LabConfig:
    """Get or create the global configuration."""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = LabConfig()
    
    return _config_instance


def update_config(new_config: LabConfig):
    """Update the global configuration."""
    global _config_instance
    _config_instance = new_config