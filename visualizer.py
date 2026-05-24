"""
Visualization and UI components for the lab monitor.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import logging

from config import (
    ActivityStatus,  # Updated
    LabConfig, 
    BoundingBox,
    get_config
)
from tracker import PersonState


logger = logging.getLogger(__name__)


class ColorPalette:
    """Manages color schemes with improved labels."""
    
    def __init__(self, config: Optional[LabConfig] = None):
        self.config = config or get_config()
    
    def get_color(self, status: ActivityStatus) -> Tuple[int, int, int]:
        """Get BGR color for a status."""
        return self.config.colors.get(status, (255, 255, 255))
    
    def get_seat_color(self, is_occupied: bool) -> Tuple[int, int, int]:
        """Get color for a seat."""
        if is_occupied:
            return (0, 255, 255)  # Yellow
        return (150, 150, 150)    # Gray
    
    def get_status_display_name(self, status: ActivityStatus) -> str:
        """Get user-friendly display name for status."""
        names = {
            ActivityStatus.SEAT_EMPTY: "Empty",
            ActivityStatus.SEATED: "Seated",
            ActivityStatus.FOCUSED: "Focused",
            ActivityStatus.DISTRACTED: "Distracted",
        }
        return names.get(status, str(status))


class FrameDrawer:
    """Handles drawing with improved labels."""
    
    def __init__(self, config: Optional[LabConfig] = None):
        self.config = config or get_config()
        self.palette = ColorPalette(config)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_small = 0.5
        self.font_scale_medium = 0.6
        self.thickness = 2
        self.line_thickness = 1
    
    def draw_seats(self, frame: np.ndarray, occupancy: Dict[str, str]) -> np.ndarray:
        """Draw seat rectangles with improved labels."""
        for seat_name, seat in self.config.seats.items():
            is_occupied = occupancy.get(seat_name, ActivityStatus.SEAT_EMPTY.value) != ActivityStatus.SEAT_EMPTY.value
            color = self.palette.get_seat_color(is_occupied)
            
            # Draw seat rectangle
            x1, y1, x2, y2 = map(int, seat.bounds.to_list())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw seat label
            status = occupancy.get(seat_name, 'Empty')
            label = f"{seat_name}: {status}"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, self.font, self.font_scale_small, self.line_thickness
            )
            
            # Text background
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color, -1
            )
            
            # Text
            cv2.putText(
                frame, label, (x1 + 5, y1 - 5),
                self.font, self.font_scale_small, (0, 0, 0),
                self.line_thickness, cv2.LINE_AA
            )
        
        return frame
    
    def draw_person(
        self, 
        frame: np.ndarray, 
        person_id: int,
        box: BoundingBox,
        state: PersonState
    ) -> np.ndarray:
        """Draw a person with improved status display."""
        color = self.palette.get_color(state.status)
        x1, y1, x2, y2 = map(int, box.to_list())
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        
        # Get display name for status
        status_display = self.palette.get_status_display_name(state.status)
        
        # Prepare info
        info_lines = [
            f"ID: {person_id}",
            f"Seat: {state.seat_name or '--'}",
            f"Status: {status_display}",
            f"Time: {state.current_seat_duration:.0f}s"
        ]
        
        # Draw info with background
        for i, line in enumerate(info_lines):
            y = y1 - 25 - (i * 20)
            
            (text_width, text_height), _ = cv2.getTextSize(
                line, self.font, self.font_scale_small, self.line_thickness
            )
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay, 
                (x1, y - text_height - 3),
                (x1 + text_width + 10, y + 3),
                color, -1
            )
            
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Text
            cv2.putText(
                frame, line, (x1 + 5, y),
                self.font, self.font_scale_small, (255, 255, 255),
                self.line_thickness, cv2.LINE_AA
            )
        
        return frame
    
    def draw_phone(self, frame: np.ndarray, box: BoundingBox) -> np.ndarray:
        """Draw a phone detection."""
        x1, y1, x2, y2 = map(int, box.to_list())
        
        # Dashed rectangle
        dash_length = 10
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), (0, 0, 255), 2)
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), (0, 0, 255), 2)
        
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), (0, 0, 255), 2)
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), (0, 0, 255), 2)
        
        # Label
        cv2.putText(
            frame, " Phone", (x1, y1 - 10),
            self.font, self.font_scale_small, (0, 0, 255),
            self.thickness, cv2.LINE_AA
        )
        
        return frame
    
    def draw_stats(
        self, 
        frame: np.ndarray, 
        stats: Dict[str, Any],
        debug_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Draw system statistics with improved layout."""
        h, w = frame.shape[:2]
        
        # Stats panel
        panel_width = 300
        panel_height = 160
        panel_margin = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (panel_margin, panel_margin),
            (panel_margin + panel_width, panel_margin + panel_height),
            (40, 40, 40), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(
            frame,
            (panel_margin, panel_margin),
            (panel_margin + panel_width, panel_margin + panel_height),
            (100, 100, 100), 1
        )
        
        # Title
        cv2.putText(
            frame, " Lab Monitor", (panel_margin + 10, panel_margin + 30),
            self.font, 0.7, (255, 255, 255),
            self.thickness, cv2.LINE_AA
        )
        
        # Statistics
        y_start = panel_margin + 60
        line_height = 25
        
        stat_lines = [
            f" Persons: {stats.get('current_persons', 0)}",
            f" Seats: {stats.get('occupied_seats', 0)}/{stats.get('total_seats', 0)}",
            f" Focused: {stats.get('focused_persons', 0)}",
            f" Distracted: {stats.get('distracted_persons', 0)}",
        ]
        
        for i, line in enumerate(stat_lines):
            y = y_start + (i * line_height)
            cv2.putText(
                frame, line, (panel_margin + 10, y),
                self.font, 0.6, (255, 255, 255),
                self.line_thickness, cv2.LINE_AA
            )
        
        # Debug info (if enabled)
        if debug_info:
            y_debug = y_start + (len(stat_lines) * line_height) + 10
            debug_lines = [
                f"FPS: {debug_info.get('fps', 0):.1f}",
                f"Detection: {debug_info.get('inference_time_ms', 0):.1f}ms",
            ]
            
            for i, line in enumerate(debug_lines):
                y = y_debug + (i * 20)
                cv2.putText(
                    frame, line, (panel_margin + 10, y),
                    self.font, 0.5, (200, 200, 200),
                    self.line_thickness - 1, cv2.LINE_AA
                )
        
        return frame
    
    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """Draw status legend."""
        h, w = frame.shape[:2]
        
        legend_width = 300
        legend_height = 150
        legend_x = w - legend_width - 10
        legend_y = 50
        
        # Legend background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (40, 40, 40), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(
            frame,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (100, 100, 100), 1
        )
        
        # Title
        cv2.putText(
            frame, "Status Legend", (legend_x + 10, legend_y + 25),
            self.font, 0.6, (255, 255, 255),
            self.line_thickness, cv2.LINE_AA
        )
        
        # Status items
        statuses = [
            ActivityStatus.SEAT_EMPTY,
            ActivityStatus.SEATED,
            ActivityStatus.FOCUSED,
            ActivityStatus.DISTRACTED,
        ]
        
        y_start = legend_y + 50
        for i, status in enumerate(statuses):
            y = y_start + (i * 25)
            
            # Color box
            color = self.palette.get_color(status)
            cv2.rectangle(
                frame,
                (legend_x + 10, y - 12),
                (legend_x + 25, y + 3),
                color, -1
            )
            
            # Border
            cv2.rectangle(
                frame,
                (legend_x + 10, y - 12),
                (legend_x + 25, y + 3),
                (255, 255, 255), 1
            )
            
            # Label
            display_name = self.palette.get_status_display_name(status)
            cv2.putText(
                frame, display_name, (legend_x + 35, y),
                self.font, 0.5, (255, 255, 255),
                self.line_thickness - 1, cv2.LINE_AA
            )
        
        return frame
    
    def draw_calibration_overlay(
        self, 
        frame: np.ndarray, 
        progress: float,
        seat_count: int
    ) -> np.ndarray:
        """Draw calibration overlay."""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        title = " Seat Calibration"
        title_size = cv2.getTextSize(title, self.font, 1.0, self.thickness + 1)[0]
        cv2.putText(
            frame, title,
            (w // 2 - title_size[0] // 2, h // 2 - 50),
            self.font, 1.0, (0, 255, 255),
            self.thickness + 1, cv2.LINE_AA
        )
        
        # Progress
        progress_text = f"{int(progress * 100)}%"
        progress_size = cv2.getTextSize(progress_text, self.font, 1.5, self.thickness + 2)[0]
        cv2.putText(
            frame, progress_text,
            (w // 2 - progress_size[0] // 2, h // 2 + 10),
            self.font, 1.5, (0, 255, 255),
            self.thickness + 2, cv2.LINE_AA
        )
        
        # Progress bar
        bar_width = 400
        bar_height = 25
        bar_x = w // 2 - bar_width // 2
        bar_y = h // 2 + 50
        
        # Background
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100), -1
        )
        
        # Fill
        fill_width = int(bar_width * progress)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            (0, 255, 255), -1
        )
        
        # Border
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (200, 200, 200), 2
        )
        
        # Info
        info_lines = [
            f"Detected {seat_count} seat(s)",
            "Sit in each seat you want to calibrate",
            "Calibration completes automatically"
        ]
        
        for i, line in enumerate(info_lines):
            y = bar_y + bar_height + 40 + (i * 30)
            text_size = cv2.getTextSize(line, self.font, 0.6, self.line_thickness)[0]
            cv2.putText(
                frame, line,
                (w // 2 - text_size[0] // 2, y),
                self.font, 0.6, (255, 255, 255),
                self.line_thickness, cv2.LINE_AA
            )
        
        return frame
    
    def draw_no_seats_warning(self, frame: np.ndarray) -> np.ndarray:
        """Draw warning when no seats are defined."""
        h, w = frame.shape[:2]
        
        # Warning bar
        bar_height = 50
        bar_y = h - bar_height
        
        # Background
        cv2.rectangle(
            frame, (0, bar_y), (w, h),
            (0, 0, 100), -1
        )
        
        # Blinking border
        blink = int(time.time() * 2) % 2 == 0
        border_color = (0, 0, 255) if blink else (0, 100, 255)
        cv2.rectangle(
            frame, (0, bar_y), (w, h),
            border_color, 3
        )
        
        # Message
        message = " No seats defined - Press 'C' to calibrate "
        text_size = cv2.getTextSize(message, self.font, 0.7, self.thickness)[0]
        cv2.putText(
            frame, message,
            (w // 2 - text_size[0] // 2, h - 15),
            self.font, 0.7, (255, 255, 255),
            self.thickness, cv2.LINE_AA
        )
        
        return frame


    
    def draw_leaderboard(self, frame: np.ndarray, leaderboard: List[Dict[str, Any]]) -> np.ndarray:
        """Draw the gamified leaderboard."""
        if not leaderboard:
            return frame
            
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_width = 300
        # Dynamic height based on number of entries (max 5)
        num_entries = min(len(leaderboard), 5)
        panel_height = 80 + (num_entries * 30)
        
        # Position: Top Right (below timestamp, similar to stats but on right side or below legend?)
        # Let's put it on the top right, shifting legend down if needed, or put it below stats on left?
        # Stats is at (10, 10). Width 300.
        # Let's put leaderboard below stats.
        stats_height = 160
        panel_x = 10
        panel_y = 10 + stats_height + 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (40, 40, 40), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (255, 215, 0), 2  # Gold border
        )
        
        # Title with Trophy Icon (text)
        cv2.putText(
            frame, "Focus Leaderboard", (panel_x + 10, panel_y + 30),
            self.font, 0.7, (255, 215, 0),  # Gold color
            self.thickness, cv2.LINE_AA
        )
        
        # Column Headers
        header_y = panel_y + 60
        cv2.putText(frame, "Rank", (panel_x + 10, header_y), self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Seat", (panel_x + 70, header_y), self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "Score", (panel_x + 220, header_y), self.font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Divider
        cv2.line(frame, (panel_x + 10, header_y + 5), (panel_x + panel_width - 10, header_y + 5), (100, 100, 100), 1)
        
        # Entries
        y_start = header_y + 25
        for i, entry in enumerate(leaderboard[:5]):
            y = y_start + (i * 30)
            rank = i + 1
            seat = entry['seat']
            score = entry['score']
            
            # Highlight top 3
            if rank == 1:
                color = (0, 215, 255) # Gold (BGR)
            elif rank == 2:
                color = (192, 192, 192) # Silver
            elif rank == 3:
                color = (0, 140, 205) # Bronze (approx)
            else:
                color = (255, 255, 255)
            
            cv2.putText(frame, f"#{rank}", (panel_x + 15, y), self.font, 0.6, color, 1, cv2.LINE_AA)
            cv2.putText(frame, seat, (panel_x + 70, y), self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(score), (panel_x + 220, y), self.font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
        return frame


class LabUI:
    """Main UI controller."""
    
    def __init__(self, config: Optional[LabConfig] = None):
        self.config = config or get_config()
        self.drawer = FrameDrawer(config)
        self.window_name = 'Lab Monitor'
        
        # Initialize window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.config.frame_width, self.config.frame_height)
        
        # UI state
        self.show_debug = False
        self.show_legend = True
        self.show_leaderboard = True
        
        logger.info("UI initialized")
    
    def draw(
        self,
        frame: np.ndarray,
        detections: List[Any],
        person_states: Dict[int, PersonState],
        stats: Dict[str, Any],
        debug_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Draw the complete UI."""
        # Get seat occupancy
        occupancy = {}
        for person_id, state in person_states.items():
            if state.seat_name and state.seconds_since_seen < 1.0:
                occupancy[state.seat_name] = state.status.value
        
        # Draw seats
        if self.config.seats:
            frame = self.drawer.draw_seats(frame, occupancy)
        else:
            frame = self.drawer.draw_no_seats_warning(frame)
        
        # Draw detections
        for det in detections:
            if det.is_person and det.track_id is not None:
                state = person_states.get(det.track_id)
                if state:
                    frame = self.drawer.draw_person(frame, det.track_id, det.box, state)
            elif det.is_phone:
                frame = self.drawer.draw_phone(frame, det.box)
        
        # Draw statistics
        frame = self.drawer.draw_stats(frame, stats, debug_info if self.show_debug else None)
        
        # Draw leaderboard
        if self.show_leaderboard:
            leaderboard = stats.get('leaderboard', [])
            frame = self.drawer.draw_leaderboard(frame, leaderboard)
        
        # Draw legend
        if self.show_legend:
            frame = self.drawer.draw_legend(frame)
        
        # Draw timestamp
        frame = self._draw_timestamp(frame)
        
        # Draw controls hint
        frame = self._draw_controls_hint(frame)
        
        return frame
    
    def _draw_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """Draw current timestamp."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        h, w = frame.shape[:2]
        
        # Background
        text_size = cv2.getTextSize(timestamp, self.drawer.font, 0.6, 1)[0]
        cv2.rectangle(
            frame,
            (w - text_size[0] - 15, 10),
            (w - 5, 10 + text_size[1] + 10),
            (40, 40, 40), -1
        )
        
        # Text
        cv2.putText(
            frame, timestamp,
            (w - text_size[0] - 10, 25),
            self.drawer.font, 0.6, (200, 200, 200),
            1, cv2.LINE_AA
        )
        
        return frame
    
    def _draw_controls_hint(self, frame: np.ndarray) -> np.ndarray:
        """Draw controls hint."""
        h, w = frame.shape[:2]
        
        hint = "C=Calibrate  F=Full  S=Snap  L=Leaderboard  R=Reset  Q=Quit"
        text_size = cv2.getTextSize(hint, self.drawer.font, 0.5, 1)[0]
        
        cv2.putText(
            frame, hint,
            (10, h - 60),
            self.drawer.font, 0.5, (200, 200, 200),
            1, cv2.LINE_AA
        )
        
        return frame
    
    def draw_calibration(
        self,
        frame: np.ndarray,
        progress: float,
        seat_count: int
    ) -> np.ndarray:
        """Draw calibration UI."""
        return self.drawer.draw_calibration_overlay(frame, progress, seat_count)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        current = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN)
        
        if current == cv2.WINDOW_NORMAL:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            logger.info("Fullscreen enabled")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.config.frame_width, self.config.frame_height)
            logger.info("Fullscreen disabled")
    
    def toggle_debug(self):
        """Toggle debug information."""
        self.show_debug = not self.show_debug
        status = "ON" if self.show_debug else "OFF"
        logger.info(f"Debug info: {status}")
        
    def toggle_leaderboard(self):
        """Toggle leaderboard visibility."""
        self.show_leaderboard = not self.show_leaderboard
        status = "ON" if self.show_leaderboard else "OFF"
        logger.info(f"Leaderboard: {status}")
    
    def get_window_name(self) -> str:
        """Get window name."""
        return self.window_name