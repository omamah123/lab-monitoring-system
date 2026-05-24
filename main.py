"""
Main application for the Lab Monitoring System.
Handles camera capture, module orchestration, and user interaction.
"""

import cv2
import time
import signal
import sys
import traceback
from typing import Optional, Tuple
import logging
from datetime import datetime

# Import refactored modules
from config import get_config, LabConfig
from detector import ObjectDetector
from tracker import LabStateManager, SeatCalibrator
from visualizer import LabUI


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LabMonitor:
    """
    Main controller for the Lab Monitoring System.
    Manages the complete pipeline from camera capture to visualization.
    """
    
    def __init__(self, config: Optional[LabConfig] = None):
        """
        Initialize the lab monitor system.
        """
        try:
            # Load configuration
            self.config = config or get_config()
            logger.info("Configuration loaded")
            
            # Initialize components
            logger.info("Initializing detector...")
            self.detector = ObjectDetector(self.config)
            
            logger.info("Initializing tracker...")
            self.tracker = LabStateManager(self.config)
            
            logger.info("Initializing calibrator...")
            self.calibrator = SeatCalibrator(self.config)
            
            logger.info("Initializing UI...")
            self.ui = LabUI(self.config)
            
            # Camera and state
            self.cap: Optional[cv2.VideoCapture] = None
            self.is_running = False
            self.is_calibrating = False
            self.is_fullscreen = False
            
            # Performance tracking
            self.frame_times = []
            self.fps = 0
            self.frame_count = 0
            
            # Signal handling
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("Lab Monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LabMonitor: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Shutting down...")
        self.is_running = False
    
    def _initialize_camera(self) -> bool:
        """Initialize the camera."""
        try:
            if isinstance(self.config.camera_index, str):
                if self.config.camera_index.startswith(('http:', 'https:', 'rtsp:')):
                    logger.info(f"Checking connection to {self.config.camera_index}...")
                    if not self._check_url_reachable(self.config.camera_index):
                        logger.error(f"Cannot reach camera URL: {self.config.camera_index}")
                        logger.error("Please check if:")
                        logger.error("1. The device is on the same network")
                        logger.error("2. The IP address and port are correct")
                        logger.error("3. The camera app is running")
                        return False

                logger.info(f"Opening camera {self.config.camera_index}")
                self.cap = cv2.VideoCapture(self.config.camera_index)
            else:
                logger.info(f"Opening camera {self.config.camera_index} (DirectShow)")
                # Use DirectShow on Windows for much faster startup
                self.cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                logger.error(f"Camera {self.config.camera_index} failed")
                
                # Try alternative indices
                for i in [1, 2, 3, 4]:
                    logger.info(f"Trying camera {i}...")
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        logger.info(f"Using camera {i}")
                        self.config.camera_index = i
                        break
                
                if not self.cap.isOpened():
                    logger.error("No camera available")
                    return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera ready: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Camera error: {e}")
            return False
    
    def _process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, dict]:
        """
        Process a single frame through the pipeline.
        """
        try:
            start_time = time.time()
            
            # 1. Detection
            detections, det_debug = self.detector.detect_with_debug(frame)
            
            # 2. Tracking & State Management
            if self.is_calibrating:
                self.calibrator.add_sample(detections)
                progress = self.calibrator.get_progress()
                
                # Draw calibration UI
                output_frame = self.ui.draw_calibration(
                    frame, 
                    progress, 
                    self.calibrator.get_current_samples()
                )
                
                # Complete calibration if time's up
                if progress >= 1.0:
                    self._complete_calibration()
                
            else:
                # Normal tracking mode
                person_states = self.tracker.update_state(detections)
                stats = self.tracker.get_statistics()
                
                # 3. Visualization
                debug_info = {
                    **det_debug,
                    "fps": self.fps,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
                
                output_frame = self.ui.draw(
                    frame, detections, person_states, stats, debug_info
                )
            
            # Update FPS
            self._update_fps(start_time)
            
            return output_frame, {}
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            # Show error on frame
            cv2.putText(frame, f"System Error", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, {"error": str(e)}
    
    def _update_fps(self, frame_start_time: float):
        """Update FPS calculation."""
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        
        # Keep last 30 frames
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        if self.frame_times:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def _complete_calibration(self):
        """Complete the calibration process."""
        try:
            logger.info("Calculating seat positions...")
            
            # Calculate new seats
            new_seats = self.calibrator.calculate_seats()
            
            if new_seats:
                # Update configuration in memory
                self.config.seats = new_seats
                self.tracker.update_seats(new_seats)
                logger.info(f"Calibration successful: {len(new_seats)} seats defined")
            else:
                logger.warning("Calibration failed: No seats detected")
            
            # Exit calibration mode
            self.is_calibrating = False
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            self.is_calibrating = False
    
    def _handle_keypress(self, key: int) -> bool:
        """
        Handle keyboard input.
        Returns: True to continue, False to exit
        """
        try:
            if key == ord('q') or key == 27:  # 'q' or ESC
                logger.info("Quit requested")
                return False
            
            elif key == ord('c') and not self.is_calibrating:
                # Start calibration
                logger.info("Starting seat calibration")
                self.is_calibrating = True
                self.calibrator.start_calibration()
                return True
            
            elif key == ord('f'):
                # Toggle fullscreen
                self.is_fullscreen = not self.is_fullscreen
                if self.is_fullscreen:
                    cv2.setWindowProperty(
                        self.ui.window_name, 
                        cv2.WND_PROP_FULLSCREEN, 
                        cv2.WINDOW_FULLSCREEN
                    )
                else:
                    cv2.setWindowProperty(
                        self.ui.window_name, 
                        cv2.WND_PROP_FULLSCREEN, 
                        cv2.WINDOW_NORMAL
                    )
                logger.info(f"Fullscreen: {self.is_fullscreen}")
                return True
            
            elif key == ord('r'):
                # Reset seats
                logger.info("Resetting all seats")
                self.config.clear_seats()
                self.tracker.update_seats({})
                return True
            
            elif key == ord('s'):
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lab_snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, self.last_frame)
                logger.info(f"Snapshot saved: {filename}")
                return True
            
            elif key == ord('d'):
                # Toggle debug info
                self.ui.toggle_debug()
                return True
            
            elif key == ord('h'):
                # Show help
                self._show_help()
                return True
            
            elif key == ord('l'):
                # Toggle leaderboard
                self.ui.toggle_leaderboard()
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Key handler error: {e}")
            return True
    
    def _show_help(self):
        """Display help information."""
        help_text = [
            "=== Lab Monitor ===",
            "Controls:",
            "  C - Calibrate seats",
            "  F - Toggle fullscreen",
            "  S - Save snapshot",
            "  R - Reset seats",
            "  D - Toggle debug info",
            "  H - Show this help",
            "  Q - Quit",
            "",
            f"Status: {'CALIBRATING' if self.is_calibrating else 'MONITORING'}",
            f"Seats: {len(self.config.seats)}",
            f"FPS: {self.fps:.1f}",
            ""
        ]
        
        print("\n".join(help_text))
    
    def run(self):
        """Run the main application loop."""
        logger.info("Starting Lab Monitor")
        
        # Initialize camera
        if not self._initialize_camera():
            logger.error("Camera initialization failed")
            return
        
        # Show help
        self._show_help()
        
        self.is_running = True
        logger.info("Entering main loop")
        
        try:
            while self.is_running and self.cap.isOpened():
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Frame capture failed")
                    time.sleep(0.1)
                    continue
                
                # Store for snapshot
                self.last_frame = frame.copy()
                self.frame_count += 1
                
                # Process frame
                output_frame, _ = self._process_frame(frame)
                
                # Display
                cv2.imshow(self.ui.window_name, output_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key pressed
                    if not self._handle_keypress(key):
                        self.is_running = False
                
                # Check window close
                if cv2.getWindowProperty(self.ui.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed")
                    self.is_running = False
                
                # Periodic status
                if self.frame_count % 100 == 0:
                    stats = self.tracker.get_statistics()
                    logger.info(
                        f"Frame {self.frame_count}: "
                        f"{stats['current_persons']} persons, "
                        f"{stats['focused_persons']} focused, "
                        f"{stats['distracted_persons']} distracted"
                    )
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        # Release camera
        if self.cap:
            self.cap.release()
            logger.info("Camera released")
        
        # Close windows
        cv2.destroyAllWindows()
        logger.info("Windows closed")
        
        logger.info("Shutdown complete")
    
    def run_test_mode(self, duration: int = 30):
        """
        Run with test pattern (no camera needed).
        """
        import numpy as np
        
        logger.info(f"Test mode for {duration} seconds")
        
        # Create test pattern
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (100, 100), (300, 300), (0, 255, 0), 2)
        cv2.putText(test_frame, "TEST MODE", (200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        start_time = time.time()
        self.is_running = True
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Process test frame
                output_frame, _ = self._process_frame(test_frame.copy())
                
                # Display
                cv2.imshow(self.ui.window_name, output_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if not self._handle_keypress(key):
                        self.is_running = False
                
                # Simulate frame rate
                time.sleep(1.0 / self.config.fps)
        
        finally:
            self._cleanup()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lab Monitoring System')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--test-duration', type=int, default=30, 
                       help='Test duration in seconds')
    parser.add_argument('--camera', type=int, help='Camera index')
    parser.add_argument('--model', type=str, help='YOLO model path')
    
    args = parser.parse_args()
    
    try:
        # Create monitor
        monitor = LabMonitor()
        
        # Apply command line overrides
        if args.camera is not None:
            monitor.config.camera_index = args.camera
        
        if args.model:
            monitor.config.model_path = args.model
        
        # Run
        if args.test:
            monitor.run_test_mode(args.test_duration)
        else:
            monitor.run()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())