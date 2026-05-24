"""
Object detection using YOLOv8.
Handles model loading, inference, and result parsing.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import logging

from config import DetectionClass, BoundingBox, get_config


logger = logging.getLogger(__name__)


class DetectionResult:
    """Structured representation of a detection."""
    
    def __init__(
        self,
        class_id: int,
        confidence: float,
        box: BoundingBox,
        track_id: Optional[int] = None
    ):
        self.class_id = class_id
        self.class_name = self._get_class_name(class_id)
        self.confidence = confidence
        self.box = box
        self.track_id = track_id
    
    @staticmethod
    def _get_class_name(class_id: int) -> str:
        try:
            return DetectionClass(class_id).name.lower().replace('_', ' ')
        except ValueError:
            return f"class_{class_id}"
    
    @property
    def is_person(self) -> bool:
        return self.class_id == DetectionClass.PERSON.value
    
    @property
    def is_phone(self) -> bool:
        return self.class_id == DetectionClass.CELL_PHONE.value
    
    def __repr__(self) -> str:
        return (f"DetectionResult(class={self.class_name}, "
                f"confidence={self.confidence:.2f}, "
                f"track_id={self.track_id})")


class ObjectDetector:
    """YOLOv8-based object detector with tracking."""
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the detector.
        
        Args:
            config: LabConfig instance. If None, uses global config.
        """
        self.config = config or get_config()
        self.model: Optional[YOLO] = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Load and initialize the YOLO model."""
        try:
            logger.info(f"Loading YOLO model from {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            
            # Warm up the model
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(
                dummy_frame,
                verbose=False,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold
            )
            logger.info("YOLO model loaded and warmed up")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Run inference
            results = self.model.track(
                frame,
                persist=True,
                classes=[DetectionClass.PERSON.value, DetectionClass.CELL_PHONE.value],
                verbose=False,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold
            )
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    for box in result.boxes:
                        try:
                            # Extract data
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Create bounding box
                            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                            bbox = BoundingBox(x1, y1, x2, y2)
                            
                            # Extract track ID if available
                            track_id = None
                            if box.id is not None:
                                track_id = int(box.id[0])
                            
                            # Filter by minimum area for persons
                            if cls_id == DetectionClass.PERSON.value:
                                if bbox.area < self.config.min_person_area:
                                    continue
                            
                            detections.append(
                                DetectionResult(
                                    class_id=cls_id,
                                    confidence=confidence,
                                    box=bbox,
                                    track_id=track_id
                                )
                            )
                            
                        except (IndexError, ValueError, TypeError) as e:
                            logger.warning(f"Failed to parse detection box: {e}")
                            continue
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_with_debug(self, frame: np.ndarray) -> Tuple[List[DetectionResult], Dict[str, Any]]:
        """
        Detect objects and return debug information.
        
        Returns:
            Tuple of (detections, debug_info)
        """
        import time
        start_time = time.time()
        
        detections = self.detect(frame)
        
        debug_info = {
            "inference_time_ms": (time.time() - start_time) * 1000,
            "total_detections": len(detections),
            "person_count": sum(1 for d in detections if d.is_person),
            "phone_count": sum(1 for d in detections if d.is_phone),
        }
        
        return detections, debug_info