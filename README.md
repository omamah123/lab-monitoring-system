# ICV Lab Monitoring System

An intelligent, computer vision-based monitoring pipeline designed to analyze activity and tracking dynamics within a laboratory environment. The system utilizes the YOLOv8 framework for robust object detection, coupled with persistent sequential tracking and real-time visualization overlays.

---

## Key Features

* **Real-Time Object Detection:** Leverages pre-trained YOLOv8 medium weights (`yolov8m.pt`) for balanced throughput and high-accuracy multi-class localization.
* **Persistent Object Tracking:** Implements frame-to-frame tracking algorithms to reliably monitor detected subjects and maintain identity consistency over time.
* **Dynamic Visualization Engine:** Generates low-latency bounding boxes, metadata labels, and historical trajectory paths directly overlayed onto the video stream.
* **Decoupled Architecture:** Centralized parameter control allowing instantaneous adjustments to detection thresholds, hardware allocation, and input/output streams.

---

##  Project Structure

```text
├── config.py          # Centralized hyperparameter, directory, and system configurations
├── detector.py        # Framework initialization and frame-by-frame inference pipeline (YOLOv8)
├── tracker.py         # Multi-object tracking logic and frame-to-frame data association
├── visualizer.py      # OpenCV canvas annotations, color-mapping, and rendering utilities
├── main.py            # Primary orchestrator and execution entry point
└── yolov8m.pt         # Serialized Ultralytics YOLOv8 medium model weights
```
---

## Prerequisites & Installation
``` ###System Requirements 
OS: Windows, macOS, or Linux

Runtime: Python 3.8 or higher

Hardware: Compute capability for CUDA acceleration is highly recommended but not mandatory (system defaults smoothly to CPU threads).
```
``` ###Setup Instructions
1.Clone the Repository
git clone [https://github.com/omamah123/lab-monitoring-system.git](https://github.com/omamah123/lab-monitoring-system.git)
cd lab-monitoring-system
2.Install Required Packages
pip install ultralytics opencv-python numpy torch torchvision
3. pip install ultralytics opencv-python numpy torch torchvision
python main.py
```
``` ###Usage
1.Configure Parameters: Open config.py to specify your target video sources (e.g., local video path or webcam stream index) and adjust your detection filters.
2.Execute the Pipeline: Launch the tracking system by running the main execution module:
python main.py
```
```###Configuration Parameters
The parameters isolated inside config.py allow you to easily fine-tune performance without altering core logic:

Inference Thresholds: Set precision parameters such as the confidence (conf) and intersection-over-union (iou) limits to filter weak detections.

Hardware Allocation: Choose your compute device manually (cuda vs cpu) or let the script auto-detect available NVIDIA hardware runtimes.

Stream Matrix: Manage video feed inputs (rtsp links, pre-recorded laboratory footage, or hardware camera index ids).
```
