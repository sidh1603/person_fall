# Multi-Scenario Detection System

A comprehensive surveillance system that can detect multiple scenarios including people, falls, fights, crowds, and fires in real-time video streams.

## Features

- **Person Detection**: Tracks and counts people in video streams
- **Fall Detection**: Identifies when a standing person falls down
- **Fight Detection**: Detects aggressive movement patterns indicative of physical altercations
- **Crowd Detection**: Alerts when the number of people exceeds a configurable threshold
- **Fire Detection**: Uses specialized YOLOv8n model to detect fires with high accuracy
- **Vehicle Detection**: Identifies and classifies vehicles (cars, trucks, buses, motorcycles)
- **Dynamic Scenario Selection**: Enable/disable detection scenarios through the web interface
- **Real-time Analytics**: Dashboard showing detection statistics and event history
- **Multi-camera Support**: Process multiple video streams simultaneously

## Models Used

- **YOLOv8s**: Main model for person detection, used for fall/fight/crowd detection
- **YOLOv8n (Fire)**: Specialized model for fire detection with higher accuracy

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the required models:
   - YOLOv8s.pt: General purpose detection
   - YOLOv8n.pt: Fire detection (trained on fire dataset)

## Usage

Run the application:
```
python new_file.py
```

Access the dashboard by opening your browser and navigating to:
```
http://localhost:5000
```

## How It Works

### Dual-Model Approach

The system uses a dual-model approach for optimal detection:

1. **YOLOv8s**: Handles general object detection with a focus on people. Used for:
   - Tracking persons across frames
   - Fall detection (based on person aspect ratio changes)
   - Fight detection (based on rapid movement patterns)
   - Crowd detection (based on person count)

2. **YOLOv8n**: Specialized for fire detection with higher sensitivity and accuracy

### Camera Assignment

- Camera 5 is dedicated to fire detection using the specialized model
- Other cameras use the general purpose YOLOv8s model

## Configuration

The system can be configured through these dictionaries:
- `ALERT_CONFIG`: Adjust thresholds for alerts
- `DETECTION_SCENARIOS`: Enable/disable detection scenarios
- `VIDEO_FILES`: Specify video sources for each camera

## Fire Detection

Fire detection uses a dedicated model that offers significant improvements:
- Higher accuracy compared to color-based methods
- Lower false positive rates
- Better detection in various lighting conditions

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Flask
- Ultralytics YOLOv8
- CUDA-compatible GPU (recommended for real-time performance)

## Note on Models

The YOLOv8n model for fire detection should be trained on a fire dataset. The model should be placed in the project root directory.