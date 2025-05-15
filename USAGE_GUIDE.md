# Multi-Scenario Detection System - Usage Guide

This guide will help you run and use the complete multi-scenario detection system.

## Running the Complete System

There are two ways to run the system:

### Option 1: Using the Run Script (Recommended)

The simplest way to run the complete system is to use the provided shell script:

```bash
./run_complete_system.sh
```

This script will:
1. Activate the appropriate conda environment
2. Start the multi-scenario detection system
3. Run the web server at http://localhost:5000

### Option 2: Manual Execution

If you prefer to run the system manually:

```bash
# Activate the conda environment
source ~/miniconda3/bin/activate fall

# Run the system
python new_file.py
```

## Accessing the Dashboard

Once the system is running, open your web browser and navigate to:
```
http://localhost:5000
```

This will display the multi-scenario detection dashboard.

## Dashboard Features

The dashboard provides:

1. **Live Camera Feeds**: View all camera streams with real-time detections
2. **Detection Controls**: Enable/disable different detection scenarios
3. **Statistics Panel**: Track counts of various detections
4. **Events List**: View a chronological list of detection events
5. **Emergency Alert Button**: Manually trigger alerts when needed

## Available Detection Scenarios

The system includes the following detection capabilities:

1. **Person Detection**: Identifies and counts people in the video streams
2. **Fall Detection**: Recognizes when a person falls down
3. **Fight Detection**: Detects aggressive movements indicative of fighting
4. **Crowd Detection**: Alerts when too many people are present
5. **Fire Detection**: Identifies fires using a specialized detection algorithm
6. **Vehicle Detection**: Identifies and classifies vehicles (cars, trucks, buses, motorcycles)

## Testing Individual Components

You can also test individual components of the system:

### Test Fire Detection Only

```bash
./run_improved_fire_test.sh
```

This will run the standalone fire detection test on the fire video.

## Configuration

If you need to modify the system configuration:

1. **Video Sources**: Edit the `VIDEO_FILES` dictionary in `new_file.py`
2. **Detection Thresholds**: Adjust parameters in `ALERT_CONFIG`
3. **Fire Detection Settings**: Modify HSV thresholds in `fire_detector.py`

## Troubleshooting

If you encounter any issues:

1. **Camera Not Loading**: Verify the video file paths in the `VIDEO_FILES` dictionary
2. **Detection Issues**: Adjust detection thresholds in the respective detector classes
3. **Environment Problems**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
4. **Web Interface Not Working**: Check if port 5000 is already in use by another application

For more detailed information, refer to the README.md file.