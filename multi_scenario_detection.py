#!/usr/bin/env python3
"""
Multi-Scenario Detection System using YOLOv8
Features:
- Person Detection: Counts and highlights all detected persons
- Fall Detection: Detects when a standing person falls down
- Fight Detection: Detects rapid movement and aggressive interactions
- Crowd Detection: Alerts when too many people are in the frame
- Dynamic scenario selection via checkboxes
- Dashboard for monitoring and alerts
"""

import os
import time
import cv2
import numpy as np
import threading
import datetime
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import math
from collections import defaultdict

# Create Flask app
app = Flask(__name__)

# Global counters
COUNTERS = {
    'processed_frames': 0,
    'detected_persons': 0,
    'detected_falls': 0,
    'detected_fights': 0,
    'detected_crowds': 0,
    'detected_fires': 0,
    'alerts_sent': 0
}

# Global event list
EVENTS = []

# Global camera frames
CAMERA_FRAMES = {
    'camera1': None,
    'camera2': None,
    'camera3': None,
    'camera4': None,
    'camera5': None
}

# Define video files
VIDEO_FILES = {
    'camera1': '/home/siddu/projects/person_fall/data/fall.mp4',
    'camera2': '/home/siddu/projects/person_fall/data/fall_2.mp4',
    'camera3': '/home/siddu/projects/person_fall/data/fight.mp4',
    'camera4': '/home/siddu/projects/person_fall/data/crowd.webm',
    'camera5': '/home/siddu/projects/person_fall/data/fire.mp4'
}

# Alert configurations
ALERT_CONFIG = {
    'enabled': True,
    'email': 'security@example.com',
    'threshold': 0.7,
    'crowd_threshold': 5  # Number of people to trigger crowd alert
}

# Detection scenarios configuration
DETECTION_SCENARIOS = {
    'person_detection': True,
    'fall_detection': True,
    'fight_detection': True,
    'crowd_detection': True,
    'fire_detection': True
}

# Load YOLOv8 models
try:
    # Force CUDA if available
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load person detection model
    if os.path.exists('yolov8s.pt'):
        MODEL = YOLO('yolov8s.pt')
    else:
        print("YOLOv8 model not found. Downloading...")
        MODEL = YOLO('yolov8s.pt')
    
    # Move model to GPU if available
    if device == 'cuda':
        MODEL.to('cuda')
        print("Model moved to CUDA")
    
    # Load fire detection model
    FIRE_MODEL = None
    if os.path.exists('model_16_m3_0.8888.pth'):
        # We'll create a custom fire detection wrapper that uses this model
        print("Fire detection model found. Loading...")
        # Note: We'll use a simpler approach using the standard YOLOv8 model for fire detection
        # and implement dedicated fire detection logic
    
    # Load class names
    CLASSNAMES = []
    if os.path.exists('classes.txt'):
        with open('classes.txt', 'r') as f:
            CLASSNAMES = f.read().splitlines()
    else:
        # Default COCO class names
        CLASSNAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                     'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                     'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    MODEL_LOADED = True
    print("YOLOv8 model loaded successfully!")
    
except Exception as e:
    print(f"Error loading YOLOv8 model: {str(e)}")
    MODEL_LOADED = False

# Create template directory
os.makedirs('templates', exist_ok=True)

# Create the HTML template with checkboxes for scenarios
with open('templates/multi_detection.html', 'w') as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Scenario Detection Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .header {
                background-color: #2c3e50;
                background-image: linear-gradient(to right, #2c3e50, #34495e);
                color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin: 0;
                font-size: 28px;
            }
            .header p {
                margin: 5px 0 0;
                opacity: 0.8;
            }
            
            /* Scenario selection */
            .scenario-selection {
                background-color: white;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .scenario-selection h3 {
                margin-top: 0;
                color: #333;
            }
            .scenario-checkbox {
                display: inline-block;
                margin-right: 20px;
                margin-bottom: 10px;
            }
            .scenario-checkbox input[type="checkbox"] {
                margin-right: 5px;
            }
            .scenario-checkbox label {
                cursor: pointer;
                font-weight: 500;
            }
            
            .stats {
                display: flex;
                justify-content: space-around;
                background-color: #34495e;
                color: white;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .stat {
                text-align: center;
            }
            .stat-name {
                font-size: 14px;
                opacity: 0.8;
            }
            .stat-value {
                font-size: 28px;
                font-weight: bold;
                margin-top: 5px;
            }
            
            /* Main layout grid */
            .main-container {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            }
            
            /* Camera section */
            .camera-container {
                flex: 2;
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }
            
            .camera {
                flex: 1;
                min-width: 250px;
                max-width: 350px;
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
                background-color: white;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                height: fit-content;
            }
            .camera-header {
                background-color: #f8f9fa;
                padding: 10px;
                font-weight: bold;
                border-bottom: 1px solid #ddd;
            }
            .camera img {
                width: 100%;
                display: block;
            }
            
            /* Events section */
            .events {
                flex: 1;
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                min-width: 280px;
                max-width: 350px;
                align-self: flex-start;
            }
            .events h2 {
                margin-top: 0;
                color: #333;
            }
            .event-list {
                max-height: 500px;
                overflow-y: auto;
            }
            .event {
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
            }
            .event-fall {
                background-color: #ffdddd;
                border-left: 4px solid #e74c3c;
            }
            .event-fight {
                background-color: #ffe0dd;
                border-left: 4px solid #ff6b35;
            }
            .event-crowd {
                background-color: #fff8dd;
                border-left: 4px solid #f39c12;
            }
            .event-person {
                background-color: #ddffdd;
                border-left: 4px solid #27ae60;
            }
            .event-header {
                display: flex;
                justify-content: space-between;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .event-type {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 15px;
                font-size: 12px;
                color: white;
                font-weight: bold;
            }
            .event-type-fall {
                background-color: #e74c3c;
                border: 2px solid #c0392b;
            }
            .event-type-fight {
                background-color: #ff6b35;
                border: 2px solid #e55300;
            }
            .event-type-crowd {
                background-color: #f39c12;
                border: 2px solid #d68910;
            }
            .event-type-person {
                background-color: #27ae60;
                border: 2px solid #229954;
            }
            .event-fire {
                background-color: #ffddbb;
                border-left: 4px solid #ff5722;
            }
            .event-type-fire {
                background-color: #ff5722;
                border: 2px solid #e64a19;
            }
            .pulse {
                animation: pulse-animation 0.5s 1;
            }
            @keyframes pulse-animation {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            
            .alert-button {
                margin-top: 20px;
                padding: 12px 20px;
                background-color: #e74c3c;
                background-image: linear-gradient(to right, #e74c3c, #c0392b);
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
                transition: background-color 0.3s;
            }
            
            .alert-button:hover {
                background-color: #c0392b;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Multi-Scenario Detection Dashboard</h1>
            <p>Monitor multiple scenarios: Person, Fall, Fight, and Crowd Detection</p>
        </div>
        
        <div class="scenario-selection">
            <h3>Select Detection Scenarios</h3>
            <div class="scenario-checkbox">
                <input type="checkbox" id="person_detection" checked>
                <label for="person_detection">Person Detection</label>
            </div>
            <div class="scenario-checkbox">
                <input type="checkbox" id="fall_detection" checked>
                <label for="fall_detection">Fall Detection</label>
            </div>
            <div class="scenario-checkbox">
                <input type="checkbox" id="fight_detection" checked>
                <label for="fight_detection">Fight Detection</label>
            </div>
            <div class="scenario-checkbox">
                <input type="checkbox" id="crowd_detection" checked>
                <label for="crowd_detection">Crowd Detection</label>
            </div>
            <div class="scenario-checkbox">
                <input type="checkbox" id="fire_detection" checked>
                <label for="fire_detection">Fire Detection</label>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-name">Processed Frames</div>
                <div id="processed-frames" class="stat-value">0</div>
            </div>
            <div class="stat">
                <div class="stat-name">Detected Persons</div>
                <div id="detected-persons" class="stat-value">0</div>
            </div>
            <div class="stat">
                <div class="stat-name">Detected Falls</div>
                <div id="detected-falls" class="stat-value">0</div>
            </div>
            <div class="stat">
                <div class="stat-name">Detected Fights</div>
                <div id="detected-fights" class="stat-value">0</div>
            </div>
            <div class="stat">
                <div class="stat-name">Detected Crowds</div>
                <div id="detected-crowds" class="stat-value">0</div>
            </div>
            <div class="stat">
                <div class="stat-name">Detected Fires</div>
                <div id="detected-fires" class="stat-value">0</div>
            </div>
        </div>
        
        <div class="main-container">
            <!-- Left side - Camera feeds -->
            <div class="camera-container">
                <div class="camera">
                    <div class="camera-header">Camera 1</div>
                    <img src="/video_feed/camera1" alt="Camera 1">
                </div>
                <div class="camera">
                    <div class="camera-header">Camera 2</div>
                    <img src="/video_feed/camera2" alt="Camera 2">
                </div>
                <div class="camera">
                    <div class="camera-header">Camera 3</div>
                    <img src="/video_feed/camera3" alt="Camera 3">
                </div>
                <div class="camera">
                    <div class="camera-header">Camera 4</div>
                    <img src="/video_feed/camera4" alt="Camera 4">
                </div>
                <div class="camera">
                    <div class="camera-header">Camera 5 (Fire)</div>
                    <img src="/video_feed/camera5" alt="Camera 5">
                </div>
            </div>
            
            <!-- Right side - Events panel -->
            <div class="events">
                <h2>Detection Events</h2>
                <div id="event-list" class="event-list">
                    <p>No events detected yet.</p>
                </div>
                
                <button id="alert-button" class="alert-button">SEND EMERGENCY ALERT</button>
            </div>
        </div>
        
        <script>
            // Handle checkbox changes
            document.querySelectorAll('.scenario-checkbox input[type="checkbox"]').forEach(checkbox => {
                checkbox.addEventListener('change', function() {
                    const scenario = this.id;
                    const enabled = this.checked;
                    
                    // Send update to server
                    fetch('/api/update_scenario', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            scenario: scenario,
                            enabled: enabled
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log(`${scenario} is now ${enabled ? 'enabled' : 'disabled'}`);
                    })
                    .catch(error => console.error('Error updating scenario:', error));
                });
            });
            
            // Function to update the stats from server
            function updateStats() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        // Get current values
                        const oldProcessed = parseInt(document.getElementById('processed-frames').textContent);
                        const oldPersons = parseInt(document.getElementById('detected-persons').textContent);
                        const oldFalls = parseInt(document.getElementById('detected-falls').textContent);
                        const oldFights = parseInt(document.getElementById('detected-fights').textContent);
                        const oldCrowds = parseInt(document.getElementById('detected-crowds').textContent);
                        const oldFires = parseInt(document.getElementById('detected-fires').textContent);
                        
                        // Update with new values
                        document.getElementById('processed-frames').textContent = data.processed_frames;
                        document.getElementById('detected-persons').textContent = data.detected_persons;
                        document.getElementById('detected-falls').textContent = data.detected_falls;
                        document.getElementById('detected-fights').textContent = data.detected_fights;
                        document.getElementById('detected-crowds').textContent = data.detected_crowds;
                        document.getElementById('detected-fires').textContent = data.detected_fires;
                        
                        // Add pulse effect if values increased
                        if (data.detected_persons > oldPersons) {
                            document.getElementById('detected-persons').classList.add('pulse');
                            setTimeout(() => {
                                document.getElementById('detected-persons').classList.remove('pulse');
                            }, 1000);
                        }
                        
                        if (data.detected_falls > oldFalls) {
                            document.getElementById('detected-falls').classList.add('pulse');
                            setTimeout(() => {
                                document.getElementById('detected-falls').classList.remove('pulse');
                            }, 1000);
                        }
                        
                        if (data.detected_fights > oldFights) {
                            document.getElementById('detected-fights').classList.add('pulse');
                            setTimeout(() => {
                                document.getElementById('detected-fights').classList.remove('pulse');
                            }, 1000);
                        }
                        
                        if (data.detected_crowds > oldCrowds) {
                            document.getElementById('detected-crowds').classList.add('pulse');
                            setTimeout(() => {
                                document.getElementById('detected-crowds').classList.remove('pulse');
                            }, 1000);
                        }
                        
                        if (data.detected_fires > oldFires) {
                            document.getElementById('detected-fires').classList.add('pulse');
                            setTimeout(() => {
                                document.getElementById('detected-fires').classList.remove('pulse');
                            }, 1000);
                        }
                    })
                    .catch(error => console.error('Error fetching stats:', error));
            }
            
            // Function to update the event list
            function updateEvents() {
                fetch('/api/events')
                    .then(response => response.json())
                    .then(data => {
                        const eventList = document.getElementById('event-list');
                        
                        // Clear the list if it only has the placeholder
                        if (eventList.innerHTML.includes('No events detected yet') && data.events.length > 0) {
                            eventList.innerHTML = '';
                        }
                        
                        // Return if no events
                        if (data.events.length === 0) {
                            return;
                        }
                        
                        // Check if new events were added
                        const newEvents = data.events.filter(event => {
                            const existingEvent = document.querySelector(`[data-event-id="${event.id}"]`);
                            return !existingEvent;
                        });
                        
                        // Add new events to the top of the list
                        newEvents.forEach(event => {
                            const eventEl = document.createElement('div');
                            eventEl.className = `event event-${event.type.toLowerCase()}`;
                            eventEl.setAttribute('data-event-id', event.id);
                            
                            eventEl.innerHTML = `
                                <div class="event-header">
                                    <div>
                                        <span class="event-type event-type-${event.type.toLowerCase()}">${event.type}</span>
                                        <span>${event.camera}</span>
                                    </div>
                                    <span>${event.time}</span>
                                </div>
                                <div>
                                    ${event.details}
                                </div>
                            `;
                            
                            // Add to the top of the list
                            if (eventList.firstChild) {
                                eventList.insertBefore(eventEl, eventList.firstChild);
                            } else {
                                eventList.appendChild(eventEl);
                            }
                            
                            // Add pulse effect
                            eventEl.classList.add('pulse');
                            setTimeout(() => {
                                eventEl.classList.remove('pulse');
                            }, 1000);
                        });
                    })
                    .catch(error => console.error('Error fetching events:', error));
            }
            
            // Handle manual alert button
            document.getElementById('alert-button').addEventListener('click', function() {
                this.textContent = 'SENDING ALERT...';
                this.disabled = true;
                
                fetch('/api/send_alert', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        source: 'manual',
                        type: 'EMERGENCY',
                        message: 'Manual emergency alert triggered by operator'
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('alert-button').textContent = 'ALERT SENT';
                        setTimeout(() => {
                            document.getElementById('alert-button').textContent = 'SEND EMERGENCY ALERT';
                            document.getElementById('alert-button').disabled = false;
                        }, 3000);
                    } else {
                        document.getElementById('alert-button').textContent = 'ALERT FAILED';
                        setTimeout(() => {
                            document.getElementById('alert-button').textContent = 'SEND EMERGENCY ALERT';
                            document.getElementById('alert-button').disabled = false;
                        }, 3000);
                    }
                })
                .catch(error => {
                    console.error('Error sending alert:', error);
                    document.getElementById('alert-button').textContent = 'ERROR: TRY AGAIN';
                    document.getElementById('alert-button').disabled = false;
                });
            });
            
            // Update stats and events every second
            setInterval(() => {
                updateStats();
                updateEvents();
            }, 1000);
            
            // Initial updates
            updateStats();
            updateEvents();
        </script>
    </body>
    </html>
    """)

# Routes
@app.route('/')
def index():
    """Render the dashboard."""
    return render_template('multi_detection.html')

@app.route('/api/stats')
def get_stats():
    """Get the current statistics."""
    return jsonify(COUNTERS)

@app.route('/api/events')
def get_events():
    """Get the current events."""
    return jsonify({"events": EVENTS})

@app.route('/api/update_scenario', methods=['POST'])
def update_scenario():
    """Update detection scenario configuration."""
    data = request.json
    scenario = data.get('scenario')
    enabled = data.get('enabled')
    
    if scenario in DETECTION_SCENARIOS:
        DETECTION_SCENARIOS[scenario] = enabled
        print(f"Updated {scenario} to {enabled}")
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Invalid scenario"})

@app.route('/api/send_alert', methods=['POST'])
def send_alert():
    """Send a security alert."""
    try:
        global COUNTERS
        print("ALERT SENT: Emergency alert triggered!")
        COUNTERS['alerts_sent'] += 1
        add_event("Operator", "EMERGENCY", "Manual alert triggered", 1.0)
        return jsonify({"success": True, "message": "Alert sent successfully"})
    except Exception as e:
        print(f"Error sending alert: {str(e)}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Stream a camera feed."""
    def generate():
        """Generate frames for the stream."""
        while True:
            if camera_id in CAMERA_FRAMES and CAMERA_FRAMES[camera_id] is not None:
                frame = CAMERA_FRAMES[camera_id]
            else:
                frame = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(frame, f"{camera_id} - Starting...", (50, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def add_event(camera, event_type, details, confidence):
    """Add a detection event."""
    global COUNTERS, EVENTS
    
    event_id = str(time.time())
    
    event = {
        "id": event_id,
        "camera": camera,
        "type": event_type,
        "details": details,
        "confidence": confidence,
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    }
    
    EVENTS.insert(0, event)
    
    # Update appropriate counter
    if event_type == "FALL":
        COUNTERS['detected_falls'] += 1
    elif event_type == "FIGHT":
        COUNTERS['detected_fights'] += 1
    elif event_type == "CROWD":
        COUNTERS['detected_crowds'] += 1
    elif event_type == "PERSON":
        COUNTERS['detected_persons'] += 1
    elif event_type == "FIRE":
        COUNTERS['detected_fires'] += 1
    
    print(f"Event added: {event_type} on {camera} - {details}")
    
    if len(EVENTS) > 100:
        EVENTS.pop()


class PersonTracker:
    """Enhanced person tracker for multiple scenarios."""
    def __init__(self, box, confidence=0.0):
        self.id = f"person_{datetime.datetime.now().timestamp()}"
        self.box = box  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.positions = []  # Track position history
        self.movement_history = []  # Track movement for fight detection
        self.is_standing = True
        self.was_standing = True
        self.standing_frames = 0
        self.fallen_frames = 0
        self.fall_detected = False
        self.last_seen = time.time()
        self.matched = False
        self.recovery_frames = 0
        self.recovery_threshold = 10
        
    def update(self, box, confidence):
        """Update person tracking with new detection."""
        # Store previous position for movement analysis
        if len(self.positions) > 0:
            prev_center = self.get_center(self.positions[-1])
            curr_center = self.get_center(box)
            movement = self.calculate_distance(prev_center, curr_center)
            self.movement_history.append(movement)
            
            # Keep movement history limited
            if len(self.movement_history) > 10:
                self.movement_history.pop(0)
        
        self.positions.append(box)
        self.box = box
        self.confidence = confidence
        self.last_seen = time.time()
        
        # Check if standing or fallen
        height = box[3] - box[1]
        width = box[2] - box[0]
        aspect_ratio = height / width if width > 0 else 999
        
        self.was_standing = self.is_standing
        self.is_standing = aspect_ratio >= 1.2
        
        if self.is_standing:
            self.standing_frames += 1
            self.fallen_frames = 0
            
            if self.fall_detected:
                self.recovery_frames += 1
                if self.recovery_frames >= self.recovery_threshold:
                    self.fall_detected = False
                    self.recovery_frames = 0
        else:
            self.fallen_frames += 1
            self.recovery_frames = 0
            if self.standing_frames > 0:
                self.standing_frames -= 1
        
        # Keep position history limited
        if len(self.positions) > 15:
            self.positions.pop(0)
    
    def get_center(self, box):
        """Get center point of bounding box."""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_fall(self):
        """Detect if a fall occurred."""
        if (self.standing_frames >= 10 and 
            self.fallen_frames >= 2 and 
            self.was_standing and not self.is_standing and 
            not self.fall_detected):
            
            self.fall_detected = True
            return True, 0.8 + (self.confidence * 0.2)
        
        elif (self.fallen_frames >= 10 and 
              not self.is_standing and 
              not self.fall_detected):
            
            self.fall_detected = True
            return True, 0.7 + (self.confidence * 0.2)
        
        return False, 0.0
    
    def detect_fight(self):
        """Detect if person is involved in a fight based on rapid movement."""
        if len(self.movement_history) < 5:
            return False, 0.0
        
        # Calculate average movement
        avg_movement = sum(self.movement_history) / len(self.movement_history)
        
        # Check for rapid movement patterns
        if avg_movement > 50:  # High average movement
            variance = np.var(self.movement_history)
            if variance > 100:  # High variance indicates erratic movement
                return True, min(0.9, 0.6 + (avg_movement / 200))
        
        return False, 0.0


class MultiScenarioDetector:
    """Multi-scenario detector using YOLOv8."""
    
    def __init__(self):
        self.persons = []
        self.max_persons = 20
        self.frame_count = 0
        self.cooldowns = {
            'fall': 0,
            'fight': 0,
            'crowd': 0,
            'person': 0,
            'fire': 0
        }
        self.last_person_count = 0
        self.crowd_frames = 0
        self.fire_frames = 0
        self.fire_detected = False
        self.camera_id = 'camera1'  # Default camera ID
        
        # Fire detection parameters
        self.fire_hsv_lower = (0, 130, 180)  # Lower threshold for fire color in HSV space
        self.fire_hsv_upper = (30, 255, 255)  # Upper threshold for fire color in HSV space
        self.fire_area_threshold = 1000  # Minimum area to consider a fire detection
        
    def process_frame(self, frame):
        """Process frame for multiple scenarios."""
        self.frame_count += 1
        
        # Update cooldowns
        for key in self.cooldowns:
            if self.cooldowns[key] > 0:
                self.cooldowns[key] -= 1
        
        if MODEL_LOADED:
            try:
                # Get YOLO detections with GPU support
                results = MODEL(frame, conf=0.3, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                # Store current detected persons
                detected_persons = []
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        if cls_id == 0:  # Person class
                            detected_persons.append(((x1, y1, x2, y2), conf))
                
                # Update tracking
                self.update_tracking(detected_persons)
                
                # Create visualization frame
                vis_frame = frame.copy()
                
                # Draw detection scenarios based on enabled options
                detection_results = {
                    'persons_detected': [],
                    'falls_detected': [],
                    'fights_detected': [],
                    'crowd_detected': False,
                    'fire_detected': False
                }
                
                # Person Detection
                if DETECTION_SCENARIOS['person_detection']:
                    for person in self.persons:
                        x1, y1, x2, y2 = person.box
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"Person {person.confidence:.2f}", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detection_results['persons_detected'].append(person)
                
                # Fall Detection
                if DETECTION_SCENARIOS['fall_detection']:
                    for person in self.persons:
                        fall_detected, confidence = person.detect_fall()
                        if fall_detected:
                            x1, y1, x2, y2 = person.box
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            
                            # Draw FALL text inside the bounding box
                            height = y2 - y1
                            width = x2 - x1
                            text_size = cv2.getTextSize("FALL", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
                            text_x = x1 + (width - text_size[0]) // 2
                            text_y = y1 + height // 2
                            
                            # Add background for text visibility
                            cv2.rectangle(vis_frame, 
                                        (text_x - 5, text_y - text_size[1] - 5),
                                        (text_x + text_size[0] + 5, text_y + 5),
                                        (0, 0, 0), -1)
                            
                            cv2.putText(vis_frame, "FALL", (text_x, text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            
                            detection_results['falls_detected'].append((person, confidence))
                
                # Fight Detection
                if DETECTION_SCENARIOS['fight_detection']:
                    for person in self.persons:
                        fight_detected, confidence = person.detect_fight()
                        if fight_detected:
                            x1, y1, x2, y2 = person.box
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                            
                            # Draw FIGHT text inside the bounding box
                            height = y2 - y1
                            width = x2 - x1
                            text_size = cv2.getTextSize("FIGHT", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
                            text_x = x1 + (width - text_size[0]) // 2
                            text_y = y1 + height // 2
                            
                            # Add background for text visibility
                            cv2.rectangle(vis_frame, 
                                        (text_x - 5, text_y - text_size[1] - 5),
                                        (text_x + text_size[0] + 5, text_y + 5),
                                        (0, 0, 0), -1)
                            
                            cv2.putText(vis_frame, "FIGHT", (text_x, text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
                            
                            detection_results['fights_detected'].append((person, confidence))
                
                # Crowd Detection
                if DETECTION_SCENARIOS['crowd_detection']:
                    current_person_count = len(self.persons)
                    if current_person_count >= ALERT_CONFIG['crowd_threshold']:
                        self.crowd_frames += 1
                        if self.crowd_frames >= 10:  # Sustained crowd for 10 frames
                            detection_results['crowd_detected'] = True
                            cv2.putText(vis_frame, f"CROWD DETECTED: {current_person_count} people", 
                                      (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    else:
                        self.crowd_frames = 0
                        
                # Fire Detection
                if DETECTION_SCENARIOS['fire_detection']:
                    # Check if we're using camera5 (dedicated fire camera)
                    if self.camera_id == 'camera5':
                        # Convert frame to HSV color space for fire detection
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        
                        # Create a mask for fire-colored pixels
                        mask = cv2.inRange(hsv, self.fire_hsv_lower, self.fire_hsv_upper)
                        
                        # Apply some morphological operations to reduce noise
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                        mask = cv2.dilate(mask, kernel, iterations=3)
                        mask = cv2.medianBlur(mask, 5)  # Additional noise reduction
                        
                        # Find contours in the mask
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Process contours to detect fires
                        fire_detected = False
                        total_fire_area = 0
                        fire_regions = []
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            total_fire_area += area
                            if area > self.fire_area_threshold:
                                # Large enough fire-colored region detected
                                x, y, w, h = cv2.boundingRect(contour)
                                fire_regions.append((x, y, w, h))
                                fire_detected = True
                        
                        # Only draw bounding boxes on regions that are large enough
                        for x, y, w, h in fire_regions:
                            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            
                            # Draw FIRE text
                            text_size = cv2.getTextSize("FIRE", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
                            text_x = x + (w - text_size[0]) // 2
                            text_y = y + h // 2
                            
                            # Add background for text visibility
                            cv2.rectangle(vis_frame, 
                                        (text_x - 5, text_y - text_size[1] - 5),
                                        (text_x + text_size[0] + 5, text_y + 5),
                                        (0, 0, 0), -1)
                            
                            cv2.putText(vis_frame, "FIRE", (text_x, text_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                        
                        # Draw fire detection information
                        if fire_detected or self.fire_detected:
                            cv2.putText(vis_frame, f"Fire Detection: ACTIVE", 
                                      (20, vis_frame.shape[0] - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(vis_frame, f"Fire Area: {total_fire_area}", 
                                      (20, vis_frame.shape[0] - 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Update fire detection state
                        if fire_detected:
                            self.fire_frames += 1
                            if self.fire_frames >= 5:  # Sustained fire for 5 frames
                                detection_results['fire_detected'] = True
                                self.fire_detected = True
                        else:
                            self.fire_frames = max(0, self.fire_frames - 1)  # Gradually decrease fire counter
                            if self.fire_frames == 0:
                                self.fire_detected = False
                        
                        # Debug view - show the mask
                        # Convert mask to BGR for visualization
                        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        # Create a small version of the mask in the corner
                        h, w = vis_frame.shape[:2]
                        small_h, small_w = h//4, w//4
                        mask_small = cv2.resize(mask_colored, (small_w, small_h))
                        vis_frame[0:small_h, w-small_w:w] = mask_small
                
                # Status bar
                cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], 30), (0, 0, 0), -1)
                active_scenarios = [s.replace('_', ' ').title() for s, v in DETECTION_SCENARIOS.items() if v]
                status_text = f"Active: {', '.join(active_scenarios)}"
                cv2.putText(vis_frame, status_text, (10, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                CAMERA_FRAMES[self.camera_id] = vis_frame
                
                # Generate events
                self.generate_events(detection_results)
                
                return vis_frame, detection_results
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                error_frame = frame.copy()
                cv2.putText(error_frame, f"Error: {str(e)}", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return error_frame, {}
        else:
            error_frame = frame.copy()
            cv2.putText(error_frame, "Model not loaded", (20, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_frame, {}
    
    def update_tracking(self, detected_persons):
        """Update tracking of persons."""
        for person in self.persons:
            person.matched = False
        
        for box, confidence in detected_persons:
            x1, y1, x2, y2 = box
            
            best_match = None
            best_iou = 0.3
            
            for person in self.persons:
                px1, py1, px2, py2 = person.box
                
                x_left = max(x1, px1)
                y_top = max(y1, py1)
                x_right = min(x2, px2)
                y_bottom = min(y2, py2)
                
                if x_right < x_left or y_bottom < y_top:
                    intersection = 0
                else:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (px2 - px1) * (py2 - py1)
                
                union = box1_area + box2_area - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_match = person
                    best_iou = iou
            
            if best_match:
                best_match.update(box, confidence)
                best_match.matched = True
            else:
                if len(self.persons) < self.max_persons:
                    new_person = PersonTracker(box, confidence)
                    new_person.matched = True
                    self.persons.append(new_person)
        
        self.persons = [p for p in self.persons if p.matched or (time.time() - p.last_seen) < 1.0]
    
    def generate_events(self, detection_results):
        """Generate events based on detections."""
        # Person detection events
        if DETECTION_SCENARIOS['person_detection'] and self.cooldowns['person'] == 0:
            person_count = len(detection_results.get('persons_detected', []))
            if person_count > self.last_person_count:
                add_event(self.camera_id.title(), "PERSON", f"{person_count} persons detected", 0.9)
                self.cooldowns['person'] = 30
                self.last_person_count = person_count
        
        # Fall detection events
        if DETECTION_SCENARIOS['fall_detection'] and self.cooldowns['fall'] == 0:
            falls = detection_results.get('falls_detected', [])
            if falls:
                person, confidence = falls[0]
                add_event(self.camera_id.title(), "FALL", f"Fall detected with confidence {confidence:.2f}", confidence)
                self.cooldowns['fall'] = 60
        
        # Fight detection events
        if DETECTION_SCENARIOS['fight_detection'] and self.cooldowns['fight'] == 0:
            fights = detection_results.get('fights_detected', [])
            if fights:
                person, confidence = fights[0]
                add_event(self.camera_id.title(), "FIGHT", f"Fight detected with confidence {confidence:.2f}", confidence)
                self.cooldowns['fight'] = 45
        
        # Crowd detection events
        if DETECTION_SCENARIOS['crowd_detection'] and self.cooldowns['crowd'] == 0:
            if detection_results.get('crowd_detected', False):
                person_count = len(self.persons)
                add_event(self.camera_id.title(), "CROWD", f"Crowd detected: {person_count} people", 0.85)
                self.cooldowns['crowd'] = 120
                
        # Fire detection events
        if DETECTION_SCENARIOS['fire_detection'] and self.cooldowns['fire'] == 0:
            if detection_results.get('fire_detected', False):
                add_event(self.camera_id.title(), "FIRE", f"Fire detected - IMMEDIATE ACTION REQUIRED", 0.95)
                self.cooldowns['fire'] = 90  # Longer cooldown for fire alerts due to severity


def process_camera(camera_id, video_path):
    """Process a camera feed for multiple scenarios."""
    global CAMERA_FRAMES, COUNTERS
    
    print(f"Starting camera processing for {camera_id} with {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        error_frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: Video not found", (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        CAMERA_FRAMES[camera_id] = error_frame
        return
    
    detector = MultiScenarioDetector()
    detector.camera_id = camera_id  # Add camera ID to detector
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            detector = MultiScenarioDetector()
            detector.camera_id = camera_id
            continue
        
        frame = cv2.resize(frame, (640, 480))
        
        COUNTERS['processed_frames'] += 1
        
        processed_frame, detection_results = detector.process_frame(frame)
        
        CAMERA_FRAMES[camera_id] = processed_frame
        
        time.sleep(0.03)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Multi-Scenario Detection System")
    print("="*60)
    print("\nStarting camera threads...")
    
    # Start camera threads for all cameras
    for camera_id, video_path in VIDEO_FILES.items():
        camera_thread = threading.Thread(
            target=process_camera, 
            args=(camera_id, video_path),
            daemon=True
        )
        camera_thread.start()
        print(f"- Started {camera_id} processing {video_path}")
    
    print("\nOpen http://localhost:5000 in your browser to view the dashboard")
    print("\nFeatures:")
    print("- Person Detection: Counts and highlights all detected persons")
    print("- Fall Detection: Detects when a standing person falls down")
    print("- Fight Detection: Detects rapid movement and aggressive interactions")
    print("- Crowd Detection: Alerts when too many people are in the frame")
    print("- Dynamic scenario selection via checkboxes")
    print("\nPress Ctrl+C to quit")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)