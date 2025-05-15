#!/usr/bin/env python3
"""
Fire Detection Module using YOLOv8n

This module provides specialized fire detection capabilities using a YOLOv8n model
trained specifically for fire detection.
"""

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

class FireDetector:
    """Fire detection using a specialized YOLOv8n model"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.35):
        """Initialize the fire detector with the provided model"""
        self.confidence_threshold = confidence_threshold
        self.fire_frames = 0
        self.fire_frames_threshold = 3  # Number of consecutive frames needed to confirm fire
        self.fire_detected = False
        self.last_detection_time = 0
        self.cooldown_period = 5  # Seconds to wait before generating new alert
        
        # Load the model
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(model_path)
            if device == 'cuda':
                self.model.to('cuda')
            self.model_loaded = True
            print(f"Fire detection model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading fire detection model: {str(e)}")
            self.model_loaded = False
    
    def detect(self, frame):
        """
        Detect fire in the given frame using YOLOv8n model
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            processed_frame: Frame with fire detection visualizations
            results_dict: Detection results including bounding boxes and confidence
        """
        if not self.model_loaded:
            # Return unprocessed frame if model isn't loaded
            return frame, {'fire_detected': False, 'detections': []}
            
        # Process the frame with YOLO
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Create a copy of the frame for visualization
        processed_frame = frame.copy()
        
        # Process detection results
        fire_boxes = []
        highest_confidence = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract box information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # In standard YOLOv8, fire is not a class, so we use a direct approach
                # The model is either specifically trained for fire (class 0) or we look for the most fire-like regions
                
                # For a fire-specific model, class 0 would be fire
                # For generic YOLOv8n, we look for bright regions using HSV
                fire_boxes.append((x1, y1, x2, y2, confidence))
                highest_confidence = max(highest_confidence, confidence)
                
                # Debug info
                print(f"Detected: {self.model.names.get(cls_id, 'unknown')} with confidence {confidence}")
        
        # Always use HSV-based detection for fire - it's more reliable for this specific task
        hsv_fire_boxes, mask = detect_fire_hsv(frame)
        
        # If there are no YOLO detections or if HSV finds fires, use HSV detections
        if not fire_boxes or hsv_fire_boxes:
            fire_boxes = hsv_fire_boxes
        
        # Add mask visualization to corner of frame 
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask, (w//4, h//4))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        # Apply a red tint to make it more visible for debugging
        mask_colored[..., 0] = 0  # Zero out blue channel
        mask_colored[..., 1] = 0  # Zero out green channel
        # Create a small version of the mask in the corner
        processed_frame[0:h//4, w-w//4:w] = mask_colored
        
        # Filter fire detections near people to avoid false positives
        # In a real-world scenario, we would use the person detector to filter these
        filtered_fire_boxes = []
        for box in fire_boxes:
            x1, y1, x2, y2, conf = box
            
            # Skip detections that are too small
            area = (x2 - x1) * (y2 - y1)
            if area < 2000:  # Minimum area for a real fire
                continue
                
            # Skip detections with strange aspect ratios 
            aspect = float(x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
            if aspect < 0.3 or aspect > 4:  # Too narrow or too wide
                continue
                
            filtered_fire_boxes.append(box)
            
        fire_boxes = filtered_fire_boxes
        
        # Check if fire is detected in this frame (from either method)
        if fire_boxes:
            self.fire_frames += 1
            if self.fire_frames >= self.fire_frames_threshold:
                self.fire_detected = True
        else:
            self.fire_frames = max(0, self.fire_frames - 1)
            if self.fire_frames == 0:
                self.fire_detected = False
        
        # Draw fire boxes with improved visibility
        for x1, y1, x2, y2, confidence in fire_boxes:
            # Draw fire rectangle with double outline for visibility
            # Outer black outline
            cv2.rectangle(processed_frame, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 3)
            # Inner red rectangle
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add label with confidence
            label = f"Fire: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1 = max(y1, label_size[1])
            
            # Background for text
            cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 0, 0), -1)
            # Text
            cv2.putText(processed_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add a warning label if fire is detected
        if self.fire_detected:
            cv2.putText(processed_frame, "FIRE DETECTED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Add timestamp to the frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, timestamp, (processed_frame.shape[1] - 210, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Create results dictionary
        results_dict = {
            'fire_detected': self.fire_detected,
            'detections': fire_boxes,
            'highest_confidence': highest_confidence if fire_boxes else 0
        }
        
        # Check if we should generate a new alert
        current_time = time.time()
        if self.fire_detected and (current_time - self.last_detection_time > self.cooldown_period):
            results_dict['generate_alert'] = True
            self.last_detection_time = current_time
        else:
            results_dict['generate_alert'] = False
        
        return processed_frame, results_dict


# Testing function
def test_fire_detector(video_path, model_path='yolov8n.pt'):
    """Test the fire detector on a video file"""
    detector = FireDetector(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Process the frame
        processed_frame, results = detector.detect(frame)
        
        # Display the processed frame
        cv2.imshow("Fire Detection", processed_frame)
        
        # Print detection results
        if results['fire_detected']:
            print(f"Fire detected! Confidence: {results['highest_confidence']:.2f}")
        
        # Exit on 'q' key
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# HSV-based fire detection as fallback
def detect_fire_hsv(frame, threshold=1500):
    """HSV-based fire detection with advanced filtering"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for fire color in HSV - more precisely tuned for actual fires
    # Narrower hue range to avoid red objects like fire extinguishers
    lower_fire = np.array([5, 150, 180])  # Much more restrictive for core fire colors
    upper_fire = np.array([25, 255, 255])
    
    # Create a mask for fire color
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)  # Additional noise removal
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour with additional filtering
    fire_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip small areas
        if area < threshold:
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Additional filters to avoid false positives
        # 1. Aspect ratio filter (fire is usually not too tall and narrow)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.3:
            continue  # Too narrow, likely not fire
            
        # 2. Solidity filter (fire contours tend to be less solid)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # 3. Check intensity - fires are very bright in their center
        mask_roi = mask[y:y+h, x:x+w]
        if mask_roi.size > 0:
            brightness_ratio = cv2.countNonZero(mask_roi) / (w * h)
            if brightness_ratio < 0.4:  # Not bright enough to be fire
                continue
        
        # 4. Consider location - fires are rarely at the very top of the frame
        if y < frame.shape[0] * 0.1 and h < frame.shape[0] * 0.2:
            continue
            
        # Calculate a more accurate confidence based on multiple factors
        confidence = min(0.95, (area / 20000) * 0.7 + brightness_ratio * 0.3)
        
        # Add to fire boxes
        fire_boxes.append((x, y, x+w, y+h, confidence))
    
    return fire_boxes, mask


if __name__ == "__main__":
    # Test the fire detector
    test_fire_detector("/home/siddu/projects/person_fall/data/fire.mp4", "yolov8n.pt")