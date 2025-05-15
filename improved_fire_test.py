#!/usr/bin/env python3
"""
Improved fire detection test script with enhanced filtering
"""

import cv2
import numpy as np
import time
import os

def improved_detect_fire_hsv(frame, threshold=1500):
    """Enhanced HSV-based fire detection with advanced filtering"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for fire color in HSV - more precisely tuned
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
        if aspect_ratio < 0.3 or aspect_ratio > 4:
            continue  # Unusual shape, likely not fire
            
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

def test_improved_fire_detection(video_path):
    """Test improved fire detection on a video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    print(f"Testing improved fire detection on {video_path}")
    
    # Create window
    cv2.namedWindow("Improved Fire Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Improved Fire Detection", 1280, 720)
    
    # Track frames with fire
    fire_frames = 0
    total_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # If end of video, loop back
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        total_frames += 1
        
        # Resize for consistent display
        frame = cv2.resize(frame, (640, 480))
        
        # Detect fire using improved HSV
        fire_boxes, fire_mask = improved_detect_fire_hsv(frame)
        
        # Draw fire boxes with improved visibility
        vis_frame = frame.copy()
        for x1, y1, x2, y2, confidence in fire_boxes:
            # Draw fire rectangle with double outline for visibility
            # Outer black outline
            cv2.rectangle(vis_frame, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 3)
            # Inner red rectangle
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add label with confidence
            label = f"Fire: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1 = max(y1, label_size[1])
            
            # Background for text
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 0, 0), -1)
            # Text
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add a warning if fire detected
        if fire_boxes:
            fire_frames += 1
            cv2.putText(vis_frame, "FIRE DETECTED!", (20, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(vis_frame, timestamp, (vis_frame.shape[1] - 210, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Create a mask display in corner
        mask_small = cv2.resize(fire_mask, (160, 120))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        mask_color[..., 0] = 0  # Remove blue channel
        mask_color[..., 1] = 0  # Remove green channel
        vis_frame[0:120, 480:640] = mask_color
        
        # Add detection stats
        cv2.rectangle(vis_frame, (0, 0), (250, 30), (0, 0, 0), -1)
        detection_rate = (fire_frames / total_frames) * 100 if total_frames > 0 else 0
        stats_text = f"Fire detection rate: {detection_rate:.1f}%"
        cv2.putText(vis_frame, stats_text, (10, 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow("Improved Fire Detection", vis_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/siddu/projects/person_fall/data/fire.mp4"
    test_improved_fire_detection(video_path)