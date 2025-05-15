#!/usr/bin/env python3
"""
Test script for fire detection to validate and debug the approach
"""

import cv2
import numpy as np
import time
import os

# Define a simple HSV-based fire detector for testing
def detect_fire_hsv(frame):
    """Detect fire using HSV color thresholding"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for fire color in HSV
    lower_fire = np.array([0, 70, 150])  # More restrictive
    upper_fire = np.array([35, 255, 255])
    
    # Create a mask for fire color
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    
    # Apply erosion and dilation to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    fire_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(0.9, area / 20000)  # Normalize confidence
            fire_boxes.append((x, y, x+w, y+h, confidence))
    
    return fire_boxes, mask

def test_fire_detection(video_path):
    """Test fire detection on a video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    print(f"Testing fire detection on {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for consistent display
        frame = cv2.resize(frame, (640, 480))
        
        # Detect fire using HSV
        fire_boxes, fire_mask = detect_fire_hsv(frame)
        
        # Draw fire boxes
        vis_frame = frame.copy()
        for x1, y1, x2, y2, confidence in fire_boxes:
            # Draw red rectangle for fire
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add label with confidence
            label = f"Fire: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add a warning if fire detected
        if fire_boxes:
            cv2.putText(vis_frame, "FIRE DETECTED!", (20, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Create a mask display in corner
        mask_small = cv2.resize(fire_mask, (160, 120))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        mask_color[..., 0] = 0  # Remove blue channel
        mask_color[..., 1] = 0  # Remove green channel
        vis_frame[0:120, 480:640] = mask_color
        
        # Display frame
        cv2.imshow("Fire Detection Test", vis_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/siddu/projects/person_fall/data/fire.mp4"
    test_fire_detection(video_path)