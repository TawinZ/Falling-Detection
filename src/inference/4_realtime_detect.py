"""
Real-time Fall Detection with LSTM
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow import keras
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *

# Load models
print("Loading models...")
yolo_model = YOLO('yolov8n-pose.pt')
lstm_model = keras.models.load_model(MODELS_DIR / 'lstm_fall_detector_final.h5')
print("✅ Models loaded!")

# Buffers
keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH + 1)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

def extract_keypoints(frame):
    """Extract keypoints from frame"""
    results = yolo_model(frame, verbose=False)
    
    if len(results[0].keypoints) == 0:
        return None
    
    kp = results[0].keypoints.xy[0].cpu().numpy()
    if len(kp) < 17:
        return None
    
    indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
    selected = kp[indices]
    
    h, w = frame.shape[:2]
    selected[:, 0] /= w
    selected[:, 1] /= h
    
    return selected

def compute_velocity(kp_prev, kp_curr):
    """Compute velocity features"""
    velocity = kp_curr - kp_prev
    velocity_x = velocity[:, 0]
    velocity_y = velocity[:, 1]
    
    hip_velocity_y = velocity_y[3:5].mean()
    
    features = [
        velocity_x.mean(),
        velocity_x.std(),
        velocity_y.mean(),
        velocity_y.std(),
        hip_velocity_y,
        np.abs(velocity_y).mean(),
        np.linalg.norm(velocity, axis=1).mean()
    ]
    
    return np.array(features)

def detect_fall(features_sequence):
    """Detect fall from sequence"""
    if len(features_sequence) < SEQUENCE_LENGTH:
        return False, 0.0
    
    # Prepare input
    seq = np.array(features_sequence[-SEQUENCE_LENGTH:])
    seq = seq.reshape(1, SEQUENCE_LENGTH, -1)
    
    # Predict
    pred = lstm_model.predict(seq, verbose=0)[0]
    fall_prob = pred[1]
    
    # Smoothing
    prediction_buffer.append(fall_prob > FALL_CONFIDENCE_THRESHOLD)
    
    if len(prediction_buffer) == SMOOTHING_WINDOW:
        fall_detected = sum(prediction_buffer) >= SMOOTHING_WINDOW * 0.6
    else:
        fall_detected = False
    
    return fall_detected, fall_prob

def draw_info(frame, fall_detected, confidence, fps):
    """Draw info on frame"""
    h, w = frame.shape[:2]
    
    # Status
    if fall_detected:
        color = (0, 0, 255)
        status = "FALL DETECTED!"
        cv2.rectangle(frame, (0, 0), (w, 100), color, -1)
    else:
        color = (0, 255, 0)
        status = "Normal"
        cv2.rectangle(frame, (0, 0), (w, 100), (50, 50, 50), -1)
    
    cv2.putText(frame, status, (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Info
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Q' to quit | 'R' to reset", (w - 400, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def main():
    print("=" * 70)
    print("🚀 Real-time Fall Detection - LSTM")
    print("=" * 70)
    print("📹 Starting camera...")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera ready!")
    print("\n💡 Controls:")
    print("   Q - Quit")
    print("   R - Reset buffers")
    print("\n" + "=" * 70 + "\n")
    
    features_sequence = []
    fps_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract keypoints
        kp = extract_keypoints(frame)
        
        if kp is not None:
            keypoints_buffer.append(kp)
            
            # Compute velocity
            if len(keypoints_buffer) >= 2:
                velocity = compute_velocity(keypoints_buffer[-2], keypoints_buffer[-1])
                features_sequence.append(velocity)
                
                # Keep only recent
                if len(features_sequence) > SEQUENCE_LENGTH:
                    features_sequence.pop(0)
            
            # Detect fall
            if len(features_sequence) >= SEQUENCE_LENGTH:
                fall_detected, confidence = detect_fall(features_sequence)
            else:
                fall_detected, confidence = False, 0.0
        else:
            fall_detected, confidence = False, 0.0
        
        # FPS
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        
        # Draw
        draw_info(frame, fall_detected, confidence, fps)
        
        cv2.imshow('LSTM Fall Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            keypoints_buffer.clear()
            prediction_buffer.clear()
            features_sequence = []
            print("🔄 Buffers reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Stopped")

if __name__ == "__main__":
    main()