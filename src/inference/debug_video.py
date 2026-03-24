"""
Debug Video Detection - แสดงทุกข้อมูลเพื่อ Debug
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
    """Extract keypoints with debug info"""
    results = yolo_model(frame, verbose=False, conf=0.5)
    
    num_detections = len(results[0].boxes)
    print(f"  Detections: {num_detections}")
    
    if num_detections > 0:
        for i, box in enumerate(results[0].boxes):
            conf = box.conf.item()
            cls = int(box.cls.item())
            print(f"    #{i}: class={cls}, conf={conf:.2f}")
    
    if len(results[0].keypoints) == 0:
        print("  ❌ No keypoints detected")
        return None
    
    if len(results[0].boxes) > 0:
        confidences = results[0].boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()
        print(f"  ✅ Using detection #{best_idx} (conf={confidences[best_idx]:.2f})")
        kp = results[0].keypoints.xy[best_idx].cpu().numpy()
    else:
        kp = results[0].keypoints.xy[0].cpu().numpy()
    
    if len(kp) < 17:
        print(f"  ❌ Incomplete keypoints: {len(kp)}/17")
        return None
    
    indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
    selected = kp[indices]
    
    h, w = frame.shape[:2]
    selected[:, 0] /= w
    selected[:, 1] /= h
    
    print(f"  ✅ Keypoints extracted: {selected.shape}")
    
    return selected


def compute_velocity(kp_prev, kp_curr):
    """Compute velocity with debug"""
    velocity = kp_curr - kp_prev
    velocity_x = velocity[:, 0]
    velocity_y = velocity[:, 1]
    
    hip_velocity_y = velocity_y[3:5].mean()
    
    print(f"  Velocity: vx={velocity_x.mean():.4f}, vy={velocity_y.mean():.4f}, hip_vy={hip_velocity_y:.4f}")
    
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


def classify_pose(keypoints, prev_keypoints=None):
    """Classify pose with debug"""
    if keypoints is None:
        return "Unknown"
    
    shoulder_y = (keypoints[1, 1] + keypoints[2, 1]) / 2
    hip_y = (keypoints[3, 1] + keypoints[4, 1]) / 2
    knee_y = (keypoints[5, 1] + keypoints[6, 1]) / 2
    
    body_height = abs(hip_y - shoulder_y)
    leg_height = abs(knee_y - hip_y)
    
    body_vertical = body_height / (body_height + abs(keypoints[3, 0] - keypoints[1, 0]) + 0.001)
    
    print(f"  Pose ratios: body_vertical={body_vertical:.2f}, leg/body={leg_height/body_height:.2f}")
    
    # Lying
    if body_vertical < 0.6:
        pose = "Lying"
    # Walking detection
    elif prev_keypoints is not None:
        ankle_movement = np.abs(keypoints[7:9] - prev_keypoints[7:9]).mean()
        print(f"  Ankle movement: {ankle_movement:.4f}")
        if ankle_movement > 0.02 and leg_height > body_height * 0.6:
            pose = "Walking"
        elif leg_height > body_height * 0.6:
            pose = "Standing"
        else:
            pose = "Sitting"
    # Standing
    elif leg_height > body_height * 0.6:
        pose = "Standing"
    # Sitting
    else:
        pose = "Sitting"
    
    print(f"  → Pose: {pose}")
    
    return pose


def detect_fall(features_sequence, pose):
    """Detect fall with debug"""
    if len(features_sequence) < SEQUENCE_LENGTH:
        print(f"  Sequence incomplete: {len(features_sequence)}/{SEQUENCE_LENGTH}")
        return False, 0.0
    
    seq = np.array(features_sequence[-SEQUENCE_LENGTH:])
    seq = seq.reshape(1, SEQUENCE_LENGTH, -1)
    
    pred = lstm_model.predict(seq, verbose=0)[0]
    fall_prob_original = pred[1]
    
    print(f"  LSTM Prediction: Not Fall={pred[0]:.3f}, Fall={fall_prob_original:.3f}")
    
    # Filter
    fall_prob = fall_prob_original
    if pose in ["Sitting", "Walking", "Standing"]:
        fall_prob = max(0, fall_prob - 0.7)
        print(f"  Pose Filter Applied: {fall_prob_original:.3f} → {fall_prob:.3f}")
    
    prediction_buffer.append(fall_prob > FALL_CONFIDENCE_THRESHOLD)
    
    if len(prediction_buffer) == SMOOTHING_WINDOW:
        fall_count = sum(prediction_buffer)
        fall_detected = fall_count >= SMOOTHING_WINDOW * 0.8
        print(f"  Smoothing: {fall_count}/{SMOOTHING_WINDOW} (need {SMOOTHING_WINDOW*0.8:.0f}) → {'FALL' if fall_detected else 'Normal'}")
    else:
        fall_detected = False
        print(f"  Smoothing buffer: {len(prediction_buffer)}/{SMOOTHING_WINDOW}")
    
    return fall_detected, fall_prob


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str, help='Path to video file')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print("=" * 70)
    print("🐛 DEBUG MODE - Fall Detection")
    print("=" * 70)
    print(f"📹 Video: {video_path.name}")
    print(f"📊 Confidence Threshold: {FALL_CONFIDENCE_THRESHOLD}")
    print(f"📊 Smoothing Window: {SMOOTHING_WINDOW}")
    print("=" * 70 + "\n")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    features_sequence = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        print(f"\n{'='*70}")
        print(f"Frame {frame_num}/{total_frames}")
        print(f"{'='*70}")
        
        kp = extract_keypoints(frame)
        
        if kp is not None:
            keypoints_buffer.append(kp)
            
            # Classify pose
            if len(keypoints_buffer) >= 2:
                pose = classify_pose(kp, keypoints_buffer[-2])
            else:
                pose = classify_pose(kp)
            
            # Compute velocity
            if len(keypoints_buffer) >= 2:
                velocity = compute_velocity(keypoints_buffer[-2], keypoints_buffer[-1])
                features_sequence.append(velocity)
                
                if len(features_sequence) > SEQUENCE_LENGTH:
                    features_sequence.pop(0)
            
            # Detect fall
            if len(features_sequence) >= SEQUENCE_LENGTH:
                fall_detected, confidence = detect_fall(features_sequence, pose)
                print(f"\n🎯 RESULT: {'🔴 FALL DETECTED' if fall_detected else '🟢 Normal'} (conf={confidence:.3f})")
            else:
                print(f"\n⏳ Buffering sequences: {len(features_sequence)}/{SEQUENCE_LENGTH}")
        
        cv2.imshow('Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Debug finished")


if __name__ == "__main__":
    main()