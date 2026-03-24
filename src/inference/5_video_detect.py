"""
Fall Detection from Video File with Screenshot Capture
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow import keras
import time
from pathlib import Path
from datetime import datetime
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

# Screenshot directory
SCREENSHOT_DIR = Path(__file__).parent.parent.parent / 'screenshots'
SCREENSHOT_DIR.mkdir(exist_ok=True)


def save_screenshot(frame, video_name, frame_num, confidence):
    """Save screenshot when fall detected"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{video_name}_frame{frame_num:04d}_{timestamp}_conf{confidence:.2f}.jpg"
    filepath = SCREENSHOT_DIR / filename
    
    cv2.imwrite(str(filepath), frame)
    print(f"📸 Screenshot saved: {filename}")
    
    return filepath


def extract_keypoints(frame):
    """Extract keypoints - strict person-only filter"""
    # เพิ่ม conf เป็น 0.7
    results = yolo_model(frame, verbose=False, conf=0.7, classes=[0])
    
    if len(results[0].keypoints) == 0:
        return None
    
    # เลือกคนที่ดีที่สุด
    best_kp = None
    best_score = 0
    
    for i in range(len(results[0].boxes)):
        if len(results[0].keypoints.xy[i]) < 17:
            continue
        
        kp = results[0].keypoints.xy[i].cpu().numpy()
        box_conf = results[0].boxes.conf[i].item()
        
        # เช็ค keypoint confidence
        if hasattr(results[0].keypoints, 'conf') and results[0].keypoints.conf is not None:
            kp_conf = results[0].keypoints.conf[i].cpu().numpy()
            
            # keypoints สำคัญ: hip, knee, ankle, shoulder
            important_indices = [5, 6, 11, 12, 13, 14, 15, 16]
            important_kp_conf = kp_conf[important_indices]
            
            # ต้องมี keypoint ครบอย่างน้อย 6/8 จุด (conf > 0.3)
            valid_kp_count = (important_kp_conf > 0.3).sum()
            if valid_kp_count < 6:
                continue  # ข้าม detection นี้
            
            # คะแนนรวม (เน้น keypoint มากขึ้น)
            avg_kp_conf = important_kp_conf.mean()
            score = box_conf * 0.3 + avg_kp_conf * 0.7
        else:
            score = box_conf
        
        # ต้องมีคะแนนสูงกว่า 0.5
        if score > 0.5 and score > best_score:
            best_score = score
            best_kp = kp
    
    if best_kp is None:
        return None
    
    indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
    selected = best_kp[indices]
    
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


def classify_pose(keypoints, prev_keypoints=None):
    """Classify pose with walking detection"""
    if keypoints is None:
        return "Unknown"
    
    shoulder_y = (keypoints[1, 1] + keypoints[2, 1]) / 2
    hip_y = (keypoints[3, 1] + keypoints[4, 1]) / 2
    knee_y = (keypoints[5, 1] + keypoints[6, 1]) / 2
    
    body_height = abs(hip_y - shoulder_y)
    leg_height = abs(knee_y - hip_y)
    
    body_vertical = body_height / (body_height + abs(keypoints[3, 0] - keypoints[1, 0]) + 0.001)
    
    if body_vertical < 0.6:
        return "Lying"
    
    if prev_keypoints is not None:
        ankle_movement = np.abs(keypoints[7:9] - prev_keypoints[7:9]).mean()
        if ankle_movement > 0.02 and leg_height > body_height * 0.6:
            return "Walking"
    
    if leg_height > body_height * 0.6:
        return "Standing"
    
    return "Sitting"


def detect_fall(features_sequence, pose):
    """Detect fall with strict pose filter"""
    if len(features_sequence) < SEQUENCE_LENGTH:
        return False, 0.0
    
    seq = np.array(features_sequence[-SEQUENCE_LENGTH:])
    seq = seq.reshape(1, SEQUENCE_LENGTH, -1)
    
    pred = lstm_model.predict(seq, verbose=0)[0]
    fall_prob = pred[1]
    
    if pose in ["Sitting", "Walking", "Standing"]:
        fall_prob = max(0, fall_prob - 0.7)
    
    prediction_buffer.append(fall_prob > FALL_CONFIDENCE_THRESHOLD)
    
    if len(prediction_buffer) == SMOOTHING_WINDOW:
        fall_detected = sum(prediction_buffer) >= SMOOTHING_WINDOW * 0.8
    else:
        fall_detected = False
    
    return fall_detected, fall_prob


def draw_skeleton(frame, keypoints):
    """Draw skeleton on frame"""
    if keypoints is None:
        return
    
    h, w = frame.shape[:2]
    kp_scaled = keypoints.copy()
    kp_scaled[:, 0] *= w
    kp_scaled[:, 1] *= h
    
    for i, kp in enumerate(kp_scaled):
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)
    
    connections = [
        (1, 2), (1, 3), (2, 4), (3, 4),
        (3, 5), (4, 6), (5, 7), (6, 8),
        (0, 1), (0, 2),
    ]
    
    for conn in connections:
        if conn[0] < len(kp_scaled) and conn[1] < len(kp_scaled):
            pt1 = (int(kp_scaled[conn[0]][0]), int(kp_scaled[conn[0]][1]))
            pt2 = (int(kp_scaled[conn[1]][0]), int(kp_scaled[conn[1]][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)


def draw_bounding_box(frame, keypoints):
    """Draw bounding box around person"""
    if keypoints is None:
        return
    
    h, w = frame.shape[:2]
    kp_scaled = keypoints.copy()
    kp_scaled[:, 0] *= w
    kp_scaled[:, 1] *= h
    
    x_min = int(kp_scaled[:, 0].min())
    x_max = int(kp_scaled[:, 0].max())
    y_min = int(kp_scaled[:, 1].min())
    y_max = int(kp_scaled[:, 1].max())
    
    padding = 20
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.rectangle(frame, (x_min, y_min - 30), (x_min + 100, y_min), (255, 0, 0), -1)
    cv2.putText(frame, "Person", (x_min + 5, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_info(frame, fall_detected, confidence, fps, frame_num, total_frames, keypoints, pose):
    """Draw info on frame"""
    h, w = frame.shape[:2]
    
    draw_skeleton(frame, keypoints)
    draw_bounding_box(frame, keypoints)
    
    # Status bar - แสดงท่าทุกเวลา
    if fall_detected:
        color = (0, 0, 255)
        status = "FALL DETECTED!"
        cv2.rectangle(frame, (0, 0), (w, 100), color, -1)
    else:
        # แสดงท่าปกติ
        if pose == "Lying":
            color = (255, 165, 0)  # ส้ม
            status = "Lying Down"
        elif pose == "Sitting":
            color = (0, 255, 255)  # ฟ้า
            status = "Sitting"
        elif pose == "Standing":
            color = (0, 255, 0)    # เขียว
            status = "Standing"
        elif pose == "Walking":
            color = (0, 200, 0)    # เขียวเข้ม
            status = "Walking"
        else:
            color = (128, 128, 128)  # เทา
            status = "Unknown"
        
        cv2.rectangle(frame, (0, 0), (w, 100), (50, 50, 50), -1)
    
    cv2.putText(frame, status, (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # แสดงท่าด้านล่าง (ใหญ่ชัดเจน)
    cv2.putText(frame, f"Pose: {pose}", (20, h - 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(frame, f"Fall Confidence: {confidence*100:.1f}%", (20, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (20, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Q' to quit | SPACE to pause", (w - 450, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


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
    print("🎥 Fall Detection from Video")
    print("=" * 70)
    print(f"📹 Video: {video_path.name}")
    print(f"📸 Screenshots will be saved to: {SCREENSHOT_DIR}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Total frames: {total_frames}")
    print(f"✅ FPS: {fps_original:.1f}")
    print("\n💡 Controls:")
    print("   Q - Quit")
    print("   SPACE - Pause/Resume")
    print("\n" + "=" * 70 + "\n")
    
    features_sequence = []
    fps_time = time.time()
    fps = 0
    frame_num = 0
    paused = False
    pose = "Unknown"
    
    # Track fall detection
    fall_detected_prev = False
    screenshot_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video finished")
                print(f"📸 Total screenshots: {screenshot_count}")
                break
            
            frame_num += 1
            
            kp = extract_keypoints(frame)
            
            if kp is not None:
                keypoints_buffer.append(kp)
                
                if len(keypoints_buffer) >= 2:
                    pose = classify_pose(kp, keypoints_buffer[-2])
                else:
                    pose = classify_pose(kp)
                
                if len(keypoints_buffer) >= 2:
                    velocity = compute_velocity(keypoints_buffer[-2], keypoints_buffer[-1])
                    features_sequence.append(velocity)
                    
                    if len(features_sequence) > SEQUENCE_LENGTH:
                        features_sequence.pop(0)
                
                if len(features_sequence) >= SEQUENCE_LENGTH:
                    fall_detected, confidence = detect_fall(features_sequence, pose)
                else:
                    fall_detected, confidence = False, 0.0
            else:
                fall_detected, confidence = False, 0.0
                pose = "Unknown"
            
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            
            draw_info(frame, fall_detected, confidence, fps, frame_num, total_frames, kp, pose)
            
            # Save screenshot when fall first detected
            if fall_detected and not fall_detected_prev:
                save_screenshot(frame, video_path.stem, frame_num, confidence)
                screenshot_count += 1
            
            fall_detected_prev = fall_detected
        
        cv2.imshow('Video Fall Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("⏸️  Paused")
            else:
                print("▶️  Resumed")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
