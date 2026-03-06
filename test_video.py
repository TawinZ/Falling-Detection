import sys
sys.path.append(".")

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from ai.pose_estimator import PoseEstimator

# ─── Config ───────────────────────────────────────────────
MODEL_PATH    = "models/cnn_fall_detector.h5"
VIDEO_PATH    = "realfall.MOV"        # เปลี่ยนเป็นชื่อไฟล์วิดีโอของคุณ
SEQUENCE_LEN      = 10
FALL_THRESHOLD    = 0.65
SMOOTH_WINDOW     = 5
PREDICT_EVERY     = 3
NO_PERSON_MAX     = 15
FALL_CONFIRM_CNT  = 5   # ต้องตรวจพบ Fall ติดกันกี่ครั้งถึงแจ้งเตือน
KEYPOINT_NAMES = [
    "nose", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# เส้นเชื่อม keypoints (skeleton)
SKELETON = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("right_hip",      "right_knee"),
    ("left_knee",      "left_ankle"),
    ("right_knee",     "right_ankle"),
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
]

# ─── Load model ───────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
estimator = PoseEstimator()
print("Model loaded.")

# ─── Open video ───────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

fps    = cap.get(cv2.CAP_PROP_FPS) or 30
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h} | FPS: {fps:.1f} | Frames: {total}")

seq_buf        = deque(maxlen=SEQUENCE_LEN)
conf_buf       = deque(maxlen=SMOOTH_WINDOW)
label          = "Analyzing..."
conf           = 0.0
is_fall        = False
frame_idx      = 0
no_person_cnt  = 0
fall_counter   = 0   # นับ Fall ติดกัน

def draw_skeleton(frame, kp_dict, color):
    """วาด skeleton บนเฟรม"""
    pts = {}
    for name, (nx, ny) in kp_dict.items():
        px, py = int(nx * w), int(ny * h)
        pts[name] = (px, py)
        cv2.circle(frame, (px, py), 5, color, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2)

def draw_hud(frame, label, conf, is_fall, frame_idx, total):
    """วาด HUD บนเฟรม"""
    overlay = frame.copy()

    # Background bar ด้านบน
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Label + Confidence
    color = (0, 0, 255) if is_fall else (0, 255, 0)
    status = f"Status: {label}"
    conf_text = f"Confidence: {conf*100:.1f}%"
    cv2.putText(frame, status,    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, conf_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 15, 200, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    filled = int(bar_w * conf)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

    # Frame counter
    progress = f"Frame: {frame_idx}/{total}"
    cv2.putText(frame, progress, (w - 180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Flash alert ถ้า Fall
    if is_fall:
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(alert_overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "⚠ FALL DETECTED", (w//2 - 160, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# ─── Main loop ────────────────────────────────────────────
print("Press 'q' to quit | Space to pause")
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Extract keypoints
        pose_data = estimator.analyze(frame)
        if pose_data and "keypoints" in pose_data:
            no_person_cnt = 0
            kp = pose_data["keypoints"]
            kp_arr = []
            for name in KEYPOINT_NAMES:
                x, y = kp[name]
                kp_arr.extend([x, y])
            seq_buf.append(kp_arr)

            color = (0, 0, 255) if is_fall else (0, 255, 0)
            draw_skeleton(frame, kp, color)
        else:
            no_person_cnt += 1
            if no_person_cnt >= NO_PERSON_MAX:
                # ไม่เจอคนนานพอ → reset เป็น Normal
                label, conf, is_fall = "Normal", 0.0, False
                conf_buf.clear()
                seq_buf.clear()
            elif seq_buf:
                seq_buf.append(seq_buf[-1])
            else:
                seq_buf.append([0.0] * 18)

        # Predict ทุก PREDICT_EVERY frame เพื่อลด slow motion
        if len(seq_buf) == SEQUENCE_LEN and frame_idx % PREDICT_EVERY == 0:
            X = np.array([list(seq_buf)], dtype=np.float32)
            pred = model.predict(X, verbose=0)[0]
            conf_buf.append(float(pred[1]))
            conf         = sum(conf_buf) / len(conf_buf)
            raw_is_fall  = conf >= FALL_THRESHOLD

            # Consecutive check
            if raw_is_fall:
                fall_counter += 1
            else:
                fall_counter = 0

            is_fall = fall_counter >= FALL_CONFIRM_CNT
            label   = "FALL" if is_fall else "Normal"

        draw_hud(frame, label, conf, is_fall, frame_idx, total)
        cv2.imshow("Fall Detection Test", frame)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
print("Done.")