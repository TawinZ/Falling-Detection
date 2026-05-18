"""Real-time fall detection via webcam using YOLOv8-pose + BiLSTM."""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow import keras
import joblib
import time
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *
from src.notification.line_notifier import send_fall_alert

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading models...")
yolo_model  = YOLO('yolov8n-pose.pt')
lstm_model  = keras.models.load_model(MODELS_DIR / MODEL_FINAL)
scaler      = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
print("Models loaded")

# ── Buffers ───────────────────────────────────────────────────────────────────
# Keep one extra keypoint frame so we can compute velocity at every step
keypoints_buffer  = deque(maxlen=SEQUENCE_LENGTH + 1)
features_buffer   = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

SCREENSHOT_DIR = Path(__file__).parent.parent.parent / 'screenshots'
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Keypoint selection: same as training (9 joints from 17-point COCO)
_KP_IDX = [0, 5, 6, 11, 12, 13, 14, 15, 16]


# ── Feature extraction (must match 1_extract_keypoints.py exactly) ────────────

def _absolute_features(kp):
    sh_cx = (kp[1, 0] + kp[2, 0]) / 2
    sh_cy = (kp[1, 1] + kp[2, 1]) / 2
    hip_cx = (kp[3, 0] + kp[4, 0]) / 2
    hip_cy = (kp[3, 1] + kp[4, 1]) / 2
    ankle_cy = (kp[7, 1] + kp[8, 1]) / 2

    torso_dx = sh_cx - hip_cx
    torso_dy = hip_cy - sh_cy
    torso_len = np.hypot(torso_dx, torso_dy) + 1e-6

    torso_cos = torso_dy / torso_len
    torso_sin = abs(torso_dx) / torso_len

    all_x, all_y = kp[:, 0], kp[:, 1]
    bbox_h = all_y.max() - all_y.min()
    bbox_w = all_x.max() - all_x.min() + 1e-6
    aspect = bbox_h / bbox_w

    head_above_hip = hip_cy - kp[0, 1]
    body_height    = abs(ankle_cy - sh_cy)
    knee_asym      = abs(kp[5, 1] - kp[6, 1])

    return np.array([
        torso_cos, torso_sin, aspect,
        hip_cy, sh_cy, head_above_hip, body_height, knee_asym,
    ], dtype=np.float32)


_WEBCAM_FPS = 30.0   # update if your webcam runs at a different fps

def _velocity_features(kp_prev, kp_curr):
    vel = (kp_curr - kp_prev) * _WEBCAM_FPS   # normalise to per-second velocity
    vx, vy = vel[:, 0], vel[:, 1]
    hip_vel_y = vy[3:5].mean()
    return np.array([
        vx.mean(), vx.std(),
        vy.mean(), vy.std(),
        hip_vel_y,
        np.abs(vy).mean(),
        np.linalg.norm(vel, axis=1).mean(),
    ], dtype=np.float32)


_REQUIRED_KP  = [5, 6, 11, 12, 13, 14, 15, 16]
_MIN_VALID_KP = 6


def extract_keypoints(frame):
    """Return normalised (9,2) keypoints, or None if skeleton incomplete."""
    results = yolo_model(frame, verbose=False, conf=0.5, classes=[0])
    if not results[0].keypoints or len(results[0].boxes) == 0:
        return None

    best_kp, best_score = None, 0.0
    for i in range(len(results[0].boxes)):
        kp = results[0].keypoints.xy[i].cpu().numpy()
        if len(kp) < 17:
            continue
        box_conf = results[0].boxes.conf[i].item()

        if results[0].keypoints.conf is not None:
            kp_conf = results[0].keypoints.conf[i].cpu().numpy()
            req_conf = kp_conf[_REQUIRED_KP]
            if (req_conf > 0.3).sum() < _MIN_VALID_KP:
                continue   # joints hidden by furniture → skip
            score = box_conf * 0.3 + req_conf.mean() * 0.7
        else:
            score = box_conf

        if score > best_score:
            best_score = score
            best_kp = kp

    if best_kp is None or best_score < 0.3:
        return None

    sel = best_kp[_KP_IDX].copy()
    h, w = frame.shape[:2]
    sel[:, 0] /= w
    sel[:, 1] /= h
    return sel


def compute_features(kp_prev, kp_curr):
    """Build one normalised 15-dim feature vector (absolute + velocity)."""
    feat = np.concatenate([_absolute_features(kp_curr),
                           _velocity_features(kp_prev, kp_curr)])
    feat[2] = np.clip(feat[2], 0.0, 10.0)   # clip aspect_ratio like in training
    return scaler.transform(feat.reshape(1, -1))[0]


# ── Pose heuristic (for display + false-positive suppression) ─────────────────

def classify_pose(kp, kp_prev=None):
    sh_cy  = (kp[1, 1] + kp[2, 1]) / 2
    hip_cy = (kp[3, 1] + kp[4, 1]) / 2
    kn_cy  = (kp[5, 1] + kp[6, 1]) / 2

    torso_dx = (kp[1, 0] + kp[2, 0]) / 2 - (kp[3, 0] + kp[4, 0]) / 2
    torso_dy = hip_cy - sh_cy
    torso_len = np.hypot(torso_dx, torso_dy) + 1e-6
    vertical_ratio = torso_dy / torso_len  # ~1 standing, ~0 lying

    if vertical_ratio < 0.5:
        return "Lying"

    leg_h   = abs(kn_cy - hip_cy)
    torso_h = abs(sh_cy - hip_cy)

    if kp_prev is not None:
        ankle_move = np.abs(kp[7:9] - kp_prev[7:9]).mean()
        if ankle_move > 0.02 and leg_h > torso_h * 0.6:
            return "Walking"

    if leg_h > torso_h * 0.6:
        return "Standing"
    return "Sitting"


# ── Fall detection ────────────────────────────────────────────────────────────

def detect_fall(pose):
    if len(features_buffer) < SEQUENCE_LENGTH:
        return False, 0.0

    seq = np.array(features_buffer).reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    # Model output is sigmoid scalar P(Fall)
    prob = float(lstm_model.predict(seq, verbose=0)[0][0])

    # Suppress obviously wrong detections
    if pose in ("Standing", "Sitting", "Walking"):
        prob = max(0.0, prob - 0.4)

    prediction_buffer.append(prob > FALL_CONFIDENCE_THRESHOLD)

    if len(prediction_buffer) == SMOOTHING_WINDOW:
        detected = sum(prediction_buffer) >= SMOOTHING_WINDOW * 0.8
    else:
        detected = False

    return detected, float(prob)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def save_screenshot(frame, frame_num, confidence):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"webcam_frame{frame_num:04d}_{ts}_conf{confidence:.2f}.jpg"
    path  = SCREENSHOT_DIR / fname
    cv2.imwrite(str(path), frame)
    print(f"Screenshot saved: {fname}")
    return path


_CONNECTIONS = [(1,2),(1,3),(2,4),(3,4),(3,5),(4,6),(5,7),(6,8),(0,1),(0,2)]

def draw_skeleton(frame, kp):
    if kp is None:
        return
    h, w = frame.shape[:2]
    pts = kp.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    for p in pts:
        cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 255, 255), -1)
    for a, b in _CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame,
                     (int(pts[a, 0]), int(pts[a, 1])),
                     (int(pts[b, 0]), int(pts[b, 1])),
                     (0, 255, 0), 2)


def draw_hud(frame, fall_detected, confidence, fps, frame_num, kp, pose):
    h, w = frame.shape[:2]
    draw_skeleton(frame, kp)

    if fall_detected:
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 255), -1)
        status, color = "FALL DETECTED!", (0, 0, 255)
    else:
        cv2.rectangle(frame, (0, 0), (w, 100), (40, 40, 40), -1)
        palette = {
            "Lying":    (0, 140, 255),
            "Sitting":  (0, 255, 255),
            "Standing": (0, 255, 0),
            "Walking":  (0, 200, 0),
        }
        color  = palette.get(pose, (128, 128, 128))
        status = pose

    cv2.putText(frame, status, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
    cv2.putText(frame, f"Pose: {pose}", (20, h - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Fall conf: {confidence*100:.1f}%", (20, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_num}  FPS: {fps:.1f}", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Q=quit  SPACE=pause", (w - 280, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Real-time Fall Detection (Webcam)")
    print("=" * 70)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    fps_time = time.time()
    fps = 0
    frame_num = 0
    paused = False
    pose = "Unknown"
    fall_prev = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame")
                break

            frame_num += 1
            kp = extract_keypoints(frame)

            if kp is not None:
                keypoints_buffer.append(kp)
                if len(keypoints_buffer) >= 2:
                    feat = compute_features(keypoints_buffer[-2], keypoints_buffer[-1])
                    features_buffer.append(feat)
                    pose = classify_pose(kp, keypoints_buffer[-2])
                else:
                    pose = classify_pose(kp)

                fall_detected, confidence = detect_fall(pose)
            else:
                fall_detected, confidence = False, 0.0
                pose = "Unknown"

            fps = 1.0 / (time.time() - fps_time + 1e-9)
            fps_time = time.time()

            draw_hud(frame, fall_detected, confidence, fps, frame_num, kp, pose)

            if fall_detected and not fall_prev:
                path = save_screenshot(frame, frame_num, confidence)
                if LINE_CHANNEL_ACCESS_TOKEN and LINE_USER_ID:
                    send_fall_alert(
                        screenshot_path=path,
                        confidence=confidence,
                        timestamp=datetime.now(),
                        token=LINE_CHANNEL_ACCESS_TOKEN,
                        user_id=LINE_USER_ID,
                        imgbb_key=IMGBB_API_KEY or None,
                    )
                    print("[LINE] Alert sent to caregiver.")
                else:
                    print("[LINE] Credentials not set — skipping notification.")

            fall_prev = fall_detected

        cv2.imshow('Fall Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
