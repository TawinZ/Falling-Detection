"""
Dual-camera fall detection from two video files.

Setup: two cameras placed at 90-degree angles, both pointing at the room center.
Fusion: soft voting — average P(Fall) from both cameras each frame.

Usage:
    python3 src/inference/dual_video_detect.py \\
        --cam_a path/to/video_a.mp4 \\
        --cam_b path/to/video_b.mp4

Optional flags:
    --threshold   fall probability threshold (default: 0.4)
    --output      save result video to this path
    --single      run only cam_a (single camera mode, no fusion)
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow import keras
import joblib
import argparse
import time
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import *

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
yolo  = YOLO('yolov8n-pose.pt')
lstm  = keras.models.load_model(MODELS_DIR / MODEL_BEST)
scaler = joblib.load(MODELS_DIR / 'feature_scaler.pkl')
print("Ready")

_KP_IDX = [0, 5, 6, 11, 12, 13, 14, 15, 16]


# ── Feature extraction ────────────────────────────────────────────────────────

_REQUIRED_KP  = [5, 6, 11, 12, 13, 14, 15, 16]
_MIN_VALID_KP = 6


def extract_keypoints(frame):
    results = yolo(frame, verbose=False, conf=0.4, classes=[0])
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
                continue   # joints hidden → skip
            score = box_conf * 0.3 + req_conf.mean() * 0.7
        else:
            score = box_conf
        if score > best_score:
            best_score, best_kp = score, kp
    if best_kp is None or best_score < 0.3:
        return None
    sel = best_kp[_KP_IDX].copy()
    h, w = frame.shape[:2]
    sel[:, 0] /= w
    sel[:, 1] /= h
    return sel


def absolute_features(kp):
    sh_cx = (kp[1,0]+kp[2,0])/2;  sh_cy = (kp[1,1]+kp[2,1])/2
    hip_cx= (kp[3,0]+kp[4,0])/2;  hip_cy= (kp[3,1]+kp[4,1])/2
    ankle_cy = (kp[7,1]+kp[8,1])/2
    tdx = sh_cx-hip_cx;  tdy = hip_cy-sh_cy
    tlen = np.hypot(tdx,tdy)+1e-6
    all_x,all_y = kp[:,0],kp[:,1]
    bbox_h = all_y.max()-all_y.min()
    bbox_w = all_x.max()-all_x.min()+1e-6
    aspect = np.clip(bbox_h/bbox_w, 0, 10)
    return np.array([tdy/tlen, abs(tdx)/tlen, aspect,
                     hip_cy, sh_cy, hip_cy-kp[0,1],
                     abs(ankle_cy-sh_cy), abs(kp[5,1]-kp[6,1])], dtype=np.float32)


def velocity_features(kp_prev, kp_curr, fps=30.0):
    vel = (kp_curr-kp_prev)*fps   # normalise to per-second velocity
    vx,vy = vel[:,0],vel[:,1]
    return np.array([vx.mean(),vx.std(),vy.mean(),vy.std(),
                     vy[3:5].mean(), np.abs(vy).mean(),
                     np.linalg.norm(vel,axis=1).mean()], dtype=np.float32)


def make_feature(kp_prev, kp_curr, fps=30.0):
    feat = np.concatenate([absolute_features(kp_curr),
                           velocity_features(kp_prev, kp_curr, fps=fps)])
    feat[2] = np.clip(feat[2], 0.0, 10.0)
    return scaler.transform(feat.reshape(1,-1))[0]


# ── Per-camera state ──────────────────────────────────────────────────────────

class CameraState:
    def __init__(self, name, fps=30.0):
        self.name = name
        self.fps   = fps
        self.kp_buf   = deque(maxlen=SEQUENCE_LENGTH+1)
        self.feat_buf  = deque(maxlen=SEQUENCE_LENGTH)
        self.pred_buf  = deque(maxlen=SMOOTHING_WINDOW)
        self.last_prob = 0.0

    def process(self, frame):
        """Return P(Fall) for this frame, or None if not enough data yet."""
        kp = extract_keypoints(frame)
        if kp is None:
            self.pred_buf.append(0.0)
            return None

        self.kp_buf.append(kp)
        if len(self.kp_buf) >= 2:
            feat = make_feature(self.kp_buf[-2], self.kp_buf[-1], fps=self.fps)
            self.feat_buf.append(feat)

        if len(self.feat_buf) < SEQUENCE_LENGTH:
            return None

        seq = np.array(self.feat_buf).reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
        prob = float(lstm.predict(seq, verbose=0)[0][0])
        self.last_prob = prob
        self.pred_buf.append(prob)
        return prob


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse(prob_a, prob_b):
    """Soft vote: average available probabilities."""
    probs = [p for p in [prob_a, prob_b] if p is not None]
    return sum(probs)/len(probs) if probs else 0.0


def smooth_decision(fused_prob, smooth_buf, threshold):
    smooth_buf.append(fused_prob)
    if len(smooth_buf) < SMOOTHING_WINDOW:
        return False
    votes = sum(1 for p in smooth_buf if p > threshold)
    return votes >= int(SMOOTHING_WINDOW * 0.8)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_frame(frame, cam_name, prob, fall):
    h, w = frame.shape[:2]
    bar_color = (0,0,255) if fall else (40,40,40)
    cv2.rectangle(frame, (0,0), (w,80), bar_color, -1)
    label = f"CAM {cam_name}  P(Fall)={prob*100:.1f}%"
    cv2.putText(frame, label, (10,55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    return frame


def draw_fusion(panel, prob_a, prob_b, fused, fall, frame_num):
    h, w = panel.shape[:2]
    panel[:] = (20,20,20)

    title_color = (0,0,255) if fall else (200,200,200)
    status = "!!! FALL DETECTED !!!" if fall else "Monitoring..."
    cv2.putText(panel, status, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, title_color, 3)

    def bar(y, label, val, color):
        bw = int((w-160)*val)
        cv2.rectangle(panel, (120,y), (120+bw, y+28), color, -1)
        cv2.putText(panel, f"{label}: {val*100:.1f}%", (10,y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 1)

    bar(80,  "Cam A",  prob_a,  (255,140,0))
    bar(120, "Cam B",  prob_b,  (0,200,255))
    bar(165, "Fused",  fused,   (0,255,0) if not fall else (0,0,255))

    cv2.line(panel, (120+int((w-160)*FALL_CONFIDENCE_THRESHOLD),75),
                    (120+int((w-160)*FALL_CONFIDENCE_THRESHOLD),195),
             (180,180,0), 2)

    cv2.putText(panel, f"Frame: {frame_num}", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140,140,140), 1)
    return panel


# ── Main ──────────────────────────────────────────────────────────────────────

def run(video_a, video_b, threshold, output_path, single_mode):
    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b)) if not single_mode else None

    if not cap_a.isOpened():
        print(f"Cannot open: {video_a}")
        return

    fps = cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay_ms = max(1, int(1000 / fps))   # for correct playback speed

    panel_h = max(H//2, 210)
    out = None
    if output_path:
        total_w = W*2 + 10 if not single_mode else W
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps,
                              (total_w, max(H, panel_h)))

    state_a = CameraState("A", fps=fps)
    state_b = CameraState("B", fps=fps) if not single_mode else None
    smooth_buf = deque(maxlen=SMOOTHING_WINDOW)

    screenshots_dir = Path(__file__).parent.parent.parent / 'screenshots'
    screenshots_dir.mkdir(exist_ok=True)

    frame_num   = 0
    fall_prev   = False
    fall_count  = 0
    display     = None
    last_display_time = time.time()
    frame_interval    = 1.0 / fps   # ideal seconds between displayed frames

    print(f"\nProcessing {'single' if single_mode else 'dual'} camera video...")
    print(f"Video fps: {fps:.1f}  |  Q=quit  SPACE=pause\n")

    paused = False

    while True:
        if not paused:
            ret_a, frame_a = cap_a.read()
            if not ret_a:
                break
            frame_num += 1

            if not single_mode:
                ret_b, frame_b = cap_b.read()
                if not ret_b:
                    frame_b = np.zeros_like(frame_a)
            else:
                frame_b = None

            prob_a = state_a.process(frame_a) or 0.0
            prob_b = state_b.process(frame_b) if state_b and frame_b is not None else None
            prob_b_val = prob_b if prob_b is not None else 0.0

            fused = fuse(prob_a, prob_b)
            fall  = smooth_decision(fused, smooth_buf, threshold)

            if fall and not fall_prev:
                fall_count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = screenshots_dir / f"fall_{ts}_f{frame_num:04d}.jpg"
                cv2.imwrite(str(save_path), frame_a)
                print(f"[FALL #{fall_count}] frame={frame_num}  "
                      f"P_A={prob_a*100:.1f}%  "
                      f"P_B={prob_b_val*100:.1f}%  "
                      f"Fused={fused*100:.1f}%")

            fall_prev = fall

            # Build annotated frame
            draw_frame(frame_a, "A", prob_a, fall)
            if frame_b is not None:
                draw_frame(frame_b, "B", prob_b_val, fall)

            panel_w = W if not single_mode else W//2
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            draw_fusion(panel, prob_a, prob_b_val, fused, fall, frame_num)

            if single_mode:
                display = frame_a
            else:
                top = np.hstack([frame_a, frame_b])
                panel_resized = cv2.resize(panel, (top.shape[1], panel_h))
                display = np.vstack([top, panel_resized]) if top.shape[0] < panel_resized.shape[0] else top
                display[-panel_h:, :panel_resized.shape[1]] = panel_resized

            if out:
                out.write(display)

            # Show frame only when enough real-time has passed → normal playback speed
            now = time.time()
            if now - last_display_time >= frame_interval:
                cv2.imshow("Dual Camera Fall Detection", display)
                last_display_time = now

        if display is not None:
            key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap_a.release()
    if cap_b:
        cap_b.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Total falls detected: {fall_count}")
    if output_path:
        print(f"Output saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_a',    required=True,  help="Video file for camera A")
    parser.add_argument('--cam_b',    default=None,   help="Video file for camera B (optional)")
    parser.add_argument('--threshold',type=float, default=FALL_CONFIDENCE_THRESHOLD)
    parser.add_argument('--output',   default=None,   help="Save output video to this path")
    parser.add_argument('--single',   action='store_true', help="Single camera mode")
    args = parser.parse_args()

    if args.cam_b is None:
        args.single = True

    run(video_a    = args.cam_a,
        video_b    = args.cam_b,
        threshold  = args.threshold,
        output_path= args.output,
        single_mode= args.single)
