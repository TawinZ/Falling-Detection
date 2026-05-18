"""
Step 1: Extract keypoints + compute rich features (15 per frame).

Features per step (frame t → t+1):
  Absolute (8): torso angle, body aspect ratio, hip height, shoulder height,
                head-above-hip offset, body height, knee asymmetry, bbox width
  Velocity (7): mean/std of x and y velocity, hip vertical velocity,
                mean absolute y velocity, mean velocity magnitude

UR-Fall labeling fix:
  Instead of marking the entire fall video as "fall=1", we auto-detect the
  actual fall window by finding the frame with peak downward hip velocity and
  labeling ±FALL_WINDOW frames around it as fall. Pre-fall walking and
  post-fall lying are labeled 0.

Montreal dataset uses existing per-frame annotations from ground_truth.py.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *

FALL_WINDOW = 15  # frames on each side of peak hip velocity to label as fall

print("Loading YOLOv8-Pose...")
yolo = YOLO('yolov8n-pose.pt')
print("Model loaded")

# 9 joints selected from 17-point COCO skeleton
# 0=nose  1=L_shoulder  2=R_shoulder  3=L_hip  4=R_hip
# 5=L_knee  6=R_knee  7=L_ankle  8=R_ankle
_KP_IDX = [0, 5, 6, 11, 12, 13, 14, 15, 16]


# Keypoints that must be visible for a reliable skeleton reading.
# If furniture occludes hips/knees/ankles, these will have low confidence
# and we skip the frame entirely rather than computing bad features.
_REQUIRED_KP  = [5, 6, 11, 12, 13, 14, 15, 16]  # shoulders, hips, knees, ankles
_MIN_VALID_KP = 6   # at least 6 of the 8 required joints must be confident


def extract_keypoints(frame):
    """Return (9,2) normalised keypoints, or None if skeleton is incomplete."""
    results = yolo(frame, verbose=False, conf=0.4, classes=[0])
    if not results[0].keypoints or len(results[0].boxes) == 0:
        return None

    best_kp, best_score = None, 0.0
    for i in range(len(results[0].boxes)):
        kp = results[0].keypoints.xy[i].cpu().numpy()
        if len(kp) < 17:
            continue

        box_conf = results[0].boxes.conf[i].item()

        # Check keypoint confidence for body joints
        if (results[0].keypoints.conf is not None):
            kp_conf = results[0].keypoints.conf[i].cpu().numpy()
            req_conf = kp_conf[_REQUIRED_KP]
            valid_count = (req_conf > 0.3).sum()
            if valid_count < _MIN_VALID_KP:
                continue   # too many joints hidden (e.g. behind furniture)
            # Score = mix of box confidence and keypoint quality
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


def absolute_features(kp):
    """
    8 pose features from a single (9, 2) keypoint frame.
    All values are normalized and dimensionless.
    """
    sh_cx = (kp[1, 0] + kp[2, 0]) / 2
    sh_cy = (kp[1, 1] + kp[2, 1]) / 2
    hip_cx = (kp[3, 0] + kp[4, 0]) / 2
    hip_cy = (kp[3, 1] + kp[4, 1]) / 2
    ankle_cy = (kp[7, 1] + kp[8, 1]) / 2

    # Torso vector: hip → shoulder  (dy > 0 when shoulder is above hip = normal)
    torso_dx = sh_cx - hip_cx
    torso_dy = hip_cy - sh_cy
    torso_len = np.hypot(torso_dx, torso_dy) + 1e-6

    # cos=1 vertical, cos=0 horizontal, cos<0 inverted
    torso_cos = torso_dy / torso_len
    # abs sin: 0 vertical, 1 horizontal (no left/right distinction needed)
    torso_sin = abs(torso_dx) / torso_len

    all_x = kp[:, 0]
    all_y = kp[:, 1]
    bbox_h = all_y.max() - all_y.min()
    bbox_w = all_x.max() - all_x.min() + 1e-6
    aspect = bbox_h / bbox_w  # >1 standing, <1 lying

    head_above_hip = hip_cy - kp[0, 1]   # positive = head above hips (normal)
    body_height = abs(ankle_cy - sh_cy)
    knee_asym = abs(kp[5, 1] - kp[6, 1])  # knee height difference (high during fall)

    return np.array([
        torso_cos,
        torso_sin,
        aspect,
        hip_cy,
        sh_cy,
        head_above_hip,
        body_height,
        knee_asym,
    ], dtype=np.float32)


FALL_WINDOW_SEC = 1.0   # label ±1 second around peak hip velocity as fall
UR_FALL_FPS     = 15.0  # UR-Fall dataset frame rate
REFERENCE_FPS   = 30.0  # all velocity features normalised to this fps


def velocity_features(kp_prev, kp_curr, fps: float = REFERENCE_FPS):
    """7 motion features, velocity normalised to REFERENCE_FPS (per-second scale)."""
    vel = (kp_curr - kp_prev) * fps   # convert per-frame → per-second equivalent
    vx, vy = vel[:, 0], vel[:, 1]
    hip_vel_y = vy[3:5].mean()
    return np.array([
        vx.mean(), vx.std(),
        vy.mean(), vy.std(),
        hip_vel_y,
        np.abs(vy).mean(),
        np.linalg.norm(vel, axis=1).mean(),
    ], dtype=np.float32)


def keypoints_to_features(kp_seq, fps: float = REFERENCE_FPS):
    """(N, 9, 2) → (N-1, 15) feature matrix with fps-normalised velocity."""
    feats = []
    for i in range(1, len(kp_seq)):
        feats.append(np.concatenate([
            absolute_features(kp_seq[i]),
            velocity_features(kp_seq[i - 1], kp_seq[i], fps=fps),
        ]))
    return np.array(feats, dtype=np.float32)


def ur_fall_labels(kp_seq, fps: float = UR_FALL_FPS):
    """Auto-detect fall window using ±FALL_WINDOW_SEC seconds around peak hip drop."""
    n = len(kp_seq) - 1
    if n <= 0:
        return np.zeros(0, dtype=np.int8)

    hip_vel = np.array([
        (kp_seq[i][3:5, 1] - kp_seq[i - 1][3:5, 1]).mean()
        for i in range(1, len(kp_seq))
    ])

    peak = int(np.argmax(hip_vel))
    window = int(fps * FALL_WINDOW_SEC)   # ±1 second in frames
    labels = np.zeros(n, dtype=np.int8)
    lo = max(0, peak - window)
    hi = min(n, peak + window + 1)
    labels[lo:hi] = 1
    return labels


def process_ur_fall():
    print("\nProcessing UR Fall dataset (PNG sequences)...")
    fall_dirs = sorted(RAW_DATA_DIR.glob('fall-*-cam0-rgb'))
    adl_dirs = sorted(RAW_DATA_DIR.glob('adl-*-cam0-rgb'))

    feat_list, lab_list = [], []
    for video_dir in tqdm(fall_dirs + adl_dirs, desc="UR folders"):
        is_fall = 'fall' in video_dir.name
        pngs = sorted(video_dir.glob('*.png'))
        if len(pngs) < 5:
            continue

        kp_seq = []
        for png in pngs:
            frame = cv2.imread(str(png))
            if frame is None:
                continue
            kp = extract_keypoints(frame)
            if kp is not None:
                kp_seq.append(kp)

        if len(kp_seq) < 5:
            continue

        feats = keypoints_to_features(kp_seq, fps=UR_FALL_FPS)
        labels = ur_fall_labels(kp_seq, fps=UR_FALL_FPS) if is_fall else np.zeros(len(feats), dtype=np.int8)
        feat_list.append(feats)
        lab_list.append(labels)

    return feat_list, lab_list


def process_montreal():
    print("\nProcessing Montreal dataset (MP4 videos)...")
    from config.ground_truth import get_label

    feat_list, lab_list = [], []
    scenarios = sorted(MONTREAL_DIR.glob('chute*'))

    for scen_dir in tqdm(scenarios, desc="Montreal scenarios"):
        scen_num = int(scen_dir.name.replace('chute', ''))
        for video_path in sorted(scen_dir.glob('*.mp4')):
            cam_num = int(video_path.stem.replace('cam', ''))
            cap = cv2.VideoCapture(str(video_path))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or REFERENCE_FPS
            kp_seq, frame_labs = [], []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    kp = extract_keypoints(frame)
                    if kp is not None:
                        kp_seq.append(kp)
                        frame_labs.append(get_label(scen_num, cam_num, frame_idx))
                except Exception:
                    pass
                frame_idx += 1
            cap.release()

            if len(kp_seq) < 5:
                continue

            feats = keypoints_to_features(kp_seq, fps=video_fps)
            labels = np.array(frame_labs[1:], dtype=np.int8)
            feat_list.append(feats)
            lab_list.append(labels)

    return feat_list, lab_list


def _extract_from_video(video_path, fps_override=None):
    """Generic: extract keypoints from any MP4/AVI video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or REFERENCE_FPS
    kp_seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extract_keypoints(frame)
        if kp is not None:
            kp_seq.append(kp)
    cap.release()
    return kp_seq, fps


def _extract_from_frames(frame_dir, fps_override=25.0):
    """Generic: extract keypoints from a directory of JPEG/PNG frames."""
    frames = sorted(frame_dir.glob('*.jpg')) + sorted(frame_dir.glob('*.jpeg')) + sorted(frame_dir.glob('*.png'))
    kp_seq = []
    for f in frames:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        kp = extract_keypoints(frame)
        if kp is not None:
            kp_seq.append(kp)
    return kp_seq, fps_override


def process_gmdcsa24():
    """
    GMDCSA-24 dataset (Zenodo).
    Structure: gmdcsa24/**/<Subject N>/Fall/<video>.mp4
                                   /ADL/<video>.mp4
    Uses direct parent folder name ('Fall' or 'ADL') to determine label.
    """
    base = RAW_DATA_DIR / 'gmdcsa24'
    if not base.exists():
        return [], []

    print("\nProcessing GMDCSA-24 dataset...")
    feat_list, lab_list = [], []

    videos = list(base.rglob('*.mp4')) + list(base.rglob('*.avi'))
    for video_path in tqdm(videos, desc="GMDCSA-24"):
        parent = video_path.parent.name.lower()   # 'Fall' or 'ADL'
        is_fall = (parent == 'fall')

        kp_seq, fps = _extract_from_video(video_path)
        if len(kp_seq) < 5:
            continue
        feats = keypoints_to_features(kp_seq, fps=fps)
        labels = ur_fall_labels(kp_seq, fps=fps) if is_fall else np.zeros(len(feats), dtype=np.int8)
        feat_list.append(feats)
        lab_list.append(labels)

    return feat_list, lab_list


def process_caucafall():
    """
    CAUCAFall dataset (Mendeley Data).
    Structure: caucafall/Subject.N/<Activity>/<video>.avi + PNG frames + .txt bbox

    Fall activities  (label=1): 'Fall backwards', 'Fall forward',
                                'Fall left', 'Fall right', 'Fall sitting'
    ADL  activities  (label=0): 'Hop', 'Kneel', 'Pick up object',
                                'Sit down', 'Walk'

    Uses os.walk with followlinks=True so the symlink is traversed correctly.
    Uses AVI video files (23 fps); PNG frames are skipped to save disk space.
    """
    import os
    base = RAW_DATA_DIR / 'caucafall'
    if not base.exists():
        return [], []

    print("\nProcessing CAUCAFall dataset...")
    feat_list, lab_list = [], []

    # Collect all AVI files, following symlinks
    avi_files = []
    for root, dirs, files in os.walk(str(base), followlinks=True):
        for f in files:
            if f.lower().endswith('.avi'):
                avi_files.append(Path(root) / f)

    for video_path in tqdm(avi_files, desc="CAUCAFall"):
        # Activity = direct parent folder name (e.g. 'Fall backwards', 'Walk')
        activity = video_path.parent.name
        is_fall  = activity.lower().startswith('fall')

        kp_seq, fps = _extract_from_video(video_path, fps_override=23.0)
        if len(kp_seq) < 5:
            continue
        feats  = keypoints_to_features(kp_seq, fps=23.0)
        labels = ur_fall_labels(kp_seq, fps=23.0) if is_fall else np.zeros(len(feats), dtype=np.int8)
        feat_list.append(feats)
        lab_list.append(labels)

    return feat_list, lab_list


def process_le2i():
    """
    Le2i Fall Detection Dataset (UBFC).
    Structure: le2i/*.zip  (one zip per environment, each containing)
      <env>/Videos/video (N).avi
      <env>/Annotation_files/video (N).txt   ← fall frame range (line1=start, line2=end)

    Videos are read one at a time via a temp file to avoid extracting
    the full dataset (~9 GB) to disk.  Annotation gives exact fall frames.
    """
    import os, tempfile, zipfile as zf_mod
    base = RAW_DATA_DIR / 'le2i'
    if not base.exists():
        return [], []

    inner_zips = sorted(base.glob('*.zip'))
    if not inner_zips:
        return [], []

    print(f"\nProcessing Le2i dataset ({len(inner_zips)} environments)...")
    feat_list, lab_list = [], []
    LE2I_FPS = 25.0

    for zip_path in tqdm(inner_zips, desc="Le2i envs"):
        with zf_mod.ZipFile(zip_path, 'r') as zf:
            avi_names = sorted(n for n in zf.namelist() if n.endswith('.avi'))

            for avi_name in avi_names:
                # Match annotation: same stem as video
                stem    = Path(avi_name).stem          # e.g. 'video (1)'
                env_dir = avi_name.split('/')[0]        # e.g. 'Home_01'
                ann_key = f"{env_dir}/Annotation_files/{stem}.txt"

                # Parse fall frame range from annotation
                fall_start, fall_end = None, None
                if ann_key in zf.namelist():
                    lines = zf.read(ann_key).decode('utf-8', errors='ignore').strip().split('\n')
                    try:
                        fall_start = int(lines[0].strip())
                        fall_end   = int(lines[1].strip())
                    except (ValueError, IndexError):
                        pass

                # Extract AVI to temp file (only ~60 MB at a time)
                with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
                    tmp.write(zf.read(avi_name))
                    tmp_path = tmp.name

                try:
                    kp_seq, _ = _extract_from_video(Path(tmp_path), fps_override=LE2I_FPS)
                finally:
                    os.unlink(tmp_path)

                if len(kp_seq) < 5:
                    continue

                feats = keypoints_to_features(kp_seq, fps=LE2I_FPS)
                n = len(feats)

                # Build per-frame labels using annotation
                labels = np.zeros(n, dtype=np.int8)
                if fall_start is not None and fall_end is not None:
                    # Convert frame indices to step indices (step i = frames i-1→i)
                    lo = max(0, fall_start - 2)
                    hi = min(n, fall_end)
                    labels[lo:hi] = 1

                feat_list.append(feats)
                lab_list.append(labels)

    return feat_list, lab_list


def main():
    print("=" * 70)
    print("Step 1: Extract Keypoints + Compute Rich Features (15/frame)")
    print("=" * 70)

    def stats(labs):
        total = sum(len(l) for l in labs)
        falls = int(sum(l.sum() for l in labs))
        return total, falls, f"{falls/total*100:.1f}%" if total else "0%"

    # ── Existing datasets ──────────────────────────────────────────
    ur_feats,  ur_labs  = process_ur_fall()
    mon_feats, mon_labs = process_montreal()

    # ── New datasets (auto-detected if downloaded) ─────────────────
    g24_feats, g24_labs = process_gmdcsa24()
    caf_feats, caf_labs = process_caucafall()
    l2i_feats, l2i_labs = process_le2i()

    # ── Print stats ────────────────────────────────────────────────
    for name, feats, labs in [
        ("UR Fall",   ur_feats,  ur_labs),
        ("Montreal",  mon_feats, mon_labs),
        ("GMDCSA-24", g24_feats, g24_labs),
        ("CAUCAFall", caf_feats, caf_labs),
        ("Le2i",      l2i_feats, l2i_labs),
    ]:
        if feats:
            t, f, r = stats(labs)
            print(f"{name:12s}: {len(feats):4d} videos | {t:7d} steps | fall={f} ({r})")

    # ── Save all ───────────────────────────────────────────────────
    np.save(PROCESSED_DATA_DIR / 'ur_features.npy',
            np.array(ur_feats, dtype=object), allow_pickle=True)
    np.save(PROCESSED_DATA_DIR / 'ur_labels.npy',
            np.array(ur_labs, dtype=object), allow_pickle=True)
    np.save(PROCESSED_DATA_DIR / 'montreal_features.npy',
            np.array(mon_feats, dtype=object), allow_pickle=True)
    np.save(PROCESSED_DATA_DIR / 'montreal_labels.npy',
            np.array(mon_labs, dtype=object), allow_pickle=True)

    # Combine new datasets into one file for Step 2
    extra_feats = g24_feats + caf_feats + l2i_feats
    extra_labs  = g24_labs  + caf_labs  + l2i_labs
    np.save(PROCESSED_DATA_DIR / 'extra_features.npy',
            np.array(extra_feats, dtype=object), allow_pickle=True)
    np.save(PROCESSED_DATA_DIR / 'extra_labels.npy',
            np.array(extra_labs,  dtype=object), allow_pickle=True)

    total_extra = len(extra_feats)
    print(f"\nExtra datasets combined: {total_extra} videos")
    print(f"Saved to: {PROCESSED_DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
