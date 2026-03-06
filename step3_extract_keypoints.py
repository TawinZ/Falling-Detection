import sys
sys.path.append(".")

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
from ai.pose_estimator import PoseEstimator

KEYPOINT_NAMES = [
    "nose", "left_shoulder", "right_shoulder", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def interpolate_missing(keypoints_list):
    """เติมค่า frame ที่จับไม่ได้ด้วย Interpolation"""
    arr = np.array(keypoints_list, dtype=np.float32)  # (frames, 18) มี NaN ตรงที่จับไม่ได้
    
    for col in range(arr.shape[1]):
        col_data = arr[:, col]
        nan_mask = np.isnan(col_data)
        
        if nan_mask.all():
            # ทั้ง column เป็น NaN → ใช้ 0 แทน
            arr[:, col] = 0.0
        elif nan_mask.any():
            # มีบางส่วนเป็น NaN → interpolate
            indices = np.arange(len(col_data))
            valid = ~nan_mask
            arr[:, col] = np.interp(indices, indices[valid], col_data[valid])
    
    return arr

def extract_keypoints(video_path, estimator):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=frame_count, desc=os.path.basename(video_path), leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pose_data = estimator.analyze(frame)

            if pose_data and "keypoints" in pose_data:
                kp_array = []
                for name in KEYPOINT_NAMES:
                    x, y = pose_data["keypoints"][name]
                    kp_array.extend([x, y])
                keypoints_list.append(kp_array)
            else:
                # จับไม่ได้ → ใส่ NaN ไว้ก่อน จะ interpolate ทีหลัง
                keypoints_list.append([np.nan] * 18)

            pbar.update(1)

    cap.release()

    if not keypoints_list:
        return np.array([])

    return interpolate_missing(keypoints_list)


os.makedirs("dataset/keypoints", exist_ok=True)
estimator = PoseEstimator()
videos = sorted(glob.glob("dataset/videos/*.mp4"))

print(f"Extracting keypoints from {len(videos)} videos...")

for video_path in videos:
    filename = os.path.basename(video_path).replace('.mp4', '')
    output_path = f"dataset/keypoints/{filename}.npy"

    if os.path.exists(output_path):
        continue

    keypoints = extract_keypoints(video_path, estimator)
    
    if len(keypoints) == 0:
        print(f"⚠️  ข้ามไฟล์ {filename} เพราะไม่มี keypoints")
        continue

    np.save(output_path, keypoints)

print("Extraction complete.")