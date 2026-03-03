# step3_extract_keypoints.py

import sys
sys.path.append(".")

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import time
from ai.pose_estimator import PoseEstimator

def extract_keypoints_from_video(video_path, estimator):
    """Extract keypoints จากวิดีโอ 1 คลิป"""
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=frame_count, desc=f"  {os.path.basename(video_path)}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_data = estimator.analyze(frame)
            
            if pose_data and "keypoints" in pose_data:
                # แปลง Keypoints เป็น Array (9 จุด x 2 = 18 ค่า)
                kp_array = []
                for name in ["nose", "left_shoulder", "right_shoulder",
                           "left_hip", "right_hip", "left_knee", 
                           "right_knee", "left_ankle", "right_ankle"]:
                    x, y = pose_data["keypoints"][name]
                    kp_array.extend([x, y])
                
                keypoints_list.append(kp_array)
            
            pbar.update(1)
    
    cap.release()
    return np.array(keypoints_list)

print("=" * 70)
print("🦴 Step 3: Extract Keypoints จากวิดีโอ")
print("=" * 70)

# สร้าง Pose Estimator
print("📦 กำลังโหลด MediaPipe Model...")
estimator = PoseEstimator()
print("✅ โหลด Model สำเร็จ")

# หาวิดีโอทั้งหมด
videos = sorted(glob.glob("dataset/videos/*.mp4"))
print(f"📹 จำนวนวิดีโอ: {len(videos)} คลิป")
print(f"⏱️  เวลาโดยประมาณ: 30-45 นาที")
print("=" * 70)

start_time = time.time()
success = 0
skipped = 0

for video_path in videos:
    filename = os.path.basename(video_path).replace('.mp4', '')
    output_path = f"dataset/keypoints/{filename}.npy"
    
    # ข้ามถ้ามีแล้ว
    if os.path.exists(output_path):
        print(f"✅ มีแล้ว: {filename}.npy")
        skipped += 1
        continue
    
    print(f"\n🎬 Processing: {filename}.mp4")
    
    # Extract
    try:
        keypoints = extract_keypoints_from_video(video_path, estimator)
        
        # บันทึก
        np.save(output_path, keypoints)
        print(f"✅ บันทึกแล้ว: {filename}.npy (shape: {keypoints.shape})")
        success += 1
    except Exception as e:
        print(f"❌ Error: {e}")

elapsed = time.time() - start_time
minutes = int(elapsed / 60)
seconds = int(elapsed % 60)

print("\n" + "=" * 70)
print(f"✅ Extract สำเร็จ: {success} ไฟล์")
print(f"⏭️  ข้าม: {skipped} ไฟล์ (มีอยู่แล้ว)")
print(f"⏱️  เวลาที่ใช้: {minutes} นาที {seconds} วินาที")
print(f"📂 Keypoints อยู่ที่: dataset/keypoints/")
print("=" * 70)
print("\n🎯 ขั้นตอนถัดไป: python3 step4_prepare_data.py")