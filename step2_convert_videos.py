# step2_convert_videos.py

import os
import zipfile
import cv2
import glob
from tqdm import tqdm
import time

def unzip_file(zip_path, extract_to):
    """แตกไฟล์ zip"""
    folder_name = os.path.basename(zip_path).replace('.zip', '')
    extract_folder = os.path.join(extract_to, folder_name)
    
    if os.path.exists(extract_folder):
        return extract_folder
    
    print(f"📂 แตกไฟล์: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ แตกเสร็จ")
    return extract_folder

def images_to_video(image_folder, output_video, fps=30):
    """แปลงภาพเป็นวิดีโอ"""
    images = sorted(glob.glob(f"{image_folder}/*.png"))
    
    if not images:
        print(f"❌ ไม่พบภาพใน {image_folder}")
        return False
    
    # อ่านภาพแรกเพื่อดูขนาด
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"❌ ไม่สามารถอ่านภาพได้")
        return False
    
    h, w, _ = frame.shape
    
    # สร้างวิดีโอ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    print(f"🎬 กำลังสร้างวิดีโอ: {os.path.basename(output_video)} ({len(images)} frames)")
    
    for img_path in tqdm(images, desc="  Processing", leave=False):
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
    
    video.release()
    print(f"✅ สร้างเสร็จ: {os.path.basename(output_video)}")
    return True

print("=" * 70)
print("🎬 Step 2: แปลงภาพเป็นวิดีโอ")
print("=" * 70)

# หาไฟล์ zip ทั้งหมด
zip_files = sorted(glob.glob("dataset/raw/*.zip"))

print(f"จำนวนไฟล์ .zip: {len(zip_files)} ไฟล์")
print(f"⏱️  เวลาโดยประมาณ: 10-15 นาที")
print("=" * 70)

start_time = time.time()
success = 0
skipped = 0

for zip_path in zip_files:
    filename = os.path.basename(zip_path).replace('.zip', '')
    output_video = f"dataset/videos/{filename}.mp4"
    
    # ข้ามถ้ามีวิดีโอแล้ว
    if os.path.exists(output_video):
        print(f"✅ มีแล้ว: {filename}.mp4")
        skipped += 1
        continue
    
    # แตก zip
    extract_folder = unzip_file(zip_path, "dataset/raw/")
    
    # แปลงเป็นวิดีโอ
    if images_to_video(extract_folder, output_video):
        success += 1
    
    print()  # บรรทัดว่าง

elapsed = time.time() - start_time
minutes = int(elapsed / 60)
seconds = int(elapsed % 60)

print("=" * 70)
print(f"✅ แปลงสำเร็จ: {success} วิดีโอ")
print(f"⏭️  ข้าม: {skipped} วิดีโอ (มีอยู่แล้ว)")
print(f"⏱️  เวลาที่ใช้: {minutes} นาที {seconds} วินาที")
print(f"📂 วิดีโออยู่ที่: dataset/videos/")
print("=" * 70)
print("\n🎯 ขั้นตอนถัดไป: python3 step3_extract_keypoints.py")