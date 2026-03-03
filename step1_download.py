# step1_download.py

import os
import urllib.request
from tqdm import tqdm
import time

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url, output):
    """ดาวน์โหลดไฟล์"""
    if os.path.exists(output):
        print(f"✅ มีแล้ว: {os.path.basename(output)}")
        return True
    
    try:
        print(f"📥 กำลังดาวน์โหลด: {os.path.basename(output)}")
        with DownloadProgress(unit='B', unit_scale=True, miniters=1) as t:
            urllib.request.urlretrieve(url, output, reporthook=t.update_to)
        print(f"✅ เสร็จ: {os.path.basename(output)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# URL ของ Dataset
base_url = "http://fenix.ur.edu.pl/~mkepski/ds/data/"

files = [
    # Fall Videos (15 คลิป)
    "fall-01-cam0-rgb.zip",
    "fall-02-cam0-rgb.zip",
    "fall-03-cam0-rgb.zip",
    "fall-04-cam0-rgb.zip",
    "fall-05-cam0-rgb.zip",
    "fall-06-cam0-rgb.zip",
    "fall-07-cam0-rgb.zip",
    "fall-08-cam0-rgb.zip",
    "fall-09-cam0-rgb.zip",
    "fall-10-cam0-rgb.zip",
    "fall-11-cam0-rgb.zip",
    "fall-12-cam0-rgb.zip",
    "fall-13-cam0-rgb.zip",
    "fall-14-cam0-rgb.zip",
    "fall-15-cam0-rgb.zip",
    
    # Normal Videos (15 คลิป)
    "adl-01-cam0-rgb.zip",
    "adl-02-cam0-rgb.zip",
    "adl-03-cam0-rgb.zip",
    "adl-04-cam0-rgb.zip",
    "adl-05-cam0-rgb.zip",
    "adl-06-cam0-rgb.zip",
    "adl-07-cam0-rgb.zip",
    "adl-08-cam0-rgb.zip",
    "adl-09-cam0-rgb.zip",
    "adl-10-cam0-rgb.zip",
    "adl-11-cam0-rgb.zip",
    "adl-12-cam0-rgb.zip",
    "adl-13-cam0-rgb.zip",
    "adl-14-cam0-rgb.zip",
    "adl-15-cam0-rgb.zip",
]

print("=" * 70)
print("📦 Step 1: ดาวน์โหลด Dataset (UR Fall Detection)")
print("=" * 70)
print(f"จำนวนไฟล์: {len(files)} ไฟล์")
print(f"ขนาดรวม: ~200-300 MB")
print(f"⏱️  เวลาโดยประมาณ: 10-20 นาที")
print("=" * 70)

start_time = time.time()
success = 0

for filename in files:
    url = base_url + filename
    output = f"dataset/raw/{filename}"
    
    if download(url, output):
        success += 1

elapsed = time.time() - start_time
minutes = int(elapsed / 60)
seconds = int(elapsed % 60)

print("\n" + "=" * 70)
print(f"✅ ดาวน์โหลดสำเร็จ: {success}/{len(files)} ไฟล์")
print(f"⏱️  เวลาที่ใช้: {minutes} นาที {seconds} วินาที")
print(f"📂 ไฟล์อยู่ที่: dataset/raw/")
print("=" * 70)
print("\n🎯 ขั้นตอนถัดไป: python3 step2_convert_videos.py")