import urllib.request
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

base_url = "http://fenix.ur.edu.pl/~mkepski/ds/data/"
fall_files = [f"fall-{i:02d}-cam0-rgb.zip" for i in range(1, 31)]
adl_files = [f"adl-{i:02d}-cam0-rgb.zip" for i in range(1, 41)]
all_files = fall_files + adl_files

print(f"Downloading {len(all_files)} files...")

for filename in all_files:
    url = base_url + filename
    
    # ข้ามถ้ามีแล้ว
    import os
    if os.path.exists(filename):
        print(f"Skip {filename} (already exists)")
        continue
    
    # ลองดาวน์โหลด 3 ครั้ง
    for attempt in range(3):
        try:
            print(f"Downloading {filename} (attempt {attempt+1}/3)...")
            urllib.request.urlretrieve(url, filename)
            print(f"  ✓ {filename}")
            break
        except Exception as e:
            if attempt < 2:
                print(f"  Retry in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  ✗ Failed: {filename}")

print("Done!")