import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
    if os.path.exists(output):
        return True
    
    try:
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output)) as t:
            urllib.request.urlretrieve(url, output, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Error downloading {os.path.basename(output)}: {e}")
        return False

base_url = "http://fenix.ur.edu.pl/~mkepski/ds/data/"

# ดาวน์โหลด 100 videos
fall_files = [f"fall-{i:02d}-cam0-rgb.zip" for i in range(1, 31)]  # 30 Fall (ทั้งหมด)
adl_files = [f"adl-{i:02d}-cam0-rgb.zip" for i in range(1, 71)]    # 70 Normal
files = fall_files + adl_files

os.makedirs("dataset/raw", exist_ok=True)

print(f"Downloading {len(files)} files...")
start_time = time.time()

for filename in files:
    download(base_url + filename, f"dataset/raw/{filename}")

elapsed = time.time() - start_time
print(f"\nCompleted in {int(elapsed/60)}m {int(elapsed%60)}s")