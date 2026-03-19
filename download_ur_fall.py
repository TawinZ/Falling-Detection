import urllib.request
import ssl
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

base_url = "http://fenix.ur.edu.pl/~mkepski/ds/data/"
fall_files = [f"fall-{i:02d}-cam0-rgb.zip" for i in range(1, 31)]
adl_files = [f"adl-{i:02d}-cam0-rgb.zip" for i in range(1, 41)]
all_files = fall_files + adl_files

print(f"Downloading {len(all_files)} files...")

for filename in all_files:
    url = base_url + filename
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"Error: {filename} - {e}")

print("Done!")