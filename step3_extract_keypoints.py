import os
import zipfile
import cv2
import glob
from tqdm import tqdm

def unzip_file(zip_path, extract_to):
    folder_name = os.path.basename(zip_path).replace('.zip', '')
    extract_folder = os.path.join(extract_to, folder_name)
    
    if not os.path.exists(extract_folder):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    return extract_folder

def images_to_video(image_folder, output_video, fps=30):
    images = sorted(glob.glob(f"{image_folder}/*.png"))
    
    if not images:
        return False
    
    frame = cv2.imread(images[0])
    if frame is None:
        return False
    
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    for img_path in tqdm(images, desc=os.path.basename(output_video), leave=False):
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
    
    video.release()
    return True

os.makedirs("dataset/videos", exist_ok=True)
zip_files = sorted(glob.glob("dataset/raw/*.zip"))

print(f"Converting {len(zip_files)} videos...")

for zip_path in zip_files:
    filename = os.path.basename(zip_path).replace('.zip', '')
    output_video = f"dataset/videos/{filename}.mp4"
    
    if os.path.exists(output_video):
        continue
    
    extract_folder = unzip_file(zip_path, "dataset/raw/")
    images_to_video(extract_folder, output_video)

print("Conversion complete.")