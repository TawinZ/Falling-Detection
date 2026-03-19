# camera/camera_reader.py

import cv2
import time

class CameraReader:

    def __init__(self, camera_url=0, camera_name="Camera"):
        self.camera_url  = camera_url
        self.camera_name = camera_name
        self.cap         = None
        self.is_open     = False

    def open(self):
        print(f"กำลังเปิดกล้อง: {self.camera_name}...")
        self.cap = cv2.VideoCapture(self.camera_url)

        if not self.cap.isOpened():
            print(f"❌ เปิดกล้องไม่ได้: {self.camera_name}")
            return False

        self.is_open = True
        print(f"✅ เปิดกล้องสำเร็จ: {self.camera_name}")
        return True

    def read_frame(self):
        if not self.is_open:
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            print(f"❌ อ่านภาพไม่ได้: {self.camera_name}")
            return False, None

        return True, frame

    def save_frame(self, frame, folder="images/"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"{folder}capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ บันทึกภาพ: {filename}")
        return filename

    def close(self):
        if self.cap:
            self.cap.release()
            self.is_open = False
            print(f"✅ ปิดกล้องแล้ว: {self.camera_name}")