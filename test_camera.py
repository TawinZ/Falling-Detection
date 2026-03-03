# test_camera.py

import sys
sys.path.append(".")

from camera.camera_reader import CameraReader
import cv2

camera = CameraReader(camera_url=0, camera_name="Webcam")

if not camera.open():
    print("❌ เปิดกล้องไม่ได้")
    exit()

print("กด Q เพื่อปิด")

while True:
    ret, frame = camera.read_frame()
    if not ret:
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.close()
cv2.destroyAllWindows()
print("ปิดระบบแล้ว")