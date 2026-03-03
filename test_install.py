# test_install.py

print("กำลังทดสอบ Libraries...")
print("-" * 40)

try:
    import cv2
    print(f"✅ OpenCV    : {cv2.__version__}")
except:
    print("❌ OpenCV ติดตั้งไม่สำเร็จ")

try:
    import mediapipe as mp
    print(f"✅ MediaPipe : {mp.__version__}")
except:
    print("❌ MediaPipe ติดตั้งไม่สำเร็จ")

try:
    import numpy as np
    print(f"✅ NumPy     : {np.__version__}")
except:
    print("❌ NumPy ติดตั้งไม่สำเร็จ")

try:
    import requests
    print(f"✅ Requests  : {requests.__version__}")
except:
    print("❌ Requests ติดตั้งไม่สำเร็จ")

print("-" * 40)
print("🎉 พร้อมใช้งานทั้งหมด!")