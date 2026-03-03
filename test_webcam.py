# test_webcam.py

import cv2

print("กำลังเปิด Webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ เปิด Webcam ไม่ได้")
    print("   ลองไปที่: System Preferences → Security → Camera")
    exit()

print("✅ Webcam พร้อม กด Q เพื่อปิด")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แสดงข้อความบนภาพ
    cv2.putText(
        frame, "Webcam OK - Press Q to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.imshow("Test Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("ปิด Webcam แล้ว ✅") 