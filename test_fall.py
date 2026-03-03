# test_fall.py

import sys
sys.path.append(".")

from camera.camera_reader import CameraReader
from ai.pose_estimator    import PoseEstimator
from ai.fall_detector     import FallDetector
import cv2
import time

POSE_COLORS = {
    "standing": (0, 255, 0),
    "sitting":  (0, 165, 255),
    "lying":    (0, 0, 255),
}

POSE_THAI = {
    "standing": "Standing",
    "sitting":  "Sitting",
    "lying":    "Lying",
}

# เริ่มต้นระบบ
camera        = CameraReader(camera_url=0, camera_name="Webcam")
estimator     = PoseEstimator()
fall_detector = FallDetector()

if not camera.open():
    exit()

print("=" * 50)
print("🚨 Fall Detection System - Testing Mode")
print("=" * 50)
print("กด Q เพื่อปิด")
print("=" * 50)

cv2.namedWindow("Fall Detection Test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = camera.read_frame()
    if not ret:
        break

    # วิเคราะห์ท่าทาง
    pose_data = estimator.analyze(frame)
    frame = estimator.draw_skeleton(frame, pose_data)

    if pose_data:
        pose = pose_data["pose"]
        pose_thai = POSE_THAI.get(pose, pose)
        color = POSE_COLORS.get(pose, (255, 255, 255))
        
        # ตรวจจับการล้ม
        fall_result = fall_detector.detect(pose)
        
        # แสดงท่าทาง
        cv2.putText(frame, f"Pose: {pose_thai}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)
        
        # แสดงสถานะการล้ม
        if fall_result["is_fall"]:
            if fall_result["event_type"] == "FALL_DETECTED":
                # แจ้งเตือนครั้งแรก
                cv2.putText(frame, "🚨 FALL DETECTED!",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 4)
                
                print(f"\n⚠️  FALL DETECTED! (Δt={fall_result['delta_time']:.2f}s)")
            
            elif fall_result["event_type"] == "STILL_LYING":
                # ยังนอนอยู่
                lying_time = fall_result["time_lying"]
                
                if lying_time > 10:
                    # ฉุกเฉิน!
                    cv2.putText(frame, f"🆘 EMERGENCY! {lying_time:.0f}s",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 4)
                    print(f"🆘 EMERGENCY! Lying for {lying_time:.0f} seconds", end="\r")
                else:
                    # เฝ้าระวัง
                    cv2.putText(frame, f"⏱️ Monitoring... {lying_time:.0f}s",
                                (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 165, 255), 3)
                    print(f"⏱️  Monitoring... {lying_time:.0f} seconds", end="\r")
        
        elif fall_result["event_type"] == "SAFE":
            # ฟื้นตัวแล้ว
            recovery_time = fall_result["recovery_time"]
            cv2.putText(frame, f"✅ SAFE! Recovered in {recovery_time:.1f}s",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)
            print(f"\n✅ SAFE! Recovered in {recovery_time:.1f} seconds")
    
    else:
        cv2.putText(frame, "No Person",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (128, 128, 128), 3)

    cv2.imshow("Fall Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.close()
cv2.destroyAllWindows()
print("\n✅ ปิดระบบแล้ว")