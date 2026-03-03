# test_pose.py

import cv2
cv2.startWindowThread()
import sys
sys.path.append(".")

from camera.camera_reader import CameraReader
from ai.pose_estimator    import PoseEstimator

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

camera    = CameraReader(camera_url=0, camera_name="Webcam")
estimator = PoseEstimator()

if not camera.open():
    exit()

print("✅ ระบบพร้อม กด Q เพื่อปิด")

cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = camera.read_frame()
    if not ret:
        break

    pose_data = estimator.analyze(frame)
    frame     = estimator.draw_skeleton(frame, pose_data)

    if pose_data:
        pose      = pose_data["pose"]
        pose_thai = POSE_THAI.get(pose, pose)
        color     = POSE_COLORS.get(pose, (255,255,255))
        angle     = estimator.calc_body_angle(pose_data["keypoints"])

        cv2.putText(frame, f"Pose: {pose_thai}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)

        cv2.putText(frame, f"Angle: {angle:.1f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

    else:
        cv2.putText(frame, "No Person",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (128, 128, 128), 3)

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.close()
cv2.destroyAllWindows()
print("✅ ปิดระบบแล้ว")