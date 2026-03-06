# compare_methods.py

import sys
sys.path.append(".")

import cv2
import time
from camera.camera_reader import CameraReader
from ai.pose_estimator import PoseEstimator
from ai.fall_detector import FallDetector
from ai.cnn_fall_detector import CNNFallDetector

def main():
    print("=" * 70)
    print("⚖️  Comparison: Rule-Based vs CNN")
    print("=" * 70)
    
    camera = CameraReader(camera_url=0)
    if not camera.open():
        print("❌ Cannot open camera")
        return
    
    pose_estimator = PoseEstimator()
    rule_detector = FallDetector()
    cnn_detector = CNNFallDetector()
    
    print("\n✅ Both systems ready!")
    print("Press Q to quit")
    print("=" * 70)
    
    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break
        
        pose_data = pose_estimator.analyze(frame)
        
        rule_result = rule_detector.detect(pose_data.get("pose", "unknown"))
        cnn_result = cnn_detector.detect(pose_data)
        
        h, w = frame.shape[:2]
        split = w // 2
        
        frame_left = frame.copy()
        frame_right = frame.copy()
        
        if pose_data and "keypoints" in pose_data:
            for name, (x, y) in pose_data["keypoints"].items():
                px_l = int(x * w)
                py_l = int(y * h)
                px_r = int(x * w)
                py_r = int(y * h)
                
                color_l = (0, 255, 0) if not rule_result["is_fall"] else (0, 0, 255)
                color_r = (0, 255, 0) if not cnn_result["is_fall"] else (0, 0, 255)
                
                cv2.circle(frame_left, (px_l, py_l), 5, color_l, -1)
                cv2.circle(frame_right, (px_r, py_r), 5, color_r, -1)
        
        cv2.putText(frame_left, "Rule-Based", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_left, rule_result["event_type"], (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame_right, "CNN", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_right, cnn_result["event_type"], (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_right, f"Conf: {cnn_result['confidence']:.2%}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        combined = cv2.hconcat([frame_left, frame_right])
        
        cv2.line(combined, (split, 0), (split, h), (255, 255, 255), 2)
        
        cv2.imshow('Comparison: Rule-Based vs CNN', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()
    print("\n✅ Comparison completed")

if __name__ == "__main__":
    main()