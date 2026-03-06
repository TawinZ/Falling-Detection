# test_cnn_realtime.py

import sys
sys.path.append(".")

import cv2
import time
from camera.camera_reader import CameraReader
from ai.pose_estimator import PoseEstimator
from ai.cnn_fall_detector import CNNFallDetector

def get_color(event_type):
    """Get color based on event type"""
    colors = {
        "SAFE": (0, 255, 0),           # Green
        "BUFFERING": (255, 165, 0),    # Orange
        "FALL_DETECTED": (0, 0, 255),  # Red
        "STILL_FALLEN": (0, 0, 255),   # Red
        "NO_DETECTION": (128, 128, 128) # Gray
    }
    return colors.get(event_type, (255, 255, 255))

def main():
    print("=" * 70)
    print("🧠 CNN Fall Detection - Real-time Test")
    print("=" * 70)
    
    camera = CameraReader(camera_url=0)
    if not camera.open():
        print("❌ Cannot open camera")
        return
    
    pose_estimator = PoseEstimator()
    fall_detector = CNNFallDetector()
    
    print("\n✅ System ready!")
    print("📹 Camera: Active")
    print("🧠 CNN Model: Loaded")
    print("\nControls:")
    print("  Q: Quit")
    print("  R: Reset detector")
    print("=" * 70)
    
    fps_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break
        
        pose_data = pose_estimator.analyze(frame)
        result = fall_detector.detect(pose_data)
        
        event_type = result["event_type"]
        confidence = result["confidence"]
        is_fall = result["is_fall"]
        
        color = get_color(event_type)
        
        if pose_data and "keypoints" in pose_data:
            for name, (x, y) in pose_data["keypoints"].items():
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                cv2.circle(frame, (px, py), 5, color, -1)
        
        frame_count += 1
        if time.time() - fps_time >= 1.0:
            fps = frame_count / (time.time() - fps_time)
            fps_time = time.time()
            frame_count = 0
        else:
            fps = 0
        
        status_text = f"Status: {event_type}"
        conf_text = f"Confidence: {confidence:.2%}"
        fps_text = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, conf_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, fps_text, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if is_fall:
            cv2.putText(frame, "🚨 FALL DETECTED!", (20, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.imshow('CNN Fall Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            fall_detector.reset()
            print("🔄 Detector reset")
    
    camera.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed")

if __name__ == "__main__":
    main()