# ai/pose_estimator.py

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import numpy as np
import cv2
import os

class PoseEstimator:

    def __init__(self):
        model_path = "pose_landmarker_lite.task"
        if not os.path.exists(model_path):
            print("❌ ไม่พบ Model กรุณาดาวน์โหลดมาวางที่ Root โปรเจคก่อน")
            exit()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # Buffer สำหรับ Smoothing (เพิ่มจาก 3 เป็น 5)
        self.pose_buffer = []
        self.buffer_size = 5  # เปลี่ยนจาก 3 เป็น 5
        # เพิ่ม Buffer สำหรับ Pose ด้วย
        self.last_poses = []
        self.pose_count = 5

    def analyze(self, frame):
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = self.landmarker.detect(mp_image)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks[0]

        keypoints = {
            "nose":           (landmarks[0].x,  landmarks[0].y),
            "left_shoulder":  (landmarks[11].x, landmarks[11].y),
            "right_shoulder": (landmarks[12].x, landmarks[12].y),
            "left_hip":       (landmarks[23].x, landmarks[23].y),
            "right_hip":      (landmarks[24].x, landmarks[24].y),
            "left_knee":      (landmarks[25].x, landmarks[25].y),
            "right_knee":     (landmarks[26].x, landmarks[26].y),
            "left_ankle":     (landmarks[27].x, landmarks[27].y),
            "right_ankle":    (landmarks[28].x, landmarks[28].y),
        }

        pose_type = self.classify_pose(keypoints)

        return {
            "keypoints": keypoints,
            "pose":      pose_type,
            "landmarks": results.pose_landmarks[0]
        }

    def classify_pose(self, keypoints):
        nose_y       = keypoints["nose"][1]
        left_hip_y   = keypoints["left_hip"][1]
        right_hip_y  = keypoints["right_hip"][1]
        left_ankle_y = keypoints["left_ankle"][1]
        right_ankle_y= keypoints["right_ankle"][1]
        
        hip_y    = (left_hip_y + right_hip_y) / 2
        ankle_y  = (left_ankle_y + right_ankle_y) / 2
        
        body_angle = self.calc_body_angle(keypoints)
        
        # เพิ่มเข้า Buffer
        self.pose_buffer.append(body_angle)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)
        
        smooth_angle = sum(self.pose_buffer) / len(self.pose_buffer)
        head_height = nose_y
        
        # คำนวณ Pose
        if head_height > 0.5:
            current_pose = "lying"
        elif head_height < 0.3 and (ankle_y - hip_y) > 0.2:
            current_pose = "standing"
        else:
            current_pose = "sitting"
        
        # Smoothing Pose ด้วย Voting
        self.last_poses.append(current_pose)
        if len(self.last_poses) > self.pose_count:
            self.last_poses.pop(0)
        
        # นับ Pose ที่เกิดบ่อยที่สุด
        pose_votes = {
            "standing": self.last_poses.count("standing"),
            "sitting":  self.last_poses.count("sitting"),
            "lying":    self.last_poses.count("lying")
        }
        
        final_pose = max(pose_votes, key=pose_votes.get)
        return final_pose

    def calc_body_angle(self, keypoints):
        ls = keypoints["left_shoulder"]
        rs = keypoints["right_shoulder"]
        lh = keypoints["left_hip"]
        rh = keypoints["right_hip"]

        shoulder_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        hip_mid      = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)

        delta_x = shoulder_mid[0] - hip_mid[0]
        delta_y = shoulder_mid[1] - hip_mid[1]
        angle   = abs(np.degrees(np.arctan2(delta_x, delta_y)))

        return angle

    def draw_skeleton(self, frame, pose_data):
        if pose_data is None:
            return frame

        if "keypoints" in pose_data:
            h, w = frame.shape[:2]
            
            # วาดจุด
            for name, (x, y) in pose_data["keypoints"].items():
                cx = int(x * w)
                cy = int(y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            
            # วาดเส้นเชื่อม
            kp = pose_data["keypoints"]
            
            def draw_line(p1_name, p2_name, color=(0, 255, 0)):
                if p1_name in kp and p2_name in kp:
                    p1 = (int(kp[p1_name][0] * w), int(kp[p1_name][1] * h))
                    p2 = (int(kp[p2_name][0] * w), int(kp[p2_name][1] * h))
                    cv2.line(frame, p1, p2, color, 2)
            
            # ลำตัว (สีเขียว)
            draw_line("left_shoulder", "right_shoulder", (0, 255, 0))
            draw_line("left_shoulder", "left_hip", (0, 255, 0))
            draw_line("right_shoulder", "right_hip", (0, 255, 0))
            draw_line("left_hip", "right_hip", (0, 255, 0))
            
            # แขนซ้าย (สีฟ้า)
            draw_line("left_shoulder", "left_hip", (255, 200, 0))
            draw_line("nose", "left_shoulder", (255, 200, 0))
            
            # แขนขวา (สีฟ้า)
            draw_line("right_shoulder", "right_hip", (255, 200, 0))
            draw_line("nose", "right_shoulder", (255, 200, 0))
            
            # ขาซ้าย (สีม่วง)
            draw_line("left_hip", "left_knee", (255, 0, 255))
            draw_line("left_knee", "left_ankle", (255, 0, 255))
            
            # ขาขวา (สีม่วง)
            draw_line("right_hip", "right_knee", (255, 0, 255))
            draw_line("right_knee", "right_ankle", (255, 0, 255))

        return frame