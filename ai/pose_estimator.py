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
        model_path = "pose_landmarker_full.task"
        if not os.path.exists(model_path):
            print("Error: pose_landmarker_full.task not found")
            exit()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        self.pose_buffer = []
        self.buffer_size = 5
        self.last_poses  = []
        self.pose_count  = 5

    def is_valid_human(self, keypoints):
        """ตรวจสอบว่า keypoints เรียงตามกายวิภาคมนุษย์จริงไหม"""
        nose_y     = keypoints["nose"][1]
        shoulder_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
        hip_y      = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
        knee_y     = (keypoints["left_knee"][1] + keypoints["right_knee"][1]) / 2
        ankle_y    = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2

        # ในภาพ y น้อย = อยู่บน, y มาก = อยู่ล่าง
        # ลำดับที่ถูกต้อง: nose < shoulder < hip < knee < ankle
        if nose_y >= shoulder_y:   return False
        if shoulder_y >= hip_y:    return False
        if hip_y >= knee_y:        return False
        if knee_y >= ankle_y:      return False

        return True

    def analyze(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = self.landmarker.detect(mp_image)

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

        # ถ้า keypoints ไม่ตรงกายวิภาคมนุษย์ → ถือเป็นวัตถุ skip
        if not self.is_valid_human(keypoints):
            return None

        pose_type = self.classify_pose(keypoints)

        return {
            "keypoints": keypoints,
            "pose":      pose_type,
            "landmarks": results.pose_landmarks[0]
        }

    def classify_pose(self, keypoints):
        nose_y        = keypoints["nose"][1]
        left_hip_y    = keypoints["left_hip"][1]
        right_hip_y   = keypoints["right_hip"][1]
        left_ankle_y  = keypoints["left_ankle"][1]
        right_ankle_y = keypoints["right_ankle"][1]

        hip_y   = (left_hip_y + right_hip_y) / 2
        ankle_y = (left_ankle_y + right_ankle_y) / 2

        body_angle = self.calc_body_angle(keypoints)

        self.pose_buffer.append(body_angle)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)

        head_height = nose_y

        if head_height > 0.5:
            current_pose = "lying"
        elif head_height < 0.3 and (ankle_y - hip_y) > 0.2:
            current_pose = "standing"
        else:
            current_pose = "sitting"

        self.last_poses.append(current_pose)
        if len(self.last_poses) > self.pose_count:
            self.last_poses.pop(0)

        pose_votes = {
            "standing": self.last_poses.count("standing"),
            "sitting":  self.last_poses.count("sitting"),
            "lying":    self.last_poses.count("lying")
        }

        return max(pose_votes, key=pose_votes.get)

    def calc_body_angle(self, keypoints):
        ls = keypoints["left_shoulder"]
        rs = keypoints["right_shoulder"]
        lh = keypoints["left_hip"]
        rh = keypoints["right_hip"]

        shoulder_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        hip_mid      = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)

        delta_x = shoulder_mid[0] - hip_mid[0]
        delta_y = shoulder_mid[1] - hip_mid[1]

        return abs(np.degrees(np.arctan2(delta_x, delta_y)))

    def draw_skeleton(self, frame, pose_data):
        if pose_data is None:
            return frame

        if "keypoints" not in pose_data:
            return frame

        h, w = frame.shape[:2]
        kp   = pose_data["keypoints"]

        for name, (x, y) in kp.items():
            cv2.circle(frame, (int(x*w), int(y*h)), 5, (0, 255, 255), -1)

        def line(a, b, color):
            if a in kp and b in kp:
                p1 = (int(kp[a][0]*w), int(kp[a][1]*h))
                p2 = (int(kp[b][0]*w), int(kp[b][1]*h))
                cv2.line(frame, p1, p2, color, 2)

        # Torso
        line("left_shoulder",  "right_shoulder", (0, 255, 0))
        line("left_shoulder",  "left_hip",        (0, 255, 0))
        line("right_shoulder", "right_hip",       (0, 255, 0))
        line("left_hip",       "right_hip",       (0, 255, 0))
        # Head
        line("nose", "left_shoulder",  (255, 200, 0))
        line("nose", "right_shoulder", (255, 200, 0))
        # Legs
        line("left_hip",   "left_knee",   (255, 0, 255))
        line("left_knee",  "left_ankle",  (255, 0, 255))
        line("right_hip",  "right_knee",  (255, 0, 255))
        line("right_knee", "right_ankle", (255, 0, 255))

        return frame