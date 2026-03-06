# ai/cnn_fall_detector.py

import numpy as np
from tensorflow.keras import models
from collections import deque

class CNNFallDetector:
    
    def __init__(self, model_path='models/cnn_fall_detector_final.h5', sequence_length=10):
        """
        CNN-based Fall Detector
        
        Args:
            model_path: path to trained model
            sequence_length: number of frames to buffer (default: 10)
        """
        self.model = models.load_model(model_path)
        self.sequence_length = sequence_length
        self.keypoints_buffer = deque(maxlen=sequence_length)
        
        self.is_fallen = False
        self.fall_start_time = None
        self.confidence_threshold = 0.7
        
    def _extract_keypoints(self, pose_data):
        """Extract 18 keypoint values from pose data"""
        if not pose_data or "keypoints" not in pose_data:
            return None
        
        kp_array = []
        keypoint_names = [
            "nose", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle"
        ]
        
        for name in keypoint_names:
            if name in pose_data["keypoints"]:
                x, y = pose_data["keypoints"][name]
                kp_array.extend([x, y])
            else:
                kp_array.extend([0.0, 0.0])
        
        return kp_array
    
    def detect(self, pose_data):
        """
        Detect fall from pose data
        
        Args:
            pose_data: dict from PoseEstimator.analyze()
            
        Returns:
            dict with:
                - is_fall: bool
                - confidence: float (0-1)
                - event_type: str
        """
        keypoints = self._extract_keypoints(pose_data)
        
        if keypoints is None:
            return {
                "is_fall": False,
                "confidence": 0.0,
                "event_type": "NO_DETECTION"
            }
        
        self.keypoints_buffer.append(keypoints)
        
        if len(self.keypoints_buffer) < self.sequence_length:
            return {
                "is_fall": False,
                "confidence": 0.0,
                "event_type": "BUFFERING"
            }
        
        sequence = np.array(list(self.keypoints_buffer))
        sequence = np.expand_dims(sequence, axis=0)
        
        prediction = self.model.predict(sequence, verbose=0)
        fall_probability = float(prediction[0][1])
        
        is_fall = fall_probability > self.confidence_threshold
        
        if is_fall and not self.is_fallen:
            self.is_fallen = True
            self.fall_start_time = None
            return {
                "is_fall": True,
                "confidence": fall_probability,
                "event_type": "FALL_DETECTED"
            }
        elif self.is_fallen:
            return {
                "is_fall": True,
                "confidence": fall_probability,
                "event_type": "STILL_FALLEN"
            }
        else:
            if self.is_fallen:
                self.is_fallen = False
            return {
                "is_fall": False,
                "confidence": fall_probability,
                "event_type": "SAFE"
            }
    
    def reset(self):
        """Reset detector state"""
        self.keypoints_buffer.clear()
        self.is_fallen = False
        self.fall_start_time = None