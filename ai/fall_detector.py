# ai/fall_detector.py

import time

class FallDetector:
    
    def __init__(self):
        # สถานะก่อนหน้า
        self.prev_pose = None
        self.prev_time = time.time()
        
        # Buffer ป้องกัน False Positive
        self.fall_buffer = []
        self.buffer_size = 3  # ต้องเกิด 2 ใน 3 ครั้ง ถึงจะถือว่าล้มจริง
        
        # สถานะการล้ม
        self.is_fallen = False
        self.fall_start_time = None
        self.fall_event_data = None
    
    def detect(self, current_pose):
        """
        ตรวจจับการล้มจากท่าทางปัจจุบัน
        
        Args:
            current_pose: "standing", "sitting", "lying"
            
        Returns:
            dict: {
                "is_fall": bool,
                "event_type": "FALL_DETECTED" / "SAFE" / None,
                "delta_time": float
            }
        """
        current_time = time.time()
        delta_time = current_time - self.prev_time
        
        result = {
            "is_fall": False,
            "event_type": None,
            "delta_time": delta_time
        }
        
        # ตรวจจับการเปลี่ยนท่า: ยืน → นอน
        if self.prev_pose == "standing" and current_pose == "lying":
            
            # เช็คความเร็ว: < 1 วินาที = น่าสงสัย
            if delta_time < 1.0:
                self.fall_buffer.append(True)
            else:
                self.fall_buffer.append(False)
        else:
            self.fall_buffer.append(False)
        
        # จำกัดขนาด Buffer
        if len(self.fall_buffer) > self.buffer_size:
            self.fall_buffer.pop(0)
        
        # นับจำนวนครั้งที่ตรวจพบการล้ม
        fall_count = self.fall_buffer.count(True)
        
        # ถ้าตรวจพบอย่างน้อย 2 ใน 3 ครั้ง → Fall จริง
        if fall_count >= 2 and not self.is_fallen:
            self.is_fallen = True
            self.fall_start_time = current_time
            self.fall_event_data = {
                "fall_time": current_time,
                "delta_time": delta_time
            }
            
            result["is_fall"] = True
            result["event_type"] = "FALL_DETECTED"
        
        # ถ้าล้มแล้วแต่กลับมายืนหรือนั่ง = ฟื้นตัว
        elif self.is_fallen and current_pose in ["standing", "sitting"]:
            recovery_time = current_time - self.fall_start_time
            
            result["is_fall"] = False
            result["event_type"] = "SAFE"
            result["recovery_time"] = recovery_time
            
            # Reset สถานะ
            self.is_fallen = False
            self.fall_start_time = None
            self.fall_buffer = []
        
        # ถ้ายังนอนอยู่
        elif self.is_fallen and current_pose == "lying":
            time_lying = current_time - self.fall_start_time
            result["is_fall"] = True
            result["event_type"] = "STILL_LYING"
            result["time_lying"] = time_lying
        
        # อัพเดทสถานะ
        self.prev_pose = current_pose
        self.prev_time = current_time
        
        return result
    
    def get_fall_duration(self):
        """คำนวณเวลาที่นอนอยู่"""
        if self.is_fallen and self.fall_start_time:
            return time.time() - self.fall_start_time
        return 0
    
    def reset(self):
        """Reset สถานะทั้งหมด"""
        self.prev_pose = None
        self.is_fallen = False
        self.fall_start_time = None
        self.fall_buffer = []