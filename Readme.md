# Fall Detection System

ระบบตรวจจับการล้มด้วย AI + แจ้งเตือน LINE

## 🎯 Features (ที่ทำเสร็จแล้ว)

- ✅ Camera Module - อ่านภาพจากกล้อง
- ✅ Pose Detection - ตรวจจับท่าทาง (ยืน/นั่ง/นอน)
- ⬜ Fall Detection - ตรวจจับการล้ม
- ⬜ LINE Alert - แจ้งเตือนผ่าน LINE

## 🛠️ เทคโนโลยี

- Python 3.13
- OpenCV 4.13
- MediaPipe 0.10
- NumPy 2.4

## 📦 การติดตั้ง

### 1. Clone Repository
```bash
git clone <YOUR_REPO_URL>
cd fall-detection
```

### 2. สร้าง Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. ติดตั้ง Libraries
```bash
pip install opencv-python mediapipe numpy requests
```

### 4. ดาวน์โหลด AI Model
ดาวน์โหลดไฟล์ `pose_landmarker_lite.task` จาก:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task


## 🚀 วิธีใช้งาน

### ทดสอบ Pose Detection
```bash
python3 test_pose.py
```

กด Q เพื่อปิด

## 📁 โครงสร้างโปรเจค
```
fall-detection/
├── camera/              # Camera Module
├── ai/                  # AI Pose Detection
├── alert/               # LINE Notification (ยังไม่ได้ทำ)
├── config/              # ตั้งค่าระบบ
├── images/              # เก็บภาพ
├── logs/                # Log ระบบ
└── test_*.py            # ไฟล์ทดสอบ
```

## 👥 ทีมพัฒนา

- นายธาวิน ฝากสาคร
- นายหัฐกรณ์ แจ้งสรีจันทร์

## 📝 License

[ใส่ License ที่ต้องการ]# Fall-Detection-Deep-Learning-Project
