# Fall Detection System with BiLSTM + LINE Notification

ระบบตรวจจับการล้มแบบ Real-time ด้วย AI + แจ้งเตือนผู้ดูแลผ่าน LINE Bot
ใช้ YOLOv8-Pose สกัด keypoints ของร่างกาย และ Bidirectional LSTM วิเคราะห์การเคลื่อนไหว

---

## ระบบทำงานอย่างไร?

```
กล้อง / วิดีโอ
       ↓
YOLOv8n-Pose  →  9 keypoints (จมูก, ไหล่, สะโพก, เข่า, ข้อเท้า)
       ↓
Feature Extraction  →  15 features / frame
  ├── 8 Pose features   (มุมลำตัว, สัดส่วน, ความสูงสะโพก ฯลฯ)
  └── 7 Velocity features  (ความเร็ว per-second)
       ↓
Sliding Window  →  sequence (30 frames, 15 features)
       ↓
Bidirectional LSTM  →  P(Fall) ∈ [0, 1]
       ↓
Smoothing (5 frames majority vote)
       ↓
FALL DETECTED  →  บันทึก Screenshot + แจ้งเตือน LINE Bot
```

---

## โครงสร้างไฟล์

```
Falling-Detection/
├── config/
│   ├── config.py              ตั้งค่าระบบ (threshold, sequence length ฯลฯ)
│   └── ground_truth.py        frame-level labels ของ Montreal dataset
│
├── data/
│   ├── raw/                   ข้อมูลดิบ (ห้ามลบ)
│   │   ├── fall-01…30/        UR Fall  — PNG sequences
│   │   ├── adl-01…40/         UR ADL   — PNG sequences
│   │   ├── dataset_montreal/  Montreal — MP4 + annotation
│   │   ├── gmdcsa24/          GMDCSA-24 — MP4 (sleeping, bending, push-ups)
│   │   ├── caucafall/         CAUCAFall — AVI (5 fall types in home env)
│   │   └── le2i/              Le2i — AVI in 4 home environments
│   └── processed/             ไฟล์ .npy (สร้างอัตโนมัติ)
│
├── models/                    โมเดลที่เทรนแล้ว (สร้างอัตโนมัติ)
│   ├── lstm_fall_detector_final.keras
│   ├── best_lstm_fall_detector.keras
│   └── feature_scaler.pkl
│
├── src/
│   ├── data_preparation/
│   │   ├── 1_extract_keypoints.py   STEP 1 — สกัด features จากทุก dataset
│   │   └── 2_create_sequences.py    STEP 2 — สร้าง sequences + normalize
│   ├── training/
│   │   └── 3_train_lstm.py          STEP 3 — เทรน Bidirectional LSTM
│   ├── inference/
│   │   ├── dual_video_detect.py     ทดสอบด้วยวิดีโอ (1 หรือ 2 กล้อง)
│   │   └── realtime_webcam_detect.py  ใช้งานจริงผ่าน webcam
│   └── notification/
│       └── line_notifier.py         ส่งแจ้งเตือน LINE Bot
│
├── run_full_pipeline.sh       รัน Steps 1→2→3 อัตโนมัติ
├── download_datasets.py       ดาวน์โหลด dataset เพิ่มเติม
├── .env.example               template สำหรับ LINE credentials
└── requirements.txt
```

---

## Dataset ที่ใช้เทรน

| Dataset | วิดีโอ | fps | สิ่งที่ครอบคลุม |
|---|---|---|---|
| UR Fall Detection | 30 fall + 40 ADL | 15 | การล้มพื้นฐาน |
| Montreal (CAMMA) | 161 clips (8 views) | 120 | การล้มหลายมุมกล้อง + frame-level label |
| GMDCSA-24 | 79 fall + 81 ADL | varies | Sleeping, bending, push-ups, pick objects |
| CAUCAFall | 50 fall + 50 ADL | 23 | ล้มหน้า/หลัง/ซ้าย/ขวา/นั่ง ในบ้านจริง |
| Le2i | ~190 videos | 25 | 4 สภาพแวดล้อมบ้านจริง (home/office/kitchen/lecture) |
| **รวม** | **491 videos** | — | Lab + Home environments |

---

## ขั้นตอนการเทรน

### 1. ติดตั้ง

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. เทรนโมเดล

```bash
# รันอัตโนมัติทีเดียว (~8 ชั่วโมง)
bash run_full_pipeline.sh

# หรือรันทีละขั้น
python3 src/data_preparation/1_extract_keypoints.py   # ~7 ชั่วโมง
python3 src/data_preparation/2_create_sequences.py    # ~1 นาที
python3 src/training/3_train_lstm.py                  # ~30 นาที
```

### 3. ทดสอบด้วยวิดีโอ

```bash
# กล้องเดียว
python3 src/inference/dual_video_detect.py \
    --cam_a test_videos/test_fall.MOV --single

# 2 กล้อง 90° (มุมห้อง)
python3 src/inference/dual_video_detect.py \
    --cam_a มุมA.mp4 --cam_b มุมB.mp4

# บันทึกผลเป็นวิดีโอ
python3 src/inference/dual_video_detect.py \
    --cam_a input.mp4 --single --output result.mp4
```

### 4. ใช้งาน Webcam จริง

```bash
python3 src/inference/realtime_webcam_detect.py
# Q = ออก  |  SPACE = หยุด/เล่นต่อ
```

### 5. ตั้งค่า LINE Bot

```bash
cp .env.example .env
# แก้ .env:
# LINE_CHANNEL_ACCESS_TOKEN=xxxx
# LINE_USER_ID=Uxxxx
# IMGBB_API_KEY=xxxx  (optional สำหรับส่งรูป)
```

---

## สถาปัตยกรรมโมเดล

```
Input  (30 frames × 15 features)
    ↓
LSTM(128) + Dropout(0.3)
    ↓
LSTM(64)  + Dropout(0.3)
    ↓
LSTM(32)  + Dropout(0.3)
    ↓
Dense(64, ReLU) → Dense(32, ReLU)
    ↓
Dense(1, Sigmoid)  →  P(Fall) ∈ [0, 1]
```

| รายการ | ค่า |
|---|---|
| Parameters | 361,122 |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam (lr=1e-4, clipnorm=0.5) |
| Class weight | Fall = 5.0× |
| Sequence length | 30 frames |
| Features | 15 (8 pose + 7 velocity) |

---

## ผลการทดสอบ (Version 6)

| วิดีโอทดสอบ | ผล |
|---|---|
| test_fall.MOV | ✅ ตรวจพบ 1 ครั้ง (92.2%) |
| test_normal.MOV | ✅ 0 false alarm |
| test_sitting.MOV | ✅ 0 false alarm |

| Metric | ค่า |
|---|---|
| val_recall | 80% |
| Fall Recall (test set) | 45% |
| Threshold ที่ใช้ | 0.80 |

---

## Model Version History

| Version | Fall Recall | สิ่งที่เปลี่ยน |
|---|---|---|
| V0 (Original) | 3% | 7 features, LSTM, UR-Fall + Montreal |
| V4 | 29% | 15 features, fix labeling, remove BatchNorm |
| V5 | 71% | FPS normalization, keypoint quality filter |
| **V6 (ปัจจุบัน)** | **80% (val)** | **+3 datasets: GMDCSA-24, CAUCAFall, Le2i** |

---

## ปรับแต่งระบบ (`config/config.py`)

| ค่า | Default | ปรับเมื่อ |
|---|---|---|
| `FALL_CONFIDENCE_THRESHOLD` | 0.80 | แจ้งเตือนผิดบ่อย → เพิ่ม / พลาดล้ม → ลด |
| `SMOOTHING_WINDOW` | 5 | ต้องการตอบสนองเร็ว → ลด |
| `SEQUENCE_LENGTH` | 30 | ต้องเทรนใหม่ทุกครั้งที่เปลี่ยน |

---

## ทีมพัฒนา

- นายธาวิน ฝากสาคร
