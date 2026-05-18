"""Configuration for Fall Detection"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MONTREAL_DIR = RAW_DATA_DIR / "dataset_montreal"
UR_FALL_PATTERN = "fall-*-cam0-rgb"
UR_ADL_PATTERN = "adl-*-cam0-rgb"

# Feature dimensions:
#   8 absolute pose features (body angle, aspect ratio, hip height, …)
#   7 velocity features      (motion speed and direction)
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 15
BATCH_SIZE = 64
EPOCHS = 100

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FALL_CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW = 5

# Model filenames
MODEL_BEST = "best_lstm_fall_detector.keras"
MODEL_FINAL = "lstm_fall_detector_final.keras"

# LINE Messaging API
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.getenv("LINE_USER_ID", "")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "")

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
