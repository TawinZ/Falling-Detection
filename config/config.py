"""Configuration for Fall Detection"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MONTREAL_DIR = RAW_DATA_DIR / "montreal"
UR_FALL_PATTERN = "fall-*-cam0-rgb"
UR_ADL_PATTERN = "adl-*-cam0-rgb"

SEQUENCE_LENGTH = 20
FEATURES_PER_FRAME = 7
BATCH_SIZE = 64
EPOCHS = 100

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FALL_CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
