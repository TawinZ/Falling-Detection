#!/bin/bash
# Full pipeline: Steps 1→2→3 with live progress display.
# Usage: bash run_full_pipeline.sh

cd "$(dirname "$0")"
source venv/bin/activate

LOG="pipeline_run.log"

echo "======================================" | tee $LOG
echo "Pipeline started: $(date)"            | tee -a $LOG
echo "======================================" | tee -a $LOG
echo "" | tee -a $LOG

# ── Verify datasets ───────────────────────────────────────────────
echo "[CHECK] Datasets available:"  | tee -a $LOG
find data/raw/fall-*-cam0-rgb -name "*.png" 2>/dev/null | wc -l | xargs echo "  UR Fall PNG frames:" | tee -a $LOG
ls data/raw/dataset_montreal/ 2>/dev/null | wc -l | xargs echo "  Montreal scenarios:" | tee -a $LOG
find data/raw/gmdcsa24 -name "*.mp4" 2>/dev/null | wc -l | xargs echo "  GMDCSA-24 videos:" | tee -a $LOG
find -L data/raw/caucafall -name "*.avi" 2>/dev/null | wc -l | xargs echo "  CAUCAFall videos:" | tee -a $LOG
ls data/raw/le2i/*.zip 2>/dev/null | wc -l | xargs echo "  Le2i inner zips:" | tee -a $LOG
echo "" | tee -a $LOG

# ── Step 1 ────────────────────────────────────────────────────────
echo "[STEP 1/3] Extracting keypoints from all datasets..." | tee -a $LOG
echo "  Started: $(date)"   | tee -a $LOG
echo "  Est. time: 6-8 hrs" | tee -a $LOG

python3 src/data_preparation/1_extract_keypoints.py 2>&1 | tee -a $LOG
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Step 1 failed." | tee -a $LOG; exit 1
fi
echo "  Finished: $(date)" | tee -a $LOG

# ── Step 2 ────────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "[STEP 2/3] Creating sequences + normalising..." | tee -a $LOG
echo "  Started: $(date)" | tee -a $LOG

python3 src/data_preparation/2_create_sequences.py 2>&1 | tee -a $LOG
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Step 2 failed." | tee -a $LOG; exit 1
fi
echo "  Finished: $(date)" | tee -a $LOG

# ── Step 3 ────────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "[STEP 3/3] Training LSTM model..." | tee -a $LOG
echo "  Started: $(date)"    | tee -a $LOG
echo "  Est. time: 30-60 min" | tee -a $LOG

python3 src/training/3_train_lstm.py 2>&1 | tee -a $LOG
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Step 3 failed." | tee -a $LOG; exit 1
fi
echo "  Finished: $(date)" | tee -a $LOG

# ── Done ──────────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "======================================" | tee -a $LOG
echo "ALL COMPLETE: $(date)"                 | tee -a $LOG
echo "======================================" | tee -a $LOG
grep -A5 "Test Set Results" $LOG | tail -8   | tee -a $LOG

osascript -e 'display notification "Training complete! Check pipeline_run.log" with title "Fall Detection" sound name "Glass"' 2>/dev/null
