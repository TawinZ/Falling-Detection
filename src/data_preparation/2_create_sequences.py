"""
Step 2: Build fixed-length sequences, normalise features, and balance classes.

Key improvements:
- Clip aspect_ratio (feat[2]) which can spike to 130+ from bad YOLO detections.
- StandardScaler fitted on train features only, applied to all splits.
  Scaler saved to models/feature_scaler.pkl for use at inference time.
- Split by VIDEO before building sequences to prevent data leakage.
- Oversample fall sequences in training set to ~35%.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *

TARGET_FALL_RATIO = 0.35
ASPECT_CLIP = 10.0   # aspect_ratio (feat[2]) physically cannot exceed ~10 for a person


def sliding_window(features, labels, seq_len):
    seqs, labs = [], []
    for i in range(len(features) - seq_len + 1):
        seqs.append(features[i: i + seq_len])
        labs.append(int(labels[i + seq_len - 1]))
    return seqs, labs


def build_split(video_indices, all_features, all_labels, seq_len):
    X, y = [], []
    for i in video_indices:
        s, l = sliding_window(all_features[i], all_labels[i], seq_len)
        X.extend(s)
        y.extend(l)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)


def oversample_falls(X, y, target_ratio):
    fall_idx  = np.where(y == 1)[0]
    nfall_idx = np.where(y == 0)[0]
    if len(fall_idx) / len(y) >= target_ratio:
        return X, y
    target_n = int(len(nfall_idx) * target_ratio / (1 - target_ratio))
    extra = np.random.choice(fall_idx, size=target_n - len(fall_idx), replace=True)
    X = np.concatenate([X, X[extra]])
    y = np.concatenate([y, y[extra]])
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def main():
    print("=" * 70)
    print("Step 2: Create Sequences + Normalise + Balance Classes")
    print("=" * 70)

    ur_feats  = np.load(PROCESSED_DATA_DIR / 'ur_features.npy',       allow_pickle=True)
    ur_labs   = np.load(PROCESSED_DATA_DIR / 'ur_labels.npy',         allow_pickle=True)
    mon_feats = np.load(PROCESSED_DATA_DIR / 'montreal_features.npy', allow_pickle=True)
    mon_labs  = np.load(PROCESSED_DATA_DIR / 'montreal_labels.npy',   allow_pickle=True)

    # Load extra datasets if available (GMDCSA-24, CAUCAFall, Le2i)
    extra_f_path = PROCESSED_DATA_DIR / 'extra_features.npy'
    if extra_f_path.exists():
        extra_feats = np.load(extra_f_path, allow_pickle=True)
        extra_labs  = np.load(PROCESSED_DATA_DIR / 'extra_labels.npy', allow_pickle=True)
        print(f"Extra datasets loaded: {len(extra_feats)} videos")
    else:
        extra_feats, extra_labs = np.array([], dtype=object), np.array([], dtype=object)
        print("No extra datasets found (run download_datasets.py to add more data)")

    # Clip aspect_ratio outliers before anything else
    for arr in list(ur_feats) + list(mon_feats) + list(extra_feats):
        arr[:, 2] = np.clip(arr[:, 2], 0.0, ASPECT_CLIP)

    all_feats = list(ur_feats) + list(mon_feats) + list(extra_feats)
    all_labs  = list(ur_labs)  + list(mon_labs)  + list(extra_labs)
    n_videos  = len(all_feats)
    print(f"Total videos: {n_videos}")

    # Video-level split (prevents leakage)
    idx = np.arange(n_videos)
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)
    val_idx,  test_idx  = train_test_split(temp_idx, test_size=0.50, random_state=42)

    print("Building sequences...")
    X_train, y_train = build_split(train_idx, all_feats, all_labs, SEQUENCE_LENGTH)
    X_val,   y_val   = build_split(val_idx,   all_feats, all_labs, SEQUENCE_LENGTH)
    X_test,  y_test  = build_split(test_idx,  all_feats, all_labs, SEQUENCE_LENGTH)

    # Fit StandardScaler on training data only
    print("Fitting StandardScaler on training features...")
    n_train, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train.reshape(-1, F))
    X_train = X_train_2d.reshape(n_train, T, F)

    X_val  = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)

    scaler_path = MODELS_DIR / 'feature_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")

    print(f"\nBefore oversampling:")
    for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        print(f"  {name}: {X.shape}  fall={int(y.sum())} ({y.mean()*100:.1f}%)")

    X_train, y_train = oversample_falls(X_train, y_train, TARGET_FALL_RATIO)

    print(f"\nAfter oversampling (train only):")
    print(f"  Train: {X_train.shape}  fall={int(y_train.sum())} ({y_train.mean()*100:.1f}%)")

    np.save(PROCESSED_DATA_DIR / 'X_train.npy', X_train)
    np.save(PROCESSED_DATA_DIR / 'X_val.npy',   X_val)
    np.save(PROCESSED_DATA_DIR / 'X_test.npy',  X_test)
    np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
    np.save(PROCESSED_DATA_DIR / 'y_val.npy',   y_val)
    np.save(PROCESSED_DATA_DIR / 'y_test.npy',  y_test)

    print(f"\nSaved to: {PROCESSED_DATA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
