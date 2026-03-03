import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split

def create_sequences(keypoints, sequence_length=10):
    sequences = []
    for i in range(len(keypoints) - sequence_length + 1):
        sequences.append(keypoints[i:i+sequence_length])
    return np.array(sequences)

fall_files = sorted(glob.glob("dataset/keypoints/fall-*.npy"))
adl_files = sorted(glob.glob("dataset/keypoints/adl-*.npy"))

fall_sequences = np.vstack([create_sequences(np.load(f)) for f in fall_files])
adl_sequences = np.vstack([create_sequences(np.load(f)) for f in adl_files])

fall_labels = np.ones(len(fall_sequences))
adl_labels = np.zeros(len(adl_sequences))

X = np.concatenate([fall_sequences, adl_sequences])
y = np.concatenate([fall_labels, adl_labels])

indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

os.makedirs("dataset/processed", exist_ok=True)

np.save("dataset/processed/X_train.npy", X_train)
np.save("dataset/processed/X_val.npy", X_val)
np.save("dataset/processed/X_test.npy", X_test)
np.save("dataset/processed/y_train.npy", y_train)
np.save("dataset/processed/y_val.npy", y_val)
np.save("dataset/processed/y_test.npy", y_test)

print(f"Data prepared: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")