"""
Step 3: Train LSTM Fall Detector — stable version.

Changes from previous attempt:
- Removed BatchNormalization (known to be unstable with LSTM on Apple Metal GPU).
- Replaced Bidirectional with standard LSTM + recurrent_dropout (simpler, stable).
- Switched to binary output (sigmoid + binary_crossentropy) — cleaner for 2-class problem.
- Lowered learning rate to 1e-4 to prevent gradient explosion after epoch 1.
- Added Recall as a training metric so we can watch fall detection quality.
- EarlyStopping now monitors val_loss; checkpoint monitors val_recall.
- Increased class_weight[1] to 5.0 to push recall on the minority fall class.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import *

print(f"TF {tf.__version__}  GPU: {tf.config.list_physical_devices('GPU')}")


def build_model(timesteps, features):
    inp = layers.Input(shape=(timesteps, features))

    # LSTM stack — no BatchNorm, no recurrent_dropout (keeps Metal GPU acceleration)
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64,  return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32,  return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)

    # Binary output: P(Fall)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=0.5),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
        ],
    )
    return model


def main():
    print("=" * 70)
    print("Step 3: Train LSTM Fall Detector (stable)")
    print("=" * 70)

    X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
    X_val   = np.load(PROCESSED_DATA_DIR / 'X_val.npy')
    X_test  = np.load(PROCESSED_DATA_DIR / 'X_test.npy')
    y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy').astype(np.float32)
    y_val   = np.load(PROCESSED_DATA_DIR / 'y_val.npy').astype(np.float32)
    y_test  = np.load(PROCESSED_DATA_DIR / 'y_test.npy').astype(np.float32)

    for name, X, y in [("Train", X_train, y_train),
                       ("Val",   X_val,   y_val),
                       ("Test",  X_test,  y_test)]:
        print(f"{name}: {X.shape}  fall={int(y.sum())} ({y.mean()*100:.1f}%)")

    # Oversampling already balanced train; class_weight gives extra push toward recall
    class_weight = {0: 1.0, 1: 5.0}
    print(f"\nClass weights: {class_weight}")

    model = build_model(timesteps=X_train.shape[1], features=X_train.shape[2])
    model.summary()

    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-7, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(MODELS_DIR / MODEL_BEST),
            monitor='val_recall', mode='max',
            save_best_only=True, verbose=1,
        ),
    ]

    print("\nTraining...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished: {int(elapsed/60)}m {int(elapsed%60)}s")
    print(f"Best val_recall : {max(history.history['val_recall']):.4f}")
    print(f"Best val_loss   : {min(history.history['val_loss']):.4f}")

    # Full evaluation on test set
    print("\nTest Set Results:")
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    print(classification_report(y_test, y_pred,
                                target_names=['Non-Fall', 'Fall'], digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"  Non-Fall → Non-Fall: {cm[0,0]}   Non-Fall → Fall: {cm[0,1]}")
    print(f"  Fall     → Non-Fall: {cm[1,0]}   Fall     → Fall: {cm[1,1]}")

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(axes,
                               ['loss',    'recall',    'precision'],
                               ['Loss',    'Recall',    'Precision']):
        ax.plot(history.history[key],          label='Train')
        ax.plot(history.history[f'val_{key}'], label='Val')
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_history.png', dpi=150)
    print(f"\nPlot saved: {MODELS_DIR / 'training_history.png'}")

    model.save(MODELS_DIR / MODEL_FINAL)
    print(f"Model saved: {MODELS_DIR / MODEL_FINAL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
