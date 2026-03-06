import numpy as np
from tensorflow.keras import models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading model...")
model = models.load_model('models/cnn_fall_detector_final.h5')

print("✅ Model loaded!")
print(f"Total parameters: {model.count_params():,}")

X_test = np.load('dataset/processed/X_test.npy')
y_test = np.load('dataset/processed/y_test.npy')

print(f"\nTest data: {X_test.shape}")

loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "=" * 60)
print(f"🎯 Test Accuracy: {acc*100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")
print("=" * 60)

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📋 Classification Report:")
print(classification_report(
    y_test, 
    y_pred_classes,
    target_names=['Not Fall', 'Fall'],
    digits=3
))

cm = confusion_matrix(y_test, y_pred_classes)
print("\n🔢 Confusion Matrix:")
print("          Predicted")
print("          Not Fall  Fall")
print(f"Not Fall    {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"Fall        {cm[1][0]:4d}    {cm[1][1]:4d}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fall', 'Fall'],
            yticklabels=['Not Fall', 'Fall'])
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=300)
print("\n✅ Confusion matrix saved: test_confusion_matrix.png")

false_positives = cm[0][1]
false_negatives = cm[1][0]
total = cm.sum()

print(f"\n📊 Error Analysis:")
print(f"   False Positives: {false_positives} ({false_positives/total*100:.2f}%)")
print(f"   False Negatives: {false_negatives} ({false_negatives/total*100:.2f}%)")

if acc > 0.95:
    print("\n⚠️  Warning: Accuracy > 95% อาจเป็น Overfitting")