"""
Phase 2 - Hackathon Test Dataset Prediction
Model: MobileNetV2 (ONNX) - Edge AI Defect Classifier
"""

import os
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
MODEL_PATH  = "/kaggle/input/models/dd1vya/edge-defect/onnx/edge-ai-defect/1/edge_defect_model.onnx"
DATASET_PATH = "/kaggle/input/datasets/dd1vya/dataset/phase2dataset/test"
IMG_SIZE    = 224
BATCH_SIZE  = 1

# =========================
# LOGGING
# =========================
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 60)
log("  PHASE 2 - HACKATHON TEST DATASET PREDICTION")
log("=" * 60)
log(f"Run started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Model path     : {MODEL_PATH}")
log(f"Dataset path   : {DATASET_PATH}")
log(f"Image size     : {IMG_SIZE}x{IMG_SIZE}")
log("")

# =========================
# TRANSFORMS
# Matches training pipeline exactly:
# - Grayscale converted to 3-channel
# - Resized to 224x224
# - ToTensor 
# =========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# =========================
# LOAD TEST DATASET
# ImageFolder discovers 9 folders alphabetically:
# idx 0=CMP, 1=LER, 2=bridge, 3=clean, 4=crack,
#     5=open, 6=other, 7=particle, 8=via
# =========================
test_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

log(f"Test folders discovered : {test_dataset.classes}")
log(f"Total test images       : {len(test_dataset)}")
log("")

# =========================
# CLASS MAPPING
# Test set has 9 folders; model trained on 8 classes.
# CMP  -> scratch  (confirmed with hackathon organizers)
# LER  -> other    (per hackathon rules: unmatched -> other)
# All other folders map 1:1 to training classes.
# =========================
train_classes = ['bridge', 'clean', 'crack', 'open', 'other', 'particle', 'scratch', 'via']

test_to_model = {
    0: 6,   # CMP      -> scratch
    1: 4,   # LER      -> other
    2: 0,   # bridge   -> bridge
    3: 1,   # clean    -> clean
    4: 2,   # crack    -> crack
    5: 3,   # open     -> open
    6: 4,   # other    -> other
    7: 5,   # particle -> particle
    8: 7    # via      -> via
}

log("Class mapping (test folder -> training class):")
for test_idx, model_idx in test_to_model.items():
    test_label  = test_dataset.classes[test_idx]
    model_label = train_classes[model_idx]
    arrow = " (organizer confirmed)" if test_label == "CMP" else \
            " (hackathon rule: unmatched -> other)" if test_label == "LER" else ""
    log(f"  {test_label:10s} -> {model_label}{arrow}")
log("")

# =========================
# LOAD ONNX MODEL
# =========================
log("Loading ONNX model...")
session    = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
log(f"Model loaded successfully.")
log(f"Model size : {model_size:.2f} MB")
log(f"Input name : {input_name}")
log("")

# =========================
# INFERENCE
# =========================
log("Running inference...")
y_true = []
y_pred = []
start_time = time.time()

for images, labels in test_loader:
    images_np = images.numpy()
    for i in range(images_np.shape[0]):
        img        = images_np[i:i+1]
        outputs    = session.run(None, {input_name: img})
        pred       = np.argmax(outputs[0], axis=1)[0]
        true_label = labels[i].item()
        mapped_true = test_to_model[true_label]
        y_true.append(mapped_true)
        y_pred.append(pred)

end_time   = time.time()
total_time = end_time - start_time

log(f"Inference completed!")
log(f"Total inference time : {total_time:.4f} seconds")
log(f"Avg time per image   : {total_time/len(y_true)*1000:.2f} ms")
log("")

# =========================
# PREDICTION DISTRIBUTION
# =========================
log("Prediction distribution:")
unique, counts = np.unique(y_pred, return_counts=True)
for u, c in zip(unique, counts):
    log(f"  {train_classes[u]:10s}: {c:3d} predictions")
log("")

# =========================
# METRICS
# =========================
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)

log("=" * 60)
log("  RESULTS SUMMARY")
log("=" * 60)
log(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
log(f"Precision : {prec:.4f} ({prec*100:.2f}%) [weighted]")
log(f"Recall    : {rec:.4f}  ({rec*100:.2f}%) [weighted]")
log(f"Model Size: {model_size:.2f} MB")
log("")

report = classification_report(
    y_true, y_pred,
    labels=list(range(8)),
    target_names=train_classes,
    zero_division=0
)
log("Per-Class Classification Report:")
log(report)

# =========================
# SAVE RESULTS TXT
# =========================
with open("results.txt", "w") as f:
    f.write("PHASE 2 PREDICTION RESULTS\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"Precision : {prec:.4f} (weighted)\n")
    f.write(f"Recall    : {rec:.4f} (weighted)\n")
    f.write(f"Model Size: {model_size:.2f} MB\n\n")
    f.write("Classification Report:\n")
    f.write(report)
log("Results saved to results.txt")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=train_classes,
    yticklabels=train_classes,
    cmap="Blues",
    linewidths=0.5,
    linecolor='gray'
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title(
    f"Confusion Matrix — 8-Class Defect Detection\n"
    f"Accuracy: {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%",
    fontsize=12
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
log("Confusion matrix saved to confusion_matrix.png")

# =========================
# PER-CLASS ACCURACY BAR CHART
# =========================
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors = ['#2ecc71' if a >= 0.5 else '#e74c3c' if a == 0 else '#f39c12'
          for a in per_class_acc]

plt.figure(figsize=(10, 5))
bars = plt.bar(train_classes, per_class_acc * 100, color=colors, edgecolor='white', linewidth=0.8)
plt.axhline(y=acc * 100, color='navy', linestyle='--', linewidth=1.5, label=f'Overall Accuracy ({acc*100:.1f}%)')
for bar, val in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.ylim(0, 115)
plt.xlabel("Defect Class", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Per-Class Accuracy — Edge AI Defect Classifier", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=150)
plt.close()
log("Per-class accuracy chart saved to per_class_accuracy.png")

# =========================
# NORMALIZED CONFUSION MATRIX
# =========================
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f",
    xticklabels=train_classes,
    yticklabels=train_classes,
    cmap="Blues",
    vmin=0, vmax=1,
    linewidths=0.5,
    linecolor='gray'
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Normalized Confusion Matrix (Row %)\nEdge AI Defect Classifier", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=150)
plt.close()
log("Normalized confusion matrix saved to confusion_matrix_normalized.png")

# =========================
# SAVE LOG FILE
# =========================
log("")
log(f"Run ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with open("run_logs.txt", "w") as f:
    f.write("\n".join(log_lines))

print("\nAll outputs saved:")
print("  - run_logs.txt")
print("  - results.txt")
print("  - confusion_matrix.png")
print("  - confusion_matrix_normalized.png")
print("  - per_class_accuracy.png")
