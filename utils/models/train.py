import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from collections import Counter

from models.mobilenet import get_mobilenet
from utils.transforms import get_train_transforms, get_val_test_transforms

DATASET_DIR = "/kaggle/input/edging/dataset"
NUM_CLASSES = 8
BATCH_SIZE = 16
EPOCHS = 40
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=get_train_transforms(IMG_SIZE)
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=get_val_test_transforms(IMG_SIZE)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

counts = Counter(train_dataset.targets)
total = sum(counts.values())
class_weights = torch.tensor(
    [total / counts[i] for i in range(NUM_CLASSES)],
    dtype=torch.float
).to(DEVICE)

model = get_mobilenet(NUM_CLASSES).to(DEVICE)

for p in model.features.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = optim.AdamW(model.classifier.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    scheduler.step()

    print(f"Epoch {epoch+1}: Val Acc = {acc:.2f}%")

    if epoch == 10:
        for p in model.features[-4:].parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "outputs/best_edge_defect_model.pth")
