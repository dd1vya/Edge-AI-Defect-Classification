import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from models.mobilenet import get_mobilenet
from utils.transforms import get_val_test_transforms

DATASET_DIR = "/kaggle/input/edging/dataset"
NUM_CLASSES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset = datasets.ImageFolder(
    f"{DATASET_DIR}/test",
    transform=get_val_test_transforms()
)
loader = DataLoader(test_dataset, batch_size=16)

model = get_mobilenet(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("outputs/best_edge_defect_model.pth"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in loader:
        preds = model(images.to(DEVICE)).argmax(1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=test_dataset.classes))
print(confusion_matrix(y_true, y_pred))
