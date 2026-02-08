from torchvision import datasets, transforms

DATASET_DIR = "/kaggle/input/edging/dataset"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root=f"{DATASET_DIR}/train",
    transform=transform
)

print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))
