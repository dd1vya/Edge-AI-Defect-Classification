# Edge-AI-Defect-Classification

## Dataset Description

### Dataset Source
The dataset used in this project is a custom / synthetic SEM (Scanning Electron Microscope) image dataset, curated specifically for semiconductor defect classification.
Images were collected and augmented to closely resemble real industrial SEM inspection data, including realistic grayscale textures, noise patterns, and defect structures.

### Number of Classes
The dataset contains 8 distinct classes, each representing a different type of semiconductor surface condition:

#### Defect Classes:

Bridge defect

Open defect

Crack

Via defect

Scratch

Particle contamination

Others

Clean (no defect)

Each class is stored in a separate folder following the ImageFolder directory structure.

### Type of Images
The dataset consists of grayscale Scanning Electron Microscope (SEM) images that are resized to a uniform resolution of 300 × 300 pixels and represented as single-channel images. These SEM images exhibit complex high-frequency textures and low-contrast regions that are characteristic of microscopic semiconductor surfaces. The defects present in the images often have irregular shapes and subtle boundaries, making classification challenging. Additionally, the dataset includes realistic noise patterns typical of SEM acquisition, which helps the model learn robust features that generalize well to real-world semiconductor inspection scenarios.

### Dataset Splitting
The dataset is divided into three subsets:

<img width="719" height="217" alt="image" src="https://github.com/user-attachments/assets/db741ed9-1d21-4ad1-ab55-3860ba4cae9c" />

Class imbalance is handled using class weighting and focal loss during training.

### Preprocessing & Augmentation
To improve robustness and generalization, the following preprocessing steps were applied:

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Enhances local contrast in SEM images without amplifying noise.

#### Data Augmentations (Training only):

Random horizontal and vertical flips

Random rotations (±12°)

Gaussian blur (simulating SEM noise)


These transformations simulate real-world variations in SEM imaging conditions such as orientation, surface roughness, and acquisition noise.

## Model Architecture

### Model Used

EfficientNet-B3

### EfficientNet
EfficientNet was selected due to its excellent balance between accuracy and computational efficiency, making it ideal for edge AI deployment in semiconductor inspection systems.

#### Key advantages:

High accuracy with fewer parameters

Scales depth, width, and resolution efficiently

Suitable for real-time or near-real-time inference

### Architecture Customization
#### 1. Input Layer Modification

Since SEM images are grayscale, the first convolution layer was modified:

 -> Original: 3-channel RGB input

 -> Modified: 1-channel grayscale input

This reduces unnecessary computation and preserves SEM-specific intensity patterns.

#### 2. Transfer Learning Strategy

The model leverages transfer learning by initializing the backbone with weights pretrained on the ImageNet dataset. These pretrained weights provide strong generic feature representations such as edges, shapes, and textures. To prevent overfitting on the relatively small SEM dataset, the backbone layers were frozen during training, and only the classifier head was updated. This strategy allows the model to adapt to the specific defect classification task while maintaining the robustness of pretrained features.

#### 3. Batch Normalization Freezing

All BatchNorm layers were frozen to:

 -> Maintain stable feature distributions

 -> Prevent training instability with small batch sizes

 -> Improve validation consistency

#### 4. Classification Head

The final fully connected layer was replaced to match the dataset:

 -> Output neurons: 8 (number of classes)

Loss optimized for class imbalance and hard samples

### Loss Function

A custom Focal Loss with Label Smoothing was used:

 -> Focal Loss focuses learning on difficult and minority classes

 -> Label smoothing prevents overconfidence and improves generalization

 -> Class weights compensate for dataset imbalance

This combination significantly improves performance on rare defect types.

### Edge AI Suitability

This architecture is designed for deployment on resource-constrained environments:

 -> Low parameter count

 -> High inference accuracy

 -> Fast convergence

 -> Compatible with ONNX / TensorRT conversion

 ### Final Performance
 
Best validation accuracy exceeds 88%

Stable convergence with minimal overfitting

Strong class-wise performance across defect categories


