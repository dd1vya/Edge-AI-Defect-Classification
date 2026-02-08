import torch.nn as nn
from torchvision import models

def get_mobilenet(num_classes):
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.last_channel, num_classes
    )
    return model
