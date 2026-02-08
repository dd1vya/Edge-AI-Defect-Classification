import torch
from models.mobilenet import get_mobilenet

IMG_SIZE = 224
NUM_CLASSES = 8

model = get_mobilenet(NUM_CLASSES)
model.load_state_dict(torch.load("outputs/best_edge_defect_model.pth"))
model.eval()

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy,
    "outputs/edge_defect_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX export complete")
