import onnx
from onnx import external_data_helper

# Load model (that references external .data)
model = onnx.load("model.onnx")

# Combine external data into the .onnx file
onnx.save_model(
    model,
    "model_merged.onnx",
    save_as_external_data=False  # ⬅️ pack everything inside
)