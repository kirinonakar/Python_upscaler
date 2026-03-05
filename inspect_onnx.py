import onnxruntime as ort
import numpy as np

model_path = "RealCUGAN_up4x-latest-no-denoise.onnx"
session = ort.InferenceSession(model_path)

print("Inputs:")
for i in session.get_inputs():
    print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

print("\nOutputs:")
for o in session.get_outputs():
    print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
