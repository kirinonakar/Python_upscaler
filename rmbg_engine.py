import os
import numpy as np
import cv2
import torch
from PIL import Image
import onnxruntime as ort

class RMBGModel:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        
        providers = ['CPUExecutionProvider']
        if "cuda" in str(device):
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3 # Silence warnings (0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape # e.g., [1, 3, 1024, 1024]
        
        # Determine input size
        # Handle cases where shape might be symbolic/None
        h = self.input_shape[2]
        w = self.input_shape[3]
        if isinstance(h, int) and h > 0:
            self.input_size = (w, h)
        else:
            # Fallback for dynamic shapes or specific models
            # BiRefNet/InSPyReNet typically use 1024x1024 or 512x512
            if "birefnet" in model_path.lower():
                self.input_size = (1024, 1024)
            elif "rmbg-2.0" in model_path.lower():
                self.input_size = (1024, 1024)
            else:
                self.input_size = (1024, 1024)
        
        print(f"[RMBG] Model Loaded: {os.path.basename(model_path)}")
        print(f"[RMBG] Input Size: {self.input_size}")
            
        # Common normalization parameters
        # Most of these models (RMBG, BiRefNet) use ImageNet normalization or similar
        # Some ONNX models have normalization built-in.
        # For simplicity, we'll try to detect or use standard.
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        
        # RMBG-1.4 specific if detected in filename
        if "rmbg-1.4" in model_path.lower():
            self.mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
            self.std = np.array([1.0, 1.0, 1.0]).reshape(1, 1, 3)

    def preprocess(self, img_np):
        # Resize to input size
        img_resized = cv2.resize(img_np, self.input_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize
        # Standard ImageNet normalization is most common for these
        # But we can try to be smart
        img_normalized = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        
        # NHWC to NCHW
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        return img_input

    def postprocess(self, result, orig_size):
        # Handle cases where model returns multiple outputs (e.g. BiRefNet)
        if isinstance(result, (list, tuple)):
            mask = result[0]
        else:
            mask = result
            
        # NCHW to HW
        if len(mask.shape) == 4: # [1, 1, H, W]
            mask = mask[0, 0]
        elif len(mask.shape) == 3: # [1, H, W]
            mask = mask[0]
            
        # Sigmoid if the model output is raw logits
        # Most salient object detection models output logits
        if mask.min() < 0 or mask.max() > 1:
            mask = 1 / (1 + np.exp(-mask))
        
        # Resize mask back to original size
        mask_resized = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)
        return mask_resized

    def __call__(self, pil_img):
        img_np = np.array(pil_img.convert("RGB"))
        orig_size = (img_np.shape[1], img_np.shape[0])
        
        img_input = self.preprocess(img_np)
        
        ort_inputs = {self.input_name: img_input}
        ort_outs = self.session.run(None, ort_inputs)
        
        # BiRefNet / InSPyReNet might return multiple masks; typically the first one is the main one.
        mask = self.postprocess(ort_outs[0], orig_size)
        
        # Apply mask to image
        mask_8bit = (mask * 255).astype(np.uint8)
        
        # Create RGBA image
        rgba_img = pil_img.convert("RGBA")
        r, g, b, _ = rgba_img.split()
        alpha = Image.fromarray(mask_8bit)
        
        result_img = Image.merge("RGBA", (r, g, b, alpha))
        return result_img

def load_rmbg_model(model_path, device="cuda"):
    # Currently specializing in ONNX as requested/common
    if model_path.lower().endswith(".onnx"):
        return RMBGModel(model_path, device)
    else:
        # For .pth/.safetensors, we would need the architecture.
        # However, for this task, we will focus on ONNX which is the most common cross-platform format.
        # If the user provides a .pth, we might try to load it via spandrel if supported, 
        # but spandrel is mostly for SR.
        raise ValueError(f"Currently only .onnx models are supported for this engine. Path: {model_path}")

def save_as_32bit_bmp(image, path):
    # PIL supports saving RGBA as BMP (which makes it 32-bit with alpha in some formats)
    # However, standard Windows BMP with alpha is often called 'BITFIELDS' format.
    # PIL's BMP writer is somewhat limited but can do it.
    image.save(path, format="BMP")
