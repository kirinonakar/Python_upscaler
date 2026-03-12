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
        
        # Custom normalization and configuration based on model name
        low_path = model_path.lower()
        
        # RMBG-1.4 specific
        if "rmbg-1.4" in low_path:
            self.mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
            self.std = np.array([1.0, 1.0, 1.0]).reshape(1, 1, 3)
        # BEN2 specific (usually follows standard ImageNet, but let's be explicit)
        elif "ben2" in low_path:
            self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        # Trendyol / BiRefNet / InSPyReNet
        else:
            self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def preprocess(self, img_np):
        # img_np is RGB (from PIL)
        ih, iw = img_np.shape[:2]
        tw, th = self.input_size
        
        # Letterbox: resize while maintaining aspect ratio
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        
        img_resized = cv2.resize(img_np, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas and pad
        # Use neutral gray (127) or black (0) for padding. Most models prefer gray or zero.
        # We'll use (0,0,0) as it's common for SOD.
        self.pad_info = {
            'top': (th - nh) // 2,
            'left': (tw - nw) // 2,
            'scale': scale,
            'orig_size': (iw, ih)
        }
        
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas[self.pad_info['top']:self.pad_info['top']+nh, 
               self.pad_info['left']:self.pad_info['left']+nw, :] = img_resized
        
        # Normalize
        img_normalized = (canvas.astype(np.float32) / 255.0 - self.mean) / self.std
        img_normalized = np.nan_to_num(img_normalized).astype(np.float32)

        # NHWC to NCHW
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def apply_guided_filter(self, I, p, r=8, eps=1e-4):
        """
        Guided Filter Implementation (Manual)
        I: guidance image (H, W, 3) 0-1
        p: filtering input image (H, W) 0-1
        r: local window radius
        eps: regularization parameter
        """
        I = I.astype(np.float32)
        p = p.astype(np.float32)

        def box_filter(img, r):
            return cv2.boxFilter(img, -1, (r, r))

        N = box_filter(np.ones_like(p), r)

        mean_I = box_filter(I, r) / N[:, :, np.newaxis]
        mean_p = box_filter(p, r) / N
        mean_Ip = box_filter(I * p[:, :, np.newaxis], r) / N[:, :, np.newaxis]
        cov_Ip = mean_Ip - mean_I * mean_p[:, :, np.newaxis]

        mean_II = box_filter(I * I, r) / N[:, :, np.newaxis]
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - (a[:, :, 0] * mean_I[:, :, 0] + a[:, :, 1] * mean_I[:, :, 1] + a[:, :, 2] * mean_I[:, :, 2])

        mean_a = box_filter(a, r) / N[:, :, np.newaxis]
        mean_b = box_filter(b, r) / N

        q = (mean_a[:, :, 0] * I[:, :, 0] + 
             mean_a[:, :, 1] * I[:, :, 1] + 
             mean_a[:, :, 2] * I[:, :, 2] + 
             mean_b)
        return np.clip(q, 0.0, 1.0)

    def postprocess(self, result, guidance_img=None):
        try:
            # result is the raw mask from model [1, 1, th, tw]
            if isinstance(result, (list, tuple)):
                mask = result[0]
            else:
                mask = result
                
            if len(mask.shape) == 4: mask = mask[0, 0]
            elif len(mask.shape) == 3: mask = mask[0]
                
            mask = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

            # Robust Activation
            v_min, v_max = mask.min(), mask.max()
            if v_min < -0.05 or v_max > 1.05:
                mask = 1.0 / (1.0 + np.exp(-np.clip(mask, -20, 20)))
            mask = np.clip(mask, 0.0, 1.0)

            # Optional: Small Dilation on low-res mask (Insurance)
            # kernel = np.ones((3, 3), np.uint8)
            # mask = cv2.dilate(mask, kernel, iterations=1)
            
            # 1. Reverse Letterbox: Crop the mask back to the valid area
            iw, ih = self.pad_info['orig_size']
            tw, th = self.input_size
            scale = self.pad_info['scale']
            nw, nh = int(iw * scale), int(ih * scale)
            
            top = self.pad_info['top']
            left = self.pad_info['left']
            
            mask_cropped = mask[top:top+nh, left:left+nw]
            
            # 2. Refined Upscaling (Guided Filter vs Linear)
            if guidance_img is not None:
                # Guidance image in RGB 0-1
                I = guidance_img.astype(np.float32) / 255.0
                # Guided filter handles resizing internally if implemented correctly, 
                # but here we resize to target first then refine.
                p = cv2.resize(mask_cropped, (iw, ih), interpolation=cv2.INTER_LINEAR)
                mask_final = self.apply_guided_filter(I, p, r=8, eps=1e-4)
            else:
                mask_final = cv2.resize(mask_cropped, (iw, ih), interpolation=cv2.INTER_LINEAR)
                
            return mask_final
            
        except Exception as e:
            print(f"[RMBG] Postprocess Error: {e}")
            # Fallback: return a black mask of the original size
            if hasattr(self, 'pad_info') and 'orig_size' in self.pad_info:
                return np.zeros(self.pad_info['orig_size'][::-1], dtype=np.float32)
            else:
                return np.zeros((1024, 1024), dtype=np.float32)

    def __call__(self, pil_img):
        try:
            img_np = np.array(pil_img.convert("RGB"))
            img_input = self.preprocess(img_np)
            
            ort_inputs = {self.input_name: img_input}
            ort_outs = self.session.run(None, ort_inputs)
            
            low_path = self.model_path.lower()
            num_outputs = len(ort_outs)
            
            # Check for NaNs in primary output (common in bad FP16 conversions)
            if np.isnan(ort_outs[0]).any():
                print(f"[RMBG] ⚠️ WARNING: Model '{os.path.basename(self.model_path)}' produced NaN values.")
                print("[RMBG] This is common with FP16 ONNX models. Please use the FP32 version if the image is black/corrupt.")

            # Model-specific output mapping
            if "ben2" in low_path:
                raw_mask = ort_outs[1] if num_outputs >= 2 else ort_outs[0]
            elif "trendyol" in low_path or "birefnet" in low_path:
                raw_mask = ort_outs[0]
            elif "inspyrenet" in low_path:
                raw_mask = ort_outs[0]
            else:
                raw_mask = ort_outs[0]
                
            # Pass original image as guidance for Guided Filter
            mask = self.postprocess(raw_mask, guidance_img=img_np)
            
            # Apply mask to image
            # Safety: ensuring mask is 0-1 float before scaling to 255
            mask_8bit = (mask * 255.0).clip(0, 255).astype(np.uint8)
            
            # Create RGBA image
            rgba_img = pil_img.convert("RGBA")
            r, g, b, _ = rgba_img.split()
            alpha = Image.fromarray(mask_8bit)
            
            result_img = Image.merge("RGBA", (r, g, b, alpha))
            return result_img
        except Exception as e:
            import traceback
            print(f"[RMBG] Inference Error with model {os.path.basename(self.model_path)}: {e}")
            traceback.print_exc()
            return pil_img.convert("RGBA") # Fallback to original
        
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
