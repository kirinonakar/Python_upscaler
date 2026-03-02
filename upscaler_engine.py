import torch
import numpy as np
import cv2
from PIL import Image
from spandrel import ModelLoader, ImageModelDescriptor
import os

def load_model(model_path, device):
    loader = ModelLoader()
    model = loader.load_from_file(model_path)
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError("Unsupported model format")
    model.to(device)
    model.eval()
    return model

def process_tiled(img_np, model, device, tile_size=512, overlap=32, progress_callback=None):
    h, w, c = img_np.shape
    scale = getattr(model, 'scale', 4)
    
    # Output canvas
    output_h, output_w = h * scale, w * scale
    output = np.zeros((output_h, output_w, c), dtype=np.float32)
    
    # Pad input to handle overlap context
    img_padded = cv2.copyMakeBorder(img_np, overlap, overlap, overlap, overlap, cv2.BORDER_REFLECT)
    
    num_tiles_y = (h + tile_size - 1) // tile_size
    num_tiles_x = (w + tile_size - 1) // tile_size
    total_tiles = num_tiles_y * num_tiles_x
    tile_count = 0
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile_count += 1
            if progress_callback:
                progress_callback(tile_count, total_tiles)
            
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            curr_tile_h = y_end - y
            curr_tile_w = x_end - x
            
            # Extract tile context from padded image
            tile_context = img_padded[y:y + curr_tile_h + 2*overlap, x:x + curr_tile_w + 2*overlap, :]
            
            # Process tile
            tile_tensor = torch.from_numpy(tile_context).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            with torch.no_grad():
                output_tile_tensor = model(tile_tensor)
            
            output_tile = output_tile_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            
            # Crop center
            sy = overlap * scale
            sx = overlap * scale
            ey = sy + curr_tile_h * scale
            ex = sx + curr_tile_w * scale
            
            res_tile = output_tile[sy:ey, sx:ex, :]
            
            # Place in output
            output[y*scale:y_end*scale, x*scale:x_end*scale, :] = res_tile
            
            # Free memory
            del output_tile_tensor
            del tile_tensor
            
    return output

def run_upscale(image, model, device, target_size=None, use_tiling=True, progress_callback=None):
    # PIL to Numpy
    img_np = np.array(image)
    if len(img_np.shape) == 2: # Grayscale to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4: # RGBA to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    if use_tiling:
        output_np = process_tiled(img_np, model, device, progress_callback=progress_callback)
    else:
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(img_tensor)
        output_np = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    
    # Convert to PIL
    output_np = (output_np * 255.0).round().astype(np.uint8)
    upscaled_pil = Image.fromarray(output_np)

    # Resize to target size if provided (Native model output might be x4, but user wanted x2)
    if target_size and upscaled_pil.size != target_size:
        upscaled_pil = upscaled_pil.resize(target_size, Image.Resampling.LANCZOS)
    
    return upscaled_pil

def get_target_size(orig_size, scale_type):
    orig_w, orig_h = orig_size
    if scale_type == "x2":
        return orig_w * 2, orig_h * 2
    elif scale_type == "x3":
        return orig_w * 3, orig_h * 3
    elif scale_type == "x4":
        return orig_w * 4, orig_h * 4
    else:
        pixel_count = 0
        if "2M" in scale_type: pixel_count = 2_000_000
        elif "3M" in scale_type: pixel_count = 3_000_000
        elif "4M" in scale_type: pixel_count = 4_000_000
        
        aspect_ratio = orig_w / orig_h
        target_h = int((pixel_count / aspect_ratio) ** 0.5)
        target_w = int(target_h * aspect_ratio)
        return target_w, target_h
