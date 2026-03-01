import gradio as gr
import torch
from spandrel import ModelLoader, ImageModelDescriptor
from PIL import Image
import numpy as np
import os
import cv2

# Set recursion limit for deep models if needed
import sys
sys.setrecursionlimit(2000)

def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        loader = ModelLoader()
        model = loader.load_from_file(model_path)
        
        if not isinstance(model, ImageModelDescriptor):
            return None, "지원하지 않는 모델 형식입니다 (spandrel 호환 모델이 아님)."

        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return None, str(e)

def process_image(input_image, model_path, scale_type, original_path=None):
    if input_image is None:
        return None, "이미지를 선택해주세요."
    if not model_path or not os.path.exists(model_path):
        return None, "유효한 모델 경로를 입력해주세요."

    model, device = load_model(model_path)
    if model is None:
        return None, f"모델 로드 실패: {device}"

    # PIL to Tensor
    img = np.array(input_image)
    if len(img.shape) == 2: # Grayscale to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: # RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Pre-process
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            output_tensor = model(img_tensor)
        
        # Post-process
        output = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        output = (output * 255.0).round().astype(np.uint8)
        upscaled_pil = Image.fromarray(output)
        
        # Determine target size
        orig_w, orig_h = input_image.size
        
        if scale_type == "x2":
            target_w, target_h = orig_w * 2, orig_h * 2
        elif scale_type == "x3":
            target_w, target_h = orig_w * 3, orig_h * 3
        elif scale_type == "x4":
            target_w, target_h = orig_w * 4, orig_h * 4
        else:
            # Pixel Target Scaling
            pixel_count = 0
            if "2M" in scale_type: pixel_count = 2_000_000
            elif "3M" in scale_type: pixel_count = 3_000_000
            elif "4M" in scale_type: pixel_count = 4_000_000
            
            aspect_ratio = orig_w / orig_h
            target_h = int((pixel_count / aspect_ratio) ** 0.5)
            target_w = int(target_h * aspect_ratio)

        # Rescale model output to requested size if necessary
        if upscaled_pil.size != (target_w, target_h):
            # Using LANCZOS for high-quality resizing
            upscaled_pil = upscaled_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # ---- 자동 저장 로직 추가 ---- #
        save_msg = ""
        if original_path:
            save_dir = os.path.dirname(original_path)
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            new_filename = f"{base_name}_upscaled.png"
            save_path = os.path.join(save_dir, new_filename)
        else:
            # 업로드한 이미지의 경우 현재 실행 디렉토리의 'output' 폴더에 저장
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f"upscaled_{timestamp}.png")
        
        try:
            upscaled_pil.save(save_path, "PNG")
            save_msg = f"\n✅ 파일 저장됨: {save_path}"
        except Exception as e:
            save_msg = f"\n❌ 저장 실패: {str(e)}"

        return upscaled_pil, f"성공! 모델: {os.path.basename(model_path)} (Device: {device}){save_msg}"
        
    except Exception as e:
        return None, f"처리 중 오류 발생: {str(e)}"

# Gradio UI with custom CSS
css = """
#main-container { max-width: 1200px; margin: auto; }
.output-msg { font-weight: bold; }
/* 이미지 박스 크기 고정 */
#input-img, #output-img { 
    height: 400px !important; 
    max-height: 400px !important; 
    overflow: hidden;
}
/* 이미지 드래그/업로드 시 박스 크기 유지 */
.gradio-container .prose { margin-bottom: 0; }
"""

def get_model_dir():
    path_file = "model_path.txt"
    if os.path.exists(path_file):
        with open(path_file, "r", encoding="utf-8") as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return os.path.dirname(os.path.abspath(__file__)) # Default fallback

def get_model_list():
    model_dir = get_model_dir()
    if not os.path.exists(model_dir):
        return []
    files = os.listdir(model_dir)
    valid_extensions = (".pth", ".safetensors")
    models = sorted([f for f in files if f.endswith(valid_extensions)])
    return models

def process_with_inputs(input_image, image_path, model_name, scale_type):
    source_image = None
    orig_path = None
    
    if image_path and os.path.exists(image_path):
        try:
            source_image = Image.open(image_path)
            orig_path = image_path
        except Exception as e:
            return None, f"이미지 경로 로드 실패: {str(e)}"
    elif input_image is not None:
        source_image = input_image
    
    if source_image is None:
        return None, "이미지를 업로드하거나 유효한 경로를 입력해주세요."
        
    if not model_name:
        return None, "모델을 선택해주세요."
    
    model_dir = get_model_dir()
    full_model_path = os.path.join(model_dir, model_name)
    return process_image(source_image, full_model_path, scale_type, original_path=orig_path)

with gr.Blocks(title="AI Image Upscaler", css=css) as demo:
    model_dir_display = get_model_dir()
    with gr.Column(elem_id="main-container"):
        gr.Markdown("# 🚀 AI Image Upscaler (Gradio)")
        gr.Markdown(f"모델 저장소: `{model_dir_display}`")
        
        with gr.Row():
            with gr.Column():
                with gr.Tab("이미지 업로드"):
                    input_img = gr.Image(type="pil", label="이미지 업로드", elem_id="input-img")
                with gr.Tab("이미지 경로 입력"):
                    input_path = gr.Textbox(label="이미지 파일 절대 경로", placeholder="C:/images/photo.jpg")
                
                model_list = get_model_list()
                model_dropdown = gr.Dropdown(
                    choices=model_list,
                    value=model_list[0] if model_list else None,
                    label="2. 모델 선택 (Select Model)",
                    interactive=True
                )
                
                scale_mode = gr.Radio(
                    choices=["x2", "x3", "x4", "2M pixel", "3M pixel", "4M pixel"],
                    value="3M pixel",
                    label="3. 출력 설정 (Scale Options)"
                )
                run_btn = gr.Button("🔥 업스케일 시작", variant="primary", size="lg")
            
            with gr.Column():
                output_img = gr.Image(label="결과 (Result)", elem_id="output-img", format="png")
                status_text = gr.Textbox(label="진행 상태 (Status)", interactive=False)

        run_btn.click(
            fn=process_with_inputs,
            inputs=[input_img, input_path, model_dropdown, scale_mode],
            outputs=[output_img, status_text]
        )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
