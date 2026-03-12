import customtkinter as ctk
import torch
from PIL import Image
import os
import sys
import threading
from tkinterdnd2 import DND_FILES, TkinterDnD
import rmbg_engine

# Set recursion limit for deep models if needed
sys.setrecursionlimit(5000)

def get_model_dir():
    path_file = "rmbg_model_path.txt"
    if os.path.exists(path_file):
        with open(path_file, "r", encoding="utf-8") as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RMBGApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        
        # Initialize tkdnd manually for CustomTkinter (from ctk_app.py reference)
        try:
            import tkinterdnd2
            import platform
            arch = "win-x64" if platform.architecture()[0] == "64bit" else "win-x32"
            dnd_dir = os.path.dirname(tkinterdnd2.__file__)
            tkdnd_path = os.path.join(dnd_dir, 'tkdnd', arch)
            
            if os.path.exists(tkdnd_path):
                self.tk.eval(f'lappend auto_path "{tkdnd_path.replace("\\", "/")}"')
                self.TkdndVersion = self.tk.call('package', 'require', 'tkdnd')
            else:
                # Fallback to default search
                self.TkdndVersion = self.tk.call('package', 'require', 'tkdnd')
        except Exception as e:
            print(f"TkinterDnD 초기화 오류: {e}")
            self.TkdndVersion = None

        self.title("AI Image Background Remover (RMBG / BiRefNet)")
        self.geometry("850x650")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.image_paths = []
        self.current_model = None
        self.loaded_model_path = None

        # -- UI Layout --
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Model Selection
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.header_frame.columnconfigure(1, weight=1)

        self.label_model = ctk.CTkLabel(self.header_frame, text="1. 모델 선택:")
        self.label_model.grid(row=0, column=0, padx=10, pady=10)

        self.model_list = self.get_model_list()
        self.model_dropdown = ctk.CTkOptionMenu(
            self.header_frame, 
            values=self.model_list if self.model_list else ["모델 없음 (.onnx)"],
            command=self.on_model_change
        )
        self.model_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # 2. Options Frame
        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.options_frame.columnconfigure(0, weight=1)

        self.bmp_var = ctk.BooleanVar(value=False)
        self.bmp_checkbox = ctk.CTkCheckBox(
            self.options_frame, 
            text="32bit BMP로 저장 (Alpha 채널 포함 - 배경투명)",
            variable=self.bmp_var
        )
        self.bmp_checkbox.grid(row=0, column=0, padx=20, pady=10, sticky="w")

        # 3. Drop Target & Preview
        self.preview_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=15, height=250)
        self.preview_frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        self.preview_frame.grid_propagate(False)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(
            self.preview_frame, 
            text="이미지나 폴더를 드래그 앤 드롭 하세요\n(PNG, JPG, BMP, WEBP 지원)", 
            font=("Arial", 14)
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # Register Drag and Drop
        self.preview_frame.drop_target_register(DND_FILES)
        self.preview_frame.dnd_bind('<<Drop>>', self.handle_drop)

        # 4. Output Folder
        self.output_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.output_frame.grid(row=3, column=0, padx=20, pady=(10, 5), sticky="ew")
        self.output_frame.columnconfigure(1, weight=1)

        self.label_output = ctk.CTkLabel(self.output_frame, text="출력 경로:")
        self.label_output.grid(row=0, column=0, padx=(0, 10), pady=0)

        self.output_var = ctk.StringVar(value="원본 폴더 (기본값)")
        self.output_entry = ctk.CTkEntry(self.output_frame, textvariable=self.output_var, state="disabled")
        self.output_entry.grid(row=0, column=1, padx=0, pady=0, sticky="ew")

        self.btn_select_output = ctk.CTkButton(self.output_frame, text="폴더 선택", width=80, command=self.select_output_folder)
        self.btn_select_output.grid(row=0, column=2, padx=(10, 0), pady=0)

        self.btn_reset_output = ctk.CTkButton(self.output_frame, text="기본값 (원본)", width=80, command=self.reset_output_folder)
        self.btn_reset_output.grid(row=0, column=3, padx=(10, 0), pady=0)

        # 5. Progress Bar
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=4, column=0, padx=20, pady=5, sticky="ew")
        self.progress_frame.columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=10)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="0 / 0", width=80)
        self.progress_label.grid(row=0, column=1, padx=10)

        # 6. Buttons
        self.footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.footer_frame.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.footer_frame.columnconfigure(0, weight=1)

        self.run_btn = ctk.CTkButton(
            self.footer_frame, 
            text="✨ 배경 제거 시작 (Batch Start)", 
            height=50, 
            font=("Arial", 18, "bold"),
            command=self.start_process_thread
        )
        self.run_btn.grid(row=0, column=0, sticky="ew")

        # 7. Status Bar
        self.status_bar = ctk.CTkLabel(self, text="준비됨", anchor="w", padx=20)
        self.status_bar.grid(row=6, column=0, sticky="ew", pady=(0, 10))

        if self.model_list:
            self.model_dropdown.set(self.model_list[0])

    def get_model_list(self):
        model_dir = get_model_dir()
        if not os.path.exists(model_dir):
            return []
        files = os.listdir(model_dir)
        # Focus on .onnx for this engine
        valid_extensions = (".onnx")
        return sorted([f for f in files if f.endswith(valid_extensions)])

    def on_model_change(self, choice):
        self.set_status(f"모델 선택됨: {choice}")

    def select_output_folder(self):
        folder = ctk.filedialog.askdirectory(title="출력 폴더 선택")
        if folder:
            self.output_var.set(folder)

    def reset_output_folder(self):
        self.output_var.set("원본 폴더 (기본값)")

    def handle_drop(self, event):
        try:
            paths = self.tk.splitlist(event.data)
        except:
            import re
            pattern = r'\{([^}]+)\}|(\S+)'
            matches = re.findall(pattern, event.data)
            paths = [m[0] if m[0] else m[1] for m in matches]
        
        valid_paths = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
        
        for path in paths:
            if os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in valid_extensions:
                    valid_paths.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in valid_extensions:
                            valid_paths.append(os.path.join(root, file))
        
        if valid_paths:
            self.image_paths = valid_paths
            count = len(valid_paths)
            self.set_status(f"{count}개의 이미지가 로드되었습니다.")
            self.progress_label.configure(text=f"0 / {count}")
            self.progress_bar.set(0)
            self.show_preview(valid_paths[0])
            
            if count > 1:
                self.preview_label.configure(text=f"📂 {os.path.basename(valid_paths[0])} 외 {count-1}개 선택됨")
        else:
            self.set_status("⚠️ 지원하는 이미지 파일이 없습니다.")

    def show_preview(self, path):
        try:
            max_w, max_h = 750, 220
            img = Image.open(path)
            orig_w, orig_h = img.size
            ratio = min(max_w / orig_w, max_h / orig_h)
            new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
            
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
            self.preview_label.configure(image=ctk_img, text="", anchor="center")
        except Exception as e:
            self.set_status(f"미리보기 실패: {e}")

    def set_status(self, text):
        self.status_bar.configure(text=text)
        self.update_idletasks()

    def load_model_if_needed(self, model_name):
        model_dir = get_model_dir()
        model_path = os.path.join(model_dir, model_name)
        if self.current_model and self.loaded_model_path == model_path:
            return self.current_model
        
        self.set_status(f"⏳ 모델 로딩 중: {model_name}...")
        try:
            model = rmbg_engine.load_rmbg_model(model_path, DEVICE)
            self.current_model = model
            self.loaded_model_path = model_path
            return model
        except Exception as e:
            self.set_status(f"❌ 모델 로드 오류: {str(e)}")
            return None

    def start_process_thread(self):
        if not self.image_paths:
            self.set_status("⚠️ 이미지를 먼저 드롭해주세요!")
            return
        
        self.run_btn.configure(state="disabled")
        threading.Thread(target=self.process_batch, daemon=True).start()

    def process_batch(self):
        model_name = self.model_dropdown.get()
        if "모델 없음" in model_name:
            self.after(0, lambda: self.set_status("⚠️ 선택된 모델이 없습니다."))
            self.after(0, lambda: self.run_btn.configure(state="normal"))
            return

        model = self.load_model_if_needed(model_name)
        if not model:
            self.after(0, lambda: self.run_btn.configure(state="normal"))
            return

        total = len(self.image_paths)
        save_bmp = self.bmp_var.get()

        for i, path in enumerate(self.image_paths):
            try:
                self.after(0, lambda p=path, idx=i: self.update_batch_ui(p, idx, total))
                
                # Inference
                image = Image.open(path)
                result_rgba = model(image)
                
                # Save
                out_dir = self.output_var.get()
                if out_dir == "원본 폴더 (기본값)":
                    out_dir = os.path.dirname(path)
                
                base_name = os.path.splitext(os.path.basename(path))[0]
                
                # Always save PNG (Standard)
                png_path = os.path.join(out_dir, f"{base_name}_no_bg.png")
                result_rgba.save(png_path, "PNG")

                # Additionally save BMP if checked
                if save_bmp:
                    bmp_path = os.path.join(out_dir, f"{base_name}_no_bg.bmp")
                    rmbg_engine.save_as_32bit_bmp(result_rgba, bmp_path)

            except Exception as e:
                self.after(0, lambda msg=f"❌ '{os.path.basename(path)}' 오류: {e}": self.set_status(msg))
        
        self.after(0, self.finish_batch)

    def update_batch_ui(self, path, current_idx, total):
        self.set_status(f"🏃 처리 중 ({current_idx+1}/{total}): {os.path.basename(path)}")
        self.progress_bar.set((current_idx + 1) / total)
        self.progress_label.configure(text=f"{current_idx + 1} / {total}")
        self.show_preview(path)

    def finish_batch(self):
        self.set_status("✅ 모든 작업이 완료되었습니다!")
        self.run_btn.configure(state="normal")

if __name__ == "__main__":
    app = RMBGApp()
    app.mainloop()
