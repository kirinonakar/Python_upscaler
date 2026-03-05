import numpy as np
import onnxruntime as ort

model_path = "4xRealWebPhoto_v4_drct-l_fp32.onnx"
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"모델이 기대하는 입력 형태: {input_shape}")

# 64x64 크기의 아주 작은 더미 이미지(가짜 데이터) 생성 (Float32 타입 강제)
dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)

print("아주 작은 데이터로 GPU 추론 시작...")
try:
    outputs = session.run(None, {input_name: dummy_input})
    print("✅ 추론 성공! 출력 형태:", outputs[0].shape)
except Exception as e:
    print("❌ 추론 실패. 에러 내용:")
    print(e)