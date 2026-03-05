import onnxruntime as ort

# 1. GPU가 정상적으로 인식되는지 확인
print("사용 가능한 Provider:", ort.get_available_providers())

# 2. 강제로 GPU 세션을 열어보기 (경로에는 실제 모델 경로를 입력하세요)
model_path = "4xRealWebPhoto_v4_drct-l_fp32.onnx"

try:
    session = ort.InferenceSession(
        model_path, 
        providers=['CUDAExecutionProvider']
    )
    print("GPU 세션 로드 성공!")
except Exception as e:
    print("GPU 세션 로드 실패. 에러 내용:")
    print(e)