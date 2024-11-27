import torch  # torch 모듈 임포트
from tensorflow.keras.models import load_model
from utils import extract_features
from diffusers import StableDiffusionPipeline
import numpy as np

# 감정 클래스별 프롬프트 정의
emotion_prompts = {
    "neutral": "a serene landscape with calm skies",
    "happy": "a cute dog with a joyful expression, surrounded by flowers",
    "sad": "a lonely tree under a gray rainy sky",
    "angry": "a stormy ocean with fierce waves crashing",
    "fear": "a dark forest with eerie lights and shadows",
    "disgust": "a messy and unclean kitchen with spoiled food",
}

def download_and_initialize_stable_diffusion():
    """
    Stable Diffusion 모델을 다운로드하고 초기화.
    """
    print("Downloading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16  # VRAM 최적화
    )
    pipe.to("cuda")  # GPU 사용
    print("Stable Diffusion model loaded successfully.")
    return pipe

def predict_emotion_and_generate_image(model, file_path, class_names, stable_diffusion_pipe):
    """
    감정을 예측하고 Stable Diffusion을 사용해 이미지를 생성.
    :param model: 감정 분류 모델
    :param file_path: 예측할 음성 파일 경로
    :param class_names: 감정 클래스 이름 리스트
    :param stable_diffusion_pipe: Stable Diffusion 파이프라인 객체
    """
    # 음성 파일에서 특징 추출
    features = extract_features(file_path)
    
    # 3차원 형태로 변환 (batch_size=1, time_steps=1, features)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    
    # 예측
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)  # 가장 높은 확률의 클래스 인덱스
    emotion = class_names[predicted_class]
    
    print(f"Predicted Emotion: {emotion}")
    
    # 감정에 맞는 프롬프트 가져오기
    prompt = emotion_prompts.get(emotion, "a calm and beautiful scene")
    print(f"Generated Prompt: {prompt}")
    
    # Stable Diffusion으로 이미지 생성
    print("Generating image with Stable Diffusion...")
    image = stable_diffusion_pipe(prompt).images[0]
    
    # 이미지 저장
    output_path = f"generated_image_{emotion}.png"
    image.save(output_path)
    print(f"Image saved as {output_path}")

def main():
    # 사전 학습된 감정 분류 모델 로드
    emotion_model = load_model('model/emotion_classifier.h5')
    
    # 감정 클래스 이름 정의 (라벨 순서와 동일하게 설정)
    class_names = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
    
    # Stable Diffusion 모델 다운로드 및 초기화
    stable_diffusion_pipe = download_and_initialize_stable_diffusion()
    
    # 예측할 음성 파일 경로
    file_path = 'datasets/test/DC_sa11.wav'
    
    # 감정 예측 및 이미지 생성
    predict_emotion_and_generate_image(emotion_model, file_path, class_names, stable_diffusion_pipe)

if __name__ == "__main__":
    main()
