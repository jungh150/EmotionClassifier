from tensorflow.keras.models import load_model
from utils import extract_features
import numpy as np

def predict_emotion(model, file_path, class_names):
    """
    주어진 음성 파일의 감정을 예측하고 출력.
    :param model: 감정 분류 모델
    :param file_path: 예측할 음성 파일 경로
    :param class_names: 감정 클래스 이름 리스트
    """
    # 음성 파일에서 특징 추출
    features = extract_features(file_path)
    
    # 3차원 형태로 변환 (batch_size=1, time_steps=1, features)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    
    # 예측
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)  # 가장 높은 확률의 클래스 인덱스
    
    print(f"Predicted Emotion: {class_names[predicted_class]}")

def main():
    # 사전 학습된 모델 로드
    model = load_model('model/emotion_classifier.h5')
    
    # 감정 클래스 이름 정의 (라벨 순서와 동일해야 함)
    class_names = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted"]
    
    # 예측할 음성 파일 경로
    file_path = 'datasets/test/DC_sa11.wav'
    
    # 감정 예측
    predict_emotion(model, file_path, class_names)

if __name__ == "__main__":
    main()