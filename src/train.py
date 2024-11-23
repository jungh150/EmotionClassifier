from utils import load_data, preprocess_data
from emotionClassifier import build_model
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # 데이터 로드 및 전처리
    csv_path = 'datasets/emotion_melpath_dataset.csv'
    file_paths, labels = load_data(csv_path)
    X, y = preprocess_data(file_paths, labels)

    # 학습 및 검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Conv1D + LSTM 모델 생성
    input_shape = (X.shape[1], X.shape[2])  # (time_steps, features)
    num_classes = len(np.unique(y))
    model = build_model(input_shape, num_classes)

    # 모델 학습
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

    # 모델 저장
    model.save('model/emotion_classifier.h5')
    print("Model saved as emotion_classifier.h5")

if __name__ == "__main__":
    main()
