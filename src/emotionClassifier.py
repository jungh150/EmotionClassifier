from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense

def build_model(input_shape, num_classes):
    """
    Conv1D + LSTM 기반 모델 정의
    :param input_shape: 입력 데이터의 형태 (time_steps, features)
    :param num_classes: 출력 클래스 수
    """
    model = Sequential()
    
    # Conv1D 블록 1
    model.add(Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Conv1D 블록 2
    model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Conv1D 블록 3
    model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # LSTM 블록
    model.add(LSTM(128, return_sequences=True))  # 첫 번째 LSTM
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))  # 두 번째 LSTM
    model.add(Dropout(0.3))
    model.add(LSTM(128))  # 마지막 LSTM
    model.add(Dropout(0.3))
    
    # Dense 블록
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # 출력층
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
