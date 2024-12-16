from dataProcessing import load_data, preprocess_data
from model import MLPModel, train_model
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # Загрузка данных
    json_file_path = 'pose_data.json'
    features, labels = load_data(json_file_path)

    # Подготовка данных
    X_train, X_val, y_train, y_val = preprocess_data(features, labels)

    # Создание модели
    input_size = X_train.shape[1]
    model = MLPModel(input_size)

    # Определение функции потерь и оптимизатора
    criterion = nn.BCELoss()  # Для бинарной классификации
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Сохранение модели
    torch.save(model.state_dict(), 'squat_model.pth')
    print("Модель сохранена в файл squat_model.pth")

if __name__ == "__main__":
    main()
