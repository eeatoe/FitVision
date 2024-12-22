from dataProcessing import load_data, preprocess_data
from model import MLPModel, train_model
import torch
import torch.nn as nn
import torch.optim as optim
from collector_Json import merge_json_files
import os
def main():
    # объединение всех json файлов
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = "dataset/poses/"
    pose_data = os.path.join(project_root, "pose_data.json")
    merge_json_files(input_dir, pose_data)
    # Проверка существования объединенного файла
    if not os.path.exists(pose_data):
        print(f"Ошибка: объединённый файл {pose_data} не был создан.")
        return
    # Загрузка данных
    json_file_path = pose_data
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
