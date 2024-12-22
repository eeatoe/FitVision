import torch
from model import MLPModel  # Импорт вашей модели
import numpy as np
import json

# Функция для загрузки тестовых данных из JSON
def load_test_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    test_entry = data[0]  # Берём первую запись
    keypoints = test_entry['keypoints']
    
    # Извлекаем координаты x, y, z для первых 22 точек
    test_data = []
    for point in list(keypoints.keys())[:22]:  # Только первые 22 точки
        for axis in ['x', 'y', 'z']:  # Все три координаты
            test_data.append(keypoints[point][axis])
    
    print(f"Количество признаков: {len(test_data)}")  # Должно быть 66
    return np.array([test_data])  # Возвращаем данные в формате (1, 66)

# Функция для тестирования модели
def test_model():
    # Путь к сохранённой модели
    model_path = 'squat_model.pth'

    # Создание модели с входом размером 66
    input_size = 66  # Входной размер для модели
    model = MLPModel(input_size)  # Создаём модель с соответствующим размером входа
    model.load_state_dict(torch.load(model_path))  # Загружаем сохранённые веса модели
    model.eval()  # Устанавливаем модель в режим тестирования

    # Загрузка тестовых данных из JSON
    json_file_path = 'src/pose_data.json'  # Путь к вашему JSON файлу
    test_data = load_test_data(json_file_path)

    # Преобразуем данные в тензор для PyTorch
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Прогнозируем результаты
    with torch.no_grad():  # Отключаем вычисление градиентов для теста
        predictions = model(test_data)
    
    # Печатаем результат
    print("Предсказания:", predictions.detach().numpy())  # Выводим результат

if __name__ == "__main__":
    test_model()
