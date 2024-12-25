import torch
from model import MLPModel  # Импорт вашей модели
import numpy as np
import json

# Функция для загрузки всех тестовых данных из JSON
def load_test_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Создаём массив для всех фреймов
    all_test_data = []
    
    # Обрабатываем каждую запись (фрейм)
    for test_entry in data:
        keypoints = test_entry['keypoints']
        
        # Проверяем и исключаем метку, если она есть
        if 'label' in test_entry:
            print("ВНИМАНИЕ: Найдена метка 'label' в данных. Она будет игнорирована.")
            del test_entry['label']  # Удаляем метку из текущей записи
        
        # Извлекаем координаты x, y, z для первых 12 точек
        test_data = []
        for point in list(keypoints.keys())[:12]:  # Только первые 12 точки
            for axis in ['x', 'y', 'z']:  # Все три координаты
                test_data.append(keypoints[point][axis])
        
        # Добавляем обработанную запись в массив
        all_test_data.append(test_data)
    
    print(f"Общее количество записей: {len(all_test_data)}")
    print(f"Количество признаков в каждой записи: {len(all_test_data[0])}")  # Должно быть 36

    return np.array(all_test_data)  # Возвращаем массив данных в формате (N, 36), где N — количество записей

# Функция для тестирования модели
def test_model():
    # Путь к сохранённой модели
    model_path = 'squat_model.pth'

    # Создание модели с входом размером 36
    input_size = 36  # Входной размер для модели
    model = MLPModel(input_size)  # Создаём модель с соответствующим размером входа
    model.load_state_dict(torch.load(model_path))  # Загружаем сохранённые веса модели
    model.eval()  # Устанавливаем модель в режим тестирования

    # Загрузка тестовых данных из JSON
    json_file_path = 'dataset/testData/IMG_6090_segment_0.json'  # Путь к вашему JSON файлу
    test_data = load_test_data(json_file_path)

    # Преобразуем данные в тензор для PyTorch
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Прогнозируем результаты
    with torch.no_grad():  # Отключаем вычисление градиентов для теста
        predictions = model(test_data)
    
    # Проверка диапазона предсказаний
    predictions_np = predictions.detach().numpy()
    if np.any(predictions_np < 0) or np.any(predictions_np > 1):
        print("ВНИМАНИЕ: Предсказания вне диапазона [0, 1]. Проверьте активацию на выходном слое.")
    
    # Печатаем результаты
    print("Предсказания для всех записей:")
    for i, pred in enumerate(predictions_np):
        print(f"Запись {i + 1}: {pred}")

if __name__ == "__main__":
    test_model()
