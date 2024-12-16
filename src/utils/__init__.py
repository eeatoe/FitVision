from model import SquatModel
import torch

# Загрузим модель
model = SquatModel()
model.load_state_dict(torch.load("squat_model.pth"))
model.eval()

# Функция для предсказания на новом кадре
def predict(input_data):
    input_tensor = torch.tensor(input_data).unsqueeze(0)  # Преобразуем данные в тензор
    output = model(input_tensor)
    prediction = (output > 0.5).float()
    return prediction.item()

# Пример использования
input_data = [0.4546349346637726, 0.3655001223087311, 0.3462199568748474,  # Пример данных
              0.35191357135772705, 0.348592072725296, -0.3180901110172272,  # Координаты ключевых точек
              # ...
              0.5178617835044861, 0.869202196598053, 0.4039325416088104]

prediction = predict(input_data)
print(f"Prediction: {'Correct' if prediction == 1 else 'Incorrect'}")