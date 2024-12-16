import torch
from dataProcessing import SquatDataset
from model import SquatModel
from torch.utils.data import DataLoader

def evaluate(model, dataloader):
    model.eval()  # Переводим модель в режим оценки
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # Для бинарной классификации
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Пример оценки модели
if __name__ == "__main__":
    dataset = SquatDataset('../../pose_data.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = SquatModel()
    model.load_state_dict(torch.load("squat_model.pth", weights_only=True))  # Загрузите обученную модель
    evaluate(model, dataloader)