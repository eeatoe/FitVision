import torch
import torch.optim as optim
import torch.nn as nn
from dataProcessing import SquatDataset
from model import SquatModel
from torch.utils.data import DataLoader

def train(model, dataloader, epochs=10):
    criterion = nn.BCELoss()  # Для бинарной классификации
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.unsqueeze(1)  # Добавляем размерность для батча (batch_size, sequence_length, features)
            labels = labels.float().unsqueeze(1)  # Преобразуем метки в нужный формат
            
            optimizer.zero_grad()  # Обнуляем градиенты
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Вычисляем ошибку
            loss.backward()  # Вычисляем градиенты
            optimizer.step()  # Обновляем параметры модели
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}")

# Пример тренировки модели
if __name__ == "__main__":
    dataset = SquatDataset("../../pose_data.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SquatModel()
    train(model, dataloader, epochs=10)
    torch.save(model.state_dict(), 'squat_model.pth')