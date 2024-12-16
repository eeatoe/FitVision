import torch
import torch.nn as nn

class SquatModel(nn.Module):
    def __init__(self):
        super(SquatModel, self).__init__()
        # Изменяем input_size на 66, так как у нас 66 признаков
        self.gru = nn.GRU(input_size=66, hidden_size=64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)  # Для бинарной классификации, используем 1 выходной нейрон
    
    def forward(self, x):
        out, _ = self.gru(x)  # Получаем выход из GRU
        out = out[:, -1, :]  # Берем последний временной шаг
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))  # Для бинарной классификации
        return out
    