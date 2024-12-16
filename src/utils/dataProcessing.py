import json
import torch
from torch.utils.data import Dataset
import numpy as np

class SquatDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame = self.data[idx]
        keypoints = frame["keypoints"]
        
        # Извлекаем координаты x, y, z для каждого сустава
        keypoints_data = []
        for joint in keypoints.values():
            keypoints_data.extend([joint["x"], joint["y"], joint["z"]])
        
        # Преобразуем список в numpy массив
        keypoints_data = np.array(keypoints_data, dtype=np.float32)
        
        # Пример: метки могут быть 1 для правильного приседания и 0 для неправильного
        label = 1  # Замените на вашу логику определения метки (например, правильность приседания)
        
        return torch.tensor(keypoints_data), torch.tensor(label, dtype=torch.long)