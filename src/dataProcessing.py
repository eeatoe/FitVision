import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch

def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    features = []
    labels = []
    for entry in data:
        keypoints = entry['keypoints']
        features.append([keypoints[point][axis] for point in keypoints for axis in ['x', 'y', 'z']])
        labels.append(entry['label'])
    print()
    return np.array(features), np.array(labels)

def preprocess_data(features, labels):
    # Нормализация данных
    features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Преобразование в тензоры PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    return X_train, X_val, y_train, y_val
