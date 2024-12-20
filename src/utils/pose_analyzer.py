import cv2
import os
import mediapipe as mp
import json

def pose_analyzer(video_path):
  cap = cv2.VideoCapture(video_path)
  pose = mp.solutions.pose.Pose(model_complexity=1) # 0 - быстрый, 1 - умеренный, 2 - медленный

  results_data = []
  frame_number = 0

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Конвертация изображения в формат RGB (MediaPipe требует RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Добавление эффекта гауссовского размытия
    frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
      # Сериализация результатов в JSON
      frame_data = {
        "frame": frame_number,
        "keypoints": {
          landmark_name.name: {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
          }
          for landmark_name, landmark in zip(mp.solutions.pose.PoseLandmark, results.pose_landmarks.landmark)
          if landmark_name.value > 10 # Исключаем координаты лица
        }
      }
      results_data.append(frame_data)

    frame_number += 1

  cap.release()
  pose.close()
  return results_data

def save_json(data, output_path):
  with open(output_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

# Главная логика
def process_videos(processed_dir, poses_dir):
  for root, _, files in os.walk(processed_dir):
    for file in files:
      if file.lower().endswith('.mp4'):
        video_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, processed_dir)
        poses_subdir = os.path.join(poses_dir, relative_path)
        os.makedirs(poses_subdir, exist_ok=True)

        print(f"Обработка видео: {video_path}")
        landmarks = pose_analyzer(video_path)

        output_json_path = os.path.join(poses_subdir, f"{file[:-4]}.json")
        save_json(landmarks, output_json_path)
        print(f"Результаты сохранены: {output_json_path}")

# Инициализация путей
processed_dir = "dataset/processed/"
poses_dir = "dataset/poses/"
print(f"Текущая рабочая директория: {os.getcwd()}")

process_videos(processed_dir, poses_dir)