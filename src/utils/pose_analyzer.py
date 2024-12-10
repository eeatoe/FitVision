import cv2
import mediapipe as mp
import json

# Инициализация MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1) # 0 - быстрый, 1 - умеренный, 2 - медленный

# Открытие видеофайла
video_path = "src/utils/20241122_124628_segment_1.mp4"
cap = cv2.VideoCapture(video_path)

# Хранение результатов
results_data = []

# Функция для анализа кадра
frame_number = 0
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # Конвертация изображения в формат RGB (MediaPipe требует RGB)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Добавление эффекта гауссовского размытия
  frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), 0)

  # Обработка кадра для определения ключевых точек
  results = pose.process(frame_rgb)

  if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    frame_data = {
      "frame": frame_number,
      "keypoints": {
        landmark_name.name: {
          "x": landmark.x,
          "y": landmark.y,
          "z": landmark.z,
          "visibility": landmark.visibility,
        }
        for landmark_name, landmark in zip(mp_pose.PoseLandmark, landmarks)
        if landmark_name.value > 10  # Исключаем координаты лица
      }
    }
    results_data.append(frame_data)

  # Рисование ключевых точек и соединений
  if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Рисование ключевых точек и соединений
    for landmark in landmarks:
      x = int(landmark.x * frame.shape[1])
      y = int(landmark.y * frame.shape[0])
      cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Рисуем точку на суставе

    # Рисуем линии между точками для тела (связи между суставами)
    for connection in mp_pose.POSE_CONNECTIONS:
      start_idx, end_idx = connection
      start = landmarks[start_idx]
      end = landmarks[end_idx]
      start_coords = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
      end_coords = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
      cv2.line(frame, start_coords, end_coords, (0, 0, 255), 2)  # Рисуем линию между точками

    # Показываем кадр с визуализацией
    cv2.imshow('Pose Estimation', frame)

    # Для выхода из цикла нажмите 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  
  frame_number += 1

cap.release()
pose.close()

# Запись в JSON-файл
output_json_path = "pose_data.json"
with open(output_json_path, "w") as json_file:
    json.dump(results_data, json_file, indent=4)