import os
from video_split import random_cut
from video_mirroring import video_mirroring

raw_dir = "dataset/raw/"
processed_dir = "dataset/processed/"

print(f"Текущая рабочая директория: {os.getcwd()}")

# Проходим по всем директориям и файлам в base_dir
for root, dirs, files in os.walk(raw_dir):
  for file in files:
    if file.lower().endswith(('.mp4')):
      video_path = root
      video_title = file

      # Путь к директории, где будут храниться нарезанные видео
      relative_path = os.path.relpath(root, raw_dir)
      processed_subdir = os.path.join(processed_dir, relative_path)

      # Создаем директорию, если в processed_dir еще нет всех директорий, которые есть в raw_dir
      if os.path.exists(processed_subdir):
        print(f"Директория '{processed_subdir}' уже существует.")
      else:
        os.makedirs(processed_subdir, exist_ok=True)
        print(f"Директория '{processed_subdir}' готова для использования.")
      
      # Нарезаем видео (параметры нарезки укажите в функции)
      random_cut(video_path, video_title, processed_subdir, 3, 10)  # Замените на нужные параметры


path = "dataset/raw/correct"
video_title = "20241122_130306.mp4"