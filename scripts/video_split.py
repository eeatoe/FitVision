import random
from moviepy import VideoFileClip

def random_cut(video_path, video_title, output_dir, min_length=3, max_length=10):
  # Загружаем видео
  clip = VideoFileClip(video_path + '/' + video_title)
  video_duration = int(clip.duration) # Округляем вниз, чтобы не было лишних проблем

  if video_duration <= 10:
    print("Видео слишком короткое")
    return

  # Рандомно создаем временные отрезки для видео
  segments_duration = [0]
  while video_duration > segments_duration[-1]:
    next_time = segments_duration[-1] + random.randint(min_length, max_length)
    segments_duration.append(min(next_time, video_duration))
    print(segments_duration)

  # Нарезаем видео по временным отрезкам
  for index in range(len(segments_duration) - 1):
    subclip = clip.subclipped(segments_duration[index], segments_duration[index + 1])
    subclip_resized = subclip.resized(width=1080)
    
    # Сохраняем результат
    output_path = f'{output_dir}/{video_title[:-4]}_segment_{index}.mp4'
    subclip_resized.write_videofile(output_path, codec="libx264", audio=False)

  clip.close()
  print("Видео успешно нарезано")