# from moviepy import VideoFileClip, vfx
from moviepy import *

def video_mirroring(video_path, video_title, output_dir):
  clip = VideoFileClip(video_path)
  clip = clip.resized(width=1080)
  mirrored_clip = clip.fx(vfx.mirror_x)

  output_path = f'{output_dir}/{video_title[:-4]}_mirrored.mp4'
  mirrored_clip.write_videofile(output_path)