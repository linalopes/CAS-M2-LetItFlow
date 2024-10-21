from moviepy.editor import VideoFileClip

video_path = "LetItFlow-RAW-Lina/IMG_6287.mov"
output_audio_path = "extracted_audio/your_audio.wav"

video = VideoFileClip(video_path)
audio = video.audio
audio.write_audiofile(output_audio_path, codec='pcm_s16le')
video.close()
