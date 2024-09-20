import os
from moviepy.editor import VideoFileClip

# Define the folder containing the .mov videos
input_folder = "LetItFlow-RAW-Martina"
output_folder = "Martina"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over all .mov files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".MOV"):
        video_path = os.path.join(input_folder, filename)
        output_audio_path = os.path.join(output_folder, filename.replace(".MOV", ".wav"))
        
        try:
            # Load the video
            print(f"Processing {filename}...")
            video = VideoFileClip(video_path)
            
            # Extract audio
            audio = video.audio
            
            # Save the extracted audio as a .wav file
            audio.write_audiofile(output_audio_path, codec='pcm_s16le')  # PCM is a common uncompressed codec for .wav
            
            # Close video clip to free up memory
            video.close()
            print(f"Audio saved: {output_audio_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Audio extraction completed!")
