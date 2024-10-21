import os
import subprocess

def extract_audio_from_mov_folder(input_folder, output_folder):
    """
    Extracts audio from all .mov files in the input folder and saves them as .wav files with pcm_s16le codec in the output folder.
    
    Parameters:
        input_folder (str): Path to the folder containing .mov files.
        output_folder (str): Path to the folder to save the extracted audio files.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".MOV"):
            input_file = os.path.join(input_folder, filename)
            # Generate output filename, replacing the .mov extension with .wav
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
            
            print(f"Processing: {input_file} -> {output_file}")
            
            # Use ffmpeg to extract audio and convert to .wav with pcm_s16le codec
            command = ['ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le', output_file]
            
            try:
                # Run ffmpeg command and capture output and errors
                result = subprocess.run(command, capture_output=True, text=True)
                
                # Print the command output
                print(f"FFmpeg Output:\n{result.stdout}")
                print(f"FFmpeg Errors (if any):\n{result.stderr}")
                
                if result.returncode == 0:
                    print(f"Successfully extracted audio from {filename}.")
                else:
                    print(f"Failed to extract audio from {filename}.")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while processing {filename}: {e}")
            except FileNotFoundError:
                print("ffmpeg is not installed or not found in PATH.")

# Define input and output folders
input_mov_folder = "mov"  # Folder containing .mov files
output_wav_folder = "wav"  # Folder to save extracted .wav files

# Call the function to process all .mov files
extract_audio_from_mov_folder(input_mov_folder, output_wav_folder)
