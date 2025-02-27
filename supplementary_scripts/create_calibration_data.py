import os
from pydub import AudioSegment
from tqdm import tqdm

input_directory = "AMI-diart/audio_files/train"
output_directory = "AMI-diart/audio_files/calibration"
os.makedirs(output_directory, exist_ok=True)


CALIBRATION_FILE_DURATION = 30 # seconds
snippet_duration_ms = CALIBRATION_FILE_DURATION * 1000

print(f"Extracting audio files from {input_directory} and writing the first {CALIBRATION_FILE_DURATION} seconds to {output_directory}.")

for filename in tqdm(os.listdir(input_directory)):
    if filename.lower().endswith(('.wav')):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        
        full_audio = AudioSegment.from_file(input_path)
        audio_snippet = full_audio[:snippet_duration_ms]
        audio_snippet.export(output_path, format='wav')