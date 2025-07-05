
#pip install datasets["audio"] s3fs soundfile

from datasets import load_dataset

import os
from datasets import  load_from_disk

# Check for required AWS environment variables
required_env_vars = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", 
    "AWS_ENDPOINT_URL_S3"
]

missing_vars = []
for var in required_env_vars:
    if not os.environ.get(var):
        missing_vars.append(var)

if missing_vars:
    error_message = f"Missing required environment variables: {', '.join(missing_vars)}\n"
    error_message += "Please export these variables before running the script:\n"
    for var in missing_vars:
        error_message += f"export {var}=<your_value>\n"
    raise EnvironmentError(error_message)

# If all variables are present, proceed with the original code
storage_options = {
    "key": os.environ["AWS_ACCESS_KEY_ID"],
    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    "client_kwargs": {"endpoint_url": os.environ["AWS_ENDPOINT_URL_S3"]}
}

PATH = "s3://burkimbia/audios/final_dataset"
ds = load_from_disk(PATH, storage_options=storage_options)
ds = ds.train_test_split(test_size=0.002, seed=42)


import os
import csv
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
import librosa
import soundfile as sf
import re
output_dir = "dataset"
wavs_dir = os.path.join(output_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)

metadata_train_path = os.path.join(output_dir, "metadata_train.csv")
metadata_eval_path = os.path.join(output_dir, "metadata_eval.csv")

def export_data(split, csv_path):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow(["audio_file", "text", "speaker_name"])

        dataset = ds[split]
        i = 0
        
        while i < len(dataset):
            try:
                sample = dataset[i]
                audio_array = sample['audio']['array']
                sampling_rate = sample['audio']['sampling_rate']
                text = sample['text']
                text = re.sub(r'^\d+\s+', '', text)
                speaker = (sample['speaker_name'].replace(" ", "")).upper()
                speaker = f"@{speaker}"
                print(speaker)
                wav_filename = f"{i:06d}.wav"
                wav_path = os.path.join(wavs_dir, wav_filename)
        

                write(wav_path, sampling_rate, np.array(audio_array, dtype=np.float32))

                writer.writerow([f"wavs/{wav_filename}", text, speaker])
                print(speaker)
                print(f"Exported {i}")
            except Exception as e:
                # Skip broken audio and continue
                print(f"Skipped {i}  : {e}")
                pass
            
            i += 1

export_data("train", metadata_train_path)
export_data("test", metadata_eval_path)
