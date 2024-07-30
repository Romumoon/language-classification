import os
import librosa
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_features(audio_file):
    audio, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    return mfcc

def process_file(audio_path, label, features_file, labels_file):
    feature = extract_features(audio_path)

    with lock:
        features = np.load(features_file, allow_pickle=True)
        labels = np.load(labels_file)

        features = np.append(features, [feature], axis=0)
        labels = np.append(labels, label)

        np.save(features_file, features)
        np.save(labels_file, labels)

data_dir = r"D:\Downloads\VoxDataBase\\"
label_to_language = {
    0: "English",
    1: "Portuguese",
    2: "Deutsch",
    3: "French"
}
language_to_label = {
    "en": 0,
    "pt": 1,
    "de": 2,
    "fr": 3
}

script_dir = os.path.dirname(__file__)


files_dir = os.path.join(script_dir, 'Files')
os.makedirs(files_dir, exist_ok=True)

features_file = os.path.join(files_dir, 'features.npy')
labels_file = os.path.join(files_dir, 'labels.npy')

if not os.path.exists(features_file):
    np.save(features_file, np.empty((0,)))
if not os.path.exists(labels_file):
    np.save(labels_file, np.empty((0,), dtype='int'))

total_files = sum(len(files) for _, _, files in os.walk(data_dir))
processed_files = 0

from threading import Lock
lock = Lock()

tasks = []

with ThreadPoolExecutor(max_workers=4) as executor:
    for language_folder, label in language_to_label.items():
        language_dir = os.path.join(data_dir, language_folder)
        for audio_file in os.listdir(language_dir):
            audio_path = os.path.join(language_dir, audio_file)
            future = executor.submit(process_file, audio_path, label, features_file, labels_file)
            tasks.append(future)

    for future in as_completed(tasks):
        processed_files += 1
        progress = (processed_files / total_files) * 100
        sys.stdout.write(f"\rProgress: {progress:.2f}% ({processed_files}/{total_files} files)")
        sys.stdout.flush()

print("\nLabel to Language mapping:", label_to_language)
print("Language to Label mapping:", language_to_label)
