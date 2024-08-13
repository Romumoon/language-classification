import os
import librosa
import numpy as np
import sys


def extract_features(audio_file, n_mfcc=20, max_frames=500):
    audio, sr = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_frames:
        padding = np.zeros((n_mfcc, max_frames - mfcc.shape[1]))
        mfcc = np.hstack((mfcc, padding))
    elif mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    return mfcc


script_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(script_dir, 'ProcessedData')

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

data_dir = r"D:\Downloads\VoxDataBase\teste\\"  # Path to the main folder containing language folders
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

total_files = sum(len(files) for _, _, files in os.walk(data_dir))
processed_files = 0

file_counter = 0

for language_folder, label in language_to_label.items():
    language_dir = os.path.join(data_dir, language_folder)
    for audio_file in os.listdir(language_dir):
        audio_path = os.path.join(language_dir, audio_file)
        feature = extract_features(audio_path)

        feature_file = os.path.join(processed_data_dir, f'feature_{file_counter}.npy')
        label_file = os.path.join(processed_data_dir, f'label_{file_counter}.npy')

        np.save(feature_file, feature)
        np.save(label_file, np.array([label]))

        processed_files += 1
        progress = (processed_files / total_files) * 100
        sys.stdout.write(f"\rProgress: {progress:.2f}% ({processed_files}/{total_files} files)")
        sys.stdout.flush()

        file_counter += 1

print("\nLabel to Language mapping:", label_to_language)
print("Language to Label mapping:", language_to_label)

processed_data_dir = 'ProcessedData'
feature_files = sorted([f for f in os.listdir(processed_data_dir) if f.startswith('feature_')])
label_files = sorted([f for f in os.listdir(processed_data_dir) if f.startswith('label_')])

features_list = []
labels_list = []

for feature_file, label_file in zip(feature_files, label_files):
    feature = np.load(os.path.join(processed_data_dir, feature_file))
    label = np.load(os.path.join(processed_data_dir, label_file))

    features_list.append(feature)
    labels_list.append(label)

features = np.stack(features_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

np.save('final_features.npy', features)
np.save('final_labels.npy', labels)
