import os
import glob
import numpy as np
from scipy import signal
from sklearn.preprocessing import LabelEncoder
import librosa


def spectrogram(filename, duration=2.0, gain=6.0, nperseg=4096):

    # Load wave file
    x, sr = librosa.load(filename, sr=None, mono=False, duration=duration)

    # rescale to botch channels
    x = gain * x
    xL, xR = x[0], x[1]

    # compute STFT for both channels
    f, t, ZL = signal.stft(xL, fs=sr, nperseg=nperseg)
    f, t, ZR = signal.stft(xR, fs=sr, nperseg=nperseg)

    # convert to log-magnitude spectrogram
    PL = 10 * np.log(np.abs(ZL))
    PR = 10 * np.log(np.abs(ZR))

    # concatenate vertically (frequency axis)
    freq = np.concatenate([PL, PR], axis=0)

    return freq


def process_subject_data(subjects_list, behaviors_list, data_path, suffix_1, suffix_2):
    features = []
    labels = []
    for subject in subjects_list:
        for behavior in behaviors_list:
            file_dir = os.path.join(data_path, subject, behavior)
            if not os.path.isdir(file_dir):
                print(f"Directory not found: {file_dir}")
                continue

            file_pattern = os.path.join(file_dir, f"*{suffix_1}_*.wav")
            files = glob.glob(file_pattern)

            for file_name in files:
                stft_s12 = spectrogram(file_name)

                file_name_s34 = file_name.replace(suffix_1, suffix_2)
                if not os.path.exists(file_name_s34):
                    print(f"File not found: {file_name_s34}")
                    continue

                stft_s34 = spectrogram(file_name_s34)
                if stft_s12 is not None and stft_s34 is not None:
                    features.append(np.concatenate([stft_s12, stft_s34], axis=1))
                    labels.append(behavior)
                else:
                    print(
                        f"Feature extraction failed for: {file_name} or {file_name_s34}"
                    )

    # stack into a 3D array (N, freq_bins, time_frames)
    X = np.stack(features, axis=0)
    y = np.array(labels)
    return X, y


def preprocess_data_stft(
    behaviors_list: list,
    train_subjects_list: list,
    test_subjects_list: list,
    data_path: str,
):
    # 1) Data extraction
    x_train, y_train = process_subject_data(
        subjects_list=train_subjects_list,
        behaviors_list=behaviors_list,
        data_path=data_path,
        suffix_1="S12",
        suffix_2="S34",
    )

    x_test, y_test = process_subject_data(
        subjects_list=test_subjects_list,
        behaviors_list=behaviors_list,
        data_path=data_path,
        suffix_1="S12",
        suffix_2="S34",
    )

    # 2) Label encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(
        y_train
    )  # e.g., ['UP' 'DOWN' 'RIGHT' 'LEFT'] -> [2 3 1 0]
    y_test_encoded = le.transform(y_test) if len(y_test) > 0 else []

    x_val_normed = None  # Placeholder
    y_val_encoded = None  # Placeholder
    return (
        x_train,
        y_train_encoded,
        x_val_normed,
        y_val_encoded,
        x_test,
        y_test_encoded,
        le.classes_,
    )
