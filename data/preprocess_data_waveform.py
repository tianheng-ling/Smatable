import os
import glob
import random
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


def extract_waveform_features(file_name: str):
    try:
        # Load wave file
        audio, _ = librosa.load(
            file_name,
            sr=None,
            mono=False,
            offset=0.25,
            duration=1,  # None->use the file's original sample rate 44.1kHz
        )
        # sample_rate = 44100
        # len(audio) = [2, 44100]
        # len(audio) / sample_rate} = 1 s
        return audio.T
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None


def process_subject_data(subjects_list, behaviors_list, data_path, suffix_1, suffix_2):
    features = []
    labels = []
    for subject in subjects_list:
        for behavior in behaviors_list:
            file_dir = os.path.join(data_path, subject, behavior)
            if not os.path.exists(file_dir):
                print(f"Directory not found: {file_dir}")
                continue

            file_pattern = os.path.join(file_dir, f"*{suffix_1}_*.wav")
            files = glob.glob(file_pattern)
            for file_name in files:
                waveform_s12 = extract_waveform_features(file_name)

                file_name_s34 = file_name.replace(suffix_1, suffix_2)
                if not os.path.exists(file_name_s34):
                    print(f"File not found: {file_name_s34}")
                    continue
                waveform_s34 = extract_waveform_features(file_name_s34)
                if waveform_s12 is not None and waveform_s34 is not None:
                    # combine X[44100, 2] + Y[44100, 2] = [44100, 4]
                    combined_features = np.hstack([waveform_s12, waveform_s34])
                    features.append(combined_features)  # [44100, 4]
                    labels.append(behavior)  # e.g., "UP"
                else:
                    print(
                        f"Feature extraction failed for: {file_name} or {file_name_s34}"
                    )
    return np.array(features), np.array(labels)


def reshape_for_scaler(x):
    return x.reshape(-1, x.shape[-1])


def reshape_back(x, shape):
    return x.reshape(shape[0], shape[1], shape[2])


def augment_waveforms(
    x_data,
    apply_noise=True,
    apply_flip=True,
    apply_amplitude_scaling=True,
    apply_time_masking=True,
    max_time_mask_ratio=0.2,
    noise_std=0.005,
):
    x_aug = []
    for x in x_data:
        x = x.copy()

        if apply_noise and random.random() < 0.5:
            noise = np.random.normal(0, noise_std, size=x.shape)
            x += noise

        if apply_flip and x.shape[-1] == 4 and random.random() < 0.3:
            x[:, [0, 1, 2, 3]] = x[:, [2, 3, 0, 1]]

        if apply_amplitude_scaling and random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            x *= scale

        if apply_time_masking and random.random() < 0.5:
            length = x.shape[0]
            mask_len = int(length * max_time_mask_ratio)
            start = random.randint(0, length - mask_len)
            x[start : start + mask_len, :] = 0

        x_aug.append(x)
    return np.array(x_aug)


def preprocess_data_waveform(
    behaviors_list: list,
    train_subjects_list: list,
    test_subjects_list: list,
    data_path: str,
    normalization_type: str,
    use_val_split: bool = False,
    val_split: float = 0.2,
    random_state: int = 42,
    apply_augmentation: bool = False,
):

    # 1. Load raw train/test data by person
    x_train, y_train = process_subject_data(
        subjects_list=train_subjects_list,
        behaviors_list=behaviors_list,
        data_path=data_path,
        suffix_1="S12",
        suffix_2="S34",
    )
    # shape of X_train: (240, 44100, 4)
    # shape of Y_train: (240,) -> ['RIGHT' 'RIGHT' ... 'LEFT' 'LEFT']

    x_test, y_test = process_subject_data(
        subjects_list=test_subjects_list,
        behaviors_list=behaviors_list,
        data_path=data_path,
        suffix_1="S12",
        suffix_2="S34",
    )
    # shape of X_test: (120, 44100, 4)
    # shape of Y_test: (120,)

    # 2. Label Encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(
        y_train
    )  # e.g., ['UP' 'DOWN' 'RIGHT' 'LEFT'] -> [2 3 1 0]
    y_test_encoded = le.transform(y_test) if len(y_test) > 0 else []

    # 3.Dataset Splitting
    if use_val_split:
        x_train, x_val, y_train_encoded, y_val_encoded = train_test_split(
            x_train,
            y_train_encoded,
            test_size=val_split,
            stratify=y_train_encoded,
            random_state=random_state,
        )
    else:
        x_val, y_val_encoded = None, None

    if apply_augmentation:
        x_train = augment_waveforms(x_train)

    # 4. Data normalization
    x_train_2d = reshape_for_scaler(x_train)
    x_test_2d = reshape_for_scaler(x_test)
    if x_val is not None:
        x_val_2d = reshape_for_scaler(x_val)

    normalization_type = normalization_type.lower()
    scaler_cls = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "maxabs": MaxAbsScaler,
    }.get(normalization_type)

    if scaler_cls is None:
        raise ValueError(f"Invalid normalization type: {normalization_type}")

    scaler = scaler_cls()
    x_train_2d_normed = scaler.fit_transform(x_train_2d)
    x_test_2d_normed = scaler.transform(x_test_2d)
    if x_val is not None:
        x_val_2d_normed = scaler.transform(x_val_2d)

    x_train = reshape_back(x_train_2d_normed, x_train.shape)
    x_test = reshape_back(x_test_2d_normed, x_test.shape)
    x_val = reshape_back(x_val_2d_normed, x_val.shape) if x_val is not None else None

    print(f"Training data shape: {x_train.shape}")
    if x_val is not None:
        print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")

    return (
        x_train,
        y_train_encoded,
        x_val,
        y_val_encoded,
        x_test,
        y_test_encoded,
        le.classes_,
    )
