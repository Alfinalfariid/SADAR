from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


MANIFEST = Path(
    "dataset/dataset_splits.csv"
)

OUTPUT = Path(
    "dataset/yamnet_calibration_audio.npz"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600

SAMPLES_PER_CLASS = 30
SEED = 42


def load_window(path, rng):

    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Sample rate salah: {sr}"
        )

    if audio.ndim == 2:
        audio = np.mean(
            audio,
            axis=1
        )

    # Kalau cukup panjang → ambil window acak
    if len(audio) >= WINDOW_SIZE:

        max_start = (
            len(audio)
            - WINDOW_SIZE
        )

        if max_start > 0:

            start = rng.integers(
                0,
                max_start + 1
            )

        else:
            start = 0

        audio = audio[
            start:
            start + WINDOW_SIZE
        ]

    # Kalau pendek → zero padding
    else:

        padding = (
            WINDOW_SIZE
            - len(audio)
        )

        audio = np.pad(
            audio,
            (0, padding)
        )

    return audio.astype(
        np.float32
    )


def main():

    df = pd.read_csv(
        MANIFEST
    )

    # Calibration hanya TRAIN
    train_df = df[
        df["split"] == "train"
    ]

    rng = np.random.default_rng(
        SEED
    )

    windows = []
    labels = []

    print()
    print("=" * 60)
    print("YAMNET REPRESENTATIVE DATASET")
    print("=" * 60)

    for class_name in sorted(
        train_df[
            "target_class"
        ].unique()
    ):

        class_df = train_df[
            train_df[
                "target_class"
            ] == class_name
        ]

        n = min(
            SAMPLES_PER_CLASS,
            len(class_df)
        )

        selected = class_df.sample(
            n=n,
            random_state=SEED
        )

        print(
            f"{class_name:<20}: "
            f"{n}"
        )

        for _, row in selected.iterrows():

            waveform = load_window(
                row["path"],
                rng
            )

            windows.append(
                waveform
            )

            labels.append(
                class_name
            )

    windows = np.stack(
        windows
    )

    labels = np.array(
        labels
    )

    np.savez_compressed(
        OUTPUT,
        audio_windows=windows,
        labels=labels
    )

    print()
    print(
        "Shape:",
        windows.shape
    )

    print(
        "dtype:",
        windows.dtype
    )

    print(
        "Output:",
        OUTPUT
    )


if __name__ == "__main__":
    main()