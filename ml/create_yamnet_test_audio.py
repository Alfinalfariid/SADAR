from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


MANIFEST = Path(
    "dataset/dataset_splits.csv"
)

OUTPUT = Path(
    "dataset/yamnet_test_audio.npz"
)

SAMPLE_RATE = 16000


CLASS_NAMES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


def main():

    df = pd.read_csv(
        MANIFEST
    )

    test_df = df[
        df["split"] == "test"
    ].copy()

    waveforms = []
    labels = []
    filenames = []

    print("=" * 60)
    print("MEMBUAT YAMNET TEST AUDIO")
    print("=" * 60)

    failed = 0

    for _, row in test_df.iterrows():

        try:

            audio, sr = sf.read(
                row["path"],
                dtype="float32",
                always_2d=False
            )

            if sr != SAMPLE_RATE:
                raise ValueError(
                    f"Sample rate {sr}"
                )

            if audio.ndim == 2:

                audio = np.mean(
                    audio,
                    axis=1
                )

            audio = audio.astype(
                np.float32
            )

            waveforms.append(
                audio
            )

            labels.append(
                CLASS_TO_ID[
                    row["target_class"]
                ]
            )

            filenames.append(
                str(row["path"])
            )

        except Exception as e:

            failed += 1

            print(
                "GAGAL:",
                row["path"],
                e
            )

    # Object array karena panjang audio
    # tiap file berbeda-beda.
    waveforms = np.array(
        waveforms,
        dtype=object
    )

    labels = np.array(
        labels,
        dtype=np.int64
    )

    filenames = np.array(
        filenames,
        dtype=object
    )

    class_names = np.array(
        CLASS_NAMES
    )

    np.savez_compressed(
        OUTPUT,
        waveforms=waveforms,
        labels=labels,
        filenames=filenames,
        class_names=class_names
    )

    print()
    print("=" * 60)

    print(
        "Jumlah test:",
        len(waveforms)
    )

    print(
        "Gagal:",
        failed
    )

    print(
        "Output:",
        OUTPUT
    )

    print()
    print("DISTRIBUSI TEST")

    for i, name in enumerate(
        CLASS_NAMES
    ):

        count = np.sum(
            labels == i
        )

        print(
            f"{name:<20}: {count}"
        )


if __name__ == "__main__":
    main()