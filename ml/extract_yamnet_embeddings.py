from pathlib import Path
import csv
import time

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub


# ============================================================
# CONFIG
# ============================================================

MANIFEST_FILE = Path(
    "dataset/dataset_splits.csv"
)

OUTPUT_DIR = Path(
    "dataset/embeddings"
)

OUTPUT_NPZ = OUTPUT_DIR / "yamnet_embeddings.npz"

OUTPUT_CSV = OUTPUT_DIR / "yamnet_embeddings_metadata.csv"

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

TARGET_SAMPLE_RATE = 16000


CLASS_NAMES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

CLASS_TO_ID = {
    class_name: index
    for index, class_name
    in enumerate(CLASS_NAMES)
}


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    waveform, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False
    )

    # Stereo -> mono
    if waveform.ndim == 2:

        waveform = np.mean(
            waveform,
            axis=1
        )

    if waveform.ndim != 1:

        raise ValueError(
            f"Format audio tidak valid: "
            f"{waveform.shape}"
        )

    if sample_rate != TARGET_SAMPLE_RATE:

        raise ValueError(
            f"Sample rate {sample_rate} Hz, "
            f"seharusnya {TARGET_SAMPLE_RATE} Hz"
        )

    if len(waveform) == 0:

        raise ValueError(
            "Audio kosong."
        )

    waveform = waveform.astype(
        np.float32
    )

    return waveform


# ============================================================
# EXTRACT YAMNET EMBEDDING
# ============================================================

def extract_embedding(
    yamnet,
    waveform
):

    waveform_tensor = (
        tf.convert_to_tensor(
            waveform,
            dtype=tf.float32
        )
    )

    scores, embeddings, spectrogram = (
        yamnet(
            waveform_tensor
        )
    )

    # embeddings:
    # [jumlah_frame, 1024]
    #
    # Karena classifier kita bekerja per-file,
    # semua frame dari satu audio dirata-ratakan.
    file_embedding = tf.reduce_mean(
        embeddings,
        axis=0
    )

    file_embedding = (
        file_embedding
        .numpy()
        .astype(np.float32)
    )

    return (
        file_embedding,
        embeddings.shape[0]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("YAMNET EMBEDDING EXTRACTION")
    print("=" * 75)

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Manifest tidak ditemukan:\n"
            f"{MANIFEST_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = pd.read_csv(
        MANIFEST_FILE
    )

    print()
    print(
        "Jumlah file:",
        len(df)
    )

    print(
        "Jumlah kelas:",
        df["target_class"].nunique()
    )

    print()
    print(
        df[
            "target_class"
        ].value_counts()
    )


    # ========================================================
    # VALIDASI CLASS
    # ========================================================

    unknown_classes = set(
        df["target_class"].unique()
    ) - set(
        CLASS_NAMES
    )

    if unknown_classes:

        raise ValueError(
            f"Ada kelas tidak dikenal: "
            f"{unknown_classes}"
        )


    # ========================================================
    # LOAD YAMNET
    # ========================================================

    print()
    print(
        "Loading YAMNet dari TensorFlow Hub..."
    )

    print(
        "Pertama kali dijalankan mungkin "
        "perlu download model."
    )

    start_load = time.time()

    yamnet = hub.load(
        YAMNET_URL
    )

    load_time = (
        time.time()
        - start_load
    )

    print(
        f"YAMNet berhasil dimuat "
        f"({load_time:.2f} detik)"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    all_embeddings = []
    all_labels = []
    all_splits = []
    all_filenames = []

    metadata_rows = []

    failed = []

    start_total = time.time()


    # ========================================================
    # PROCESS
    # ========================================================

    for index, row in df.iterrows():

        filename = row[
            "filename"
        ]

        class_name = row[
            "target_class"
        ]

        split = row[
            "split"
        ]

        audio_path = Path(
            row["path"]
        )


        try:

            waveform = load_audio(
                audio_path
            )

            embedding, frame_count = (
                extract_embedding(
                    yamnet,
                    waveform
                )
            )


            # ================================================
            # VALIDASI EMBEDDING
            # ================================================

            if embedding.shape != (1024,):

                raise ValueError(
                    f"Embedding shape salah: "
                    f"{embedding.shape}"
                )


            label_id = (
                CLASS_TO_ID[
                    class_name
                ]
            )


            all_embeddings.append(
                embedding
            )

            all_labels.append(
                label_id
            )

            all_splits.append(
                split
            )

            all_filenames.append(
                filename
            )


            metadata_rows.append({

                "filename":
                    filename,

                "target_class":
                    class_name,

                "label_id":
                    label_id,

                "split":
                    split,

                "dataset":
                    row[
                        "dataset"
                    ],

                "source_group":
                    row[
                        "source_group"
                    ],

                "yamnet_frames":
                    int(
                        frame_count
                    ),
            })


        except Exception as error:

            failed.append(
                {
                    "filename":
                        filename,

                    "error":
                        str(error),
                }
            )

            print()
            print(
                f"ERROR {filename}: "
                f"{error}"
            )


        # ================================================
        # PROGRESS
        # ================================================

        current = index + 1

        if (
            current % 50 == 0
            or current == len(df)
        ):

            elapsed = (
                time.time()
                - start_total
            )

            print(
                f"[{current:4}/{len(df)}] "
                f"selesai | "
                f"{elapsed:.1f} detik"
            )


    # ========================================================
    # CONVERT ARRAY
    # ========================================================

    X = np.stack(
        all_embeddings
    ).astype(
        np.float32
    )

    y = np.array(
        all_labels,
        dtype=np.int64
    )

    splits = np.array(
        all_splits
    )

    filenames = np.array(
        all_filenames
    )


    # ========================================================
    # SAVE NPZ
    # ========================================================

    np.savez_compressed(

        OUTPUT_NPZ,

        embeddings=X,

        labels=y,

        splits=splits,

        filenames=filenames,

        class_names=np.array(
            CLASS_NAMES
        ),
    )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    total_time = (
        time.time()
        - start_total
    )

    print()
    print("=" * 75)
    print("HASIL EKSTRAKSI")
    print("=" * 75)

    print(
        "Embedding shape :",
        X.shape
    )

    print(
        "Label shape     :",
        y.shape
    )

    print(
        "Berhasil        :",
        len(X)
    )

    print(
        "Gagal           :",
        len(failed)
    )

    print(
        f"Waktu total     : "
        f"{total_time:.2f} detik"
    )


    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    print()
    print("JUMLAH EMBEDDING PER SPLIT")
    print("-" * 45)

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        count = np.sum(
            splits == split_name
        )

        print(
            f"{split_name:<15}: "
            f"{count}"
        )


    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    print()
    print("JUMLAH EMBEDDING PER KELAS")
    print("-" * 45)

    for class_name in CLASS_NAMES:

        class_id = (
            CLASS_TO_ID[
                class_name
            ]
        )

        count = np.sum(
            y == class_id
        )

        print(
            f"{class_id} "
            f"{class_name:<20}: "
            f"{count}"
        )


    # ========================================================
    # FAILED SUMMARY
    # ========================================================

    if failed:

        print()
        print("FILE GAGAL")
        print("-" * 45)

        for item in failed:

            print(
                item["filename"],
                "->",
                item["error"]
            )


    print()
    print(
        "Embedding:"
    )

    print(
        OUTPUT_NPZ
    )

    print()

    print(
        "Metadata:"
    )

    print(
        OUTPUT_CSV
    )

    print()


if __name__ == "__main__":
    main()