
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf


# ============================================
# KONFIGURASI
# ============================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT.parent
    / "mobile/assets/models"
    / "yamnet_embedding_dynamic_int8.tflite"
)

DATASET_ROOT = (
    ROOT / "dataset/robustness_v3"
)

BACKGROUND_ORIGINAL = (
    DATASET_ROOT
    / "background_chunks"
    / "run_20261027_120154_693387"
    / "background_train_manifest.csv"
)

BACKGROUND_AUGMENTED = (
    DATASET_ROOT
    / "augmented_background"
    / "run_20261102_060308_345348"
    / "augmented_train_manifest.csv"
)

ALARM_COMBINED = (
    DATASET_ROOT
    / "augmented_alarm"
    / "run_20261102_063619_102882"
    / "alarm_train_manifest_v3.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "results/robustness_v3/embedding_extraction"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680
EMBEDDING_DIM = 1024

CLASS_NAMES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================
# UTILITAS
# ============================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_path(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    return path


def load_manifest(path, expected_count, name):
    if not path.is_file():
        raise FileNotFoundError(path)

    data = pd.read_csv(path).fillna("")

    required = {
        "path",
        "target_class",
        "split",
        "source_group",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"{name}: kolom hilang {sorted(missing)}"
        )

    if len(data) != expected_count:
        raise ValueError(
            f"{name}: diharapkan {expected_count} "
            f"segmen, ditemukan {len(data)}"
        )

    if not (data["split"] == "train").all():
        raise ValueError(
            f"{name}: terdapat data bukan train"
        )

    if data["source_group"].astype(str).eq("").any():
        raise ValueError(
            f"{name}: ada source_group kosong"
        )

    data = data.copy()
    data["manifest_origin"] = name

    return data


def prepare_manifest():
    background_original = load_manifest(
        BACKGROUND_ORIGINAL,
        45,
        "background_original"
    )

    background_augmented = load_manifest(
        BACKGROUND_AUGMENTED,
        45,
        "background_augmented"
    )

    alarm = load_manifest(
        ALARM_COMBINED,
        58,
        "alarm_original_and_augmented"
    )

    if not (
        background_original["target_class"]
        == "background"
    ).all():
        raise ValueError(
            "Label background original salah."
        )

    if not (
        background_augmented["target_class"]
        == "background"
    ).all():
        raise ValueError(
            "Label background augmentation salah."
        )

    if not (
        alarm["target_class"] == "alarm"
    ).all():
        raise ValueError(
            "Label alarm salah."
        )

    data = pd.concat(
        [
            background_original,
            background_augmented,
            alarm,
        ],
        ignore_index=True
    ).fillna("")

    data["resolved_path"] = data["path"].map(
        lambda value: str(resolve_path(value))
    )

    if data["resolved_path"].duplicated().any():
        raise ValueError(
            "Ada file audio yang masuk dua kali."
        )

    excluded = {
        "alarm_02_clip_001.wav",
        "alarm_02_clip_011.wav",
    }

    for value in data["resolved_path"]:
        path = Path(value)

        if not path.is_file():
            raise FileNotFoundError(path)

        if path.name in excluded:
            raise ValueError(
                f"Segmen senyap masih ditemukan: {path}"
            )

        audio_info = sf.info(path)

        if (
            audio_info.samplerate != SAMPLE_RATE
            or audio_info.channels != 1
            or audio_info.frames != 48000
        ):
            raise ValueError(
                "Audio harus WAV mono 16 kHz "
                f"durasi 3 detik: {path}"
            )

    return data


def create_interpreter():
    os.environ.setdefault(
        "TF_CPP_MIN_LOG_LEVEL", "2"
    )

    try:
        from tflite_runtime.interpreter import (
            Interpreter
        )

        backend = "tflite_runtime"

    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
        backend = "tensorflow_lite"

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        num_threads=2
    )

    input_details = (
        interpreter.get_input_details()[0]
    )

    input_shape = tuple(
        int(x) for x in input_details["shape"]
    )

    # Sama seperti Flutter: 15600 sampel.
    if input_shape != (WINDOW_SIZE,):
        interpreter.resize_tensor_input(
            input_details["index"],
            [WINDOW_SIZE],
            strict=False
        )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()[0]
    )

    output_details = (
        interpreter.get_output_details()[0]
    )

    if input_details["dtype"] != np.float32:
        raise ValueError(
            "YAMNet input tidak FLOAT32."
        )

    if output_details["dtype"] != np.float32:
        raise ValueError(
            "YAMNet embedding output bukan FLOAT32."
        )

    output_size = int(
        np.prod(output_details["shape"])
    )

    if output_size != EMBEDDING_DIM:
        raise ValueError(
            f"Output embedding {output_size}, "
            f"seharusnya {EMBEDDING_DIM}"
        )

    print("Backend:", backend)
    print("Input shape:", input_details["shape"])
    print("Output shape:", output_details["shape"])

    return (
        interpreter,
        input_details,
        output_details,
        backend
    )


def extract_embedding(
    interpreter,
    input_details,
    output_details,
    window
):
    input_shape = tuple(
        int(x) for x in input_details["shape"]
    )

    input_array = np.asarray(
        window,
        dtype=np.float32
    ).reshape(input_shape)

    interpreter.set_tensor(
        input_details["index"],
        input_array
    )

    interpreter.invoke()

    embedding = interpreter.get_tensor(
        output_details["index"]
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(-1)

    if embedding.size != EMBEDDING_DIM:
        raise ValueError(
            "Dimensi embedding tidak sesuai."
        )

    if not np.all(np.isfinite(embedding)):
        raise ValueError(
            "Embedding mengandung NaN/Infinity."
        )

    return embedding.copy()


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("ROBUSTNESS V3 - YAMNET EMBEDDING")
    print("=" * 60)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    data = prepare_manifest()

    print("Total segmen:", len(data))
    print(
        "Background:",
        int((data["target_class"] == "background").sum())
    )
    print(
        "Alarm:",
        int((data["target_class"] == "alarm").sum())
    )
    print(
        "Source group:",
        data["source_group"].nunique()
    )

    (
        interpreter,
        input_details,
        output_details,
        backend
    ) = create_interpreter()

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    output_dir = OUTPUT_ROOT / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    X = []
    y = []
    records = []

    for segment_index, row in data.iterrows():
        audio_path = Path(row["resolved_path"])

        audio, sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True
        )

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Sample rate salah: {audio_path}"
            )

        waveform = audio[:, 0]

        label = str(row["target_class"])
        label_id = CLASS_TO_ID[label]

        count = 0

        for start in range(
            0,
            len(waveform) - WINDOW_SIZE + 1,
            HOP_SIZE
        ):
            end = start + WINDOW_SIZE

            window = waveform[start:end]

            embedding = extract_embedding(
                interpreter,
                input_details,
                output_details,
                window
            )

            X.append(embedding)
            y.append(label_id)

            augmentation = str(
                row.get("augmentation", "")
            )

            if not augmentation:
                if row["manifest_origin"] == (
                    "background_augmented"
                ):
                    augmentation = (
                        "gain_plus_light_noise"
                    )
                else:
                    augmentation = "none"

            records.append({
                "embedding_index": len(X) - 1,
                "segment_index": segment_index,
                "path": str(audio_path),
                "target_class": label,
                "class_id": label_id,
                "split": "train",
                "source_group": row["source_group"],
                "manifest_origin": row["manifest_origin"],
                "augmentation": augmentation,
                "window_start_sec": round(
                    start / SAMPLE_RATE, 4
                ),
                "window_end_sec": round(
                    end / SAMPLE_RATE, 4
                ),
                "window_rms": round(
                    float(
                        np.sqrt(
                            np.mean(
                                window.astype(
                                    np.float64
                                ) ** 2
                            )
                        )
                    ),
                    6
                ),
            })

            count += 1

        if count != 5:
            raise RuntimeError(
                f"{audio_path.name}: "
                f"ditemukan {count} window, "
                "seharusnya 5."
            )

        if (segment_index + 1) % 10 == 0:
            print(
                f"Progres: {segment_index + 1}"
                f"/{len(data)} segmen"
            )

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.shape != (740, EMBEDDING_DIM):
        raise RuntimeError(
            f"Shape tidak sesuai: {X.shape}"
        )

    if y.shape != (740,):
        raise RuntimeError(
            f"Shape label tidak sesuai: {y.shape}"
        )

    metadata = pd.DataFrame(records)

    features_path = (
        output_dir / "v3_new_embeddings.npz"
    )

    metadata_path = (
        output_dir / "v3_embedding_metadata.csv"
    )

    np.savez_compressed(
        features_path,
        X=X,
        y=y
    )

    metadata.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig"
    )

    class_counts = {
        label: int(
            (metadata["target_class"] == label).sum()
        )
        for label in ("alarm", "background")
    }

    summary = {
        "stage": "robustness_v3_embedding_extraction",
        "run_id": run_id,
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "backend": backend,
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "embedding_dimension": EMBEDDING_DIM,
        "total_segments": len(data),
        "total_embeddings": len(X),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "class_counts": class_counts,
        "split": "train_only",
        "source_groups": int(
            data["source_group"].nunique()
        ),
        "manifests": [
            str(BACKGROUND_ORIGINAL),
            str(BACKGROUND_AUGMENTED),
            str(ALARM_COMBINED),
        ],
        "previous_models_modified": False,
        "training_performed": False,
    }

    summary_path = (
        output_dir / "embedding_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EKSTRAKSI EMBEDDING V3 SELESAI")
    print("=" * 60)

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)
    print("Alarm embeddings:", class_counts["alarm"])
    print(
        "Background embeddings:",
        class_counts["background"]
    )
    print("Model:", MODEL_PATH.name)
    print("Fitur:", features_path)
    print("Metadata:", metadata_path)
    print("Ringkasan:", summary_path)

    print("\nSTATUS:")
    print("- Hanya data TRAIN")
    print("- Source group tersimpan")
    print("- Model sebelumnya tidak berubah")
    print("- Belum dilakukan training V3")


if __name__ == "__main__":
    main()
