
from pathlib import Path
import hashlib
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parent

V1_EMBEDDINGS = ROOT / (
    "dataset/robustness_v1/embeddings/"
    "fixed_window_dynamic_int8.npz"
)

V1_PROVENANCE = ROOT / (
    "results/robustness_v1/training_v1/"
    "fixed_window_provenance.csv"
)

HARDNEG_MANIFEST = ROOT / (
    "dataset/robustness_v2/"
    "hard_negative_recordings.csv"
)

YAMNET_PATH = ROOT / (
    "../mobile/assets/models/"
    "yamnet_embedding_dynamic_int8.tflite"
)

OUTPUT_DIR = ROOT / (
    "dataset/robustness_v2/embeddings"
)

RESULT_DIR = ROOT / (
    "results/robustness_v2"
)

OUTPUT_NPZ = OUTPUT_DIR / (
    "combined_fixed_window_int8.npz"
)

OUTPUT_PROVENANCE = RESULT_DIR / (
    "combined_provenance.csv"
)

OUTPUT_SUMMARY = RESULT_DIR / (
    "combined_dataset_summary.csv"
)

OUTPUT_CONFIG = RESULT_DIR / (
    "merge_config.json"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680
RMS_THRESHOLD = 0.008

MAX_WINDOWS_PER_RECORDING = 64

CLASSES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

BACKGROUND_ID = CLASSES.index("background")


# ============================================================
# UTILITIES
# ============================================================

def file_hash(path):

    sha = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha.update(chunk)

    return sha.hexdigest()


def confirm_true(value):

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def resolve_audio(value):

    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio tidak ditemukan: {path}"
        )

    return path


# ============================================================
# LOAD EMBEDDING V1
# ============================================================

def load_v1():

    print()
    print("[1/4] MEMUAT EMBEDDING V1")

    with np.load(
        V1_EMBEDDINGS,
        allow_pickle=False
    ) as data:

        required = {
            "embeddings",
            "labels",
            "splits",
            "sample_weights",
            "class_names",
        }

        if not required.issubset(data.files):
            raise ValueError(
                "Struktur NPZ V1 tidak lengkap."
            )

        X = data["embeddings"].astype(
            np.float32
        )

        y = data["labels"].astype(
            np.int64
        )

        splits = data["splits"].astype(str)

        weights = data["sample_weights"].astype(
            np.float32
        )

        class_names = data["class_names"].astype(
            str
        ).tolist()

    if class_names != CLASSES:
        raise ValueError(
            "Urutan label V1 berbeda."
        )

    if X.ndim != 2 or X.shape[1] != 1024:
        raise ValueError(
            "Embedding V1 bukan 1024 dimensi."
        )

    if not (
        len(X) == len(y)
        == len(splits)
        == len(weights)
    ):
        raise ValueError(
            "Jumlah array V1 tidak sesuai."
        )

    provenance = pd.read_csv(
        V1_PROVENANCE
    )

    if len(provenance) != len(X):
        raise ValueError(
            "Jumlah provenance V1 tidak sesuai "
            "dengan jumlah embedding."
        )

    if (
        provenance["split"].astype(str).to_numpy()
        != splits
    ).any():
        raise ValueError(
            "Urutan split NPZ dan provenance "
            "tidak sesuai."
        )

    expected_labels = [
        CLASSES[int(index)]
        for index in y
    ]

    if (
        provenance["class"].astype(str).tolist()
        != expected_labels
    ):
        raise ValueError(
            "Urutan label NPZ dan provenance "
            "tidak sesuai."
        )

    print("Total embedding V1 :", len(X))

    print(
        "Dataset lama        :",
        int(
            (
                provenance["origin"] == "original"
            ).sum()
        )
    )

    print(
        "DEMAND              :",
        int(
            (
                provenance["origin"] == "demand_pseudo"
            ).sum()
        )
    )

    return X, y, splits, weights, provenance


# ============================================================
# LOAD YAMNET
# ============================================================

def load_yamnet():

    interpreter = tf.lite.Interpreter(
        model_path=str(YAMNET_PATH),
        num_threads=2
    )

    inp = interpreter.get_input_details()[0]

    interpreter.resize_tensor_input(
        inp["index"],
        [WINDOW_SIZE],
        strict=False
    )

    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    if (
        list(inp["shape"]) != [WINDOW_SIZE]
        or int(np.prod(out["shape"])) != 1024
        or inp["dtype"] != np.float32
        or out["dtype"] != np.float32
    ):
        raise ValueError(
            "YAMNet TFLite tidak sesuai."
        )

    return (
        interpreter,
        inp["index"],
        out["index"]
    )


def extract_embedding(bundle, waveform):

    model, input_id, output_id = bundle

    model.set_tensor(
        input_id,
        waveform.reshape(
            WINDOW_SIZE
        ).astype(np.float32)
    )

    model.invoke()

    embedding = model.get_tensor(
        output_id
    ).reshape(-1).astype(np.float32)

    if (
        embedding.shape != (1024,)
        or not np.isfinite(embedding).all()
    ):
        raise ValueError(
            "Embedding baru tidak valid."
        )

    return embedding


# ============================================================
# AMBIL WINDOW AKTIF
# ============================================================

def select_active_windows(audio):

    if len(audio) < WINDOW_SIZE:
        return [], 0

    candidates = []

    for start in range(
        0,
        len(audio) - WINDOW_SIZE + 1,
        HOP_SIZE
    ):

        window = audio[
            start:start + WINDOW_SIZE
        ]

        values = window.astype(
            np.float64
        )

        rms = float(
            np.sqrt(
                np.mean(values ** 2)
            )
        )

        if rms >= RMS_THRESHOLD:
            candidates.append(start)

    count = min(
        len(candidates),
        MAX_WINDOWS_PER_RECORDING
    )

    if count == 0:
        return [], 0

    indices = np.linspace(
        0,
        len(candidates) - 1,
        count,
        dtype=int
    )

    chosen = [
        candidates[index]
        for index in sorted(set(indices))
    ]

    return chosen, len(candidates)


# ============================================================
# TAMBAHKAN HARD-NEGATIVE V2
# ============================================================

def extract_hardneg():

    print()
    print("[2/4] EKSTRAKSI HARD-NEGATIVE V2")

    manifest = pd.read_csv(
        HARDNEG_MANIFEST
    )

    required = {
        "recording_id",
        "audio_path",
        "session_id",
        "source_group",
        "scene",
        "split",
        "verified_background",
        "label_status",
    }

    if not required.issubset(manifest.columns):
        raise ValueError(
            "Kolom manifest hard-negative kurang: "
            f"{required - set(manifest.columns)}"
        )

    if manifest["recording_id"].duplicated().any():
        raise ValueError(
            "Ada recording_id yang duplikat."
        )

    cross = (
        manifest.groupby("source_group")["split"]
        .nunique()
    )

    if (cross > 1).any():
        raise ValueError(
            "Session hard-negative melintasi "
            "train dan test."
        )

    # File test tidak digunakan untuk training.
    train = manifest[
        manifest["split"] == "train"
    ].copy()

    train = train[
        train["verified_background"].apply(
            confirm_true
        )
    ]

    train = train[
        train["label_status"]
        == "background_confirmed_by_recorder"
    ]

    if train.empty:
        raise ValueError(
            "Tidak ada rekaman background "
            "training yang dikonfirmasi."
        )

    yamnet = load_yamnet()

    embeddings = []
    rows = []

    for record in train.itertuples(
        index=False
    ):

        path = resolve_audio(
            record.audio_path
        )

        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=False
        )

        if sr != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(
                f"Audio harus mono 16 kHz: {path}"
            )

        chosen, total_active = (
            select_active_windows(audio)
        )

        for start in chosen:

            window = audio[
                start:start + WINDOW_SIZE
            ]

            embedding = extract_embedding(
                yamnet,
                window
            )

            embeddings.append(
                embedding
            )

            rows.append({
                "source_file": str(path),
                "source_group": record.source_group,
                "window_start_sec": round(
                    start / SAMPLE_RATE,
                    5
                ),
                "class": "background",
                "split": "train",
                "origin": "hard_negative_v2",
                "label_status": (
                    "background_confirmed_by_recorder"
                ),
            })

        print()
        print("Rekaman :", record.recording_id)
        print("Window aktif :", total_active)
        print("Embedding terpilih :", len(chosen))

    if not embeddings:
        raise ValueError(
            "Tidak ada embedding hard-negative "
            "yang berhasil dibuat."
        )

    return (
        np.stack(embeddings).astype(
            np.float32
        ),
        pd.DataFrame(rows)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DANGER SOUND AI - MERGE ROBUSTNESS V2")
    print("=" * 70)

    required = [
        V1_EMBEDDINGS,
        V1_PROVENANCE,
        HARDNEG_MANIFEST,
        YAMNET_PATH,
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Lindungi hasil jika script dijalankan ulang.
    for path in [
        OUTPUT_NPZ,
        OUTPUT_PROVENANCE,
        OUTPUT_SUMMARY,
        OUTPUT_CONFIG,
    ]:

        if path.exists():
            raise FileExistsError(
                f"File sudah ada, tidak ditimpa: {path}"
            )

    X_old, y_old, split_old, weight_old, meta_old = (
        load_v1()
    )

    X_new, meta_new = extract_hardneg()

    print()
    print("[3/4] MENGGABUNGKAN DATASET")

    X = np.concatenate(
        [X_old, X_new],
        axis=0
    )

    y = np.concatenate([
        y_old,
        np.full(
            len(X_new),
            BACKGROUND_ID,
            dtype=np.int64
        )
    ])

    splits = np.concatenate([
        split_old,
        np.full(
            len(X_new),
            "train",
            dtype=str
        )
    ])

    weights = np.concatenate([
        weight_old,
        np.ones(
            len(X_new),
            dtype=np.float32
        )
    ])

    metadata = pd.concat(
        [meta_old, meta_new],
        ignore_index=True
    )

    if len(metadata) != len(X):
        raise ValueError(
            "Jumlah metadata dan embedding berbeda."
        )

    leakage = (
        metadata.groupby("source_group")["split"]
        .nunique()
    )

    if (leakage > 1).any():
        raise ValueError(
            "DATA LEAKAGE: source_group lintas split."
        )

    # Dataset test dan validation tidak berubah.
    for split_name in ["validation", "test"]:

        before = int(
            (split_old == split_name).sum()
        )

        after = int(
            (splits == split_name).sum()
        )

        if before != after:
            raise ValueError(
                f"Split {split_name} berubah!"
            )

    print("Embedding lama :", len(X_old))
    print("Embedding baru :", len(X_new))
    print("TOTAL GABUNGAN :", len(X))
    print("Bentuk array   :", X.shape)

    print()
    print("[4/4] MENYIMPAN DATASET V2")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=X,
        labels=y,
        splits=splits,
        sample_weights=weights,
        class_names=np.asarray(CLASSES)
    )

    metadata.to_csv(
        OUTPUT_PROVENANCE,
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        metadata.groupby(
            ["split", "class", "origin"],
            as_index=False
        )
        .size()
        .rename(
            columns={"size": "embeddings"}
        )
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "experiment": "robustness_v2",
        "type": "embedding_merge",
        "yamnet_variant": "dynamic_int8",
        "yamnet_sha256": file_hash(YAMNET_PATH),
        "v1_embedding_sha256": file_hash(
            V1_EMBEDDINGS
        ),
        "hard_negative_manifest_sha256": file_hash(
            HARDNEG_MANIFEST
        ),
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "rms_threshold": RMS_THRESHOLD,
        "max_windows_per_recording":
            MAX_WINDOWS_PER_RECORDING,
        "old_embedding_count": int(len(X_old)),
        "new_embedding_count": int(len(X_new)),
        "total_embeddings": int(len(X)),
        "source_group_leakage": 0,
        "notes": [
            "V1 embeddings are reused without modification.",
            "Only confirmed hard-negative training recordings are added.",
            "Reserved test recordings are not added.",
            "Original validation and test splits are unchanged.",
            "DEMAND labels remain pseudo-labels.",
            "Original audio clip labels were inherited per window.",
            "The V2 classifier has not yet been trained."
        ]
    }

    OUTPUT_CONFIG.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("DISTRIBUSI DATASET GABUNGAN V2")
    print("=" * 70)

    print(
        summary.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("MERGE DATASET ROBUSTNESS V2 BERHASIL")
    print("=" * 70)

    print("Embedding :", OUTPUT_NPZ)
    print("Provenance:", OUTPUT_PROVENANCE)
    print("Summary   :", OUTPUT_SUMMARY)
    print("Config    :", OUTPUT_CONFIG)

    print()
    print("Source group lintas split: 0")
    print("Dataset dan model lama tetap aman.")
    print("Siap untuk training classifier V2.")


if __name__ == "__main__":
    main()
