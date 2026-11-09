
from pathlib import Path
import hashlib
import json
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# KONFIGURASI - BASELINE TIDAK DIUBAH
# ============================================================

ROOT = Path(__file__).resolve().parent

BASE_MANIFEST = ROOT / "dataset/dataset_splits.csv"

DEMAND_MANIFEST = ROOT / (
    "dataset/robustness_v1/splits/demand_partition_v1.csv"
)

DEMAND_WINDOWS = ROOT / (
    "results/robustness_v1/demand_window_rms.csv"
)

YAMNET_FILE = ROOT / (
    "../mobile/assets/models/"
    "yamnet_embedding_dynamic_int8.tflite"
)

DATA_DIR = ROOT / "dataset/robustness_v1/embeddings"
MODEL_DIR = ROOT / "models/robustness_v1"
RESULT_DIR = ROOT / "results/robustness_v1/training_v1"

EMBEDDING_FILE = (
    DATA_DIR / "fixed_window_dynamic_int8.npz"
)

BEST_MODEL = MODEL_DIR / "classifier_best.keras"
FINAL_MODEL = MODEL_DIR / "classifier_final.keras"

TFLITE_FILE = (
    MODEL_DIR / "danger_classifier_fp32.tflite"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680
RMS_THRESHOLD = 0.008

MAX_WINDOWS_PER_ORIGINAL_FILE = 8
MAX_DEMAND_WINDOWS_PER_SCENE = 80

# Bobot lebih rendah karena DEMAND masih pseudo-label.
DEMAND_TRAIN_SAMPLE_WEIGHT = 0.6

SEED = 42
EPOCHS = 60
BATCH_SIZE = 32

CLASSES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

LABEL_ID = {
    name: index
    for index, name in enumerate(CLASSES)
}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}


# ============================================================
# UTILITIES
# ============================================================

def sha256(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def assert_inputs():

    required = [
        BASE_MANIFEST,
        DEMAND_MANIFEST,
        DEMAND_WINDOWS,
        YAMNET_FILE,
    ]

    for path in required:

        if not path.is_file():
            raise FileNotFoundError(
                f"File diperlukan tidak ditemukan: {path}"
            )

    # Cegah model eksperimen baru tertimpa.
    outputs = [
        EMBEDDING_FILE,
        BEST_MODEL,
        FINAL_MODEL,
        TFLITE_FILE,
    ]

    for path in outputs:

        if path.exists():
            raise FileExistsError(
                "Artefak eksperimen sudah ada dan "
                f"tidak akan ditimpa: {path}"
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_audio(value):

    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio tidak ditemukan: {path}"
        )

    return path


def read_audio(path):

    waveform, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False
    )

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Sample rate audio {path} = "
            f"{sample_rate}, seharusnya 16000."
        )

    if waveform.ndim == 2:

        waveform = waveform.mean(
            axis=1,
            dtype=np.float32
        )

    if waveform.ndim != 1 or len(waveform) == 0:
        raise ValueError(
            f"Audio kosong atau tidak valid: {path}"
        )

    return waveform.astype(
        np.float32,
        copy=False
    )


def rms(window):

    values = window.astype(np.float64)

    return float(
        np.sqrt(
            np.mean(values * values)
        )
    )


# ============================================================
# FIXED WINDOW
# ============================================================

def choose_windows(waveform, limit):

    # Audio lebih pendek dari 0,975 detik
    # diberi zero-padding.
    if len(waveform) < WINDOW_SIZE:

        padded = np.zeros(
            WINDOW_SIZE,
            dtype=np.float32
        )

        padded[:len(waveform)] = waveform

        return [(0, padded)]

    starts = list(
        range(
            0,
            len(waveform) - WINDOW_SIZE + 1,
            HOP_SIZE
        )
    )

    count = min(
        limit,
        len(starts)
    )

    # Pilih window yang tersebar sepanjang audio.
    # Deterministik untuk reproducibility.
    picks = sorted(
        set(
            np.linspace(
                0,
                len(starts) - 1,
                count,
                dtype=int
            )
        )
    )

    return [
        (
            starts[index],
            waveform[
                starts[index]:
                starts[index] + WINDOW_SIZE
            ]
        )
        for index in picks
    ]


# ============================================================
# YAMNET TFLITE
# ============================================================

def load_yamnet():

    model = tf.lite.Interpreter(
        model_path=str(YAMNET_FILE),
        num_threads=2
    )

    input_detail = model.get_input_details()[0]

    # Resize tensor dinamis seperti pada Flutter.
    model.resize_tensor_input(
        input_detail["index"],
        [WINDOW_SIZE],
        strict=False
    )

    model.allocate_tensors()

    input_detail = model.get_input_details()[0]
    output_detail = model.get_output_details()[0]

    if (
        list(input_detail["shape"]) != [WINDOW_SIZE]
        or int(np.prod(output_detail["shape"])) != 1024
    ):
        raise ValueError(
            "Shape YAMNet tidak sesuai 15600 -> 1024."
        )

    if (
        input_detail["dtype"] != np.float32
        or output_detail["dtype"] != np.float32
    ):
        raise ValueError(
            "Input/output YAMNet harus float32."
        )

    print(
        "YAMNet TFLite siap:",
        input_detail["shape"],
        "->",
        output_detail["shape"]
    )

    return (
        model,
        input_detail["index"],
        output_detail["index"]
    )


def embedding_for(
    yamnet,
    input_id,
    output_id,
    window
):

    waveform = np.asarray(
        window,
        dtype=np.float32
    ).reshape(WINDOW_SIZE)

    yamnet.set_tensor(
        input_id,
        waveform
    )

    yamnet.invoke()

    embedding = yamnet.get_tensor(
        output_id
    ).reshape(-1).astype(np.float32)

    if (
        embedding.shape != (1024,)
        or not np.isfinite(embedding).all()
    ):
        raise ValueError(
            "Embedding YAMNet tidak valid."
        )

    return embedding


# ============================================================
# VALIDASI MANIFEST BASELINE
# ============================================================

def validate_base(df):

    required = {
        "path",
        "target_class",
        "split",
        "source_group",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "Kolom dataset_splits.csv kurang: "
            f"{required - set(df.columns)}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError(
            "Manifest baseline memiliki nilai kosong."
        )

    if not set(df["target_class"]).issubset(LABEL_ID):
        raise ValueError(
            "Ada kelas baseline tidak dikenal."
        )

    if not set(df["split"]).issubset(VALID_SPLITS):
        raise ValueError(
            "Ada nama split baseline tidak dikenal."
        )

    cross = (
        df.groupby("source_group")["split"]
        .nunique()
    )

    if (cross > 1).any():
        raise ValueError(
            "Source group baseline melintasi split."
        )


# ============================================================
# EKSTRAKSI EMBEDDING
# ============================================================

def get_features():

    yamnet, input_id, output_id = load_yamnet()

    base = pd.read_csv(BASE_MANIFEST)

    validate_base(base)

    demand = pd.read_csv(DEMAND_MANIFEST)
    windows = pd.read_csv(DEMAND_WINDOWS)

    required_demand = [
        "segment_id",
        "scene",
        "audio_path",
        "split",
        "start_sec",
        "end_sec",
        "source_group",
    ]

    for column in required_demand:

        if column not in demand.columns:
            raise ValueError(
                f"Kolom DEMAND tidak ditemukan: {column}"
            )

    required_windows = [
        "segment_id",
        "scene",
        "start_sec",
        "activity",
    ]

    for column in required_windows:

        if column not in windows.columns:
            raise ValueError(
                f"Kolom window tidak ditemukan: {column}"
            )

    if demand["segment_id"].duplicated().any():
        raise ValueError(
            "Ada segment_id DEMAND duplikat."
        )

    if (
        demand.groupby("source_group")["split"]
        .nunique() > 1
    ).any():
        raise ValueError(
            "Source DEMAND melintasi split."
        )

    X = []
    y = []
    splits = []
    weights = []
    metadata = []

    # ========================================================
    # SIMPAN SATU EMBEDDING
    # ========================================================

    def add(
        window,
        label,
        split,
        source,
        source_group,
        position,
        origin,
        weight
    ):

        if rms(window) < RMS_THRESHOLD:
            return False

        embedding = embedding_for(
            yamnet,
            input_id,
            output_id,
            window
        )

        X.append(embedding)
        y.append(LABEL_ID[label])
        splits.append(split)
        weights.append(weight)

        metadata.append({
            "source_file": str(source),
            "source_group": str(source_group),
            "window_start_sec": round(
                position / SAMPLE_RATE,
                5
            ),
            "class": label,
            "split": split,
            "origin": origin,
            "label_status": (
                "clip_label_inherited_unverified_per_window"
                if origin == "original"
                else "demand_pseudo_background_unverified"
            ),
        })

        return True

    # ========================================================
    # DATASET LAMA
    # ========================================================

    print()
    print("[1/3] EKSTRAKSI EMBEDDING DATASET LAMA")
    print("=" * 65)

    count_base = 0

    for index, row in enumerate(
        base.itertuples(index=False),
        start=1
    ):

        audio_path = resolve_audio(row.path)

        waveform = read_audio(audio_path)

        selected_windows = choose_windows(
            waveform,
            MAX_WINDOWS_PER_ORIGINAL_FILE
        )

        for offset, window in selected_windows:

            added = add(
                window=window,
                label=row.target_class,
                split=row.split,
                source=audio_path,
                source_group=row.source_group,
                position=offset,
                origin="original",
                weight=1.0
            )

            count_base += int(added)

        if index % 200 == 0 or index == len(base):

            print(
                f"Audio {index}/{len(base)} | "
                f"Embedding: {count_base}"
            )

    # ========================================================
    # DATASET DEMAND - HANYA TRAIN
    # ========================================================

    print()
    print("[2/3] EKSTRAKSI EMBEDDING DEMAND TRAIN")
    print("=" * 65)

    demand_train = demand[
        demand["split"] == "train"
    ].copy()

    if demand_train.empty:
        raise ValueError(
            "Tidak ada DEMAND untuk training."
        )

    active = windows[
        windows["activity"] == "ACTIVE"
    ].copy()

    train_segments = demand_train[
        [
            "segment_id",
            "scene",
            "source_group",
            "audio_path",
            "start_sec",
            "end_sec",
        ]
    ].rename(
        columns={
            "start_sec": "segment_start_sec",
            "end_sec": "segment_end_sec",
        }
    )

    active = active.merge(
        train_segments,
        on=["segment_id", "scene"],
        how="inner",
        validate="many_to_one"
    )

    # Pastikan window berada di dalam
    # segmen yang telah diseleksi.
    active = active[
        (
            active["start_sec"]
            >= active["segment_start_sec"] - 1e-5
        )
        &
        (
            active["start_sec"] + WINDOW_SIZE / SAMPLE_RATE
            <= active["segment_end_sec"] + 1e-5
        )
    ]

    count_demand = 0

    for scene, group in active.groupby(
        "scene",
        sort=True
    ):

        group = group.sort_values("start_sec")

        count = min(
            MAX_DEMAND_WINDOWS_PER_SCENE,
            len(group)
        )

        picked_indices = sorted(
            set(
                np.linspace(
                    0,
                    len(group) - 1,
                    count,
                    dtype=int
                )
            )
        )

        chosen = group.iloc[picked_indices]

        cached_path = None
        waveform = None

        scene_count = 0

        for row in chosen.itertuples(index=False):

            audio_path = resolve_audio(
                row.audio_path
            )

            if audio_path != cached_path:

                waveform = read_audio(
                    audio_path
                )

                cached_path = audio_path

            position = int(
                round(
                    float(row.start_sec)
                    * SAMPLE_RATE
                )
            )

            window = waveform[
                position:
                position + WINDOW_SIZE
            ]

            if len(window) != WINDOW_SIZE:
                raise ValueError(
                    f"Window DEMAND tidak lengkap: {scene}"
                )

            added = add(
                window=window,
                label="background",
                split="train",
                source=audio_path,
                source_group=row.source_group,
                position=position,
                origin="demand_pseudo",
                weight=DEMAND_TRAIN_SAMPLE_WEIGHT
            )

            scene_count += int(added)
            count_demand += int(added)

        print(
            f"{scene:<12}: {scene_count} embedding"
        )

    # ========================================================
    # SAVE DATASET EMBEDDING
    # ========================================================

    print()
    print("[3/3] MENYIMPAN EMBEDDING")
    print("=" * 65)

    if not X:
        raise ValueError(
            "Tidak ada embedding terkumpul."
        )

    X = np.stack(X).astype(np.float32)

    y = np.asarray(
        y,
        dtype=np.int64
    )

    splits = np.asarray(
        splits,
        dtype=str
    )

    weights = np.asarray(
        weights,
        dtype=np.float32
    )

    metadata_df = pd.DataFrame(metadata)

    leakage = (
        metadata_df.groupby("source_group")["split"]
        .nunique()
    )

    if (leakage > 1).any():
        raise ValueError(
            "DATA LEAKAGE: source_group lintas split."
        )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        present_labels = set(
            y[splits == split]
        )

        if present_labels != set(range(len(CLASSES))):
            raise ValueError(
                f"Split {split} tidak memiliki "
                f"seluruh kelas: {present_labels}"
            )

    np.savez_compressed(
        EMBEDDING_FILE,
        embeddings=X,
        labels=y,
        splits=splits,
        sample_weights=weights,
        class_names=np.asarray(CLASSES)
    )

    metadata_df.to_csv(
        RESULT_DIR / "fixed_window_provenance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("Embedding shape :", X.shape)
    print("Dataset lama    :", count_base)
    print("DEMAND train    :", count_demand)

    print()
    print("DISTRIBUSI EMBEDDING")
    print("-" * 65)

    print(
        metadata_df.groupby(
            ["split", "class", "origin"]
        ).size().to_string()
    )

    return (
        X,
        y,
        splits,
        weights,
        metadata_df,
        count_demand
    )


# ============================================================
# CLASSIFIER
# ============================================================

def build_classifier():

    inputs = tf.keras.Input(
        shape=(1024,),
        name="yamnet_embedding"
    )

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    outputs = tf.keras.layers.Dense(
        6,
        activation="softmax",
        name="sound_class"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="danger_sound_robustness_classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# EVALUASI DAN CONFUSION MATRIX
# ============================================================

def save_evaluation(
    name,
    model,
    X,
    y,
    metadata
):

    probabilities = model.predict(
        X,
        batch_size=128,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    metrics = {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "windows_evaluated": int(len(y)),
    }

    matrix = confusion_matrix(
        y,
        predictions,
        labels=list(range(6))
    )

    pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES
    ).to_csv(
        RESULT_DIR / f"{name}_confusion_matrix.csv"
    )

    report = classification_report(
        y,
        predictions,
        labels=list(range(6)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).T.to_csv(
        RESULT_DIR / f"{name}_classification_report.csv"
    )

    prediction_df = metadata.reset_index(
        drop=True
    ).copy()

    prediction_df["actual_label"] = [
        CLASSES[i] for i in y
    ]

    prediction_df["predicted_label"] = [
        CLASSES[i] for i in predictions
    ]

    prediction_df["predicted_confidence"] = (
        probabilities.max(axis=1)
    )

    for i, class_name in enumerate(CLASSES):

        prediction_df[
            f"prob_{class_name}"
        ] = probabilities[:, i]

    prediction_df.to_csv(
        RESULT_DIR / f"{name}_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(matrix)

    fig.colorbar(image, ax=ax)

    ax.set_xticks(
        range(6),
        CLASSES,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(6),
        CLASSES
    )

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual (label klip diwariskan)")
    ax.set_title(f"Confusion Matrix - {name}")

    for i in range(6):
        for j in range(6):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9
            )

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / f"{name}_confusion_matrix.png",
        dpi=220
    )

    plt.close(fig)

    print(
        name.upper(),
        "| Accuracy:",
        f"{metrics['accuracy']:.2%}",
        "| Macro F1:",
        f"{metrics['macro_f1']:.2%}"
    )

    return metrics


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    assert_inputs()

    (
        X,
        y,
        splits,
        source_weights,
        metadata,
        demand_count
    ) = get_features()

    masks = {
        name: splits == name
        for name in VALID_SPLITS
    }

    train_labels = y[
        masks["train"]
    ]

    class_ids = np.arange(6)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=train_labels
    )

    sample_weights = (
        source_weights[masks["train"]]
        * class_weights[train_labels]
    )

    print()
    print("=" * 70)
    print("MULAI RETRAINING CLASSIFIER ROBUSTNESS V1")
    print("=" * 70)

    model = build_classifier()

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_MODEL),
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=4,
            factor=0.5,
            min_lr=1e-6
        ),
    ]

    history = model.fit(
        X[masks["train"]],
        train_labels,
        sample_weight=sample_weights,
        validation_data=(
            X[masks["validation"]],
            y[masks["validation"]]
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    pd.DataFrame(
        history.history
    ).to_csv(
        RESULT_DIR / "training_history.csv",
        index_label="epoch_zero_based"
    )

    # Gunakan model dengan val_loss terbaik.
    model = tf.keras.models.load_model(
        BEST_MODEL
    )

    model.save(
        FINAL_MODEL
    )

    # ========================================================
    # GRAFIK TRAINING
    # ========================================================

    hist = pd.DataFrame(
        history.history
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        hist["loss"],
        label="Training loss"
    )

    ax.plot(
        hist["val_loss"],
        label="Validation loss"
    )

    ax.set_xlabel("Epoch (0-based)")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / "training_loss.png",
        dpi=220
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        hist["accuracy"],
        label="Training accuracy"
    )

    ax.plot(
        hist["val_accuracy"],
        label="Validation accuracy"
    )

    ax.set_xlabel("Epoch (0-based)")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULT_DIR / "training_accuracy.png",
        dpi=220
    )

    plt.close(fig)

    # ========================================================
    # VALIDATION DAN TEST
    # ========================================================

    metrics = {}

    for name in [
        "validation",
        "test",
    ]:

        metrics[name] = save_evaluation(
            name,
            model,
            X[masks[name]],
            y[masks[name]],
            metadata.loc[masks[name]]
        )

    # ========================================================
    # EXPORT CLASSIFIER TFLITE FP32
    # ========================================================

    print()
    print("EXPORT TFLITE FP32...")

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converted = converter.convert()

    TFLITE_FILE.write_bytes(
        converted
    )

    # Verifikasi input/output model baru.
    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_FILE)
    )

    interpreter.allocate_tensors()

    t_input = interpreter.get_input_details()[0]
    t_output = interpreter.get_output_details()[0]

    if (
        list(t_input["shape"]) != [1, 1024]
        or int(np.prod(t_output["shape"])) != 6
    ):
        raise ValueError(
            "Shape classifier TFLite baru tidak sesuai."
        )

    # ========================================================
    # PERBANDINGAN KERAS VS TFLITE
    # ========================================================

    probe = X[masks["test"]][:1]

    interpreter.set_tensor(
        t_input["index"],
        probe
    )

    interpreter.invoke()

    lite_pred = interpreter.get_tensor(
        t_output["index"]
    )

    keras_pred = model(
        probe,
        training=False
    ).numpy()

    difference = float(
        np.max(
            np.abs(lite_pred - keras_pred)
        )
    )

    print(
        "Max absolute error Keras vs TFLite:",
        round(difference, 7)
    )

    if difference > 0.01:
        raise ValueError(
            "Output Keras dan TFLite terlalu berbeda."
        )

    # ========================================================
    # LABEL JSON
    # ========================================================

    labels = {
        str(i): class_name
        for i, class_name in enumerate(CLASSES)
    }

    (
        MODEL_DIR / "labels.json"
    ).write_text(
        json.dumps(
            labels,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # SAVE CONFIG DAN METRICS
    # ========================================================

    config = {
        "experiment": "robustness_v1_prototype",
        "seed": SEED,
        "classes": CLASSES,
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "rms_threshold": RMS_THRESHOLD,
        "max_windows_per_original_file":
            MAX_WINDOWS_PER_ORIGINAL_FILE,
        "max_demand_windows_per_scene":
            MAX_DEMAND_WINDOWS_PER_SCENE,
        "demand_training_sample_weight":
            DEMAND_TRAIN_SAMPLE_WEIGHT,
        "demand_train_embeddings": demand_count,
        "yamnet_sha256": sha256(YAMNET_FILE),
        "base_manifest_sha256": sha256(BASE_MANIFEST),
        "demand_manifest_sha256": sha256(DEMAND_MANIFEST),
        "model_tflite_sha256": sha256(TFLITE_FILE),
        "tensorflow_version": tf.__version__,
        "metrics": metrics,
        "limitations": [
            "Label klip diwariskan ke setiap window.",
            "Belum tersedia anotasi event setiap window.",
            "DEMAND pseudo-background belum diverifikasi manusia.",
            "DEMAND hanya digunakan untuk training.",
            "Validation dan test DEMAND tidak digunakan sebagai accuracy supervised.",
            "Metrik fixed-window tidak dapat dibandingkan langsung dengan baseline per-file.",
            "Diperlukan pengujian independen dan false alert per jam sebelum klaim robustness."
        ]
    }

    (
        MODEL_DIR / "training_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    (
        RESULT_DIR / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("RETRAINING ROBUSTNESS V1 SELESAI")
    print("=" * 70)

    print("Model terbaik :", BEST_MODEL)
    print("Model akhir   :", FINAL_MODEL)
    print("TFLite FP32   :", TFLITE_FILE)
    print("Embedding     :", EMBEDDING_FILE)
    print("Hasil metrik  :", RESULT_DIR)

    print()
    print(
        "MODEL BASELINE TIDAK DITIMPA."
    )

    print(
        "JANGAN PASANG TFLITE BARU KE FLUTTER "
        "SEBELUM EVALUASI ROBUSTNESS."
    )


if __name__ == "__main__":
    main()
