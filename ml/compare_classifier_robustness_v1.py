
from pathlib import Path
import hashlib
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parent

EMBEDDINGS = ROOT / (
    "dataset/robustness_v1/embeddings/"
    "fixed_window_dynamic_int8.npz"
)

DEMAND_MANIFEST = ROOT / (
    "dataset/robustness_v1/splits/"
    "demand_partition_v1.csv"
)

DEMAND_WINDOWS = ROOT / (
    "results/robustness_v1/demand_window_rms.csv"
)

DEMAND_AUDIO = ROOT / (
    "dataset/robustness_v1/demand_review"
)

YAMNET_PATH = ROOT / (
    "../mobile/assets/models/"
    "yamnet_embedding_dynamic_int8.tflite"
)

MODELS = {
    "baseline": ROOT / (
        "../mobile/assets/models/"
        "danger_classifier_fp32.tflite"
    ),
    "robustness_v1": ROOT / (
        "models/robustness_v1/"
        "danger_classifier_fp32.tflite"
    ),
}

OUT = ROOT / (
    "results/robustness_v1/comparison_v1"
)

CLASSES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

THRESHOLDS = {
    "alarm": 0.70,
    "glass_breaking": 0.80,
    "scream": 0.70,
    "siren": 0.70,
    "vehicle_horn": 0.70,
}

CONFIRM = {
    "alarm": 2,
    "glass_breaking": 1,
    "scream": 2,
    "siren": 2,
    "vehicle_horn": 2,
}

WINDOW = 15600
SR = 16000


def digest(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1048576),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def interpreter(path, input_shape, output_length):

    model = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2
    )

    inp = model.get_input_details()[0]

    if list(inp["shape"]) != input_shape:
        model.resize_tensor_input(
            inp["index"],
            input_shape,
            strict=False
        )

    model.allocate_tensors()

    inp = model.get_input_details()[0]
    out = model.get_output_details()[0]

    valid_shape = (
        list(inp["shape"]) == input_shape
        and int(np.prod(out["shape"])) == output_length
    )

    if not valid_shape:
        raise ValueError(
            f"Shape model salah: {path.name}"
        )

    if (
        inp["dtype"] != np.float32
        or out["dtype"] != np.float32
    ):
        raise ValueError(
            f"Tensor bukan FP32: {path.name}"
        )

    return (
        model,
        inp["index"],
        out["index"],
        tuple(input_shape),
    )


def predict(bundle, values):

    model, input_id, output_id, shape = bundle

    data = np.asarray(
        values,
        dtype=np.float32
    ).reshape(shape)

    model.set_tensor(input_id, data)
    model.invoke()

    result = model.get_tensor(
        output_id
    ).ravel()

    if not np.isfinite(result).all():
        raise ValueError(
            "Output model mengandung NaN atau Inf."
        )

    return result


def classify(scores):

    index = int(np.argmax(scores))
    label = CLASSES[index]
    confidence = float(scores[index])

    passes = (
        label in THRESHOLDS
        and confidence >= THRESHOLDS[label]
    )

    return label, confidence, passes


# =====================================================
# TEST DATASET LAMA PADA EMBEDDING YANG SAMA
# =====================================================

def evaluate_test(npz, models):

    X = npz["embeddings"]
    y = npz["labels"]
    splits = npz["splits"].astype(str)

    if list(npz["class_names"].astype(str)) != CLASSES:
        raise ValueError(
            "Urutan kelas tidak sesuai."
        )

    if not (len(X) == len(y) == len(splits)):
        raise ValueError(
            "Jumlah data embedding tidak sesuai."
        )

    mask = splits == "test"

    X_test = X[mask]
    y_test = y[mask]

    if len(X_test) == 0:
        raise ValueError(
            "Test split kosong."
        )

    metrics = []
    prediction_rows = []

    for name, model in models.items():

        print(
            "Evaluasi test classifier:",
            name
        )

        probabilities = np.stack([
            predict(model, embedding)
            for embedding in X_test
        ])

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                labels=list(range(6)),
                average="macro",
                zero_division=0,
            )
        )

        background_mask = y_test == 1

        background_horn = int(
            (
                (predictions == 5)
                & background_mask
            ).sum()
        )

        background_danger = int(
            (
                (predictions != 1)
                & background_mask
            ).sum()
        )

        background_above_threshold = sum(
            bool(classify(scores)[2])
            for scores in probabilities[background_mask]
        )

        metrics.append({
            "model": name,
            "test_windows": len(y_test),
            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions
                )
            ),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "background_windows": int(
                background_mask.sum()
            ),
            "background_predicted_horn":
                background_horn,
            "background_predicted_any_danger":
                background_danger,
            "background_danger_above_threshold":
                background_above_threshold,
        })

        report = classification_report(
            y_test,
            predictions,
            labels=list(range(6)),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report).T.to_csv(
            OUT / (
                f"{name}_test_classification_report.csv"
            )
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=list(range(6))
        )

        pd.DataFrame(
            matrix,
            index=CLASSES,
            columns=CLASSES
        ).to_csv(
            OUT / (
                f"{name}_test_confusion_matrix.csv"
            )
        )

        for index, (real, scores) in enumerate(
            zip(y_test, probabilities)
        ):

            label, confidence, passed = classify(
                scores
            )

            prediction_rows.append({
                "model": name,
                "test_window_index": index,
                "actual_label": CLASSES[int(real)],
                "predicted_label": label,
                "confidence": confidence,
                "passes_threshold": bool(passed),
            })

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(prediction_rows),
    )


# =====================================================
# DEMAND TEST: TBUS DAN STRAFFIC
# =====================================================

def evaluate_demand(yamnet, models):

    manifest = pd.read_csv(
        DEMAND_MANIFEST
    )

    windows = pd.read_csv(
        DEMAND_WINDOWS
    )

    selected = manifest.loc[
        manifest["split"] == "test",
        ["segment_id", "scene"]
    ]

    if selected["segment_id"].duplicated().any():
        raise ValueError(
            "Ada segment_id test yang duplikat."
        )

    candidates = windows.merge(
        selected,
        on=["segment_id", "scene"],
        how="inner"
    )

    candidates = candidates.sort_values(
        ["scene", "start_sec"]
    ).reset_index(drop=True)

    if candidates.empty:
        raise ValueError(
            "Tidak ada window DEMAND test."
        )

    rows = []

    for scene, group in candidates.groupby(
        "scene",
        sort=True
    ):

        audio_path = (
            DEMAND_AUDIO / f"{scene}_ch01.wav"
        )

        if not audio_path.is_file():
            raise FileNotFoundError(
                audio_path
            )

        audio, rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False
        )

        if rate != SR or audio.ndim != 1:
            raise ValueError(
                f"Format audio tidak sesuai: {scene}"
            )

        for item in group.itertuples(index=False):

            active = (
                str(item.activity) == "ACTIVE"
            )

            if active:

                position = int(
                    round(
                        float(item.start_sec) * SR
                    )
                )

                waveform = audio[
                    position:position + WINDOW
                ]

                if len(waveform) != WINDOW:
                    raise ValueError(
                        f"Window tidak lengkap: {scene}"
                    )

                embedding = predict(
                    yamnet,
                    waveform
                )

            for name, model in models.items():

                if active:
                    scores = predict(
                        model,
                        embedding
                    )
                else:
                    scores = np.array([
                        0, 1, 0, 0, 0, 0
                    ])

                label, confidence, passed = (
                    classify(scores)
                )

                rows.append({
                    "model": name,
                    "scene": scene,
                    "segment_id": item.segment_id,
                    "start_sec": float(
                        item.start_sec
                    ),
                    "activity": str(item.activity),
                    "top1_label": label,
                    "confidence": confidence,
                    "passes_threshold": bool(passed),
                    "confirmation_candidate": False,
                })

        print(
            f"DEMAND test {scene}: "
            f"{len(group)} window"
        )

    df = pd.DataFrame(rows)

    df = df.sort_values([
        "model",
        "scene",
        "segment_id",
        "start_sec"
    ]).reset_index(drop=True)

    # Simulasi konfirmasi sederhana per segmen.
    # Belum mereplikasi latch Flutter sepenuhnya.
    for _, group in df.groupby(
        ["model", "segment_id"],
        sort=False
    ):

        previous = None
        consecutive = 0

        for idx, item in group.iterrows():

            label = (
                item.top1_label
                if item.passes_threshold
                else None
            )

            if label is None:

                previous = None
                consecutive = 0

            else:

                if label == previous:
                    consecutive += 1
                else:
                    consecutive = 1

                previous = label

                if consecutive == CONFIRM[label]:
                    df.at[
                        idx,
                        "confirmation_candidate"
                    ] = True

    summary = (
        df.groupby(
            ["model", "scene"],
            as_index=False
        )
        .agg(
            windows=("segment_id", "size"),
            active_windows=(
                "activity",
                lambda x: int(
                    (x == "ACTIVE").sum()
                )
            ),
            danger_top1=(
                "top1_label",
                lambda x: int(
                    (x != "background").sum()
                )
            ),
            above_threshold=(
                "passes_threshold",
                "sum"
            ),
            confirmation_candidates=(
                "confirmation_candidate",
                "sum"
            ),
        )
    )

    return summary, df


# =====================================================
# MAIN
# =====================================================

def main():

    required = [
        EMBEDDINGS,
        DEMAND_MANIFEST,
        DEMAND_WINDOWS,
        YAMNET_PATH,
        *MODELS.values(),
    ]

    for path in required:

        if not path.is_file():
            raise FileNotFoundError(path)

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("PERBANDINGAN CLASSIFIER LAMA VS BARU")
    print("=" * 70)

    models = {
        name: interpreter(
            path,
            [1, 1024],
            6
        )
        for name, path in MODELS.items()
    }

    yamnet = interpreter(
        YAMNET_PATH,
        [WINDOW],
        1024
    )

    with np.load(
        EMBEDDINGS,
        allow_pickle=False
    ) as npz:

        test_summary, test_rows = (
            evaluate_test(
                npz,
                models
            )
        )

    demand_summary, demand_rows = (
        evaluate_demand(
            yamnet,
            models
        )
    )

    test_summary.to_csv(
        OUT / "test_model_comparison.csv",
        index=False
    )

    test_rows.to_csv(
        OUT / "test_window_predictions.csv",
        index=False
    )

    demand_summary.to_csv(
        OUT / "demand_test_scene_comparison.csv",
        index=False
    )

    demand_rows.to_csv(
        OUT / "demand_test_window_predictions.csv",
        index=False
    )

    metadata = {
        "purpose": (
            "Compare classifiers on identical "
            "fixed-window embeddings"
        ),
        "limitations": [
            "DEMAND labels remain unverified.",
            "DEMAND split selected after baseline inspection.",
            "Confirmation does not simulate Flutter latch.",
            "Original labels inherited at window level.",
            "The classifiers used different training preprocessing."
        ],
        "file_sha256": {
            str(path): digest(path)
            for path in required
        }
    }

    (
        OUT / "experiment_config.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("HASIL PERBANDINGAN TEST 6 KELAS")
    print("=" * 70)

    print(
        test_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIK DEMAND TEST")
    print("=" * 70)

    print(
        demand_summary.to_string(
            index=False
        )
    )

    print()
    print("Hasil disimpan di:", OUT)
    print("Model baseline tidak diubah.")


if __name__ == "__main__":
    main()
