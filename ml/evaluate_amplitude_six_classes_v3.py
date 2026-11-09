
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ==================================================
# KONFIGURASI
# ==================================================

ROOT = Path(__file__).resolve().parent

MANIFEST = ROOT / "dataset" / "dataset_splits.csv"

MODEL_DIR = ROOT.parent / "mobile" / "assets" / "models"

YAMNET = (
    MODEL_DIR / "yamnet_embedding_dynamic_int8.tflite"
)

CLASSIFIER = (
    MODEL_DIR / "danger_classifier_fp32.tflite"
)

EXPECTED_YAMNET = (
    "d1e8efac7c78a7929db7049547eeab55264cad6837696e2c28280a8e4483114a"
)

EXPECTED_CLASSIFIER = (
    "90fc6136ceeae9382ff9745d4ec98a19582afebe5829e96a051b3dfa4a20ca37"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680
RMS_THRESHOLD = 0.008
MAX_WINDOWS = 8

CLASS_NAMES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

CLASS_TO_ID = {
    label: i
    for i, label in enumerate(CLASS_NAMES)
}

BACKGROUND_ID = CLASS_TO_ID["background"]
HORN_ID = CLASS_TO_ID["vehicle_horn"]
ALARM_ID = CLASS_TO_ID["alarm"]

DANGER_THRESHOLDS = {
    "alarm": 0.70,
    "glass_breaking": 0.80,
    "scream": 0.70,
    "siren": 0.70,
    "vehicle_horn": 0.70,
}

# RMS cap tidak memperkuat audio lemah.
# Semua pilihan hanya digunakan untuk eksperimen.
MODES = {
    "original": ("fixed", 1.00),
    "gain_070": ("fixed", 0.70),
    "gain_050": ("fixed", 0.50),
    "gain_030": ("fixed", 0.30),
    "rms_cap_005": ("cap", 0.05),
    "rms_cap_003": ("cap", 0.03),
}


# ==================================================
# VERIFIKASI MODEL
# ==================================================

def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1048576),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_models():
    for path, expected in [
        (YAMNET, EXPECTED_YAMNET),
        (CLASSIFIER, EXPECTED_CLASSIFIER),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha256(path)

        if actual.lower() != expected:
            raise RuntimeError(
                f"Hash berbeda: {path.name}\n"
                f"Aktual: {actual}\n"
                f"Seharusnya: {expected}\n"
                "Pengujian dibatalkan agar tidak "
                "memakai model yang salah."
            )

        print(f"VERIFIED: {path.name}")


# ==================================================
# LOAD TFLITE
# ==================================================

def load_model(path, shape, output_size):
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2,
    )

    inp = interpreter.get_input_details()[0]

    if tuple(inp["shape"]) != tuple(shape):
        interpreter.resize_tensor_input(
            inp["index"],
            shape,
            strict=False,
        )

    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    if tuple(inp["shape"]) != tuple(shape):
        raise RuntimeError(
            f"Input shape tidak sesuai: {path.name}"
        )

    if int(np.prod(out["shape"])) != output_size:
        raise RuntimeError(
            f"Output shape tidak sesuai: {path.name}"
        )

    if (
        inp["dtype"] != np.float32
        or out["dtype"] != np.float32
    ):
        raise RuntimeError(
            f"Input/output bukan FLOAT32: {path.name}"
        )

    return (
        interpreter,
        inp["index"],
        out["index"],
    )


def predict(window, yamnet_bundle, classifier_bundle):
    yamnet, y_input, y_output = yamnet_bundle
    classifier, c_input, c_output = classifier_bundle

    yamnet.set_tensor(
        y_input,
        window.astype(np.float32),
    )

    yamnet.invoke()

    embedding = yamnet.get_tensor(
        y_output
    ).reshape(-1)

    if embedding.shape != (1024,):
        raise RuntimeError(
            "Embedding bukan 1024 fitur."
        )

    classifier.set_tensor(
        c_input,
        embedding.reshape(1, 1024).astype(np.float32),
    )

    classifier.invoke()

    probabilities = classifier.get_tensor(
        c_output
    ).reshape(-1)

    if (
        probabilities.shape != (6,)
        or not np.isfinite(probabilities).all()
        or not np.isclose(
            float(probabilities.sum()),
            1.0,
            atol=0.02,
        )
    ):
        raise RuntimeError(
            "Output classifier tidak valid."
        )

    return probabilities


# ==================================================
# LOAD AUDIO
# ==================================================

def resolve_audio(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def read_audio(path):
    wave, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"Sample rate bukan 16 kHz: {path}"
        )

    if wave.ndim == 2:
        wave = wave.mean(
            axis=1,
            dtype=np.float32,
        )

    if wave.ndim != 1 or len(wave) == 0:
        raise RuntimeError(
            f"Audio invalid: {path}"
        )

    return wave.astype(np.float32)


# ==================================================
# PEMILIHAN WINDOW
# ==================================================

def select_windows(wave):
    # Mengikuti choose_windows() training V1.

    if len(wave) < WINDOW_SIZE:
        padded = np.zeros(
            WINDOW_SIZE,
            dtype=np.float32,
        )

        padded[:len(wave)] = wave
        candidates = [(0, padded)]

    else:
        starts = list(
            range(
                0,
                len(wave) - WINDOW_SIZE + 1,
                HOP_SIZE,
            )
        )

        count = min(MAX_WINDOWS, len(starts))

        picks = sorted(
            set(
                np.linspace(
                    0,
                    len(starts) - 1,
                    count,
                    dtype=int,
                )
            )
        )

        candidates = [
            (
                starts[i],
                wave[
                    starts[i]:starts[i] + WINDOW_SIZE
                ],
            )
            for i in picks
        ]

    # Filter pada audio asli terlebih dahulu.
    # Window yang terpilih dipakai semua mode.
    return [
        (start, window)
        for start, window in candidates
        if float(
            np.sqrt(
                np.mean(
                    window.astype(np.float64) ** 2
                )
            )
        ) >= RMS_THRESHOLD
    ]


# ==================================================
# PERUBAHAN AMPLITUDO
# ==================================================

def transform(window, original_rms, method, value):
    if method == "fixed":
        scale = value

    else:
        scale = min(
            1.0,
            value / max(original_rms, 1e-12),
        )

    changed = (
        window * scale
    ).astype(np.float32)

    return changed, scale


# ==================================================
# EVALUASI
# ==================================================

def evaluate(split):
    verify_models()

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    manifest = pd.read_csv(MANIFEST)

    required = {
        "path",
        "target_class",
        "split",
        "source_group",
    }

    if not required.issubset(manifest.columns):
        raise RuntimeError(
            "Kolom manifest hilang: "
            f"{required - set(manifest.columns)}"
        )

    leakage = (
        manifest.groupby("source_group")["split"]
        .nunique()
    )

    if (leakage > 1).any():
        raise RuntimeError(
            "Source group melintasi split."
        )

    subset = manifest.loc[
        manifest["split"] == split
    ].copy()

    if subset.empty:
        raise RuntimeError(
            f"Tidak ada data split {split}."
        )

    if not set(
        subset["target_class"]
    ).issubset(CLASS_TO_ID):
        raise RuntimeError(
            "Ada nama kelas tidak dikenal."
        )

    print(
        f"Split {split.upper()} | "
        f"{len(subset)} audio asal"
    )

    yamnet = load_model(
        YAMNET,
        [WINDOW_SIZE],
        1024,
    )

    classifier = load_model(
        CLASSIFIER,
        [1, 1024],
        6,
    )

    output_dir = (
        ROOT
        / "results"
        / "amplitude_six_classes_v3"
        / (
            f"{split}_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    rows = []
    selected_count = 0

    for file_no, row in enumerate(
        subset.itertuples(index=False),
        1,
    ):
        path = resolve_audio(row.path)
        waveform = read_audio(path)

        label_true = str(row.target_class)
        true_id = CLASS_TO_ID[label_true]

        chosen = select_windows(waveform)
        selected_count += len(chosen)

        for start, original_window in chosen:
            original_rms = float(
                np.sqrt(
                    np.mean(
                        original_window.astype(
                            np.float64
                        ) ** 2
                    )
                )
            )

            for mode_name, (method, value) in MODES.items():
                changed, scale = transform(
                    original_window,
                    original_rms,
                    method,
                    value,
                )

                changed_rms = original_rms * scale
                filtered = changed_rms < RMS_THRESHOLD

                if filtered:
                    # Seperti filter senyap aplikasi:
                    # tidak menjalankan classifier.
                    predicted_id = BACKGROUND_ID
                    confidence = 0.0
                    candidate = "background"

                else:
                    probs = predict(
                        changed,
                        yamnet,
                        classifier,
                    )

                    predicted_id = int(
                        np.argmax(probs)
                    )

                    confidence = float(
                        probs[predicted_id]
                    )

                    predicted_name = CLASS_NAMES[
                        predicted_id
                    ]

                    candidate = (
                        predicted_name
                        if (
                            predicted_name
                            in DANGER_THRESHOLDS
                            and confidence
                            >= DANGER_THRESHOLDS[
                                predicted_name
                            ]
                        )
                        else "background"
                    )

                rows.append({
                    "mode": mode_name,
                    "source_file": str(path),
                    "source_group": str(
                        row.source_group
                    ),
                    "start_sec": round(
                        start / SAMPLE_RATE,
                        5,
                    ),
                    "actual": label_true,
                    "predicted": CLASS_NAMES[
                        predicted_id
                    ],
                    "confidence": confidence,
                    "candidate_without_confirmation":
                        candidate,
                    "original_rms": original_rms,
                    "scale": scale,
                    "processed_rms": changed_rms,
                    "rms_filtered": filtered,
                    "actual_id": true_id,
                    "predicted_id": predicted_id,
                })

        if (
            file_no % 50 == 0
            or file_no == len(subset)
        ):
            print(
                f"Audio {file_no}/{len(subset)} "
                f"| window asal: {selected_count}"
            )

    # ==========================================
    # SIMPAN PREDIKSI
    # ==========================================

    details = pd.DataFrame(rows)

    details.to_csv(
        output_dir / "all_window_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = []
    classes_summary = []

    for mode_name in MODES:
        part = details.loc[
            details["mode"] == mode_name
        ]

        y_true = part[
            "actual_id"
        ].to_numpy(dtype=int)

        y_pred = part[
            "predicted_id"
        ].to_numpy(dtype=int)

        prec, rec, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(range(6)),
                zero_division=0,
            )
        )

        bg = part.loc[
            part["actual"] == "background"
        ]

        alarm = part.loc[
            part["actual"] == "alarm"
        ]

        raw_bg_fp = int(
            (
                bg["predicted"] != "background"
            ).sum()
        )

        candidate_bg_fp = int(
            (
                bg["candidate_without_confirmation"]
                != "background"
            ).sum()
        )

        alarm_as_horn = int(
            (
                alarm["predicted"]
                == "vehicle_horn"
            ).sum()
        )

        summary.append({
            "mode": mode_name,
            "windows": len(part),
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "macro_precision": float(
                np.mean(prec)
            ),
            "macro_recall": float(
                np.mean(rec)
            ),
            "macro_f1": float(
                np.mean(f1)
            ),
            "recall_alarm": float(
                rec[ALARM_ID]
            ),
            "recall_scream": float(
                rec[CLASS_TO_ID["scream"]]
            ),
            "recall_siren": float(
                rec[CLASS_TO_ID["siren"]]
            ),
            "recall_vehicle_horn": float(
                rec[HORN_ID]
            ),
            "recall_glass_breaking": float(
                rec[CLASS_TO_ID["glass_breaking"]]
            ),
            "recall_background": float(
                rec[BACKGROUND_ID]
            ),
            "alarm_as_horn": alarm_as_horn,
            "background_raw_false_danger":
                raw_bg_fp,
            "background_candidate_above_threshold":
                candidate_bg_fp,
            "rms_filtered_windows": int(
                part["rms_filtered"].sum()
            ),
        })

        for i, class_name in enumerate(
            CLASS_NAMES
        ):
            classes_summary.append({
                "mode": mode_name,
                "class": class_name,
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            })

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(6)),
        )

        pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(
            output_dir
            / f"confusion_{mode_name}.csv"
        )

    # ==========================================
    # HASIL PERBANDINGAN
    # ==========================================

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        output_dir / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        classes_summary
    ).to_csv(
        output_dir / "per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        summary_df["mode"],
        summary_df["macro_f1"],
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title(
        f"SADAR V3: Uji Amplitudo ({split})"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "macro_f1_comparison.png",
        dpi=180,
    )

    plt.close(fig)

    audit = {
        "experiment":
            "V3 amplitude diagnostic; no retraining",
        "split": split,
        "original_files": int(len(subset)),
        "selected_windows": selected_count,
        "yamnet_sha256": sha256(YAMNET),
        "classifier_sha256": sha256(CLASSIFIER),
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "window_limit_per_file": MAX_WINDOWS,
        "note": (
            "Windows chosen and RMS-gated on "
            "original audio. All variants reuse "
            "the same windows. Windows filtered "
            "after scaling are assigned background. "
            "Threshold candidates are not real "
            "app alerts; confirmation and latch "
            "are not simulated."
        ),
    }

    (
        output_dir / "audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    columns = [
        "mode",
        "windows",
        "accuracy",
        "macro_f1",
        "recall_alarm",
        "recall_scream",
        "recall_siren",
        "recall_vehicle_horn",
        "recall_background",
        "alarm_as_horn",
        "background_candidate_above_threshold",
        "rms_filtered_windows",
    ]

    print(
        "\n=== HASIL PERBANDINGAN ENAM KELAS ==="
    )

    print(
        summary_df[columns]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nWindow terpilih:",
        selected_count,
    )

    if split == "test":
        print(
            "Acuan V3 test lama: "
            "1546 window; accuracy sekitar 0.8551."
        )

        actual = summary_df.loc[
            summary_df["mode"] == "original",
            "accuracy",
        ].iloc[0]

        if (
            selected_count != 1546
            or abs(actual - 0.8551) > 0.005
        ):
            print(
                "PERINGATAN: Hasil original berbeda "
                "dari acuan V3 lama. Periksa "
                "perbedaannya sebelum menafsirkan gain."
            )

    print("\nOutput:", output_dir)

    print(
        "Sumber/model tidak diubah. "
        "Tidak ada training."
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    args = parser.parse_args()

    evaluate(args.split)
