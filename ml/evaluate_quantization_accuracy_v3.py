
from datetime import datetime
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================
# KONFIGURASI SADAR
# ============================================

ROOT = Path(__file__).resolve().parent

SPLITS = ROOT / "dataset/dataset_splits.csv"

YAMNET = {
    "FP32": (
        ROOT / "models/yamnet_embedding_fp32.tflite"
    ),
    "Dynamic_INT8": (
        ROOT / "models/yamnet_embedding_dynamic_int8.tflite"
    ),
}

CLASSIFIER = (
    ROOT.parent
    / "mobile/assets/models/danger_classifier_fp32.tflite"
)

CLASSES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

WINDOW = 15600
HOP = 7680
SAMPLE_RATE = 16000


# ============================================
# UTILITIES
# ============================================

def hash_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_interpreter(
    path,
    input_shape,
    output_shape,
):
    model = tf.lite.Interpreter(
        model_path=str(path)
    )

    model.resize_tensor_input(
        model.get_input_details()[0]["index"],
        input_shape,
    )

    model.allocate_tensors()

    input_tensor = model.get_input_details()[0]
    output_tensor = model.get_output_details()[0]

    if (
        list(input_tensor["shape"]) != input_shape
        or list(output_tensor["shape"]) != output_shape
        or input_tensor["dtype"] != np.float32
        or output_tensor["dtype"] != np.float32
    ):
        raise ValueError(
            f"Shape atau dtype tidak sesuai: {path.name}"
        )

    return (
        model,
        input_tensor["index"],
        output_tensor["index"],
    )


def infer(engine, x):
    model, in_id, out_id = engine

    model.set_tensor(in_id, x)
    model.invoke()

    return model.get_tensor(out_id).copy()


# ============================================
# AUDIO WINDOW
# ============================================

def iter_windows(wav_file):
    signal, rate = sf.read(
        wav_file,
        dtype="float32",
        always_2d=True,
    )

    if rate != SAMPLE_RATE:
        raise ValueError(
            f"Audio belum 16 kHz: {wav_file} ({rate})"
        )

    signal = signal.mean(axis=1).astype(np.float32)

    if (
        len(signal) == 0
        or not np.isfinite(signal).all()
    ):
        raise ValueError(
            f"Audio tidak valid: {wav_file}"
        )

    if len(signal) < WINDOW:
        yield 0, np.pad(
            signal,
            (0, WINDOW - len(signal)),
        ).astype(np.float32)

    else:
        for start in range(
            0,
            len(signal) - WINDOW + 1,
            HOP,
        ):
            yield start, np.ascontiguousarray(
                signal[start:start + WINDOW]
            )


# ============================================
# CONFUSION MATRIX CSV + PNG
# ============================================

def save_cm(
    cm,
    name,
    csv_dir,
    png_dir,
    normalized=False,
):
    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(
        csv_dir / f"{name}.csv",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(
        cm,
        cmap="Blues",
        vmin=0,
        vmax=1 if normalized else None,
    )

    ax.set_xticks(
        range(6),
        labels=CLASSES,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        range(6),
        labels=CLASSES,
    )

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(name.replace("_", " "))

    for i in range(6):
        for j in range(6):
            value = cm[i, j]

            ax.text(
                j,
                i,
                (
                    f"{value:.2f}"
                    if normalized
                    else str(int(value))
                ),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if im.norm(value) > 0.5
                    else "black"
                ),
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    fig.savefig(
        png_dir / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================
# MAIN EXPERIMENT
# ============================================

def main():
    for file in [
        SPLITS,
        CLASSIFIER,
        *YAMNET.values(),
    ]:
        if not file.is_file():
            raise FileNotFoundError(file)

    df = pd.read_csv(SPLITS)

    test = df[
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower() == "test"
    ].copy()

    if len(test) != 300:
        raise ValueError(
            f"Jumlah WAV test bukan 300: {len(test)}"
        )

    files = [
        ROOT / str(s).strip().replace("\\", "/")
        for s in test["path"]
    ]

    missing = [
        str(p)
        for p in files
        if not p.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} WAV belum ditemukan: "
            f"{missing[:3]}"
        )

    if not set(
        test["target_class"]
    ).issubset(CLASSES):
        raise ValueError(
            "Ada label test di luar enam kelas"
        )

    # ========================================
    # OUTPUT DIRECTORIES
    # ========================================

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    output = (
        ROOT
        / "results/quantization_accuracy_v3"
        / run_id
    )

    csv_dir = output / "csv"
    png_dir = output / "png"
    meta_dir = output / "metadata"

    for folder in [
        csv_dir,
        png_dir,
        meta_dir,
    ]:
        folder.mkdir(
            parents=True,
            exist_ok=False,
        )

    # ========================================
    # LOAD MODELS
    # ========================================

    yamnet = {
        name: load_interpreter(
            path,
            [WINDOW],
            [1024],
        )
        for name, path in YAMNET.items()
    }

    classifier = load_interpreter(
        CLASSIFIER,
        [1, 1024],
        [1, 6],
    )

    rows = []
    clips = []

    print("=" * 60)
    print("SADAR - EVALUASI FP32 VS DYNAMIC INT8")
    print("=" * 60)
    print("WAV test: 300")
    print("Classifier: V3 tetap")
    print("Training ulang: TIDAK")
    print()

    # ========================================
    # PREDICTIONS
    # ========================================

    for clip_id, (item, wav) in enumerate(
        zip(test.itertuples(index=False), files),
        1,
    ):
        true_id = CLASSES.index(
            str(item.target_class)
        )

        clip_probs = {
            name: []
            for name in YAMNET
        }

        for win_id, (start, audio) in enumerate(
            iter_windows(wav),
            1,
        ):
            for name, model in yamnet.items():
                embedding = infer(
                    model,
                    audio,
                )

                prob = infer(
                    classifier,
                    embedding.reshape(1, 1024),
                )[0]

                if not np.isfinite(prob).all():
                    raise ValueError(
                        f"Output tidak valid: {item.filename}"
                    )

                pred = int(np.argmax(prob))

                rows.append({
                    "clip_id": clip_id,
                    "filename": item.filename,
                    "source_group": item.source_group,
                    "window_index": win_id,
                    "start_sec": start / SAMPLE_RATE,
                    "model": name,
                    "true_id": true_id,
                    "true_label": CLASSES[true_id],
                    "pred_id": pred,
                    "pred_label": CLASSES[pred],
                    "confidence": float(prob[pred]),
                    **{
                        f"prob_{c}": float(prob[i])
                        for i, c in enumerate(CLASSES)
                    },
                })

                clip_probs[name].append(prob)

        # Prediksi per audio:
        # rata-rata probabilitas seluruh window.

        for name, probabilities in clip_probs.items():
            mean_prob = np.mean(
                probabilities,
                axis=0,
            )

            pred = int(np.argmax(mean_prob))

            clips.append({
                "clip_id": clip_id,
                "filename": item.filename,
                "model": name,
                "true_id": true_id,
                "true_label": CLASSES[true_id],
                "pred_id": pred,
                "pred_label": CLASSES[pred],
                "window_count": len(probabilities),
            })

        if clip_id % 25 == 0:
            print(
                f"Progres: {clip_id}/300 WAV"
            )

    # ========================================
    # SAVE PREDICTIONS
    # ========================================

    windows = pd.DataFrame(rows)
    clip_table = pd.DataFrame(clips)

    windows.to_csv(
        csv_dir / "predictions_per_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clip_table.to_csv(
        csv_dir / "predictions_per_clip.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================
    # CLASSIFICATION METRICS
    # ========================================

    metrics = []
    class_metrics = []

    for unit, table in [
        ("window", windows),
        ("clip", clip_table),
    ]:
        for name in YAMNET:
            subset = table[
                table["model"] == name
            ]

            actual = subset[
                "true_id"
            ].to_numpy()

            predicted = subset[
                "pred_id"
            ].to_numpy()

            p, r, f1, _ = (
                precision_recall_fscore_support(
                    actual,
                    predicted,
                    labels=list(range(6)),
                    average="macro",
                    zero_division=0,
                )
            )

            metrics.append({
                "unit": unit,
                "model": name,
                "count": len(subset),
                "accuracy": accuracy_score(
                    actual,
                    predicted,
                ),
                "macro_precision": p,
                "macro_recall": r,
                "macro_f1": f1,
            })

            report = classification_report(
                actual,
                predicted,
                labels=list(range(6)),
                target_names=CLASSES,
                zero_division=0,
                output_dict=True,
            )

            pd.DataFrame(report).T.to_csv(
                csv_dir
                / f"classification_report_{unit}_{name}.csv",
                encoding="utf-8-sig",
            )

            for c in CLASSES:
                class_metrics.append({
                    "unit": unit,
                    "model": name,
                    "class": c,
                    "precision": report[c]["precision"],
                    "recall": report[c]["recall"],
                    "f1": report[c]["f1-score"],
                    "support": report[c]["support"],
                })

            cm = confusion_matrix(
                actual,
                predicted,
                labels=list(range(6)),
            )

            prefix = f"confusion_{unit}_{name}"

            save_cm(
                cm,
                prefix,
                csv_dir,
                png_dir,
            )

            totals = cm.sum(
                axis=1,
                keepdims=True,
            )

            normal = np.divide(
                cm,
                totals,
                out=np.zeros_like(
                    cm,
                    dtype=float,
                ),
                where=totals != 0,
            )

            save_cm(
                normal,
                prefix + "_normalized",
                csv_dir,
                png_dir,
                normalized=True,
            )

    # ========================================
    # SAVE METRIC CSV
    # ========================================

    metrics_df = pd.DataFrame(metrics)
    classes_df = pd.DataFrame(class_metrics)

    metrics_df.to_csv(
        csv_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    classes_df.to_csv(
        csv_dir / "per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================
    # PAIRED COMPARISON
    # ========================================

    paired = windows.pivot(
        index=["clip_id", "window_index"],
        columns="model",
        values="pred_id",
    ).reset_index()

    paired["same_prediction"] = (
        paired["FP32"] ==
        paired["Dynamic_INT8"]
    )

    paired.to_csv(
        csv_dir / "paired_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired[
        ~paired["same_prediction"]
    ].to_csv(
        csv_dir / "prediction_disagreements.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================
    # ACCURACY + F1 GRAPH
    # ========================================

    for unit in ["window", "clip"]:
        part = metrics_df[
            metrics_df["unit"] == unit
        ].set_index("model").loc[
            list(YAMNET)
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        positions = np.arange(2)

        a = ax.bar(
            positions - 0.18,
            part["accuracy"],
            0.36,
            label="Accuracy",
        )

        b = ax.bar(
            positions + 0.18,
            part["macro_f1"],
            0.36,
            label="Macro F1",
        )

        ax.bar_label(a, fmt="%.3f")
        ax.bar_label(b, fmt="%.3f")

        ax.set_xticks(
            positions,
            ["FP32", "Dynamic INT8"],
        )

        ax.set_ylim(0, 1.08)
        ax.set_title(
            f"Kinerja klasifikasi per {unit}"
        )
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            png_dir
            / f"accuracy_macro_f1_{unit}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================
    # RECALL GRAPH
    # ========================================

    recall = classes_df[
        classes_df["unit"] == "window"
    ].pivot(
        index="class",
        columns="model",
        values="recall",
    ).reindex(CLASSES)

    ax = recall.plot(
        kind="bar",
        figsize=(11, 5),
        rot=30,
    )

    ax.set_ylim(0, 1)
    ax.set_title(
        "Recall setiap kelas (per window)"
    )
    ax.set_ylabel("Recall")

    plt.tight_layout()

    plt.savefig(
        png_dir / "per_class_recall.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================
    # METADATA
    # ========================================

    metadata = {
        "experiment": (
            "SADAR quantization accuracy comparison "
            "with frozen V3 classifier"
        ),
        "run_id": run_id,
        "test_wav": len(test),
        "windows_per_model": len(windows) // 2,
        "yamnet_fp32_sha256": hash_file(
            YAMNET["FP32"]
        ),
        "yamnet_dynamic_int8_sha256": hash_file(
            YAMNET["Dynamic_INT8"]
        ),
        "classifier_v3_sha256": hash_file(
            CLASSIFIER
        ),
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW,
        "hop_size": HOP,
        "clip_prediction": (
            "argmax of mean probabilities "
            "over all windows"
        ),
        "short_clip_handling": (
            "zero pad to 15600 samples"
        ),
        "retraining": False,
        "limitations": [
            (
                "V3 classifier trained on Dynamic "
                "INT8 embeddings, possibly favoring INT8"
            ),
            (
                "Test dataset was reused during "
                "model development"
            ),
            (
                "This measures classification only, "
                "not real-time alert accuracy or latency"
            ),
        ],
    }

    (
        meta_dir / "evaluation_summary.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================
    # FINAL OUTPUT
    # ========================================

    print("\n" + "=" * 60)
    print(
        "EVALUASI KETEPATAN "
        "FP32 VS DYNAMIC INT8 SELESAI"
    )
    print("=" * 60)

    print(
        "Jumlah window per model:",
        len(windows) // 2,
    )

    if len(windows) // 2 != 1546:
        print(
            "CATATAN: Jumlah window berbeda "
            "dari 1546 test V3 lama."
        )
        print(
            "Periksa segmentasi sebelum "
            "membandingkan antar-eksperimen."
        )

    print("\nMETRIK:")
    print(metrics_df.to_string(index=False))

    print(
        "\nPersentase prediksi kedua model sama:",
        f"{paired['same_prediction'].mean() * 100:.2f}%",
    )

    print("\nFolder hasil:", output)

    print(
        "Total file:",
        len(list(output.rglob("*.*"))),
    )

    print(
        "Model V3 tidak diubah; "
        "tidak ada training ulang."
    )


if __name__ == "__main__":
    main()
