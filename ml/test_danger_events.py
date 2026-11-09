
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf


ROOT = Path(__file__).resolve().parent

MANIFEST = ROOT / "dataset/dataset_splits.csv"

OUT = ROOT / (
    "results/robustness_v2/danger_event_test_v1"
)

YAMNET = ROOT / (
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
    "robustness_v2": ROOT / (
        "models/robustness_v2/"
        "danger_classifier_fp32.tflite"
    ),
}

CLASSES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

THRESHOLD = {
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

SR = 16000
WINDOW = 15600
HOP = 7680
RMS_LIMIT = 0.008
LATCH = 2.5


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def load_model(path, shape, output_size):

    if not path.is_file():
        raise FileNotFoundError(path)

    model = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2
    )

    item = model.get_input_details()[0]

    if list(item["shape"]) != shape:
        model.resize_tensor_input(
            item["index"],
            shape,
            strict=False
        )

    model.allocate_tensors()

    inp = model.get_input_details()[0]
    out = model.get_output_details()[0]

    if (
        list(inp["shape"]) != shape
        or int(np.prod(out["shape"])) != output_size
        or inp["dtype"] != np.float32
        or out["dtype"] != np.float32
    ):
        raise ValueError(
            f"Shape atau dtype tidak sesuai: {path.name}"
        )

    return (
        model,
        inp["index"],
        out["index"],
        tuple(shape)
    )


def predict(bundle, values):

    model, inp, out, shape = bundle

    data = np.asarray(
        values,
        dtype=np.float32
    ).reshape(shape)

    model.set_tensor(inp, data)
    model.invoke()

    scores = model.get_tensor(
        out
    ).reshape(-1)

    if not np.isfinite(scores).all():
        raise ValueError(
            "Prediksi mengandung NaN atau Inf."
        )

    return scores


def main():

    required_files = [
        MANIFEST,
        YAMNET,
        *MODELS.values(),
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    if (OUT / "clip_summary.csv").exists():
        raise FileExistsError(
            "Eksperimen sudah tersimpan, "
            f"tidak akan ditimpa: {OUT}"
        )

    data = pd.read_csv(MANIFEST)

    required_cols = {
        "path",
        "target_class",
        "split",
        "source_group",
    }

    if not required_cols.issubset(data.columns):
        raise ValueError(
            f"Kolom manifest kurang: "
            f"{required_cols - set(data.columns)}"
        )

    test = data[
        data["split"].astype(str) == "test"
    ].copy()

    if (
        test.empty
        or not set(test["target_class"]).issubset(
            set(CLASSES)
        )
    ):
        raise ValueError(
            "Split test kosong atau kelas tidak sesuai."
        )

    leakage = (
        data.groupby("source_group")["split"]
        .nunique()
    )

    if (leakage > 1).any():
        raise ValueError(
            "Ada source_group lintas split."
        )

    print("=" * 70)
    print("DANGER SOUND AI - EVENT-LEVEL TEST")
    print("=" * 70)

    print("Memuat YAMNet INT8...")

    yamnet = load_model(
        YAMNET,
        [WINDOW],
        1024
    )

    classifiers = {
        name: load_model(path, [1, 1024], 6)
        for name, path in MODELS.items()
    }

    for name in classifiers:
        print("Classifier siap:", name)

    clip_rows = []
    alerts = []
    window_rows = []

    print()
    print("MENGUJI DATASET TEST")
    print("-" * 70)

    for number, record in enumerate(
        test.itertuples(index=False),
        start=1
    ):

        path = Path(str(record.path))

        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        audio, sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=False
        )

        if sample_rate != SR:
            raise ValueError(
                f"Audio bukan 16 kHz: {path}"
            )

        if audio.ndim == 2:
            audio = audio.mean(
                axis=1
            ).astype(np.float32)

        if audio.ndim != 1:
            raise ValueError(
                f"Audio tidak mono: {path}"
            )

        states = {
            name: {
                "last": None,
                "streak": 0,
                "next": 0.0,
                "events": [],
            }
            for name in MODELS
        }

        windows_count = 0
        active_count = 0

        for start in range(
            0,
            len(audio) - WINDOW + 1,
            HOP
        ):

            window = audio[
                start:start + WINDOW
            ]

            values = window.astype(
                np.float64
            )

            rms = float(
                np.sqrt(
                    np.mean(values * values)
                )
            )

            active = rms >= RMS_LIMIT

            embedding = (
                predict(yamnet, window)
                if active
                else None
            )

            windows_count += 1
            active_count += int(active)

            end_time = (
                start + WINDOW
            ) / SR

            for name, classifier in (
                classifiers.items()
            ):

                if active:

                    scores = predict(
                        classifier,
                        embedding
                    )

                    index = int(
                        np.argmax(scores)
                    )

                    label = CLASSES[index]

                    confidence = float(
                        scores[index]
                    )

                else:
                    label = "background"
                    confidence = 1.0

                passed = (
                    active
                    and label in THRESHOLD
                    and confidence >= THRESHOLD.get(
                        label, 2.0
                    )
                )

                state = states[name]

                if not passed:
                    state["last"] = None
                    state["streak"] = 0

                elif state["last"] == label:
                    state["streak"] += 1

                else:
                    state["last"] = label
                    state["streak"] = 1

                fired = bool(
                    passed
                    and state["streak"] >= CONFIRM[label]
                    and end_time >= state["next"]
                )

                if fired:

                    state["next"] = (
                        end_time + LATCH
                    )

                    state["events"].append(
                        (label, end_time)
                    )

                    alerts.append({
                        "model": name,
                        "audio": str(path),
                        "actual_clip_label":
                            record.target_class,
                        "time_sec": round(
                            end_time, 4
                        ),
                        "predicted_label": label,
                    })

                window_rows.append({
                    "model": name,
                    "audio": str(path),
                    "actual_clip_label":
                        record.target_class,
                    "start_sec": round(
                        start / SR, 4
                    ),
                    "rms": rms,
                    "predicted_label": label,
                    "confidence": confidence,
                    "passed_threshold": passed,
                    "simulated_alert": fired,
                })

        for name, state in states.items():

            found = [
                event[0]
                for event in state["events"]
            ]

            clip_rows.append({
                "model": name,
                "audio": str(path),
                "source_group": record.source_group,
                "actual_clip_label":
                    record.target_class,
                "duration_sec": round(
                    len(audio) / SR, 4
                ),
                "total_windows": windows_count,
                "active_windows": active_count,
                "alert_events": len(found),
                "any_danger_alert": bool(found),
                "correct_class_alert":
                    record.target_class in found,
                "first_alert_sec": (
                    round(
                        state["events"][0][1], 4
                    )
                    if found
                    else np.nan
                ),
            })

        if (
            number % 50 == 0
            or number == len(test)
        ):
            print(
                f"Audio test {number}/{len(test)} selesai"
            )

    # ==========================================
    # REKAP HASIL
    # ==========================================

    clips = pd.DataFrame(clip_rows)

    class_rows = []
    overall_rows = []

    for model, part in clips.groupby(
        "model",
        sort=False
    ):

        for label, group in part.groupby(
            "actual_clip_label",
            sort=False
        ):

            class_rows.append({
                "model": model,
                "class": label,
                "clips": len(group),
                "clips_with_any_alert": int(
                    group["any_danger_alert"].sum()
                ),
                "clips_with_correct_class_alert":
                    int(
                        group["correct_class_alert"].sum()
                    ),
                "any_alert_rate": float(
                    group["any_danger_alert"].mean()
                ),
                "correct_class_alert_rate": float(
                    group["correct_class_alert"].mean()
                ),
            })

        danger = part[
            part["actual_clip_label"] != "background"
        ]

        background = part[
            part["actual_clip_label"] == "background"
        ]

        overall_rows.append({
            "model": model,
            "danger_clips": len(danger),
            "danger_clips_alerted": int(
                danger["any_danger_alert"].sum()
            ),
            "danger_clip_alert_recall": (
                float(
                    danger["any_danger_alert"].mean()
                )
                if len(danger)
                else np.nan
            ),
            "background_clips": len(background),
            "background_clips_with_alert": int(
                background["any_danger_alert"].sum()
            ),
            "background_clip_alert_rate": (
                float(
                    background["any_danger_alert"].mean()
                )
                if len(background)
                else np.nan
            ),
            "total_alert_events_on_background": int(
                background["alert_events"].sum()
            ),
            "danger_clips_shorter_than_2_windows":
                int(
                    (danger["total_windows"] < 2).sum()
                ),
        })

    # ==========================================
    # SIMPAN HASIL
    # ==========================================

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    clips.to_csv(
        OUT / "clip_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        alerts,
        columns=[
            "model",
            "audio",
            "actual_clip_label",
            "time_sec",
            "predicted_label",
        ]
    ).to_csv(
        OUT / "alert_events.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(window_rows).to_csv(
        OUT / "window_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    class_summary = pd.DataFrame(
        class_rows
    )

    class_summary.to_csv(
        OUT / "per_class_clip_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    overall = pd.DataFrame(
        overall_rows
    )

    overall.to_csv(
        OUT / "summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "experiment": "danger_event_test_v1",
        "test_source": str(MANIFEST),
        "test_clips": len(test),
        "sample_rate": SR,
        "window_samples": WINDOW,
        "hop_samples": HOP,
        "rms_threshold": RMS_LIMIT,
        "thresholds": THRESHOLD,
        "confirmations": CONFIRM,
        "latch_seconds": LATCH,
        "sha256": {
            str(path): sha256(path)
            for path in required_files
        },
        "limitations": [
            "Clip-level labels; event timestamps unavailable.",
            "Existing test set already used in development.",
            "Alert simulation approximates Flutter.",
            "Short clips may not satisfy confirmation.",
            "No actual Android notification is measured.",
        ],
    }

    (OUT / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("HASIL DETEKSI PER REKAMAN")
    print("=" * 70)

    print(
        overall.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("HASIL PER KELAS")
    print("=" * 70)

    print(
        class_summary.to_string(index=False)
    )

    print()
    print("Hasil disimpan:", OUT)
    print("Semua model tetap aman.")


if __name__ == "__main__":
    main()
