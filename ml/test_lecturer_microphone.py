
from pathlib import Path
import hashlib
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf


# ===============================================
# LOKASI PROJECT
# ===============================================

ROOT = Path(__file__).resolve().parent

MANIFEST = ROOT / (
    "dataset/robustness_v2/"
    "hard_negative_recordings.csv"
)

OUTPUT_DIR = ROOT / (
    "results/robustness_v2/"
    "lecturer_microphone_test"
)

YAMNET_PATH = ROOT / (
    "../mobile/assets/models/"
    "yamnet_embedding_dynamic_int8.tflite"
)

MODEL_PATHS = {
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

# Sesuai session perekaman dosen sebelumnya.
TARGET_SESSION = "classroom_independent_test_01"

SR = 16000
WINDOW = 15600
HOP = 7680
RMS_THRESHOLD = 0.008
LATCH_SEC = 2.5

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

CONFIRMATIONS = {
    "alarm": 2,
    "glass_breaking": 1,
    "scream": 2,
    "siren": 2,
    "vehicle_horn": 2,
}


# ===============================================
# UTILITIES
# ===============================================

def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1048576),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def resolve_audio(value):

    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def rms_value(waveform):

    a = waveform.astype(np.float64)

    return float(
        np.sqrt(np.mean(a * a))
    )


# ===============================================
# LOAD TFLITE
# ===============================================

def load_tflite(path, input_shape, output_size):

    if not path.is_file():
        raise FileNotFoundError(
            f"Model belum ditemukan:\n{path}"
        )

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

    assert list(inp["shape"]) == input_shape

    assert int(
        np.prod(out["shape"])
    ) == output_size

    assert inp["dtype"] == np.float32
    assert out["dtype"] == np.float32

    return (
        model,
        inp["index"],
        out["index"],
        tuple(input_shape)
    )


def infer(bundle, values):

    model, input_id, output_id, shape = bundle

    x = np.asarray(
        values,
        dtype=np.float32
    ).reshape(shape)

    model.set_tensor(input_id, x)

    model.invoke()

    result = model.get_tensor(
        output_id
    ).reshape(-1)

    if not np.isfinite(result).all():
        raise ValueError(
            "Output model mengandung NaN atau Inf."
        )

    return result


# ===============================================
# SIMULASI ATURAN NOTIFIKASI
# ===============================================

def simulate_alerts(predictions):

    predictions = predictions.copy()

    predictions["simulated_alert"] = False

    events = []

    for (model, recording_id), group in (
        predictions.groupby(
            ["model", "recording_id"],
            sort=False
        )
    ):

        group = group.sort_values("start_sec")

        last_label = None
        streak = 0
        next_allowed_time = 0.0

        for index, row in group.iterrows():

            label = (
                row["predicted_label"]
                if row["passes_threshold"]
                else None
            )

            if label is None:

                last_label = None
                streak = 0
                continue

            if label == last_label:
                streak += 1
            else:
                last_label = label
                streak = 1

            required = CONFIRMATIONS[label]

            alert_time = (
                float(row["start_sec"])
                + WINDOW / SR
            )

            if (
                streak >= required
                and alert_time >= next_allowed_time
            ):

                predictions.at[
                    index, "simulated_alert"
                ] = True

                events.append({
                    "model": model,
                    "recording_id": recording_id,
                    "time_sec": round(
                        alert_time, 3
                    ),
                    "predicted_label": label,
                    "confidence": float(
                        row["confidence"]
                    )
                })

                next_allowed_time = (
                    alert_time + LATCH_SEC
                )

    return predictions, pd.DataFrame(
        events,
        columns=[
            "model",
            "recording_id",
            "time_sec",
            "predicted_label",
            "confidence"
        ]
    )


# ===============================================
# MAIN
# ===============================================

def main():

    print("=" * 70)
    print("DANGER SOUND AI")
    print("REAL-WORLD LECTURER MICROPHONE TEST")
    print("=" * 70)

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    manifest = pd.read_csv(MANIFEST)

    required_columns = {
        "recording_id",
        "session_id",
        "split",
        "verified_background",
        "label_status",
        "audio_path",
    }

    if not required_columns.issubset(
        manifest.columns
    ):
        raise ValueError(
            "Kolom manifest tidak lengkap."
        )

    recordings = manifest[
        (manifest["session_id"] == TARGET_SESSION)
        & (manifest["split"] == "test")
    ].copy()

    if recordings.empty:

        print("\nSession tersedia:")
        print(
            manifest[
                ["session_id", "split"]
            ].to_string(index=False)
        )

        raise ValueError(
            f"Session test '{TARGET_SESSION}' "
            "tidak ditemukan."
        )

    verified = (
        recordings["verified_background"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    label_ok = (
        recordings["label_status"]
        == "background_confirmed_by_recorder"
    )

    if not (verified & label_ok).all():
        raise ValueError(
            "Ada rekaman test yang belum "
            "dikonfirmasi sebagai background."
        )

    # ===========================================
    # LOAD MODEL
    # ===========================================

    print("\nMemuat YAMNet Dynamic INT8...")

    yamnet = load_tflite(
        YAMNET_PATH,
        [WINDOW],
        1024
    )

    classifiers = {}

    for name, path in MODEL_PATHS.items():

        classifiers[name] = load_tflite(
            path,
            [1, 1024],
            6
        )

        print("Classifier siap:", name)

    # ===========================================
    # EVALUASI
    # ===========================================

    all_rows = []
    total_duration = 0.0
    total_windows = 0
    active_windows = 0
    audio_hashes = {}

    for recording in recordings.itertuples(
        index=False
    ):

        audio_path = resolve_audio(
            recording.audio_path
        )

        audio, sr = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False
        )

        if sr != SR or audio.ndim != 1:
            raise ValueError(
                "Audio harus mono 16 kHz: "
                f"{audio_path.name}"
            )

        duration = len(audio) / SR

        total_duration += duration

        audio_hashes[
            recording.recording_id
        ] = sha256(audio_path)

        print()
        print("Rekaman :", recording.recording_id)
        print("Durasi  :", round(duration, 2), "detik")

        scene_total = 0
        scene_active = 0

        for start in range(
            0,
            len(audio) - WINDOW + 1,
            HOP
        ):

            window = audio[
                start:start + WINDOW
            ]

            rms = rms_value(window)

            active = rms >= RMS_THRESHOLD

            embedding = None

            if active:

                # YAMNet hanya dipanggil sekali
                # untuk ketiga classifier.

                embedding = infer(
                    yamnet,
                    window
                )

                scene_active += 1

            for model_name, classifier in (
                classifiers.items()
            ):

                if active:

                    scores = infer(
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

                threshold = THRESHOLDS.get(
                    label
                )

                passes = (
                    active
                    and threshold is not None
                    and confidence >= threshold
                )

                all_rows.append({
                    "model": model_name,
                    "recording_id": (
                        recording.recording_id
                    ),
                    "start_sec": round(
                        start / SR, 5
                    ),
                    "end_sec": round(
                        (start + WINDOW) / SR, 5
                    ),
                    "rms": rms,
                    "active": active,
                    "predicted_label": label,
                    "confidence": confidence,
                    "passes_threshold": bool(
                        passes
                    ),
                })

            scene_total += 1

        total_windows += scene_total
        active_windows += scene_active

        print(
            "Total window:",
            scene_total
        )

        print(
            "Window aktif:",
            scene_active
        )

    predictions = pd.DataFrame(
        all_rows
    )

    if predictions.empty:
        raise ValueError(
            "Tidak ada window untuk dievaluasi."
        )

    # ===========================================
    # SIMULASI ALERT
    # ===========================================

    predictions, events = simulate_alerts(
        predictions
    )

    # ===========================================
    # SUMMARY
    # ===========================================

    summaries = []

    for model_name, group in (
        predictions.groupby("model")
    ):

        horn_top1 = int(
            (
                group["predicted_label"]
                == "vehicle_horn"
            ).sum()
        )

        horn_threshold = int(
            (
                (group["predicted_label"]
                 == "vehicle_horn")
                & group["passes_threshold"]
            ).sum()
        )

        alert_count = int(
            group["simulated_alert"].sum()
        )

        hourly_rate = (
            alert_count * 3600 / total_duration
            if total_duration > 0
            else 0
        )

        summaries.append({
            "model": model_name,
            "duration_sec": round(
                total_duration, 2
            ),
            "total_windows": total_windows,
            "active_windows": active_windows,
            "horn_top1": horn_top1,
            "horn_above_threshold": horn_threshold,
            "all_danger_above_threshold": int(
                group["passes_threshold"].sum()
            ),
            "simulated_alerts": alert_count,
            "estimated_alerts_per_hour": round(
                hourly_rate, 2
            )
        })

    summary = pd.DataFrame(
        summaries
    )

    # ===========================================
    # SIMPAN HASIL
    # ===========================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_csv(
        OUTPUT_DIR / "window_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    events.to_csv(
        OUTPUT_DIR / "alert_events.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "experiment": "lecturer_microphone_test",
        "target_session": TARGET_SESSION,
        "ground_truth": (
            "background_confirmed_by_recorder"
        ),
        "sample_rate": SR,
        "window_size": WINDOW,
        "hop_size": HOP,
        "rms_threshold": RMS_THRESHOLD,
        "thresholds": THRESHOLDS,
        "confirmations": CONFIRMATIONS,
        "latch_seconds": LATCH_SEC,
        "yamnet_sha256": sha256(YAMNET_PATH),
        "classifier_sha256": {
            name: sha256(path)
            for name, path in MODEL_PATHS.items()
        },
        "audio_sha256": audio_hashes,
        "limitations": [
            "Label background based on recorder verification.",
            "Overlapping windows are correlated.",
            "Simulation approximates Flutter confirmation and latch.",
            "Rate per hour extrapolated from short recordings.",
            "No actual Android notification measured."
        ]
    }

    (
        OUTPUT_DIR / "experiment_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("HASIL PENGUJIAN REKAMAN DOSEN")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("FILE HASIL:")
    print(OUTPUT_DIR / "model_comparison.csv")
    print(OUTPUT_DIR / "window_predictions.csv")
    print(OUTPUT_DIR / "alert_events.csv")
    print(OUTPUT_DIR / "experiment_config.json")

    print()
    print(
        "CATATAN: simulated_alerts merupakan "
        "estimasi berdasarkan aturan yang "
        "dikonfigurasi, bukan notifikasi "
        "Android yang diamati secara langsung."
    )

    print()
    print("Model lama dan baru tidak diubah.")


if __name__ == "__main__":
    main()
