
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import tensorflow as tf

from scipy.signal import resample_poly


# ==========================================================
# KONFIGURASI
# ==========================================================

ROOT = Path(__file__).resolve().parent

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

OUTPUT_ROOT = ROOT / (
    "results/robustness_v2/live_siren_test"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680

RMS_THRESHOLD = 0.008
LATCH_SECONDS = 2.5

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


# ==========================================================
# UTILITIES
# ==========================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1048576),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def calculate_rms(waveform):

    values = waveform.astype(np.float64)

    return float(
        np.sqrt(np.mean(values * values))
    )


def resample_window(waveform, source_rate):

    gcd = math.gcd(
        source_rate,
        SAMPLE_RATE
    )

    result = resample_poly(
        waveform,
        SAMPLE_RATE // gcd,
        source_rate // gcd
    ).astype(np.float32)

    if len(result) < WINDOW_SIZE:

        result = np.pad(
            result,
            (0, WINDOW_SIZE - len(result))
        )

    return result[:WINDOW_SIZE]


# ==========================================================
# LOAD MODEL TFLITE
# ==========================================================

def load_model(path, target_shape, output_size):

    if not path.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {path}"
        )

    model = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2
    )

    input_info = model.get_input_details()[0]

    if list(input_info["shape"]) != target_shape:

        model.resize_tensor_input(
            input_info["index"],
            target_shape,
            strict=False
        )

    model.allocate_tensors()

    input_info = model.get_input_details()[0]
    output_info = model.get_output_details()[0]

    if list(input_info["shape"]) != target_shape:
        raise ValueError(
            f"Input shape salah: {path.name}"
        )

    if int(np.prod(output_info["shape"])) != output_size:
        raise ValueError(
            f"Output shape salah: {path.name}"
        )

    if (
        input_info["dtype"] != np.float32
        or output_info["dtype"] != np.float32
    ):
        raise ValueError(
            f"Input/output bukan FP32: {path.name}"
        )

    return (
        model,
        input_info["index"],
        output_info["index"],
        tuple(target_shape)
    )


def inference(bundle, data):

    model, input_id, output_id, shape = bundle

    values = np.asarray(
        data,
        dtype=np.float32
    ).reshape(shape)

    model.set_tensor(
        input_id,
        values
    )

    model.invoke()

    result = model.get_tensor(
        output_id
    ).reshape(-1)

    if not np.isfinite(result).all():
        raise ValueError(
            "Output AI mengandung NaN atau Inf."
        )

    return result


# ==========================================================
# PREDIKSI DAN SIMULASI ALERT
# ==========================================================

def predict_window(
    model_name,
    model,
    embedding,
    active,
    start_sec,
    rms,
    state
):

    if active:

        probabilities = inference(
            model,
            embedding
        )

        winner = int(
            np.argmax(probabilities)
        )

        label = CLASSES[winner]

        confidence = float(
            probabilities[winner]
        )

    else:

        label = "background"
        confidence = 1.0

    threshold = THRESHOLDS.get(label)

    passes = (
        active
        and threshold is not None
        and confidence >= threshold
    )

    if not passes:

        state["last_label"] = None
        state["streak"] = 0

    elif label == state["last_label"]:

        state["streak"] += 1

    else:

        state["last_label"] = label
        state["streak"] = 1

    end_sec = start_sec + WINDOW_SIZE / SAMPLE_RATE

    triggered = False

    if passes:

        required = CONFIRMATIONS[label]

        if (
            state["streak"] >= required
            and end_sec >= state["next_allowed"]
        ):

            triggered = True

            state["next_allowed"] = (
                end_sec + LATCH_SECONDS
            )

    return {
        "model": model_name,
        "start_sec": round(start_sec, 5),
        "end_sec": round(end_sec, 5),
        "rms": round(rms, 6),
        "active": bool(active),
        "predicted_label": label,
        "confidence": round(confidence, 6),
        "passes_threshold": bool(passes),
        "simulated_alert": bool(triggered),
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seconds",
        type=float,
        default=60
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None
    )

    parser.add_argument(
        "--list-devices",
        action="store_true"
    )

    args = parser.parse_args()

    if args.list_devices:

        print(sd.query_devices())
        return

    if not (5 <= args.seconds <= 1800):

        raise ValueError(
            "Durasi harus antara 5 dan 1800 detik."
        )

    print("=" * 70)
    print("DANGER SOUND AI - LIVE MICROPHONE TEST")
    print("=" * 70)

    # ======================================================
    # MEMUAT MODEL
    # ======================================================

    print()
    print("Memuat model TFLite...")

    yamnet = load_model(
        YAMNET_PATH,
        [WINDOW_SIZE],
        1024
    )

    classifiers = {}

    for name, path in MODEL_PATHS.items():

        classifiers[name] = load_model(
            path,
            [1, 1024],
            6
        )

        print("Model siap:", name)

    # ======================================================
    # KONFIGURASI MIKROFON
    # ======================================================

    device = sd.query_devices(
        args.device,
        "input"
    )

    native_rate = int(
        round(device["default_samplerate"])
    )

    if native_rate <= 0:
        raise ValueError(
            "Sample rate mikrofon tidak valid."
        )

    native_window = int(
        round(
            WINDOW_SIZE * native_rate / SAMPLE_RATE
        )
    )

    native_hop = int(
        round(
            HOP_SIZE * native_rate / SAMPLE_RATE
        )
    )

    sd.check_input_settings(
        device=args.device,
        channels=1,
        samplerate=native_rate
    )

    print()
    print("Mikrofon        :", device["name"])
    print("Sample rate     :", native_rate)
    print("Window target   :", WINDOW_SIZE)
    print("Hop target      :", HOP_SIZE)
    print("Durasi pengujian:", args.seconds, "detik")

    print()
    print(
        "Putar rekaman sirene ambulans "
        "melalui speaker saat tes dimulai."
    )

    input(
        "\nTekan ENTER untuk memulai tes..."
    )

    # ======================================================
    # STATUS SIMULASI
    # ======================================================

    states = {
        name: {
            "last_label": None,
            "streak": 0,
            "next_allowed": 0.0,
        }
        for name in classifiers
    }

    rows = []

    recorded_blocks = []

    buffer = np.empty(
        0,
        dtype=np.float32
    )

    captured_samples = 0
    window_start_sample = 0
    window_count = 0
    overflow_count = 0

    print()
    print("=" * 70)
    print("PENGUJIAN LIVE DIMULAI")
    print("=" * 70)

    # ======================================================
    # BACA MIKROFON SECARA BERKELANJUTAN
    # ======================================================

    try:

        with sd.InputStream(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            device=args.device,
            blocksize=native_hop
        ) as stream:

            while (
                captured_samples / native_rate
                < args.seconds
            ):

                block, overflowed = stream.read(
                    native_hop
                )

                if overflowed:
                    overflow_count += 1

                mono = block[:, 0].copy()

                recorded_blocks.append(mono)

                captured_samples += len(mono)

                buffer = np.concatenate([
                    buffer,
                    mono
                ])

                while len(buffer) >= native_window:

                    audio_window = resample_window(
                        buffer[:native_window],
                        native_rate
                    )

                    start_sec = (
                        window_start_sample / native_rate
                    )

                    rms = calculate_rms(
                        audio_window
                    )

                    active = (
                        rms >= RMS_THRESHOLD
                    )

                    embedding = None

                    if active:

                        embedding = inference(
                            yamnet,
                            audio_window
                        )

                    current = []

                    for name, model in classifiers.items():

                        row = predict_window(
                            name,
                            model,
                            embedding,
                            active,
                            start_sec,
                            rms,
                            states[name]
                        )

                        rows.append(row)
                        current.append(row)

                        if row["simulated_alert"]:

                            print(
                                f"\n[ALERT SIMULASI] "
                                f"{name} | "
                                f"{row['predicted_label']} | "
                                f"{row['confidence']:.2%} | "
                                f"detik {row['end_sec']:.2f}"
                            )

                    window_count += 1

                    # Tampilkan ringkasan setiap 5 window
                    # agar terminal tidak terlalu ramai.
                    if window_count % 5 == 0:

                        print()
                        print(
                            f"Window {window_count} | "
                            f"t={start_sec:.1f}s | "
                            f"RMS={rms:.4f}"
                        )

                        for row in current:

                            print(
                                f"  {row['model']:<14} "
                                f"{row['predicted_label']:<17} "
                                f"{row['confidence']:.2%}"
                            )

                    buffer = buffer[native_hop:]

                    window_start_sample += native_hop

    except KeyboardInterrupt:

        print()
        print("Tes dihentikan melalui keyboard.")

    # ======================================================
    # SIMPAN AUDIO DAN PREDIKSI
    # ======================================================

    if not recorded_blocks or not rows:

        print(
            "Tidak ada data cukup untuk dianalisis."
        )
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    out = OUTPUT_ROOT / f"run_{timestamp}"

    out.mkdir(
        parents=True,
        exist_ok=False
    )

    recorded_audio = np.concatenate(
        recorded_blocks
    )

    audio_16k = resample_window_full(
        recorded_audio,
        native_rate
    )

    audio_path = out / "microphone_capture_16k.wav"

    sf.write(
        str(audio_path),
        audio_16k,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    predictions = pd.DataFrame(rows)

    events = predictions[
        predictions["simulated_alert"]
    ].copy()

    summary_rows = []

    duration = (
        captured_samples / native_rate
    )

    for name, group in predictions.groupby(
        "model",
        sort=False
    ):

        siren = (
            group["predicted_label"] == "siren"
        )

        summary_rows.append({
            "model": name,
            "duration_sec": round(duration, 2),
            "total_windows": len(group),
            "active_windows": int(
                group["active"].sum()
            ),
            "siren_top1": int(
                siren.sum()
            ),
            "siren_above_threshold": int(
                (
                    siren
                    & group["passes_threshold"]
                ).sum()
            ),
            "all_danger_above_threshold": int(
                group["passes_threshold"].sum()
            ),
            "simulated_siren_alerts": int(
                (
                    siren
                    & group["simulated_alert"]
                ).sum()
            ),
            "simulated_total_alerts": int(
                group["simulated_alert"].sum()
            ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    predictions.to_csv(
        out / "window_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    events.to_csv(
        out / "alert_events.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        out / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "experiment": "live_microphone_siren_test",
        "capture_device": device["name"],
        "native_sample_rate": native_rate,
        "target_sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "rms_threshold": RMS_THRESHOLD,
        "latch_seconds": LATCH_SECONDS,
        "thresholds": THRESHOLDS,
        "confirmations": CONFIRMATIONS,
        "duration_sec": duration,
        "audio_overflow_count": overflow_count,
        "yamnet_sha256": sha256_file(YAMNET_PATH),
        "classifier_sha256": {
            name: sha256_file(path)
            for name, path in MODEL_PATHS.items()
        },
        "limitations": [
            "Sirene yang diputar belum dianotasi waktu mulai dan berakhir.",
            "Simulasi alert bukan notifikasi Android aktual.",
            "Resampling menggunakan scipy pada window audio.",
            "Aturan latch adalah pendekatan terhadap Flutter.",
            "Audio berasal dari mikrofon komputer, bukan mikrofon Android."
        ]
    }

    (out / "experiment_config.json").write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("HASIL LIVE SIREN TEST")
    print("=" * 70)

    print(
        summary.to_string(index=False)
    )

    print()
    print("Audio overflow:", overflow_count)
    print("Hasil disimpan:", out)
    print()
    print("Model baseline dan V1/V2 tidak diubah.")


def resample_window_full(waveform, source_rate):

    gcd = math.gcd(
        source_rate,
        SAMPLE_RATE
    )

    return resample_poly(
        waveform,
        SAMPLE_RATE // gcd,
        source_rate // gcd
    ).astype(np.float32)


if __name__ == "__main__":
    main()
