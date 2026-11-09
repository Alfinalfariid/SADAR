
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import json
import math

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent

AUDIO_DIR = ROOT / "dataset/realworld_diagnostics_v3"

AUDIO_FILES = [
    AUDIO_DIR / "02-11-2026 21.49.wav",
    AUDIO_DIR / "02-11-2026 21.50.wav",
]

MODEL_DIR = ROOT.parent / "mobile/assets/models"

YAMNET_PATH = (
    MODEL_DIR / "yamnet_embedding_dynamic_int8.tflite"
)

CLASSIFIER_PATH = (
    MODEL_DIR / "danger_classifier_fp32.tflite"
)

EXPECTED_YAMNET_HASH = (
    "d1e8efac7c78a7929db7049547eeab55264cad6837696e2c28280a8e4483114a"
)

EXPECTED_CLASSIFIER_HASH = (
    "90fc6136ceeae9382ff9745d4ec98a19582afebe5829e96a051b3dfa4a20ca37"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680
RMS_THRESHOLD = 0.008

CLASS_NAMES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_models():
    for path, expected in [
        (YAMNET_PATH, EXPECTED_YAMNET_HASH),
        (CLASSIFIER_PATH, EXPECTED_CLASSIFIER_HASH),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

        actual = file_hash(path)

        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"Hash model tidak sesuai: {path.name}\n"
                f"Ditemukan: {actual}\n"
                f"Diharapkan: {expected}\n"
                "Pengujian dibatalkan agar tidak "
                "menggunakan model yang keliru."
            )

        print(f"MODEL VERIFIED: {path.name}")


def load_interpreter(path, expected_input):
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2,
    )

    input_info = interpreter.get_input_details()[0]

    current_shape = tuple(
        int(x) for x in input_info["shape"]
    )

    if current_shape != tuple(expected_input):
        interpreter.resize_tensor_input(
            input_info["index"],
            expected_input,
            strict=False,
        )

    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    if input_info["dtype"] != np.float32:
        raise RuntimeError("Input model bukan FLOAT32.")

    if output_info["dtype"] != np.float32:
        raise RuntimeError("Output model bukan FLOAT32.")

    return interpreter, input_info, output_info


def read_audio(path):
    waveform, original_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    # Stereo menjadi mono.
    mono = waveform.mean(
        axis=1,
        dtype=np.float32,
    )

    if original_sr != SAMPLE_RATE:
        factor = math.gcd(
            original_sr,
            SAMPLE_RATE,
        )

        mono = resample_poly(
            mono,
            SAMPLE_RATE // factor,
            original_sr // factor,
        ).astype(np.float32)

    if len(mono) < WINDOW_SIZE:
        raise ValueError(
            f"Audio terlalu pendek: {path.name}"
        )

    return mono, original_sr


def run_embedding(
    interpreter,
    input_info,
    output_info,
    window,
):
    interpreter.set_tensor(
        input_info["index"],
        window.astype(np.float32),
    )

    interpreter.invoke()

    embedding = interpreter.get_tensor(
        output_info["index"]
    ).reshape(-1).astype(np.float32)

    if embedding.shape != (1024,):
        raise RuntimeError(
            f"Dimensi embedding salah: {embedding.shape}"
        )

    return embedding


def run_classifier(
    interpreter,
    input_info,
    output_info,
    embedding,
):
    interpreter.set_tensor(
        input_info["index"],
        embedding.reshape(1, 1024),
    )

    interpreter.invoke()

    probabilities = interpreter.get_tensor(
        output_info["index"]
    ).reshape(-1).astype(np.float32)

    if probabilities.shape != (6,):
        raise RuntimeError(
            f"Output classifier salah: {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Output classifier tidak valid."
        )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=0.02,
    ):
        raise RuntimeError(
            "Output classifier bukan probabilitas "
            "softmax yang diharapkan."
        )

    return probabilities


def main():
    print("=" * 65)
    print("SADAR V3 - EVALUASI ALARM KONDISI NYATA")
    print("=" * 65)

    verify_models()

    yamnet, y_in, y_out = load_interpreter(
        YAMNET_PATH,
        [WINDOW_SIZE],
    )

    classifier, c_in, c_out = load_interpreter(
        CLASSIFIER_PATH,
        [1, 1024],
    )

    timestamp = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )

    output_dir = (
        ROOT
        / "results/realworld_alarm_v3"
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    records = []
    summaries = []

    for path in AUDIO_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)

        waveform, original_sr = read_audio(path)

        print("\n" + "-" * 65)
        print("FILE:", path.name)
        print("Original SR:", original_sr)
        print(
            "Duration:",
            round(len(waveform) / SAMPLE_RATE, 3),
            "seconds",
        )

        counts = Counter()
        rms_values = []
        active_windows = 0
        skipped_windows = 0

        starts = range(
            0,
            len(waveform) - WINDOW_SIZE + 1,
            HOP_SIZE,
        )

        for index, start in enumerate(starts):
            window = waveform[
                start:start + WINDOW_SIZE
            ]

            rms = float(
                np.sqrt(
                    np.mean(
                        window.astype(np.float64) ** 2
                    )
                )
            )

            rms_values.append(rms)

            record = {
                "file": path.name,
                "window": index,
                "start_sec": round(
                    start / SAMPLE_RATE,
                    4,
                ),
                "end_sec": round(
                    (start + WINDOW_SIZE) / SAMPLE_RATE,
                    4,
                ),
                "rms": rms,
            }

            if rms < RMS_THRESHOLD:
                skipped_windows += 1
                record["top1"] = "silence_filtered"
                record["top1_confidence"] = 0.0
                record["top2"] = ""
                record["top2_confidence"] = 0.0

                for label in CLASS_NAMES:
                    record[f"prob_{label}"] = None

                records.append(record)
                continue

            embedding = run_embedding(
                yamnet,
                y_in,
                y_out,
                window,
            )

            probs = run_classifier(
                classifier,
                c_in,
                c_out,
                embedding,
            )

            ranked = np.argsort(probs)[::-1]

            top1 = int(ranked[0])
            top2 = int(ranked[1])

            label = CLASS_NAMES[top1]

            active_windows += 1
            counts[label] += 1

            record["top1"] = label
            record["top1_confidence"] = float(
                probs[top1]
            )
            record["top2"] = CLASS_NAMES[top2]
            record["top2_confidence"] = float(
                probs[top2]
            )

            for class_id, class_name in enumerate(
                CLASS_NAMES
            ):
                record[f"prob_{class_name}"] = float(
                    probs[class_id]
                )

            records.append(record)

        print("Window aktif:", active_windows)
        print("Window senyap:", skipped_windows)
        print(
            "RMS rata-rata window:",
            round(float(np.mean(rms_values)), 6),
        )

        print("\nDISTRIBUSI PREDIKSI TOP-1:")

        percentages = {}

        for label in CLASS_NAMES:
            count = counts[label]

            percent = (
                100.0 * count / active_windows
                if active_windows
                else 0.0
            )

            percentages[label] = round(
                percent,
                2,
            )

            print(
                f"{label:<17} "
                f"{count:>3} window | "
                f"{percent:6.2f}%"
            )

        summaries.append({
            "file": path.name,
            "original_sample_rate": original_sr,
            "active_windows": active_windows,
            "silence_filtered_windows":
                skipped_windows,
            "mean_window_rms": float(
                np.mean(rms_values)
            ),
            "top1_counts": dict(counts),
            "top1_percentages": percentages,
        })

    df = pd.DataFrame(records)

    csv_path = (
        output_dir / "window_predictions.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    json_path = (
        output_dir / "summary.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "model": "SADAR Classifier V3",
                "yamnet": "Dynamic INT8",
                "sample_rate": SAMPLE_RATE,
                "window_size": WINDOW_SIZE,
                "hop_size": HOP_SIZE,
                "rms_threshold": RMS_THRESHOLD,
                "decision_mode": (
                    "raw_classifier_without_"
                    "confirmation_or_latch"
                ),
                "files": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("EVALUASI SELESAI")
    print("=" * 65)
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Model tidak diubah.")
    print("Tidak dilakukan training.")


if __name__ == "__main__":
    main()
