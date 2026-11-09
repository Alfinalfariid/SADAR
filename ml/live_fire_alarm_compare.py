
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly


# ==========================================================
# KONFIGURASI PROJECT
# ==========================================================

ROOT = Path(__file__).resolve().parent

YAMNET_PATH = ROOT / (
    "../mobile/assets/models/"
    "yamnet_embedding_dynamic_int8.tflite"
)

CLASSIFIER_PATHS = {
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
    "results/robustness_v2/live_fire_alarm_test"
)

SR = 16000
WINDOW = 15600
HOP = 7680

RMS_MIN = 0.008
LATCH = 2.5

CUE_START = 5.0
CUE_STOP = 25.0

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


# ==========================================================
# UTILITIES
# ==========================================================

def hash_file(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def create_interpreter(
    path,
    expected_input,
    expected_output
):

    if not path.is_file():
        raise FileNotFoundError(
            f"Model belum ditemukan: {path}"
        )

    model = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2
    )

    input_spec = model.get_input_details()[0]

    if list(input_spec["shape"]) != expected_input:

        model.resize_tensor_input(
            input_spec["index"],
            expected_input,
            strict=False
        )

    model.allocate_tensors()

    input_spec = model.get_input_details()[0]
    output_spec = model.get_output_details()[0]

    if list(input_spec["shape"]) != expected_input:
        raise ValueError(
            f"Input shape salah: {path}"
        )

    if (
        int(np.prod(output_spec["shape"]))
        != expected_output
    ):
        raise ValueError(
            f"Output size salah: {path}"
        )

    if (
        input_spec["dtype"] != np.float32
        or output_spec["dtype"] != np.float32
    ):
        raise ValueError(
            f"Tipe data bukan float32: {path}"
        )

    return (
        model,
        input_spec["index"],
        output_spec["index"],
        tuple(expected_input)
    )


def infer(bundle, data):

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
            "Output TFLite mengandung NaN atau Inf"
        )

    return result


def phase_for(start, end):

    if end <= CUE_START:
        return "before_alarm"

    if start >= CUE_STOP:
        return "after_alarm"

    if (
        start >= CUE_START
        and end <= CUE_STOP
    ):
        return "during_alarm"

    return "transition"


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seconds",
        type=float,
        default=45.0
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

    if args.seconds < 31:
        raise ValueError(
            "Gunakan --seconds minimal 31 "
            "(disarankan 45)."
        )

    print("=" * 65)
    print("DANGER SOUND AI - LIVE FIRE ALARM TEST")
    print("=" * 65)

    # ======================================================
    # LOAD MODEL
    # ======================================================

    print("\nMemuat ketiga classifier dan YAMNet...")

    yamnet = create_interpreter(
        YAMNET_PATH,
        [WINDOW],
        1024
    )

    classifiers = {}

    for name, path in CLASSIFIER_PATHS.items():

        classifiers[name] = create_interpreter(
            path,
            [1, 1024],
            6
        )

        print("Model siap:", name)

    # ======================================================
    # SETUP MIKROFON
    # ======================================================

    device = sd.query_devices(
        args.device,
        "input"
    )

    native_sr = int(
        round(device["default_samplerate"])
    )

    if native_sr <= 0:
        raise ValueError(
            "Sample rate mikrofon tidak valid"
        )

    sd.check_input_settings(
        device=args.device,
        channels=1,
        samplerate=native_sr
    )

    print("\nMikrofon:", device["name"])
    print("Sample rate mikrofon:", native_sr)
    print("Durasi rekaman:", args.seconds, "detik")

    print("\nSKENARIO:")
    print("0-5 detik   : Jangan putar suara alarm")
    print("5-25 detik  : Putar rekaman FIRE ALARM")
    print("25-45 detik : Hentikan suara alarm")

    print(
        "\nGunakan volume sedang."
    )

    print(
        "Rekaman dianalisis SETELAH selesai."
    )

    input(
        "\nTekan ENTER saat siap memulai... "
    )

    # ======================================================
    # REKAM SUARA MIKROFON
    # ======================================================

    chunks = []

    captured = 0

    cue_start_sent = False
    cue_stop_sent = False

    overflow_count = 0

    blocksize = 1024

    target_samples = int(
        round(args.seconds * native_sr)
    )

    print(
        "\nPENGUJIAN DIMULAI",
        flush=True
    )

    try:

        with sd.InputStream(
            device=args.device,
            samplerate=native_sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize
        ) as stream:

            while captured < target_samples:

                block, overflow = stream.read(
                    blocksize
                )

                overflow_count += int(
                    bool(overflow)
                )

                remaining = (
                    target_samples - captured
                )

                chunk = block[
                    :remaining, 0
                ].copy()

                chunks.append(chunk)

                captured += len(chunk)

                elapsed = (
                    captured / native_sr
                )

                if (
                    not cue_start_sent
                    and elapsed >= CUE_START
                ):

                    cue_start_sent = True

                    print(
                        "\n>>> SEKARANG PUTAR FIRE ALARM! <<<",
                        flush=True
                    )

                if (
                    not cue_stop_sent
                    and elapsed >= CUE_STOP
                ):

                    cue_stop_sent = True

                    print(
                        "\n>>> SEKARANG HENTIKAN FIRE ALARM! <<<",
                        flush=True
                    )

    except KeyboardInterrupt:

        print(
            "\nRekaman dihentikan lebih awal."
        )

    if not chunks:
        raise RuntimeError(
            "Tidak ada audio yang terekam."
        )

    # ======================================================
    # RESAMPLING KE 16 KHZ
    # ======================================================

    original_audio = np.concatenate(
        chunks
    )

    gcd = math.gcd(
        native_sr,
        SR
    )

    audio = resample_poly(
        original_audio,
        SR // gcd,
        native_sr // gcd
    ).astype(np.float32)

    duration = len(audio) / SR

    # ======================================================
    # SIMPAN AUDIO
    # ======================================================

    run_dir = OUTPUT_ROOT / (
        datetime.now().strftime(
            "run_%Y%m%d_%H%M%S_%f"
        )
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    wav_path = run_dir / (
        "fire_alarm_microphone_16k.wav"
    )

    sf.write(
        wav_path,
        audio,
        SR,
        subtype="PCM_16"
    )

    # ======================================================
    # ANALISIS KETIGA MODEL
    # ======================================================

    states = {
        name: {
            "last": None,
            "streak": 0,
            "next_allowed": 0.0
        }
        for name in classifiers
    }

    rows = []

    print(
        "\nMenganalisis rekaman 16 kHz "
        "dengan tiga model..."
    )

    for window_no, start_sample in enumerate(
        range(
            0,
            len(audio) - WINDOW + 1,
            HOP
        ),
        start=1
    ):

        segment = audio[
            start_sample:start_sample + WINDOW
        ]

        rms = float(
            np.sqrt(
                np.mean(
                    segment.astype(np.float64) ** 2
                )
            )
        )

        active = rms >= RMS_MIN

        embedding = (
            infer(yamnet, segment)
            if active
            else None
        )

        start_sec = start_sample / SR

        end_sec = (
            start_sample + WINDOW
        ) / SR

        phase = phase_for(
            start_sec,
            end_sec
        )

        for name, classifier in (
            classifiers.items()
        ):

            if active:

                scores = infer(
                    classifier,
                    embedding
                )

                class_index = int(
                    np.argmax(scores)
                )

                label = CLASSES[class_index]

                confidence = float(
                    scores[class_index]
                )

            else:

                label = "background"
                confidence = 1.0

            passed = bool(
                active
                and label in THRESHOLDS
                and confidence >= THRESHOLDS.get(
                    label,
                    2.0
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

            alerted = False

            if (
                passed
                and state["streak"] >= CONFIRM[label]
                and end_sec >= state["next_allowed"]
            ):

                alerted = True

                state["next_allowed"] = (
                    end_sec + LATCH
                )

                print(
                    f"[ALERT SIMULASI] "
                    f"{name:<14} | "
                    f"{label:<16} | "
                    f"{confidence:.1%} | "
                    f"t={end_sec:.2f}s"
                )

            rows.append({
                "model": name,
                "window_no": window_no,
                "start_sec": round(
                    start_sec, 4
                ),
                "end_sec": round(
                    end_sec, 4
                ),
                "phase": phase,
                "rms": round(rms, 6),
                "active": active,
                "predicted_label": label,
                "confidence": round(
                    confidence, 6
                ),
                "passes_threshold": passed,
                "simulated_alert": alerted
            })

    if not rows:
        raise ValueError(
            "Rekaman terlalu pendek "
            "untuk satu window."
        )

    # ======================================================
    # HASIL PREDIKSI
    # ======================================================

    predictions = pd.DataFrame(rows)

    events = predictions[
        predictions["simulated_alert"]
    ].copy()

    summary_rows = []

    for name, group in predictions.groupby(
        "model",
        sort=False
    ):

        during = group[
            group["phase"] == "during_alarm"
        ]

        alert_during = during[
            during["simulated_alert"]
        ]

        alarm_during = during[
            during["predicted_label"] == "alarm"
        ]

        alarm_above = alarm_during[
            alarm_during["passes_threshold"]
        ]

        correct_alert = alert_during[
            alert_during["predicted_label"]
            == "alarm"
        ]

        wrong_alert = alert_during[
            alert_during["predicted_label"]
            != "alarm"
        ]

        first_alarm = (
            correct_alert["end_sec"].min()
            if len(correct_alert)
            else np.nan
        )

        summary_rows.append({
            "model": name,

            "total_windows": len(group),

            "active_windows": int(
                group["active"].sum()
            ),

            "playback_windows": len(during),

            "alarm_top1_playback": len(
                alarm_during
            ),

            "alarm_above_threshold_playback": len(
                alarm_above
            ),

            "horn_top1_playback": int(
                (
                    during["predicted_label"]
                    == "vehicle_horn"
                ).sum()
            ),

            "alarm_alerts_playback": len(
                correct_alert
            ),

            "wrong_class_alerts_playback": len(
                wrong_alert
            ),

            "any_alerts_playback": len(
                alert_during
            ),

            "alerts_before_playback": int(
                (
                    (
                        group["phase"]
                        == "before_alarm"
                    )
                    & group["simulated_alert"]
                ).sum()
            ),

            "alerts_after_playback": int(
                (
                    (
                        group["phase"]
                        == "after_alarm"
                    )
                    & group["simulated_alert"]
                ).sum()
            ),

            "first_alarm_alert_sec": first_alarm
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # ======================================================
    # SIMPAN HASIL CSV
    # ======================================================

    predictions.to_csv(
        run_dir / "window_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    events.to_csv(
        run_dir / "alert_events.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        run_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================================
    # SIMPAN KONFIGURASI PENELITIAN
    # ======================================================

    config = {
        "experiment": "live_fire_alarm_test",

        "capture_device": device["name"],

        "native_sample_rate": native_sr,

        "target_sample_rate": SR,

        "duration_sec": duration,

        "manual_cue_start_sec": CUE_START,

        "manual_cue_stop_sec": CUE_STOP,

        "manual_cues_are_approximate": True,

        "window_samples": WINDOW,

        "hop_samples": HOP,

        "rms_threshold": RMS_MIN,

        "thresholds": THRESHOLDS,

        "confirmation_windows": CONFIRM,

        "latch_sec": LATCH,

        "input_overflow_count": overflow_count,

        "wav_sha256": hash_file(wav_path),

        "yamnet_sha256": hash_file(
            YAMNET_PATH
        ),

        "classifier_sha256": {
            name: hash_file(path)
            for name, path in CLASSIFIER_PATHS.items()
        },

        "limitations": [
            "Playback timing is manually controlled "
            "and approximate.",

            "Alerts are offline simulations, "
            "not actual Android notifications.",

            "A single alarm may generate "
            "repeated alerts.",

            "This test is not a latency benchmark."
        ]
    }

    (
        run_dir / "experiment_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # ======================================================
    # TAMPILKAN HASIL
    # ======================================================

    print("\n" + "=" * 65)
    print("HASIL LIVE FIRE ALARM TEST")
    print("=" * 65)

    print(
        summary.to_string(index=False)
    )

    print(
        "\nAudio overflow:",
        overflow_count
    )

    print(
        "Hasil disimpan:",
        run_dir
    )

    print(
        "Model tidak diubah."
    )


if __name__ == "__main__":
    main()
