
from collections import Counter
from datetime import datetime
from pathlib import Path
from math import gcd
import csv
import hashlib
import json

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

MODELS = PROJECT / "mobile" / "assets" / "models"

AUDIO_DIR = (
    ROOT / "dataset" / "realworld_diagnostics_v3"
)

YAMNET_FILE = (
    MODELS / "yamnet_embedding_dynamic_int8.tflite"
)

V3_FILE = (
    MODELS / "danger_classifier_fp32.tflite"
)

V4_FILE = (
    ROOT / "models" / "candidates"
    / "danger_classifier_fp32_v4.tflite"
)

EXPECTED_YAMNET_HASH = (
    "d1e8efac7c78a7929db7049547eeab55264cad6837696e2c28280a8e4483114a"
)

EXPECTED_V3_HASH = (
    "90fc6136ceeae9382ff9745d4ec98a19582afebe5829e96a051b3dfa4a20ca37"
)

CLASS_NAMES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

RECORDINGS = [
    ("02-11-2026 21.49.wav", "2_meter"),
    ("02-11-2026 21.50.wav", "30_cm"),
]

SR = 16000
WIN = 15600
HOP = 7680
RMS_MIN = 0.008


def digest(path):
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_interpreter(path, desired_input_shape):
    if not path.is_file():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2,
    )

    inp = interpreter.get_input_details()[0]

    interpreter.resize_tensor_input(
        inp["index"],
        desired_input_shape,
        strict=False,
    )

    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]

    if inp["dtype"] != np.float32:
        raise RuntimeError(
            f"Input bukan float32: {path.name}"
        )

    return interpreter, inp["index"]


def forward(
    interpreter,
    input_idx,
    samples,
    expected_dims,
):
    interpreter.set_tensor(
        input_idx,
        samples.astype(np.float32),
    )

    interpreter.invoke()

    possible = []

    for info in interpreter.get_output_details():
        array = interpreter.get_tensor(
            info["index"]
        )

        if array.size == expected_dims:
            possible.append(
                array.reshape(-1)
            )

    if len(possible) != 1:
        raise RuntimeError(
            "Output model tidak sesuai atau ambigu: "
            f"{expected_dims} nilai, "
            f"{len(possible)} kandidat."
        )

    result = possible[0].astype(np.float32)

    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Prediksi mengandung NaN/Inf."
        )

    return result


def read_audio(path):
    signal, source_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    mono = signal.mean(
        axis=1,
        dtype=np.float32,
    )

    if source_sr != SR:
        divisor = gcd(
            int(source_sr),
            SR,
        )

        mono = resample_poly(
            mono,
            SR // divisor,
            int(source_sr) // divisor,
        ).astype(np.float32)

    return mono, source_sr


def main():
    for path in [
        YAMNET_FILE,
        V3_FILE,
        V4_FILE,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if digest(YAMNET_FILE) != EXPECTED_YAMNET_HASH:
        raise RuntimeError(
            "YAMNet berbeda dari eksperimen V3."
        )

    if digest(V3_FILE) != EXPECTED_V3_HASH:
        raise RuntimeError(
            "Classifier V3 berbeda dari baseline."
        )

    print("MODEL VERIFIED: YAMNet dan V3")
    print("V4 SHA256:", digest(V4_FILE))

    yamnet, yi = load_interpreter(
        YAMNET_FILE,
        [WIN],
    )

    v3, v3i = load_interpreter(
        V3_FILE,
        [1, 1024],
    )

    v4, v4i = load_interpreter(
        V4_FILE,
        [1, 1024],
    )

    out = (
        ROOT
        / "results"
        / "realworld_v3_v4"
        / datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    )

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    rows = []
    summaries = []

    for filename, distance in RECORDINGS:
        path = AUDIO_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        wave, source_sr = read_audio(path)

        counts = {
            "V3": Counter(),
            "V4": Counter(),
        }

        strong_horns = {
            "V3": 0,
            "V4": 0,
        }

        active_count = 0
        quiet_count = 0

        for win_index, start in enumerate(
            range(
                0,
                len(wave) - WIN + 1,
                HOP,
            )
        ):
            clip = wave[start:start + WIN]

            rms = float(
                np.sqrt(
                    np.mean(
                        clip.astype(np.float64) ** 2
                    )
                )
            )

            if rms < RMS_MIN:
                quiet_count += 1
                continue

            active_count += 1

            embedding = forward(
                yamnet,
                yi,
                clip,
                1024,
            )

            for model_name, interpreter, inp in [
                ("V3", v3, v3i),
                ("V4", v4, v4i),
            ]:
                probs = forward(
                    interpreter,
                    inp,
                    embedding.reshape(1, 1024),
                    6,
                )

                pred_id = int(
                    np.argmax(probs)
                )

                pred = CLASS_NAMES[pred_id]

                counts[model_name][pred] += 1

                if (
                    pred == "vehicle_horn"
                    and float(probs[pred_id]) >= 0.70
                ):
                    strong_horns[model_name] += 1

                row = {
                    "file": filename,
                    "distance": distance,
                    "model": model_name,
                    "window": win_index,
                    "start_sec": round(
                        start / SR, 4
                    ),
                    "end_sec": round(
                        (start + WIN) / SR, 4
                    ),
                    "rms": rms,
                    "top1": pred,
                    "top1_confidence": float(
                        probs[pred_id]
                    ),
                }

                row.update({
                    "prob_" + name: float(probs[i])
                    for i, name in enumerate(
                        CLASS_NAMES
                    )
                })

                rows.append(row)

        for model_name in ("V3", "V4"):
            result = {
                "file": filename,
                "distance": distance,
                "model": model_name,
                "source_sr": source_sr,
                "active_windows": active_count,
                "quiet_windows": quiet_count,
                "strong_horn_windows_ge_070":
                    strong_horns[model_name],
            }

            result.update({
                "count_" + name:
                    counts[model_name][name]
                for name in CLASS_NAMES
            })

            result.update({
                "pct_" + name: round(
                    100
                    * counts[model_name][name]
                    / max(active_count, 1),
                    2,
                )
                for name in CLASS_NAMES
            })

            summaries.append(result)

    with (
        out / "paired_window_predictions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    with (
        out / "summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0]),
        )

        writer.writeheader()
        writer.writerows(summaries)

    (
        out / "models_and_method.json"
    ).write_text(
        json.dumps({
            "yamnet_sha256": digest(YAMNET_FILE),
            "v3_sha256": digest(V3_FILE),
            "v4_sha256": digest(V4_FILE),
            "sample_rate": SR,
            "window_size": WIN,
            "hop_size": HOP,
            "rms_threshold": RMS_MIN,
            "evaluation": (
                "Diagnostic comparison, "
                "not an independent blind test"
            ),
        }, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== HASIL AUDIO NYATA V3 VS V4 ==="
    )

    for item in summaries:
        print(
            f"{item['distance']:8} "
            f"{item['model']}: "
            f"aktif={item['active_windows']}, "
            f"alarm={item['count_alarm']} "
            f"({item['pct_alarm']:.2f}%), "
            f"klakson={item['count_vehicle_horn']} "
            f"({item['pct_vehicle_horn']:.2f}%), "
            "klakson>=70%="
            f"{item['strong_horn_windows_ge_070']}, "
            f"background={item['count_background']}, "
            f"sirene={item['count_siren']}"
        )

    print("\nOutput:", out)

    print(
        "Model dan WAV asli tidak diubah."
    )


if __name__ == "__main__":
    main()
