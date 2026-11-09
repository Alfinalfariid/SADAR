
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from evaluate_realworld_alarm_v3 import (
    AUDIO_DIR,
    CLASSIFIER_PATH,
    CLASS_NAMES,
    HOP_SIZE,
    RMS_THRESHOLD,
    ROOT,
    SAMPLE_RATE,
    WINDOW_SIZE,
    YAMNET_PATH,
    load_interpreter,
    read_audio,
    run_classifier,
    run_embedding,
    verify_models,
)


CASES = [
    ("30cm_original", "02-11-2026 21.50.wav", 1.0),
    ("30cm_gain_070", "02-11-2026 21.50.wav", 0.70),
    ("30cm_gain_050", "02-11-2026 21.50.wav", 0.50),
    ("30cm_gain_044", "02-11-2026 21.50.wav", 0.44),
    ("30cm_gain_030", "02-11-2026 21.50.wav", 0.30),
    ("2m_original", "02-11-2026 21.49.wav", 1.0),
]


def main():
    verify_models()

    yamnet, y_input, y_output = load_interpreter(
        YAMNET_PATH, [WINDOW_SIZE]
    )

    classifier, c_input, c_output = load_interpreter(
        CLASSIFIER_PATH, [1, 1024]
    )

    output_dir = (
        ROOT
        / "results"
        / "realworld_alarm_v3"
        / (
            "gain_sweep_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    window_rows = []
    summary_rows = []

    for case_name, file_name, gain in CASES:
        audio, original_sr = read_audio(
            AUDIO_DIR / file_name
        )

        audio = (
            audio * gain
        ).astype(np.float32)

        counts = Counter()
        active_count = 0
        filtered_count = 0

        strong_horn_count = 0
        adjacent_strong_horn_pairs = 0
        previous_strong_horn = False

        active_rms = []

        for number, start in enumerate(
            range(
                0,
                len(audio) - WINDOW_SIZE + 1,
                HOP_SIZE,
            )
        ):
            window = audio[
                start:start + WINDOW_SIZE
            ]

            rms = float(
                np.sqrt(
                    np.mean(
                        window.astype(np.float64) ** 2
                    )
                )
            )

            row = {
                "case": case_name,
                "file": file_name,
                "gain": gain,
                "window": number,
                "start_sec": round(
                    start / SAMPLE_RATE, 4
                ),
                "rms": rms,
            }

            if rms < RMS_THRESHOLD:
                filtered_count += 1
                previous_strong_horn = False

                row.update(
                    top1="silence_filtered",
                    confidence=0.0,
                )

                window_rows.append(row)
                continue

            embedding = run_embedding(
                yamnet,
                y_input,
                y_output,
                window,
            )

            probs = run_classifier(
                classifier,
                c_input,
                c_output,
                embedding,
            )

            top_index = int(np.argmax(probs))
            label = CLASS_NAMES[top_index]
            confidence = float(
                probs[top_index]
            )

            counts[label] += 1
            active_count += 1
            active_rms.append(rms)

            strong_horn = (
                label == "vehicle_horn"
                and confidence >= 0.70
            )

            if strong_horn:
                strong_horn_count += 1

            if (
                strong_horn
                and previous_strong_horn
            ):
                adjacent_strong_horn_pairs += 1

            previous_strong_horn = strong_horn

            row.update(
                top1=label,
                confidence=confidence,
                prob_alarm=float(probs[0]),
                prob_vehicle_horn=float(probs[5]),
            )

            window_rows.append(row)

        alarm_count = counts["alarm"]
        horn_count = counts["vehicle_horn"]

        summary = {
            "case": case_name,
            "gain": gain,
            "original_sample_rate": original_sr,
            "active_windows": active_count,
            "filtered_windows": filtered_count,
            "mean_active_rms": (
                float(np.mean(active_rms))
                if active_rms
                else 0.0
            ),
            "alarm_top1": alarm_count,
            "alarm_pct": (
                round(
                    100 * alarm_count / active_count,
                    2,
                )
                if active_count
                else 0.0
            ),
            "horn_top1": horn_count,
            "horn_pct": (
                round(
                    100 * horn_count / active_count,
                    2,
                )
                if active_count
                else 0.0
            ),
            "horn_top1_ge_70": strong_horn_count,
            "adjacent_high_horn_pairs": (
                adjacent_strong_horn_pairs
            ),
        }

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    window_df = pd.DataFrame(window_rows)

    summary_df.to_csv(
        output_dir / "gain_sweep_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    window_df.to_csv(
        output_dir / "gain_sweep_per_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== HASIL PENGUJIAN GAIN V3 ===")

    print(
        summary_df.to_string(index=False)
    )

    print("\nFile hasil:", output_dir)
    print("Model asli dan WAV tidak diubah.")

    print(
        "Adjacent pairs hanya indikasi, "
        "bukan simulasi lengkap latch Android."
    )


if __name__ == "__main__":
    main()
