
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import soundfile as sf


# ==================================================
# CONFIG
# ==================================================

INPUT_FILE = Path(
    "results/robustness_v1/"
    "demand_candidate_background_guarded.csv"
)

OUTPUT_DIR = Path(
    "results/robustness_v1"
)

WINDOW_CSV = (
    OUTPUT_DIR / "demand_window_rms.csv"
)

SEGMENT_CSV = (
    OUTPUT_DIR / "demand_segment_activity.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "demand_window_activity_summary.csv"
)

SAMPLE_RATE = 16000

WINDOW_SIZE = 15600

HOP_SIZE = 7680

RMS_THRESHOLD = 0.008


# ==================================================
# RMS
# ==================================================

def calculate_rms(audio):

    audio = np.asarray(
        audio,
        dtype=np.float64
    )

    return float(
        np.sqrt(
            np.mean(audio * audio)
        )
    )


# ==================================================
# SAVE CSV
# ==================================================

def save_csv(path, rows):

    if not rows:
        raise ValueError(
            f"Tidak ada data untuk {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print("DEMAND - REALTIME WINDOW ACTIVITY AUDIT")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "segment_id",
        "scene",
        "source_group",
        "audio_path",
        "start_sec",
        "end_sec",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}"
        )

    if df["segment_id"].duplicated().any():
        raise ValueError(
            "Ada segment_id duplikat."
        )

    window_rows = []
    segment_rows = []
    summary_rows = []

    # ==================================================
    # PROCESS PER SCENE
    # ==================================================

    for scene, scene_df in df.groupby(
        "scene",
        sort=True
    ):

        print()
        print("-" * 70)
        print("LINGKUNGAN:", scene)
        print("-" * 70)

        scene_windows = 0
        scene_active = 0
        scene_inactive = 0

        for _, row in scene_df.iterrows():

            audio_path = Path(
                row["audio_path"]
            )

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio tidak ditemukan: {audio_path}"
                )

            with sf.SoundFile(
                str(audio_path),
                mode="r"
            ) as audio_file:

                if audio_file.samplerate != SAMPLE_RATE:
                    raise ValueError(
                        f"Sample rate salah: {audio_path}"
                    )

                if audio_file.channels != 1:
                    raise ValueError(
                        f"Audio tidak mono: {audio_path}"
                    )

                start_sample = int(
                    round(
                        float(row["start_sec"])
                        * SAMPLE_RATE
                    )
                )

                end_sample = int(
                    round(
                        float(row["end_sec"])
                        * SAMPLE_RATE
                    )
                )

                expected_samples = (
                    end_sample - start_sample
                )

                audio_file.seek(
                    start_sample
                )

                audio = audio_file.read(
                    frames=expected_samples,
                    dtype="float32",
                    always_2d=False
                )

            if len(audio) != expected_samples:
                raise ValueError(
                    f"Panjang audio tidak sesuai: "
                    f"{row['segment_id']}"
                )

            if len(audio) < WINDOW_SIZE:
                raise ValueError(
                    f"Segmen terlalu pendek: "
                    f"{row['segment_id']}"
                )

            window_rms_values = []

            active_count = 0
            inactive_count = 0

            window_index = 0

            # ======================================
            # WINDOW 15600, HOP 7680
            # ======================================

            for offset in range(
                0,
                len(audio) - WINDOW_SIZE + 1,
                HOP_SIZE
            ):

                window = audio[
                    offset:offset + WINDOW_SIZE
                ]

                rms = calculate_rms(
                    window
                )

                active = (
                    rms >= RMS_THRESHOLD
                )

                if active:
                    active_count += 1
                else:
                    inactive_count += 1

                window_index += 1

                window_rms_values.append(
                    rms
                )

                window_rows.append({
                    "scene": scene,
                    "source_group": row["source_group"],
                    "segment_id": row["segment_id"],
                    "window_index": window_index,
                    "start_sec": round(
                        float(row["start_sec"])
                        + offset / SAMPLE_RATE,
                        5
                    ),
                    "end_sec": round(
                        float(row["start_sec"])
                        + (offset + WINDOW_SIZE)
                        / SAMPLE_RATE,
                        5
                    ),
                    "rms": round(rms, 7),
                    "threshold": RMS_THRESHOLD,
                    "activity": (
                        "ACTIVE"
                        if active
                        else "BELOW_THRESHOLD"
                    ),
                })

            total_windows = len(
                window_rms_values
            )

            segment_rows.append({
                "scene": scene,
                "source_group": row["source_group"],
                "segment_id": row["segment_id"],
                "total_windows": total_windows,
                "active_windows": active_count,
                "inactive_windows": inactive_count,
                "active_ratio": round(
                    active_count / total_windows,
                    4
                ),
                "mean_rms": round(
                    float(np.mean(window_rms_values)),
                    7
                ),
                "max_rms": round(
                    float(np.max(window_rms_values)),
                    7
                ),
                "min_rms": round(
                    float(np.min(window_rms_values)),
                    7
                ),
            })

            scene_windows += total_windows
            scene_active += active_count
            scene_inactive += inactive_count

        active_pct = (
            100 * scene_active / scene_windows
            if scene_windows
            else 0
        )

        summary_rows.append({
            "scene": scene,
            "source_group": f"DEMAND_{scene}",
            "segments": len(scene_df),
            "total_windows": scene_windows,
            "active_windows": scene_active,
            "below_threshold_windows": scene_inactive,
            "active_pct": round(active_pct, 2),
            "sample_rate": SAMPLE_RATE,
            "window_size": WINDOW_SIZE,
            "hop_size": HOP_SIZE,
            "rms_threshold": RMS_THRESHOLD,
        })

        print("Segmen terpilih :", len(scene_df))
        print("Total window    :", scene_windows)
        print("Window aktif    :", scene_active)
        print("Window rendah   :", scene_inactive)
        print("Persentase aktif:", round(active_pct, 2), "%")

    # ==================================================
    # SAVE
    # ==================================================

    save_csv(
        WINDOW_CSV,
        window_rows
    )

    save_csv(
        SEGMENT_CSV,
        segment_rows
    )

    save_csv(
        SUMMARY_CSV,
        summary_rows
    )

    total_windows = sum(
        row["total_windows"]
        for row in summary_rows
    )

    total_active = sum(
        row["active_windows"]
        for row in summary_rows
    )

    total_inactive = sum(
        row["below_threshold_windows"]
        for row in summary_rows
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    print()
    print("=" * 70)
    print("RINGKASAN AUDIT WINDOW")
    print("=" * 70)

    print("Total segmen  :", len(segment_rows))
    print("Total window  :", total_windows)
    print("Window aktif  :", total_active)
    print("Window rendah :", total_inactive)

    print(
        "Aktif (%)     :",
        round(
            100 * total_active / total_windows,
            2
        ) if total_windows else 0
    )

    print()
    print("File hasil:")
    print(WINDOW_CSV)
    print(SEGMENT_CSV)
    print(SUMMARY_CSV)

    print()
    print(
        "CATATAN: Pemeriksaan ini hanya "
        "mengukur aktivitas berdasarkan RMS."
    )

    print(
        "Tidak ada label baru yang diberikan, "
        "dan dataset/model baseline tetap aman."
    )


if __name__ == "__main__":
    main()
