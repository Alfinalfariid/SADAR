
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent

SOURCE_DIR = (
    ROOT
    / "dataset/robustness_v3/alarm_curated"
    / "run_20261102_063122_274252"
)

MANIFEST = (
    SOURCE_DIR / "alarm_train_manifest_curated.csv"
)

OUTPUT_ROOT = (
    ROOT / "results/robustness_v3/alarm_final_review"
)

SAMPLE_RATE = 16000
GAP_SECONDS = 0.8

EXCLUDED = {
    "alarm_02_clip_001.wav",
    "alarm_02_clip_011.wav",
}


def main():
    print("=" * 60)
    print("ROBUSTNESS V3 - FINAL ALARM REVIEW")
    print("=" * 60)

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    data = pd.read_csv(MANIFEST)

    if len(data) != 29:
        raise ValueError(
            f"Harus ada 29 segmen, ditemukan {len(data)}"
        )

    if not (data["target_class"] == "alarm").all():
        raise ValueError("Ada label selain alarm.")

    if not (data["split"] == "train").all():
        raise ValueError("Ada data selain train.")

    data = data.copy()
    data["filename"] = data["path"].map(
        lambda value: Path(str(value)).name
    )

    if data["filename"].duplicated().any():
        raise ValueError("Ada nama file duplikat.")

    if set(data["filename"]) & EXCLUDED:
        raise ValueError(
            "Segmen yang dikeluarkan masih ada!"
        )

    data = data.sort_values(
        "filename"
    ).reset_index(drop=True)

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    output_dir = OUTPUT_ROOT / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    gap = np.zeros(
        int(GAP_SECONDS * SAMPLE_RATE),
        dtype=np.float32
    )

    montage_parts = []
    records = []
    cursor_samples = 0

    for index, row in data.iterrows():
        path = Path(str(row["path"]))

        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        audio, sr = sf.read(
            path,
            dtype="float32",
            always_2d=True
        )

        if sr != SAMPLE_RATE or audio.shape[1] != 1:
            raise ValueError(
                f"Format salah: {path.name}"
            )

        waveform = audio[:, 0]

        if (
            len(waveform) == 0
            or not np.all(np.isfinite(waveform))
        ):
            raise ValueError(
                f"Audio tidak valid: {path.name}"
            )

        start_sec = cursor_samples / SAMPLE_RATE

        montage_parts.append(waveform)
        cursor_samples += len(waveform)

        end_sec = cursor_samples / SAMPLE_RATE

        if index < len(data) - 1:
            montage_parts.append(gap)
            cursor_samples += len(gap)

        rms = float(
            np.sqrt(
                np.mean(
                    waveform.astype(np.float64) ** 2
                )
            )
        )

        records.append({
            "number": index + 1,
            "filename": path.name,
            "source_group": row["source_group"],
            "montage_start_sec": round(start_sec, 3),
            "montage_end_sec": round(end_sec, 3),
            "rms": round(rms, 6),
            "review_status": "pending",
            "notes": "",
        })

        print(
            f"{index + 1:02d}. "
            f"{path.name} "
            f"[{start_sec:.1f}-{end_sec:.1f}s]"
        )

    montage = np.concatenate(montage_parts)

    montage_path = (
        output_dir / "alarm_montage.wav"
    )

    sf.write(
        montage_path,
        montage,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    review_path = (
        output_dir / "alarm_review_index.csv"
    )

    pd.DataFrame(records).to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("FINAL REVIEW AUDIO BERHASIL DIBUAT")
    print("=" * 60)
    print("Segmen diperiksa nanti:", len(records))
    print("Durasi montage:", round(
        len(montage) / SAMPLE_RATE, 2
    ), "detik")
    print("Audio:", montage_path)
    print("Indeks:", review_path)
    print("\nDataset asli tidak diubah.")
    print("Belum ada augmentation atau training.")


if __name__ == "__main__":
    main()
