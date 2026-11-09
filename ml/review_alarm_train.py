
from pathlib import Path
from datetime import datetime
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "dataset/dataset_splits.csv"

OUTPUT = ROOT / (
    "results/robustness_v3/alarm_train_review"
) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")

SR = 16000
SAMPLE_COUNT = 20
GAP_SECONDS = 1.0


def read_audio(path):
    audio, original_sr = sf.read(
        path,
        dtype="float32",
        always_2d=True
    )

    audio = audio.mean(axis=1)

    if original_sr != SR:
        common = gcd(original_sr, SR)

        audio = resample_poly(
            audio,
            SR // common,
            original_sr // common
        )

    return audio.astype(np.float32)


def main():
    data = pd.read_csv(MANIFEST)

    required = {
        "path", "target_class",
        "split", "source_group"
    }

    if not required.issubset(data.columns):
        raise ValueError("Kolom dataset tidak lengkap.")

    alarm_train = data[
        (data["target_class"] == "alarm")
        & (data["split"] == "train")
    ].copy()

    # Ambil satu contoh tiap source group,
    # agar sampel lebih beragam.
    alarm_train = alarm_train.drop_duplicates(
        subset=["source_group"]
    )

    sample = alarm_train.sample(
        n=min(SAMPLE_COUNT, len(alarm_train)),
        random_state=42
    ).reset_index(drop=True)

    OUTPUT.mkdir(parents=True, exist_ok=False)

    clips_dir = OUTPUT / "clips"
    clips_dir.mkdir()

    gap = np.zeros(
        int(GAP_SECONDS * SR),
        dtype=np.float32
    )

    combined = [gap]
    current_sample = len(gap)

    rows = []

    for i, row in sample.iterrows():
        path = Path(str(row["path"]))

        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        audio = read_audio(path)

        number = i + 1

        output_clip = (
            clips_dir / f"alarm_train_{number:02d}.wav"
        )

        sf.write(
            output_clip,
            audio,
            SR,
            subtype="PCM_16"
        )

        start_sec = current_sample / SR
        end_sec = (
            current_sample + len(audio)
        ) / SR

        combined.extend([audio, gap])
        current_sample += len(audio) + len(gap)

        rows.append({
            "number": number,
            "source_group": row["source_group"],
            "original_file": str(path),
            "clip_file": str(output_clip),
            "duration_sec": round(
                len(audio) / SR, 3
            ),
            "montage_start_sec": round(
                start_sec, 3
            ),
            "montage_end_sec": round(
                end_sec, 3
            ),
            "heard_sound_type": "",
            "suitable_for_target_alarm": "",
            "notes": ""
        })

        print(
            f"{number:02d}. "
            f"{path.name} "
            f"({len(audio) / SR:.2f} detik)"
        )

    montage = np.concatenate(combined)

    montage_file = OUTPUT / "alarm_train_montage.wav"

    sf.write(
        montage_file,
        montage,
        SR,
        subtype="PCM_16"
    )

    review_file = OUTPUT / "alarm_train_review.csv"

    pd.DataFrame(rows).to_csv(
        review_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("AUDIT AUDIO ALARM TRAIN SELESAI")
    print("=" * 60)

    print("Audio diperiksa:", len(sample))
    print("Montage:", montage_file)
    print("CSV review:", review_file)

    print(
        "\nDataset asli dan semua model tidak diubah."
    )


if __name__ == "__main__":
    main()
