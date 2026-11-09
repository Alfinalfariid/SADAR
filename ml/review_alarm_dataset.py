
from pathlib import Path
from datetime import datetime
import math

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent

MANIFEST = ROOT / "dataset/dataset_splits.csv"

OUTPUT = ROOT / (
    "results/robustness_v2/"
    "alarm_dataset_audit/manual_review"
) / datetime.now().strftime(
    "run_%Y%m%d_%H%M%S"
)

TARGET_SR = 16000
GAP_SECONDS = 1.0


def load_audio(path):

    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    audio = audio.mean(axis=1)

    if sr != TARGET_SR:

        divisor = math.gcd(
            sr,
            TARGET_SR
        )

        audio = resample_poly(
            audio,
            TARGET_SR // divisor,
            sr // divisor
        )

    return audio.astype(np.float32)


def main():

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    data = pd.read_csv(MANIFEST)

    required = {
        "path",
        "target_class",
        "split",
        "source_group"
    }

    if not required.issubset(data.columns):
        raise ValueError(
            f"Kolom kurang: "
            f"{required - set(data.columns)}"
        )

    test_alarm = data[
        (
            data["target_class"]
            .astype(str)
            .str.lower()
            .eq("alarm")
        )
        &
        (
            data["split"]
            .astype(str)
            .str.lower()
            .eq("test")
        )
    ].copy()

    test_alarm = test_alarm.sort_values(
        ["source_group", "path"]
    ).reset_index(drop=True)

    if test_alarm.empty:
        raise ValueError(
            "Tidak ada audio alarm pada split test."
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=False
    )

    clips_dir = OUTPUT / "individual_clips"

    clips_dir.mkdir()

    silence = np.zeros(
        int(GAP_SECONDS * TARGET_SR),
        dtype=np.float32
    )

    montage = [silence.copy()]

    current_sample = len(silence)

    rows = []

    print("=" * 65)
    print("MANUAL REVIEW DATASET ALARM - TEST")
    print("=" * 65)

    for index, row in test_alarm.iterrows():

        source_path = Path(
            str(row["path"])
        )

        if not source_path.is_absolute():
            source_path = ROOT / source_path

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        audio = load_audio(source_path)

        number = index + 1

        review_path = (
            clips_dir /
            f"alarm_test_{number:02d}.wav"
        )

        sf.write(
            str(review_path),
            audio,
            TARGET_SR,
            subtype="PCM_16"
        )

        start_sec = current_sample / TARGET_SR

        end_sec = (
            current_sample + len(audio)
        ) / TARGET_SR

        montage.append(audio)
        montage.append(silence.copy())

        current_sample += (
            len(audio) + len(silence)
        )

        rows.append({
            "number": number,
            "source_group": row["source_group"],
            "original_path": str(source_path),
            "review_clip": str(review_path),
            "duration_sec": round(
                len(audio) / TARGET_SR,
                4
            ),
            "montage_start_sec": round(
                start_sec, 3
            ),
            "montage_end_sec": round(
                end_sec, 3
            ),
            "heard_sound_type": "",
            "label_correct": "",
            "notes": ""
        })

        print(
            f"{number:02d}. "
            f"Durasi {len(audio)/TARGET_SR:.2f}s "
            f"| {source_path.name}"
        )

    combined = np.concatenate(montage)

    combined = np.clip(
        combined,
        -1.0,
        1.0
    )

    montage_path = (
        OUTPUT / "alarm_test_montage.wav"
    )

    sf.write(
        str(montage_path),
        combined,
        TARGET_SR,
        subtype="PCM_16"
    )

    review_csv = (
        OUTPUT / "alarm_manual_review.csv"
    )

    pd.DataFrame(rows).to_csv(
        review_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 65)
    print("SELESAI")
    print("=" * 65)

    print("Jumlah audio :", len(test_alarm))
    print(
        "Durasi montage:",
        round(len(combined)/TARGET_SR, 2),
        "detik"
    )

    print("\nAudio gabungan:")
    print(montage_path)

    print("\nCSV penilaian:")
    print(review_csv)

    print(
        "\nSemua file dataset asli "
        "dan model tetap aman."
    )


if __name__ == "__main__":
    main()
