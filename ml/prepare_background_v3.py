
from pathlib import Path
from math import gcd
import csv

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    ROOT / "dataset/robustness_v3/raw_background"
)

OUTPUT_DIR = (
    ROOT / "dataset/robustness_v3/prepared_background"
)

REPORT_DIR = (
    ROOT / "results/robustness_v3/background_preparation"
)

TARGET_SR = 16000


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.wav"))

    if not files:
        print("ERROR: Tidak ada file WAV ditemukan.")
        print("Folder:", INPUT_DIR)
        return

    print("=" * 60)
    print("ROBUSTNESS V3 - BACKGROUND PREPARATION")
    print("=" * 60)

    rows = []

    for path in files:
        audio, original_sr = sf.read(
            path,
            dtype="float32",
            always_2d=True
        )

        original_channels = audio.shape[1]

        # Stereo/multichannel menjadi mono
        mono = audio.mean(axis=1)

        # Resampling ke 16 kHz jika diperlukan
        if original_sr != TARGET_SR:
            factor = gcd(original_sr, TARGET_SR)

            mono = resample_poly(
                mono,
                TARGET_SR // factor,
                original_sr // factor
            )

        mono = mono.astype(np.float32)

        if len(mono) == 0:
            raise ValueError(f"Audio kosong: {path.name}")

        if not np.all(np.isfinite(mono)):
            raise ValueError(
                f"Audio memiliki nilai tidak valid: {path.name}"
            )

        duration = len(mono) / TARGET_SR
        rms = float(np.sqrt(np.mean(mono ** 2)))
        peak = float(np.max(np.abs(mono)))

        if peak > 1.0:
            raise ValueError(
                f"Amplitudo melebihi rentang PCM16: {path.name}"
            )

        output_path = OUTPUT_DIR / path.name

        sf.write(
            output_path,
            mono,
            TARGET_SR,
            subtype="PCM_16"
        )

        status = "OK"

        if peak >= 0.99:
            status = "REVIEW_POSSIBLE_CLIPPING"
        elif rms < 0.008:
            status = "REVIEW_LOW_VOLUME"

        rows.append({
            "filename": path.name,
            "original_sample_rate": original_sr,
            "original_channels": original_channels,
            "duration_sec": round(duration, 3),
            "prepared_sample_rate": TARGET_SR,
            "prepared_channels": 1,
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "target_class": "background",
            "usage": "train_only",
            "status": status,
            "prepared_file": str(output_path)
        })

        print(f"\nFile: {path.name}")
        print(f"  Sample rate asal : {original_sr} Hz")
        print(f"  Channel asal     : {original_channels}")
        print(f"  Durasi           : {duration:.2f} detik")
        print(f"  RMS              : {rms:.4f}")
        print(f"  Peak             : {peak:.4f}")
        print(f"  Status           : {status}")

    report_path = (
        REPORT_DIR / "background_preparation_report.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("PREPARATION SELESAI")
    print("=" * 60)
    print("Total file :", len(rows))
    print("Output     :", OUTPUT_DIR)
    print("Report CSV :", report_path)
    print("\nAudio asli tidak diubah.")
    print("Belum ada training atau augmentation.")


if __name__ == "__main__":
    main()
