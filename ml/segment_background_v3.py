
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    ROOT / "dataset/robustness_v3/prepared_background"
)

OUTPUT_ROOT = (
    ROOT / "dataset/robustness_v3/background_chunks"
)

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3
SEGMENT_SIZE = SAMPLE_RATE * SEGMENT_SECONDS
MIN_RMS = 0.008

FILES = {
    "humming_test_01.wav": "humming",
    "speech_test_01.wav": "speech",
}


def main():
    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    output_dir = OUTPUT_ROOT / run_id
    clips_dir = output_dir / "clips"

    clips_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    rows = []
    rejected = 0

    print("=" * 60)
    print("ROBUSTNESS V3 - SEGMENT BACKGROUND")
    print("=" * 60)

    for filename, sound_type in FILES.items():
        input_path = INPUT_DIR / filename

        if not input_path.is_file():
            raise FileNotFoundError(
                f"File tidak ditemukan: {input_path}"
            )

        audio, sr = sf.read(
            input_path,
            dtype="float32",
            always_2d=True
        )

        if sr != SAMPLE_RATE or audio.shape[1] != 1:
            raise ValueError(
                f"Format belum mono 16 kHz: {filename}"
            )

        audio = audio[:, 0]

        if not np.all(np.isfinite(audio)):
            raise ValueError(
                f"Audio tidak valid: {filename}"
            )

        total_segments = len(audio) // SEGMENT_SIZE
        accepted = 0

        for i in range(total_segments):
            start = i * SEGMENT_SIZE
            end = start + SEGMENT_SIZE

            segment = audio[start:end]

            rms = float(
                np.sqrt(np.mean(segment ** 2))
            )

            if rms < MIN_RMS:
                rejected += 1
                continue

            accepted += 1

            clip_name = (
                f"bg_{sound_type}_{accepted:03d}.wav"
            )

            clip_path = clips_dir / clip_name

            sf.write(
                clip_path,
                segment,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            rows.append({
                "path": str(clip_path),
                "target_class": "background",
                "split": "train",
                "source_group": "v3_voice_session_01",
                "original_file": filename,
                "sound_type": sound_type,
                "start_sec": round(
                    start / SAMPLE_RATE, 3
                ),
                "duration_sec": SEGMENT_SECONDS,
                "rms": round(rms, 6)
            })

        print(f"\nFile: {filename}")
        print(f"  Segmen tersedia : {total_segments}")
        print(f"  Segmen diterima : {accepted}")

    if not rows:
        raise RuntimeError(
            "Tidak ada segmen yang memenuhi syarat."
        )

    manifest_path = (
        output_dir / "background_train_manifest.csv"
    )

    pd.DataFrame(rows).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("SEGMENTASI SELESAI")
    print("=" * 60)
    print("Total segmen diterima:", len(rows))
    print("Segmen RMS rendah:", rejected)
    print("Folder output:", clips_dir)
    print("Manifest:", manifest_path)
    print("\nSplit: TRAIN ONLY")
    print("Source group sama untuk kedua rekaman.")
    print("Dataset asli dan model tidak diubah.")
    print("Belum ada augmentation atau training.")


if __name__ == "__main__":
    main()
