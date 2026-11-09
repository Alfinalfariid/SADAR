
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent

SOURCE_DIR = (
    ROOT
    / "dataset"
    / "robustness_v3"
    / "background_chunks"
    / "run_20261027_120154_693387"
)

OUTPUT_ROOT = (
    ROOT
    / "dataset"
    / "robustness_v3"
    / "augmented_background"
)

SAMPLE_RATE = 16000
SEED = 42


def main():
    manifest_path = (
        SOURCE_DIR / "background_train_manifest.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    required_columns = {
        "path",
        "target_class",
        "split",
        "source_group",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom manifest tidak lengkap: {missing}"
        )

    if df.empty:
        raise ValueError("Manifest kosong.")

    if not (df["split"] == "train").all():
        raise ValueError(
            "Augmentation hanya boleh untuk data train."
        )

    if not (df["target_class"] == "background").all():
        raise ValueError(
            "Ada label selain background."
        )

    rng = np.random.default_rng(SEED)

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    output_dir = OUTPUT_ROOT / run_id
    clips_dir = output_dir / "clips"

    clips_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    results = []

    print("=" * 60)
    print("ROBUSTNESS V3 - BACKGROUND AUGMENTATION")
    print("=" * 60)

    for index, row in df.iterrows():
        audio_path = Path(str(row["path"]))

        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        audio, sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )

        if sr != SAMPLE_RATE or audio.shape[1] != 1:
            raise ValueError(
                f"Audio harus mono 16 kHz: {audio_path.name}"
            )

        waveform = audio[:, 0]

        if (
            len(waveform) == 0
            or not np.all(np.isfinite(waveform))
        ):
            raise ValueError(
                f"Audio tidak valid: {audio_path.name}"
            )

        # Augmentation 1: perubahan volume ringan
        gain = float(rng.uniform(0.90, 1.10))
        modified = waveform * gain

        # Augmentation 2: tambahan noise ringan
        snr_db = float(rng.uniform(28.0, 35.0))

        signal_rms = float(
            np.sqrt(np.mean(modified ** 2))
        )

        noise_rms = signal_rms / (
            10 ** (snr_db / 20.0)
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_rms,
            size=len(modified),
        )

        augmented = modified + noise

        # Hindari clipping
        peak = float(np.max(np.abs(augmented)))

        if peak > 0.99:
            augmented *= 0.99 / peak

        augmented = augmented.astype(np.float32)

        name = f"aug_bg_{index + 1:03d}.wav"
        output_path = clips_dir / name

        sf.write(
            output_path,
            augmented,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        augmented_rms = float(
            np.sqrt(np.mean(augmented ** 2))
        )

        results.append({
            "path": str(output_path),
            "target_class": "background",
            "split": "train",
            "source_group": row["source_group"],
            "parent_segment": str(audio_path),
            "augmentation": "gain_plus_light_noise",
            "gain": round(gain, 4),
            "snr_db": round(snr_db, 2),
            "rms": round(augmented_rms, 6),
            "sample_rate": SAMPLE_RATE,
            "seed": SEED,
        })

    report_path = (
        output_dir / "augmented_train_manifest.csv"
    )

    pd.DataFrame(results).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Segmen asli       : {len(df)}")
    print(f"Segmen augmentation: {len(results)}")
    print(f"Total tersedia    : {len(df) + len(results)}")
    print(f"Sample rate       : {SAMPLE_RATE} Hz")
    print(f"Random seed       : {SEED}")
    print(f"Folder output     : {clips_dir}")
    print(f"Manifest          : {report_path}")

    print("\nSTATUS:")
    print("- Split TRAIN ONLY")
    print("- Source group dipertahankan")
    print("- File asli tidak diubah")
    print("- Model sebelumnya tidak diubah")
    print("- Training belum dilakukan")


if __name__ == "__main__":
    main()
