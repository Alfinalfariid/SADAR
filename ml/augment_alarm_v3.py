
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import soundfile as sf


# ============================================
# CONFIGURATION
# ============================================

ROOT = Path(__file__).resolve().parent

SOURCE_DIR = (
    ROOT
    / "dataset/robustness_v3/alarm_curated"
    / "run_20261102_063122_274252"
)

SOURCE_MANIFEST = (
    SOURCE_DIR / "alarm_train_manifest_curated.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "dataset/robustness_v3/augmented_alarm"
)

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3
SEGMENT_SIZE = SAMPLE_RATE * SEGMENT_SECONDS

SEED = 42

GAIN_MIN = 0.90
GAIN_MAX = 1.10

SNR_MIN_DB = 30.0
SNR_MAX_DB = 38.0

EXCLUDED_FILES = {
    "alarm_02_clip_001.wav",
    "alarm_02_clip_011.wav",
}


# ============================================
# HELPERS
# ============================================

def calculate_rms(audio):
    values = np.asarray(
        audio,
        dtype=np.float64
    )

    return float(
        np.sqrt(np.mean(values ** 2))
    )


def resolve_path(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    return path


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("ROBUSTNESS V3 - ALARM AUGMENTATION")
    print("=" * 60)

    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            SOURCE_MANIFEST
        )

    data = pd.read_csv(
        SOURCE_MANIFEST
    ).fillna("")

    required_columns = {
        "path",
        "target_class",
        "split",
        "source_group",
        "original_file",
        "source_url",
        "creator",
        "license",
    }

    missing = (
        required_columns - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Kolom manifest tidak lengkap: {missing}"
        )

    if len(data) != 29:
        raise ValueError(
            "Manifest harus berisi 29 segmen. "
            f"Ditemukan: {len(data)}"
        )

    if not (
        data["target_class"] == "alarm"
    ).all():
        raise ValueError(
            "Ada data selain kelas alarm."
        )

    if not (
        data["split"] == "train"
    ).all():
        raise ValueError(
            "Augmentation hanya untuk train."
        )

    if data["source_group"].eq("").any():
        raise ValueError(
            "Ada source_group kosong."
        )

    for column in (
        "source_url",
        "creator",
        "license",
    ):
        if data[column].eq("").any():
            raise ValueError(
                f"Metadata {column} belum lengkap."
            )

    data = data.copy()

    data["filename"] = data["path"].map(
        lambda value: Path(str(value)).name
    )

    if data["filename"].duplicated().any():
        raise ValueError(
            "Ada nama file duplikat."
        )

    if set(data["filename"]) & EXCLUDED_FILES:
        raise ValueError(
            "Segmen senyap masih ada "
            "di manifest training!"
        )

    data = data.sort_values(
        "filename"
    ).reset_index(drop=True)

    # Validasi audio sebelum membuat output
    for _, row in data.iterrows():
        path = resolve_path(row["path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        info = sf.info(path)

        if (
            info.samplerate != SAMPLE_RATE
            or info.channels != 1
            or info.frames != SEGMENT_SIZE
        ):
            raise ValueError(
                f"Format tidak sesuai: {path}"
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

    original_records = []
    augmented_records = []
    augmentation_report = []

    # ========================================
    # AUGMENTATION
    # ========================================

    for index, row in data.iterrows():
        input_path = resolve_path(row["path"])

        audio, sr = sf.read(
            input_path,
            dtype="float32",
            always_2d=True
        )

        if (
            sr != SAMPLE_RATE
            or audio.shape[1] != 1
        ):
            raise ValueError(
                f"Audio tidak sesuai: {input_path}"
            )

        waveform = audio[:, 0]

        if (
            len(waveform) != SEGMENT_SIZE
            or not np.all(np.isfinite(waveform))
        ):
            raise ValueError(
                f"Audio tidak valid: {input_path}"
            )

        # ------------------------------------
        # ORIGINAL RECORD
        # ------------------------------------

        original_record = row.drop(
            labels=["filename"]
        ).to_dict()

        original_record["path"] = str(
            input_path
        )

        original_record["augmentation"] = "none"

        original_record["parent_segment"] = ""

        original_record["manual_review_status"] = (
            "approved_by_user"
        )

        original_records.append(
            original_record
        )

        # ------------------------------------
        # AUGMENTATION 1: LIGHT GAIN
        # ------------------------------------

        gain = float(
            rng.uniform(
                GAIN_MIN,
                GAIN_MAX
            )
        )

        modified = (
            waveform.astype(np.float64)
            * gain
        )

        # ------------------------------------
        # AUGMENTATION 2: LIGHT NOISE
        # ------------------------------------

        snr_db = float(
            rng.uniform(
                SNR_MIN_DB,
                SNR_MAX_DB
            )
        )

        signal_rms = calculate_rms(
            modified
        )

        target_noise_rms = (
            signal_rms /
            (10 ** (snr_db / 20.0))
        )

        noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=len(modified)
        )

        noise_rms = calculate_rms(
            noise
        )

        if noise_rms > 0:
            noise *= (
                target_noise_rms
                / noise_rms
            )

        augmented = modified + noise

        # ------------------------------------
        # CLIPPING PROTECTION
        # ------------------------------------

        peak = float(
            np.max(np.abs(augmented))
        )

        peak_scaled = False

        if peak > 0.99:
            augmented *= (
                0.99 / peak
            )

            peak_scaled = True

        augmented = augmented.astype(
            np.float32
        )

        # ------------------------------------
        # SAVE AUDIO
        # ------------------------------------

        output_name = (
            f"aug_alarm_{index + 1:03d}.wav"
        )

        output_path = (
            clips_dir / output_name
        )

        sf.write(
            output_path,
            augmented,
            SAMPLE_RATE,
            subtype="PCM_16"
        )

        # ------------------------------------
        # AUGMENTED RECORD
        # ------------------------------------

        augmented_record = (
            original_record.copy()
        )

        augmented_record["path"] = str(
            output_path
        )

        augmented_record["augmentation"] = (
            "gain_plus_light_noise"
        )

        augmented_record["parent_segment"] = (
            str(input_path)
        )

        augmented_record["gain"] = round(
            gain, 6
        )

        augmented_record["snr_db"] = round(
            snr_db, 4
        )

        augmented_record["seed"] = SEED

        augmented_records.append(
            augmented_record
        )

        augmentation_report.append({
            "filename": output_name,
            "parent_segment": input_path.name,
            "source_group": row["source_group"],
            "gain": round(gain, 6),
            "snr_db": round(snr_db, 4),
            "rms_before": round(
                calculate_rms(waveform), 6
            ),
            "rms_after": round(
                calculate_rms(augmented), 6
            ),
            "peak_after": round(
                float(np.max(np.abs(augmented))), 6
            ),
            "peak_scaled": peak_scaled,
            "sample_rate": SAMPLE_RATE,
            "duration_sec": SEGMENT_SECONDS,
        })

        print(
            f"[{index + 1:02d}/29] "
            f"{input_path.name} "
            f"-> {output_name}"
        )

    # ========================================
    # SAVE MANIFESTS
    # ========================================

    augmented_manifest = (
        output_dir
        / "augmented_alarm_manifest.csv"
    )

    combined_manifest = (
        output_dir
        / "alarm_train_manifest_v3.csv"
    )

    report_path = (
        output_dir
        / "augmentation_report.csv"
    )

    pd.DataFrame(
        augmented_records
    ).to_csv(
        augmented_manifest,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        original_records
        + augmented_records
    ).to_csv(
        combined_manifest,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        augmentation_report
    ).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "stage": "robustness_v3_alarm_augmentation",
        "source_manifest": str(SOURCE_MANIFEST),
        "original_segments": len(original_records),
        "augmented_segments": len(augmented_records),
        "total_training_segments": (
            len(original_records)
            + len(augmented_records)
        ),
        "unique_source_groups": int(
            data["source_group"].nunique()
        ),
        "sample_rate": SAMPLE_RATE,
        "segment_seconds": SEGMENT_SECONDS,
        "seed": SEED,
        "gain_range": [
            GAIN_MIN,
            GAIN_MAX
        ],
        "snr_db_range": [
            SNR_MIN_DB,
            SNR_MAX_DB
        ],
        "split": "train_only",
        "excluded_files": sorted(
            EXCLUDED_FILES
        ),
        "original_files_modified": False,
        "previous_models_modified": False,
        "training_performed": False,
    }

    summary_path = (
        output_dir / "augmentation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("ALARM AUGMENTATION SELESAI")
    print("=" * 60)

    print(
        "Segmen asli:",
        len(original_records)
    )

    print(
        "Segmen augmentation:",
        len(augmented_records)
    )

    print(
        "Total training:",
        len(original_records)
        + len(augmented_records)
    )

    print(
        "Sumber independen:",
        data["source_group"].nunique()
    )

    print("Output:", clips_dir)
    print("Manifest gabungan:", combined_manifest)
    print("Laporan:", report_path)
    print("Ringkasan:", summary_path)

    print("\nSTATUS:")
    print("- TRAIN ONLY")
    print("- Source group dipertahankan")
    print("- Segmen senyap tidak digunakan")
    print("- Audio asli tidak diubah")
    print("- Model Baseline, V1, V2 aman")
    print("- Belum dilakukan training")


if __name__ == "__main__":
    main()
