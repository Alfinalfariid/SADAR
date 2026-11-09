
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent

REPORT_FILE = (
    ROOT
    / "results/robustness_v3/alarm_preparation"
    / "run_20261102_061558_144471"
    / "alarm_preparation_report.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "dataset/robustness_v3/alarm_chunks"
)

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 3
SEGMENT_SIZE = SAMPLE_RATE * SEGMENT_SECONDS
MIN_RMS = 0.008


# Metadata diverifikasi dari Wikimedia Commons.
SOURCES = {
    "Activated_fire_alarm_(sound).ogg": {
        "creator": "Bidgee",
        "license": "CC BY-SA 3.0 AU",
        "source_url": (
            "https://commons.wikimedia.org/wiki/"
            "File:Activated_fire_alarm_(sound).ogg"
        ),
    },
    "Smoke_alarm.ogg": {
        "creator": "cori",
        "license": "Public Domain",
        "source_url": (
            "https://commons.wikimedia.org/wiki/"
            "File:Smoke_alarm.ogg"
        ),
    },
    "WWS_Firealarm.ogg": {
        "creator": (
            "Work With Sounds / Torsten Nilsson"
        ),
        "license": "CC BY 4.0",
        "source_url": (
            "https://commons.wikimedia.org/wiki/"
            "File:WWS_Firealarm.ogg"
        ),
    },
}


def calculate_rms(audio):
    values = audio.astype(np.float64)
    return float(np.sqrt(np.mean(values ** 2)))


def main():
    print("=" * 60)
    print("ROBUSTNESS V3 - ALARM SEGMENTATION")
    print("=" * 60)

    if not REPORT_FILE.is_file():
        raise FileNotFoundError(REPORT_FILE)

    data = pd.read_csv(
        REPORT_FILE
    ).fillna("")

    required = {
        "path",
        "original_file",
        "source_group",
        "original_sha256",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Kolom manifest tidak lengkap: {missing}"
        )

    if data.empty:
        raise ValueError("Report alarm kosong.")

    unknown = set(
        data["original_file"]
    ) - set(SOURCES)

    if unknown:
        raise ValueError(
            f"Sumber belum diverifikasi: {unknown}"
        )

    if data["original_sha256"].duplicated().any():
        raise ValueError(
            "Ada rekaman asli duplikat dalam manifest."
        )

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    output_dir = OUTPUT_ROOT / run_id
    clips_dir = output_dir / "clips"

    clips_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    records = []
    source_records = []
    rejected_records = []

    for source_index, row in data.iterrows():
        source_name = str(row["original_file"])
        metadata = SOURCES[source_name]

        audio_path = Path(str(row["path"]))

        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        audio, sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True
        )

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Sample rate salah: {source_name}"
            )

        if audio.shape[1] != 1:
            raise ValueError(
                f"Audio belum mono: {source_name}"
            )

        waveform = audio[:, 0]

        if not np.all(np.isfinite(waveform)):
            raise ValueError(
                f"Audio tidak valid: {source_name}"
            )

        source_group = str(row["source_group"])

        total_segments = (
            len(waveform) // SEGMENT_SIZE
        )

        accepted = 0
        rejected = 0

        print(f"\nSumber: {source_name}")

        for segment_index in range(total_segments):
            start = segment_index * SEGMENT_SIZE
            end = start + SEGMENT_SIZE

            segment = waveform[start:end]

            rms = calculate_rms(segment)

            if rms < MIN_RMS:
                rejected += 1

                rejected_records.append({
                    "source_group": source_group,
                    "original_file": source_name,
                    "start_sec": start / SAMPLE_RATE,
                    "rms": rms,
                    "reason": "LOW_RMS",
                })

                continue

            accepted += 1

            clip_name = (
                f"alarm_{source_index + 1:02d}"
                f"_clip_{segment_index + 1:03d}.wav"
            )

            clip_path = clips_dir / clip_name

            sf.write(
                clip_path,
                segment,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            records.append({
                "path": str(clip_path),
                "target_class": "alarm",
                "split": "train",
                "source_group": source_group,
                "original_file": source_name,
                "original_sha256": str(
                    row["original_sha256"]
                ),
                "source_url": metadata["source_url"],
                "creator": metadata["creator"],
                "license": metadata["license"],
                "start_sec": round(
                    start / SAMPLE_RATE, 3
                ),
                "end_sec": round(
                    end / SAMPLE_RATE, 3
                ),
                "duration_sec": SEGMENT_SECONDS,
                "rms": round(rms, 6),
                "manual_source_review": (
                    "approved_by_user"
                ),
                "segment_review": "pending",
            })

        source_records.append({
            "original_file": source_name,
            "source_group": source_group,
            "source_url": metadata["source_url"],
            "creator": metadata["creator"],
            "license": metadata["license"],
            "duration_sec": round(
                len(waveform) / SAMPLE_RATE, 3
            ),
            "accepted_segments": accepted,
            "rejected_segments": rejected,
            "split": "train",
            "manual_source_review": (
                "approved_by_user"
            ),
        })

        print("Segmen tersedia :", total_segments)
        print("Segmen diterima :", accepted)
        print("Segmen ditolak  :", rejected)

    if not records:
        raise RuntimeError(
            "Tidak ada segmen alarm yang diterima."
        )

    manifest_path = (
        output_dir / "alarm_train_manifest.csv"
    )

    sources_path = (
        output_dir / "alarm_sources.csv"
    )

    pd.DataFrame(records).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(source_records).to_csv(
        sources_path,
        index=False,
        encoding="utf-8-sig"
    )

    if rejected_records:
        pd.DataFrame(rejected_records).to_csv(
            output_dir / "rejected_segments.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("\n" + "=" * 60)
    print("ALARM SEGMENTATION SELESAI")
    print("=" * 60)

    print("Sumber asli:", len(source_records))
    print("Segmen diterima:", len(records))
    print("Segmen ditolak:", len(rejected_records))
    print("Folder output:", clips_dir)
    print("Manifest:", manifest_path)
    print("Metadata sumber:", sources_path)

    print("\nCATATAN:")
    print("- TRAIN ONLY")
    print("- Source group dipertahankan")
    print("- Lisensi dan sumber tercatat")
    print("- Review per segmen masih diperlukan")
    print("- Audio asli tidak diubah")
    print("- Model lama tidak diubah")
    print("- Belum dilakukan training")


if __name__ == "__main__":
    main()
