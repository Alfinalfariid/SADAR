
from pathlib import Path
from datetime import datetime
import json
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parent

SOURCE_DIR = (
    ROOT
    / "dataset/robustness_v3/alarm_chunks"
    / "run_20261102_062325_130939"
)

SOURCE_MANIFEST = (
    SOURCE_DIR / "alarm_train_manifest.csv"
)

OUTPUT_ROOT = (
    ROOT / "dataset/robustness_v3/alarm_curated"
)

# Segmen yang telah diperiksa secara manual.
# Jangan gunakan sebagai contoh positif alarm.
EXCLUDED_FILES = {
    "alarm_02_clip_001.wav": (
        "Bagian awal Smoke Alarm: "
        "tidak terdengar bunyi alarm"
    ),
    "alarm_02_clip_011.wav": (
        "Bagian akhir Smoke Alarm: "
        "tidak terdengar bunyi alarm"
    ),
}


def resolve_path(value):
    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    return path


def main():
    print("=" * 60)
    print("ROBUSTNESS V3 - MANUAL ALARM CURATION")
    print("=" * 60)

    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)

    data = pd.read_csv(SOURCE_MANIFEST)

    required_columns = {
        "path",
        "target_class",
        "split",
        "source_group",
        "original_file",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Kolom manifest tidak lengkap: {missing}"
        )

    if len(data) != 31:
        raise ValueError(
            "Manifest sumber diharapkan berisi "
            f"31 segmen, ditemukan {len(data)}."
        )

    if not (data["target_class"] == "alarm").all():
        raise ValueError("Ada label selain alarm.")

    if not (data["split"] == "train").all():
        raise ValueError("Ada data selain train.")

    filenames = [
        Path(str(value)).name
        for value in data["path"]
    ]

    if len(filenames) != len(set(filenames)):
        raise ValueError(
            "Ada nama segmen duplikat."
        )

    not_found = set(EXCLUDED_FILES) - set(filenames)

    if not_found:
        raise ValueError(
            f"Segmen pengecualian tidak ditemukan: "
            f"{sorted(not_found)}"
        )

    # Validasi semua file sebelum membuat hasil.
    for value in data["path"]:
        path = resolve_path(value)

        if not path.is_file():
            raise FileNotFoundError(path)

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    output_dir = OUTPUT_ROOT / run_id
    clips_dir = output_dir / "clips"

    clips_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    retained = []
    excluded = []

    for _, row in data.iterrows():
        record = row.to_dict()

        original_path = resolve_path(
            record["path"]
        )

        filename = original_path.name

        if filename in EXCLUDED_FILES:
            record["curation_status"] = (
                "excluded_manual_review"
            )
            record["exclusion_reason"] = (
                EXCLUDED_FILES[filename]
            )

            excluded.append(record)

            print(f"[EXCLUDE] {filename}")
            continue

        destination = clips_dir / filename

        shutil.copy2(
            original_path,
            destination
        )

        # Path baru menunjuk folder hasil kurasi.
        record["path"] = str(destination)

        # Pertahankan lokasi segmen sebelumnya
        # untuk audit dan reproduksibilitas.
        record["source_segment_path"] = str(
            original_path
        )

        record["curation_status"] = (
            "retained_candidate"
        )

        record["exclusion_reason"] = ""

        retained.append(record)

        print(f"[KEEP]    {filename}")

    if len(retained) != 29 or len(excluded) != 2:
        raise RuntimeError(
            "Jumlah hasil kurasi tidak sesuai."
        )

    curated_manifest = (
        output_dir
        / "alarm_train_manifest_curated.csv"
    )

    excluded_report = (
        output_dir
        / "excluded_segments_manual.csv"
    )

    pd.DataFrame(retained).to_csv(
        curated_manifest,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(excluded).to_csv(
        excluded_report,
        index=False,
        encoding="utf-8-sig"
    )

    source_metadata = (
        SOURCE_DIR / "alarm_sources.csv"
    )

    if source_metadata.is_file():
        shutil.copy2(
            source_metadata,
            output_dir / "alarm_sources.csv"
        )

    summary = {
        "stage": "robustness_v3_alarm_curation",
        "source_manifest": str(SOURCE_MANIFEST),
        "total_before": len(data),
        "retained_candidates": len(retained),
        "excluded_manual_review": len(excluded),
        "excluded_files": EXCLUDED_FILES,
        "split": "train_only",
        "original_files_modified": False,
        "model_modified": False,
        "remaining_segment_review": (
            "pending_for_unreviewed_segments"
        ),
    }

    summary_path = (
        output_dir / "curation_summary.json"
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
    print("ALARM CURATION SELESAI")
    print("=" * 60)

    print("Segmen sebelumnya :", len(data))
    print("Segmen dikeluarkan:", len(excluded))
    print("Segmen tersisa    :", len(retained))
    print("Folder hasil      :", clips_dir)
    print("Manifest baru     :", curated_manifest)
    print("Laporan exclude   :", excluded_report)
    print("Ringkasan         :", summary_path)

    print("\nCATATAN:")
    print("- Kedua segmen senyap tidak ikut dataset baru")
    print("- Dataset asli tidak diubah")
    print("- Source group tetap dipertahankan")
    print("- Segmen lain tetap kandidat training")
    print("- Belum dilakukan augmentation atau training")


if __name__ == "__main__":
    main()
