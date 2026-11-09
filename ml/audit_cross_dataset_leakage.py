
from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent

MANIFEST = (
    ROOT / "dataset" / "dataset_splits.csv"
)

OUTPUT = (
    ROOT
    / "results"
    / "cross_dataset_leakage_audit"
    / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )
)

ESC_PATTERN = re.compile(
    r"^[1-5]-(\d+)-[A-Z]-\d+\.wav$",
    re.IGNORECASE,
)

URBAN_PATTERN = re.compile(
    r"^(?:urbansound8k_)?"
    r"(\d+)-\d+-\d+-\d+\.wav$",
    re.IGNORECASE,
)


def identify_source(path, original_group):
    filename = str(path).replace(
        "\\", "/"
    ).split("/")[-1]

    esc = ESC_PATTERN.match(filename)

    if esc:
        source_id = esc.group(1)

        return (
            f"FREESOUND_{source_id}",
            "ESC50",
            source_id,
        )

    urban = URBAN_PATTERN.match(filename)

    if urban:
        source_id = urban.group(1)

        return (
            f"FREESOUND_{source_id}",
            "UrbanSound8K",
            source_id,
        )

    # Untuk dataset lain, pertahankan
    # source_group aslinya.
    return (
        f"OTHER_{original_group}",
        "OTHER",
        "",
    )


def main():
    print("=" * 65)
    print("SADAR - AUDIT KEBOCORAN SUMBER DATA")
    print("=" * 65)

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    df = pd.read_csv(
        MANIFEST,
        dtype=str,
    ).fillna("")

    required = {
        "path",
        "target_class",
        "split",
        "source_group",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom manifest hilang: {missing}"
        )

    if df["source_group"].eq("").any():
        raise ValueError(
            "Ada source_group kosong."
        )

    if not set(df["split"]).issubset({
        "train", "validation", "test"
    }):
        raise ValueError(
            "Nama split tidak dikenal."
        )

    # Pemeriksaan source_group asli
    original_leakage = (
        df.groupby("source_group")["split"]
        .nunique()
    )

    original_leakage = original_leakage[
        original_leakage > 1
    ]

    # Identifikasi Freesound ID lintas dataset
    identified = [
        identify_source(
            row.path,
            row.source_group,
        )
        for row in df.itertuples(index=False)
    ]

    df["canonical_source"] = [
        item[0] for item in identified
    ]

    df["dataset_origin"] = [
        item[1] for item in identified
    ]

    df["freesound_id"] = [
        item[2] for item in identified
    ]

    OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    df.to_csv(
        OUTPUT / "all_sources_audited.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Sumber canonical yang masuk beberapa split
    split_counts = (
        df.groupby("canonical_source")["split"]
        .nunique()
    )

    leaked_keys = set(
        split_counts[
            split_counts > 1
        ].index
    )

    cross_rows = df[
        df["canonical_source"].isin(
            leaked_keys
        )
    ].copy()

    cross_rows.to_csv(
        OUTPUT / "cross_split_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summaries = []

    for source, group in df.groupby(
        "canonical_source"
    ):
        datasets = sorted(
            group["dataset_origin"].unique()
        )

        splits = sorted(
            group["split"].unique()
        )

        summary = {
            "canonical_source": source,
            "freesound_id": (
                group["freesound_id"].iloc[0]
            ),
            "datasets": ", ".join(datasets),
            "splits": ", ".join(splits),
            "file_count": len(group),
            "train_count": int(
                (group["split"] == "train").sum()
            ),
            "validation_count": int(
                (group["split"] == "validation").sum()
            ),
            "test_count": int(
                (group["split"] == "test").sum()
            ),
            "classes": ", ".join(
                sorted(
                    group["target_class"].unique()
                )
            ),
            "cross_split": len(splits) > 1,
            "cross_dataset": (
                "ESC50" in datasets
                and "UrbanSound8K" in datasets
            ),
        }

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUTPUT / "source_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    leaked = (
        summary_df[
            summary_df["cross_split"]
        ]
        .sort_values(
            [
                "file_count",
                "canonical_source",
            ],
            ascending=[False, True],
        )
    )

    leaked.to_csv(
        OUTPUT / "cross_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shared = summary_df[
        summary_df["cross_dataset"]
    ].copy()

    shared.to_csv(
        OUTPUT / "shared_esc50_urbansound.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_affected = cross_rows[
        cross_rows["split"] == "test"
    ]

    validation_affected = cross_rows[
        cross_rows["split"] == "validation"
    ]

    train_affected = cross_rows[
        cross_rows["split"] == "train"
    ]

    audit = {
        "total_manifest_rows": len(df),
        "total_canonical_sources": int(
            df["canonical_source"].nunique()
        ),
        "original_group_cross_split_count":
            len(original_leakage),
        "cross_dataset_shared_sources": len(shared),
        "canonical_cross_split_sources": len(leaked),
        "affected_train_files": len(train_affected),
        "affected_validation_files":
            len(validation_affected),
        "affected_test_files":
            len(test_affected),
        "scope": (
            "Original dataset_splits.csv only; "
            "ESC-50 and UrbanSound8K grouped by "
            "their Freesound source IDs. "
            "Augmentation and later V2/V3 "
            "provenance are not yet audited."
        ),
    }

    (
        OUTPUT / "audit_summary.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RINGKASAN AUDIT ===")

    for key, value in audit.items():
        print(f"{key}: {value}")

    print(
        "\n=== SUMBER YANG MELINTASI SPLIT ==="
    )

    columns = [
        "canonical_source",
        "datasets",
        "train_count",
        "validation_count",
        "test_count",
        "classes",
    ]

    if leaked.empty:
        print(
            "Tidak ditemukan pada aturan "
            "identifikasi yang diuji."
        )
    else:
        print(
            leaked[columns]
            .to_string(index=False)
        )

    print("\n=== STATUS TEST SET ===")

    if len(test_affected):
        print(
            "PERHATIAN:",
            len(test_affected),
            "file test mempunyai sumber yang "
            "juga muncul pada split lain."
        )
    else:
        print(
            "Tidak ditemukan overlap sumber "
            "pada test untuk cakupan audit ini."
        )

    print("\nHasil disimpan:")
    print(OUTPUT)

    print("\nTidak ada dataset atau model diubah.")


if __name__ == "__main__":
    main()
