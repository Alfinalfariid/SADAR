
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parent

PREDICTIONS_CSV = (
    ROOT
    / "results/amplitude_six_classes_v3"
    / "validation_20261102_224039"
    / "all_window_predictions.csv"
)

LEAKAGE_CSV = (
    ROOT
    / "results/cross_dataset_leakage_audit"
    / "run_20261103_003631"
    / "cross_split_details.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results/v3_leakage_impact"
    / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )
)

CLASSES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

ESC_PATTERN = re.compile(
    r"^[1-5]-(\d+)-[A-Z]-\d+\.wav$",
    re.IGNORECASE,
)

US8K_PATTERN = re.compile(
    r"^(?:urbansound8k_)?"
    r"(\d+)-\d+-\d+-\d+\.wav$",
    re.IGNORECASE,
)


def source_id(value):
    filename = re.split(
        r"[\\/]",
        str(value),
    )[-1]

    match = ESC_PATTERN.match(filename)

    if match:
        return f"FREESOUND_{match.group(1)}"

    match = US8K_PATTERN.match(filename)

    if match:
        return f"FREESOUND_{match.group(1)}"

    return ""


def evaluate_group(name, data):
    if data.empty:
        return {
            "group": name,
            "windows": 0,
            "source_files": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
        }, []

    actual = data["actual"].astype(str)
    predicted = data["predicted"].astype(str)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=CLASSES,
            zero_division=0,
        )
    )

    result = {
        "group": name,
        "windows": len(data),
        "source_files": data["source_file"].nunique(),
        "accuracy": accuracy_score(
            actual,
            predicted,
        ),
        "macro_f1": float(np.mean(f1)),
        "background_to_danger": int(
            (
                (actual == "background")
                & (predicted != "background")
            ).sum()
        ),
        "alarm_as_horn": int(
            (
                (actual == "alarm")
                & (predicted == "vehicle_horn")
            ).sum()
        ),
        "siren_as_horn": int(
            (
                (actual == "siren")
                & (predicted == "vehicle_horn")
            ).sum()
        ),
    }

    per_class = []

    for index, label in enumerate(CLASSES):
        per_class.append({
            "group": name,
            "class": label,
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        })

    return result, per_class


def main():
    print("=" * 65)
    print("SADAR V3 - DAMPAK SOURCE LEAKAGE")
    print("=" * 65)

    for path in [
        PREDICTIONS_CSV,
        LEAKAGE_CSV,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(
        PREDICTIONS_CSV,
        dtype={"source_file": str},
    )

    predictions = predictions[
        predictions["mode"] == "original"
    ].copy()

    if len(predictions) != 1314:
        raise RuntimeError(
            "Jumlah prediksi validasi tidak sesuai. "
            f"Ditemukan {len(predictions)}, "
            "diharapkan 1314."
        )

    leakage = pd.read_csv(
        LEAKAGE_CSV,
        dtype=str,
    )

    if "canonical_source" not in leakage.columns:
        raise RuntimeError(
            "Kolom canonical_source tidak ditemukan."
        )

    compromised_sources = set(
        leakage["canonical_source"]
        .dropna()
        .astype(str)
        .tolist()
    )

    predictions["canonical_source"] = (
        predictions["source_file"]
        .map(source_id)
    )

    predictions["source_leakage"] = (
        predictions["canonical_source"]
        .isin(compromised_sources)
    )

    # Ketiga subset berasal dari prediksi
    # V3 yang sama. Tidak ada inferensi ulang.
    all_data = predictions

    clean_data = predictions[
        ~predictions["source_leakage"]
    ].copy()

    affected_data = predictions[
        predictions["source_leakage"]
    ].copy()

    groups = {
        "all_validation": all_data,
        "without_known_overlap": clean_data,
        "known_overlap_only": affected_data,
    }

    summaries = []
    class_rows = []

    for name, subset in groups.items():
        result, per_class = evaluate_group(
            name,
            subset,
        )

        summaries.append(result)
        class_rows.extend(per_class)

    summary = pd.DataFrame(summaries)
    class_metrics = pd.DataFrame(class_rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_metrics.to_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    affected_data.to_csv(
        OUTPUT_DIR / "affected_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    affected_files = (
        affected_data[
            [
                "source_file",
                "canonical_source",
                "actual",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["canonical_source", "source_file"]
        )
    )

    affected_files.to_csv(
        OUTPUT_DIR / "affected_source_files.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== PERBANDINGAN METRIK V3 ===")

    print(
        summary.round(4).to_string(index=False)
    )

    print("\n=== RECALL PER KELAS ===")

    recall_table = class_metrics.pivot(
        index="class",
        columns="group",
        values="recall",
    )

    print(
        recall_table.round(4).to_string()
    )

    print("\n=== DATA TERDAMPAK ===")

    print(
        "Sumber bocor teridentifikasi:",
        len(compromised_sources),
    )

    print(
        "Window validasi terdampak:",
        len(affected_data),
    )

    print(
        "File validasi terdampak:",
        len(affected_files),
    )

    print(
        "Window validasi tersisa:",
        len(clean_data),
    )

    print("\n=== STATUS ===")

    print(
        "Tidak ada model atau data asli "
        "yang diubah."
    )

    print(
        "Hasil hanya mengecualikan overlap "
        "ESC-50/UrbanSound8K yang diketahui."
    )

    print(
        "Audit V2/V3 dan sumber augmentasi "
        "masih diperlukan."
    )

    print("\nOutput:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
