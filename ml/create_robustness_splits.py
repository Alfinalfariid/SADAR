
from pathlib import Path
import hashlib
import json

import pandas as pd


# ==================================================
# KONFIGURASI
# ==================================================

INPUT_CANDIDATES = Path(
    "results/robustness_v1/"
    "demand_candidate_background_guarded.csv"
)

INPUT_WINDOWS = Path(
    "results/robustness_v1/demand_window_rms.csv"
)

OUTPUT_ROOT = Path(
    "dataset/robustness_v1/splits"
)

RESULT_ROOT = Path(
    "results/robustness_v1"
)

OUTPUT_MANIFEST = (
    OUTPUT_ROOT / "demand_partition_v1.csv"
)

OUTPUT_SUMMARY = (
    RESULT_ROOT / "demand_partition_v1_summary.csv"
)

OUTPUT_CONFIG = (
    RESULT_ROOT / "demand_partition_v1_config.json"
)


# ==================================================
# PEMBAGIAN BERDASARKAN SUMBER REKAMAN
# ==================================================

SCENE_SPLITS = {
    "NPARK": "train",
    "OMEETING": "train",
    "OOFFICE": "train",
    "PCAFETER": "train",
    "PSTATION": "train",
    "TCAR": "train",
    "TMETRO": "train",

    "SPSQUARE": "validation",

    "TBUS": "test",
    "STRAFFIC": "test",
}


# ==================================================
# SHA256
# ==================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print("DANGER SOUND AI - ROBUSTNESS SPLIT V1")
    print("=" * 70)

    for path in [
        INPUT_CANDIDATES,
        INPUT_WINDOWS
    ]:

        if not path.is_file():
            raise FileNotFoundError(path)

    # Jangan menimpa manifest eksperimen
    # jika sudah pernah dibuat.
    for path in [
        OUTPUT_MANIFEST,
        OUTPUT_SUMMARY,
        OUTPUT_CONFIG
    ]:

        if path.exists():
            raise FileExistsError(
                f"Output sudah ada: {path}\n"
                "File eksperimen tidak akan ditimpa."
            )

    candidates = pd.read_csv(
        INPUT_CANDIDATES
    )

    windows = pd.read_csv(
        INPUT_WINDOWS
    )

    # ==========================================
    # VALIDASI
    # ==========================================

    required_candidates = {
        "segment_id",
        "scene",
        "source_group",
        "audio_path",
        "start_sec",
        "end_sec",
    }

    required_windows = {
        "segment_id",
        "activity",
    }

    if not required_candidates.issubset(
        candidates.columns
    ):
        raise ValueError(
            "Kolom kandidat tidak lengkap."
        )

    if not required_windows.issubset(
        windows.columns
    ):
        raise ValueError(
            "Kolom window tidak lengkap."
        )

    if candidates["segment_id"].duplicated().any():
        raise ValueError(
            "Ada segment_id duplikat."
        )

    unknown = (
        set(candidates["scene"])
        - set(SCENE_SPLITS)
    )

    if unknown:
        raise ValueError(
            f"Lingkungan belum dipetakan: {unknown}"
        )

    # ==========================================
    # HITUNG WINDOW AKTIF PER SEGMEN
    # ==========================================

    windows["is_active"] = (
        windows["activity"] == "ACTIVE"
    )

    activity = (
        windows.groupby("segment_id")
        .agg(
            total_windows=("is_active", "size"),
            active_windows=("is_active", "sum"),
        )
        .reset_index()
    )

    candidates = candidates.merge(
        activity,
        on="segment_id",
        how="left",
        validate="one_to_one"
    )

    if candidates[
        ["total_windows", "active_windows"]
    ].isna().any().any():
        raise ValueError(
            "Sebagian segmen tidak memiliki data window."
        )

    # ==========================================
    # TETAPKAN SPLIT
    # ==========================================

    candidates["split"] = (
        candidates["scene"]
        .map(SCENE_SPLITS)
    )

    candidates["label_status"] = (
        "PSEUDO_BACKGROUND_UNVERIFIED"
    )

    candidates["duration_sec"] = (
        candidates["end_sec"]
        - candidates["start_sec"]
    )

    candidates["source_id"] = (
        candidates["scene"].apply(
            lambda scene: f"DEMAND_{scene}_ch01"
        )
    )

    # ==========================================
    # CEK KEBOCORAN SOURCE
    # ==========================================

    leakage = (
        candidates.groupby("source_id")["split"]
        .nunique()
    )

    problematic = leakage[
        leakage > 1
    ]

    if len(problematic) > 0:
        raise ValueError(
            "DATA LEAKAGE TERDETEKSI:\n"
            f"{problematic}"
        )

    # ==========================================
    # REKAP
    # ==========================================

    summary = (
        candidates.groupby(
            ["split", "scene"],
            as_index=False
        )
        .agg(
            segments=("segment_id", "size"),
            total_windows=("total_windows", "sum"),
            active_windows=("active_windows", "sum"),
            duration_sec=("duration_sec", "sum")
        )
    )

    summary["duration_minutes"] = (
        summary["duration_sec"] / 60
    ).round(2)

    candidates = candidates.sort_values(
        ["split", "scene", "start_sec"]
    ).reset_index(drop=True)

    # ==========================================
    # SIMPAN OUTPUT
    # ==========================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "experiment": "robustness_v1",
        "method": "scene_level_partition",
        "scene_splits": SCENE_SPLITS,
        "source_group_leakage": len(problematic),
        "candidate_sha256": sha256_file(
            INPUT_CANDIDATES
        ),
        "windows_sha256": sha256_file(
            INPUT_WINDOWS
        ),
        "notes": [
            "Partition only; no training performed.",
            "All DEMAND candidate labels are unverified.",
            "No DEMAND source crosses splits.",
            "Partition chosen after exploratory baseline inspection.",
            "Independent verified final test is still required.",
            "Baseline models and dataset are unchanged."
        ]
    }

    OUTPUT_CONFIG.write_text(
        json.dumps(
            config,
            indent=4
        ),
        encoding="utf-8"
    )

    # ==========================================
    # TAMPILKAN HASIL
    # ==========================================

    print()
    print("HASIL PEMBAGIAN PER LINGKUNGAN")
    print("-" * 70)

    print(
        summary.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("RINGKASAN PER SPLIT")
    print("=" * 70)

    totals = (
        summary.groupby("split")[
            [
                "segments",
                "total_windows",
                "active_windows",
            ]
        ]
        .sum()
    )

    print(totals.to_string())

    print()
    print(
        "Source group lintas split:",
        len(problematic)
    )

    print()
    print("Manifest :", OUTPUT_MANIFEST)
    print("Summary  :", OUTPUT_SUMMARY)
    print("Config   :", OUTPUT_CONFIG)

    print()
    print(
        "DATASET DAN MODEL BASELINE "
        "TIDAK DIUBAH."
    )


if __name__ == "__main__":
    main()
