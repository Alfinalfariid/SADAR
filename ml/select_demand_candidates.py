
from pathlib import Path

import pandas as pd


# =====================================================
# KONFIGURASI
# =====================================================

INPUT_FILE = Path(
    "results/robustness_v1/demand_auto_screen.csv"
)

OUTPUT_DIR = Path(
    "results/robustness_v1"
)

SUMMARY_FILE = OUTPUT_DIR / (
    "demand_candidate_summary.csv"
)

SELECTED_FILE = OUTPUT_DIR / (
    "demand_candidate_background_guarded.csv"
)

EXCLUDED_FILE = OUTPUT_DIR / (
    "demand_excluded_segments.csv"
)

# Ambang konservatif untuk eksperimen.
# Bukan threshold yang sudah tervalidasi.
MAX_TARGET_SCORE = 0.04

# Keluarkan segmen yang berbatasan langsung
# dengan segmen terindikasi bahaya.
GUARD_SECONDS = 10.0

REFERENCE_RMS = 0.008


# =====================================================
# LOAD DATA
# =====================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "segment_id",
        "scene",
        "source_group",
        "audio_path",
        "start_sec",
        "end_sec",
        "status",
        "rms",
        "max_target_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom tidak tersedia: {missing}"
        )

    if df["segment_id"].duplicated().any():
        raise ValueError(
            "Ada segment_id yang duplikat."
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Terdapat nilai kosong pada kolom penting."
        )

    allowed_status = {
        "CANDIDATE_BACKGROUND_UNVERIFIED",
        "REVIEW_TARGET_POSSIBLE",
    }

    invalid = set(df["status"]) - allowed_status

    if invalid:
        raise ValueError(
            f"Status tidak dikenal: {invalid}"
        )

    df = df.sort_values(
        ["scene", "start_sec"]
    ).reset_index(drop=True)

    return df


# =====================================================
# SELEKSI KANDIDAT
# =====================================================

def select_candidates(df):

    selected_ids = set()
    excluded_reasons = {}

    for scene, group in df.groupby("scene"):

        suspicious = group[
            group["status"] == "REVIEW_TARGET_POSSIBLE"
        ]

        suspicious_starts = (
            suspicious["start_sec"]
            .astype(float)
            .tolist()
        )

        for index, row in group.iterrows():

            segment_id = row["segment_id"]

            if row["status"] == "REVIEW_TARGET_POSSIBLE":

                excluded_reasons[segment_id] = (
                    "target_possible"
                )
                continue

            score = float(
                row["max_target_score"]
            )

            if score >= MAX_TARGET_SCORE:

                excluded_reasons[segment_id] = (
                    "conservative_score_filter"
                )
                continue

            start = float(row["start_sec"])

            near_suspicious = any(
                abs(start - suspicious_start)
                <= GUARD_SECONDS + 0.001
                for suspicious_start in suspicious_starts
            )

            if near_suspicious:

                excluded_reasons[segment_id] = (
                    "adjacent_to_suspicious_segment"
                )
                continue

            selected_ids.add(segment_id)

    selected = df[
        df["segment_id"].isin(selected_ids)
    ].copy()

    selected["selection_status"] = (
        "PSEUDO_BACKGROUND_UNVERIFIED"
    )

    selected["selection_method"] = (
        "yamnet_screen_guarded_v1"
    )

    excluded = df[
        ~df["segment_id"].isin(selected_ids)
    ].copy()

    excluded["exclusion_reason"] = (
        excluded["segment_id"]
        .map(excluded_reasons)
    )

    return selected, excluded


# =====================================================
# SUMMARY
# =====================================================

def create_summary(df, selected):

    rows = []

    for scene, group in df.groupby("scene"):

        chosen = selected[
            selected["scene"] == scene
        ]

        total = len(group)

        initial = int(
            (
                group["status"]
                == "CANDIDATE_BACKGROUND_UNVERIFIED"
            ).sum()
        )

        flagged = int(
            (
                group["status"]
                == "REVIEW_TARGET_POSSIBLE"
            ).sum()
        )

        low_rms = int(
            (
                chosen["rms"]
                < REFERENCE_RMS
            ).sum()
        )

        rows.append({
            "scene": scene,
            "total_segments": total,
            "initial_candidates": initial,
            "flagged_segments": flagged,
            "conservative_candidates": len(chosen),
            "selected_low_rms": low_rms,
            "selected_duration_sec": len(chosen) * 10,
        })

    return pd.DataFrame(rows)


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 70)
    print("DEMAND - CONSERVATIVE BACKGROUND SELECTION")
    print("=" * 70)

    df = load_data()

    selected, excluded = select_candidates(df)

    summary = create_summary(
        df,
        selected
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    selected.to_csv(
        SELECTED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    excluded.to_csv(
        EXCLUDED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("HASIL PER LINGKUNGAN")
    print("-" * 70)

    print(
        summary.to_string(index=False)
    )

    print()
    print("=" * 70)
    print("RINGKASAN SELEKSI")
    print("=" * 70)

    print("Total segmen         :", len(df))

    print(
        "Kandidat awal        :",
        (
            df["status"]
            == "CANDIDATE_BACKGROUND_UNVERIFIED"
        ).sum()
    )

    print(
        "Terindikasi target   :",
        (
            df["status"]
            == "REVIEW_TARGET_POSSIBLE"
        ).sum()
    )

    print(
        "Kandidat konservatif :",
        len(selected)
    )

    print(
        "Segmen dikecualikan  :",
        len(excluded)
    )

    print(
        "Durasi kandidat      :",
        round(len(selected) * 10 / 60, 2),
        "menit"
    )

    print()
    print("ALASAN PENGECUALIAN")
    print("-" * 70)

    print(
        excluded["exclusion_reason"]
        .value_counts()
        .to_string()
    )

    print()
    print("HASIL DISIMPAN:")
    print(SELECTED_FILE)
    print(EXCLUDED_FILE)
    print(SUMMARY_FILE)

    print()
    print(
        "PERHATIAN: Hasil masih berupa kandidat "
        "pseudo-background, bukan ground truth."
    )

    print(
        "Tidak ada perubahan pada dataset atau "
        "model baseline."
    )


if __name__ == "__main__":
    main()
