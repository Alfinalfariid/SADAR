
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import soundfile as sf


# ============================================
# KONFIGURASI
# ============================================

ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    ROOT
    / "results"
    / "amplitude_six_classes_v3"
    / "validation_20261102_224039"
    / "all_window_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "v4_review"
    / datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )
)

WINDOW_SECONDS = 15600 / 16000


def safe_filename(name):
    return re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        name,
    )[:35]


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    data = pd.read_csv(INPUT_CSV)

    needed = {
        "mode",
        "source_file",
        "start_sec",
        "actual",
        "predicted",
        "confidence",
        "candidate_without_confirmation",
    }

    missing = needed - set(data.columns)

    if missing:
        raise ValueError(
            f"Kolom CSV hilang: {sorted(missing)}"
        )

    data = data[
        data["mode"] == "original"
    ].copy()

    if data.empty:
        raise ValueError(
            "Tidak ditemukan baris mode=original."
        )

    data["confidence"] = pd.to_numeric(
        data["confidence"],
        errors="coerce",
    )

    data["start_sec"] = pd.to_numeric(
        data["start_sec"],
        errors="coerce",
    )

    errors = data[
        data["actual"] != data["predicted"]
    ]

    # Ringkasan sumber audio bermasalah
    by_source = (
        errors.groupby(
            ["actual", "predicted", "source_file"],
            dropna=False,
        )
        .agg(
            wrong_windows=("predicted", "size"),
            highest_confidence=("confidence", "max"),
        )
        .reset_index()
        .sort_values(
            "wrong_windows",
            ascending=False,
        )
    )

    # Pilih sumber audio yang paling relevan.
    # Maksimal satu window per sumber pada setiap kategori.
    groups = [
        (
            "background_kandidat_bahaya",
            (
                (data["actual"] == "background")
                & (
                    data["candidate_without_confirmation"]
                    != "background"
                )
            ),
            10,
        ),
        (
            "alarm_salah",
            (
                (data["actual"] == "alarm")
                & (data["predicted"] != "alarm")
            ),
            10,
        ),
        (
            "sirene_jadi_klakson",
            (
                (data["actual"] == "siren")
                & (
                    data["predicted"]
                    == "vehicle_horn"
                )
            ),
            8,
        ),
        (
            "sirene_jadi_background",
            (
                (data["actual"] == "siren")
                & (
                    data["predicted"]
                    == "background"
                )
            ),
            8,
        ),
        (
            "klakson_jadi_sirene",
            (
                (data["actual"] == "vehicle_horn")
                & (data["predicted"] == "siren")
            ),
            8,
        ),
        (
            "teriakan_salah",
            (
                (data["actual"] == "scream")
                & (data["predicted"] != "scream")
            ),
            8,
        ),
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    clip_dir = OUTPUT_DIR / "audio_review"
    clip_dir.mkdir()

    by_source.to_csv(
        OUTPUT_DIR / "kesalahan_per_sumber.csv",
        index=False,
        encoding="utf-8-sig",
    )

    review = []

    for title, condition, limit in groups:
        chosen = (
            data.loc[condition]
            .sort_values(
                "confidence",
                ascending=False,
            )
            .drop_duplicates(
                subset=["source_file"]
            )
            .head(limit)
        )

        print(
            f"{title}: {len(chosen)} sumber "
            "untuk diperiksa"
        )

        for item in chosen.itertuples(
            index=False
        ):
            source = Path(
                str(item.source_file)
            )

            entry = {
                "kategori_audit": title,
                "sumber": str(source),
                "label_dataset": str(item.actual),
                "prediksi_model": str(item.predicted),
                "confidence": round(
                    float(item.confidence),
                    5,
                ),
                "window_mulai_detik": float(
                    item.start_sec
                ),
                "audio_review": "",
                "status": "belum_diperiksa",
                "label_setelah_didengar": "",
                "catatan": "",
            }

            if not source.is_file():
                entry["status"] = (
                    "file_sumber_tidak_ditemukan"
                )
                review.append(entry)
                continue

            try:
                wave, sample_rate = sf.read(
                    str(source),
                    dtype="float32",
                    always_2d=False,
                )

                if wave.ndim == 2:
                    wave = wave.mean(axis=1)

                if (
                    wave.ndim != 1
                    or len(wave) == 0
                ):
                    raise ValueError(
                        "Audio tidak valid."
                    )

                # Tambahkan konteks sekitar
                # window salah klasifikasi.
                begin_sec = max(
                    0.0,
                    float(item.start_sec) - 0.8,
                )

                end_sec = min(
                    len(wave) / sample_rate,
                    float(item.start_sec)
                    + WINDOW_SECONDS
                    + 0.8,
                )

                start_index = int(
                    begin_sec * sample_rate
                )

                end_index = int(
                    end_sec * sample_rate
                )

                clip = wave[
                    start_index:end_index
                ]

                if len(clip) == 0:
                    raise ValueError(
                        "Cuplikan kosong."
                    )

                serial = len(review) + 1

                filename = (
                    f"{serial:03d}_"
                    f"{safe_filename(title)}_"
                    f"{safe_filename(source.stem)}.wav"
                )

                destination = (
                    clip_dir / filename
                )

                sf.write(
                    str(destination),
                    clip,
                    sample_rate,
                    subtype="PCM_16",
                )

                entry["audio_review"] = str(
                    destination
                )

            except Exception as error:
                entry["status"] = (
                    f"gagal_ekspor: {error}"
                )

            review.append(entry)

    review_df = pd.DataFrame(review)

    review_df.to_csv(
        OUTPUT_DIR / "daftar_review_audio.csv",
        index=False,
        encoding="utf-8-sig",
    )

    success_count = int(
        (
            review_df["status"]
            == "belum_diperiksa"
        ).sum()
    )

    print("\n" + "=" * 60)
    print("AUDIT KESALAHAN V3 SELESAI")
    print("=" * 60)

    print(
        "Window salah:",
        len(errors),
        "dari",
        len(data),
    )

    print(
        "Cuplikan terdaftar:",
        len(review_df),
    )

    print(
        "Cuplikan berhasil diekspor:",
        success_count,
    )

    print("\nFolder hasil:")
    print(OUTPUT_DIR)

    print("\nStatus:")
    print("- Model V3 tidak diubah")
    print("- Dataset asli tidak diubah")
    print("- Split train/validation/test tidak diubah")
    print("- Tidak melakukan training")


if __name__ == "__main__":
    main()
