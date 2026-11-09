
from pathlib import Path
import csv
import io
import math
import zipfile

import numpy as np
import soundfile as sf


# ==================================================
# KONFIGURASI
# ==================================================

INPUT_ROOT = Path(
    "dataset/raw/DEMAND_16k"
)

OUTPUT_ROOT = Path(
    "dataset/robustness_v1/demand_review"
)

RESULT_ROOT = Path(
    "results/robustness_v1"
)

INVENTORY_FILE = (
    RESULT_ROOT / "demand_selected_channels.csv"
)

REVIEW_FILE = (
    RESULT_ROOT / "demand_review_segments.csv"
)

TARGET_SR = 16000
REVIEW_SEGMENT_SECONDS = 10


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)
    print("PERSIAPAN AUDIO DEMAND UNTUK REVIEW")
    print("=" * 70)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {INPUT_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    archives = sorted(
        INPUT_ROOT.glob("*_16k.zip")
    )

    inventory_rows = []
    review_rows = []

    total_seconds = 0.0

    for archive_path in archives:

        scene = archive_path.stem.replace(
            "_16k", ""
        )

        print()
        print("-" * 70)
        print("Lingkungan:", scene)

        with zipfile.ZipFile(
            archive_path, "r"
        ) as archive:

            candidates = [
                name
                for name in archive.namelist()
                if Path(name).name.lower() == "ch01.wav"
            ]

            if len(candidates) != 1:
                raise ValueError(
                    f"{scene}: ch01.wav tidak unik "
                    f"atau tidak ditemukan. "
                    f"Jumlah: {len(candidates)}"
                )

            member_name = candidates[0]

            # Baca file WAV asli dari ZIP.
            wav_bytes = archive.read(
                member_name
            )

        # ======================================
        # VALIDASI AUDIO
        # ======================================

        audio, sr = sf.read(
            io.BytesIO(wav_bytes),
            dtype="float32",
            always_2d=False
        )

        if sr != TARGET_SR:
            raise ValueError(
                f"{scene}: sample rate {sr}"
            )

        if audio.ndim != 1:
            raise ValueError(
                f"{scene}: audio tidak mono"
            )

        if len(audio) == 0:
            raise ValueError(
                f"{scene}: audio kosong"
            )

        duration = len(audio) / sr

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio.astype(np.float64))
                )
            )
        )

        peak = float(
            np.max(np.abs(audio))
        )

        # ======================================
        # SIMPAN AUDIO ASLI
        # ======================================

        destination = (
            OUTPUT_ROOT
            / f"{scene}_ch01.wav"
        )

        # Tidak menimpa file yang sudah ada.
        if destination.exists():

            if destination.read_bytes() != wav_bytes:
                raise FileExistsError(
                    f"File berbeda sudah ada: {destination}"
                )

            print("File sudah tersedia, digunakan ulang.")

        else:

            destination.write_bytes(
                wav_bytes
            )

            print("File berhasil diekstrak.")

        # ======================================
        # INVENTARIS
        # ======================================

        inventory_rows.append({
            "scene": scene,
            "archive": archive_path.name,
            "member": member_name,
            "output_path": str(destination),
            "sample_rate": sr,
            "num_samples": len(audio),
            "duration_sec": round(duration, 4),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "review_status": "PENDING_REVIEW",
        })

        # ======================================
        # BUAT DAFTAR SEGMENT REVIEW 10 DETIK
        # ======================================

        segment_count = math.ceil(
            duration / REVIEW_SEGMENT_SECONDS
        )

        for index in range(segment_count):

            start_sec = (
                index * REVIEW_SEGMENT_SECONDS
            )

            end_sec = min(
                (index + 1) * REVIEW_SEGMENT_SECONDS,
                duration
            )

            review_rows.append({
                "segment_id": (
                    f"{scene}_{index + 1:03d}"
                ),
                "scene": scene,
                "source_group": (
                    f"DEMAND_{scene}"
                ),
                "audio_path": str(destination),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "review_status": "PENDING_REVIEW",
                "target_sound_present": "",
                "notes": "",
            })

        total_seconds += duration

        print("Sample rate :", sr)
        print("Durasi      :", round(duration, 3), "detik")
        print("RMS         :", round(rms, 6))
        print("Peak        :", round(peak, 6))
        print("Segmen      :", segment_count)
        print("Output      :", destination)

    # ==========================================
    # SIMPAN CSV
    # ==========================================

    if inventory_rows and not INVENTORY_FILE.exists():

        with open(
            INVENTORY_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    inventory_rows[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(inventory_rows)

    if review_rows and not REVIEW_FILE.exists():

        with open(
            REVIEW_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    review_rows[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(review_rows)

    # ==========================================
    # SUMMARY
    # ==========================================

    print()
    print("=" * 70)
    print("RINGKASAN PERSIAPAN DEMAND")
    print("=" * 70)

    print(
        "Jumlah lingkungan :",
        len(inventory_rows)
    )

    print(
        "Audio terpilih    :",
        len(inventory_rows)
    )

    print(
        "Total durasi      :",
        round(total_seconds / 60, 2),
        "menit"
    )

    print(
        "Total segmen review:",
        len(review_rows)
    )

    print()
    print("Folder audio:")
    print(OUTPUT_ROOT)

    print()
    print("Inventaris:")
    print(INVENTORY_FILE)

    print()
    print("Daftar segmen untuk review:")
    print(REVIEW_FILE)

    print()
    print(
        "PERHATIAN: Belum ada audio yang "
        "diberi label background secara otomatis."
    )

    print(
        "Dataset dan model baseline tidak diubah."
    )


if __name__ == "__main__":
    main()
