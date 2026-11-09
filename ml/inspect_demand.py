from pathlib import Path
import csv
import zipfile

# ==========================================
# KONFIGURASI
# ==========================================

DEMAND_ROOT = Path(
    "dataset/raw/DEMAND_16k"
)

OUTPUT_FILE = Path(
    "results/robustness_v1/demand_zip_audit.csv"
)

EXPECTED_FILES = [
    "STRAFFIC_16k.zip",
    "PCAFETER_16k.zip",
    "OMEETING_16k.zip",
    "TCAR_16k.zip",
    "TMETRO_16k.zip",
    "TBUS_16k.zip",
    "PSTATION_16k.zip",
    "OOFFICE_16k.zip",
    "SPSQUARE_16k.zip",
    "NPARK_16k.zip",
]


def mb(size):
    return round(size / (1024 * 1024), 2)


def main():

    print("=" * 65)
    print("AUDIT DATASET DEMAND 16 KHZ")
    print("=" * 65)

    if not DEMAND_ROOT.is_dir():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {DEMAND_ROOT}"
        )

    rows = []

    for filename in EXPECTED_FILES:

        path = DEMAND_ROOT / filename

        row = {
            "filename": filename,
            "status": "",
            "zip_size_mb": 0,
            "wav_count": 0,
            "wav_uncompressed_mb": 0,
            "example_wav": "",
        }

        if not path.exists():
            row["status"] = "MISSING"
            rows.append(row)
            print(f"[MISSING] {filename}")
            continue

        row["zip_size_mb"] = mb(
            path.stat().st_size
        )

        try:
            if not zipfile.is_zipfile(path):
                raise zipfile.BadZipFile(
                    "Format ZIP tidak valid"
                )

            with zipfile.ZipFile(path, "r") as archive:

                wav_files = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and info.filename.lower().endswith(".wav")
                ]

                row["wav_count"] = len(wav_files)

                row["wav_uncompressed_mb"] = mb(
                    sum(info.file_size for info in wav_files)
                )

                if wav_files:
                    row["example_wav"] = wav_files[0].filename
                    row["status"] = "OK"
                else:
                    row["status"] = "NO_WAV"

            print(
                f"[{row['status']}] {filename} | "
                f"{row['zip_size_mb']} MB | "
                f"{row['wav_count']} WAV"
            )

        except Exception as error:
            row["status"] = f"ERROR: {error}"
            print(f"[ERROR] {filename}: {error}")

        rows.append(row)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    valid = sum(
        row["status"] == "OK"
        for row in rows
    )

    print()
    print("=" * 65)
    print("RINGKASAN AUDIT")
    print("=" * 65)

    print(f"ZIP diharapkan : {len(EXPECTED_FILES)}")
    print(f"ZIP berisi WAV : {valid}")
    print(
        "Total WAV      :",
        sum(row["wav_count"] for row in rows)
    )
    print(
        "Total ZIP (MB) :",
        round(
            sum(row["zip_size_mb"] for row in rows),
            2
        )
    )

    print()
    print("Hasil audit disimpan ke:")
    print(OUTPUT_FILE)

    print()
    print(
        "Catatan: audit ini memeriksa struktur ZIP, "
        "belum memvalidasi isi audio atau label."
    )


if __name__ == "__main__":
    main()