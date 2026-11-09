from pathlib import Path
import csv
import shutil


# ============================================================
# PATH
# ============================================================

DATASET_ROOT = Path(
    "dataset/raw/urbansound8k"
)

AUDIO_ROOT = DATASET_ROOT / "audio"

META_FILE = (
    DATASET_ROOT
    / "metadata"
    / "UrbanSound8K.csv"
)

OUTPUT_ROOT = Path(
    "dataset/processed"
)

MANIFEST_FILE = Path(
    "dataset/urbansound8k_manifest.csv"
)


# ============================================================
# MAPPING CLASS
# ============================================================

CLASS_MAPPING = {
    "siren": "siren",
    "car_horn": "vehicle_horn",
}


# ============================================================
# VALIDASI
# ============================================================

def validate_paths():

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan:\n"
            f"{DATASET_ROOT}"
        )

    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(
            f"Folder audio tidak ditemukan:\n"
            f"{AUDIO_ROOT}"
        )

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Metadata tidak ditemukan:\n"
            f"{META_FILE}"
        )


# ============================================================
# COPY AUDIO
# ============================================================

def copy_audio(
    filename,
    fold,
    target_class
):

    source = (
        AUDIO_ROOT
        / f"fold{fold}"
        / filename
    )

    destination_folder = (
        OUTPUT_ROOT
        / target_class
    )

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Prefix agar tidak bentrok dengan ESC-50
    new_filename = (
        f"urbansound8k_{filename}"
    )

    destination = (
        destination_folder
        / new_filename
    )

    if not source.exists():

        print(
            f"WARNING file tidak ditemukan: "
            f"{source}"
        )

        return None

    shutil.copy2(
        source,
        destination
    )

    return new_filename


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PREPARE URBANSOUND8K")
    print("=" * 60)

    validate_paths()

    counts = {
        "siren": 0,
        "vehicle_horn": 0,
    }

    manifest_rows = []

    with open(
        META_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            original_class = (
                row["class"]
            )

            if (
                original_class
                not in CLASS_MAPPING
            ):
                continue

            target_class = (
                CLASS_MAPPING[
                    original_class
                ]
            )

            new_filename = copy_audio(
                row["slice_file_name"],
                row["fold"],
                target_class
            )

            if new_filename is None:
                continue

            counts[
                target_class
            ] += 1

            manifest_rows.append({

                "filename":
                    new_filename,

                "original_filename":
                    row["slice_file_name"],

                "dataset":
                    "UrbanSound8K",

                "original_class":
                    original_class,

                "target_class":
                    target_class,

                "fold":
                    row["fold"],

                "fsID":
                    row["fsID"],

                "start":
                    row["start"],

                "end":
                    row["end"],

                "salience":
                    row["salience"],
            })


    # ========================================================
    # SIMPAN MANIFEST
    # ========================================================

    fieldnames = [
        "filename",
        "original_filename",
        "dataset",
        "original_class",
        "target_class",
        "fold",
        "fsID",
        "start",
        "end",
        "salience",
    ]

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )


    print()
    print("=" * 60)
    print("HASIL URBANSOUND8K")
    print("=" * 60)

    total = 0

    for class_name, count in (
        counts.items()
    ):

        print(
            f"{class_name:<20}: "
            f"{count} file"
        )

        total += count

    print("-" * 60)

    print(
        f"{'TOTAL':<20}: "
        f"{total} file"
    )

    print()

    print(
        "Manifest:"
    )

    print(
        MANIFEST_FILE
    )

    print()


if __name__ == "__main__":
    main()