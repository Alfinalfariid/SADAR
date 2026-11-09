from pathlib import Path
import csv
import shutil


# ============================================================
# PATH
# ============================================================

ESC50_ROOT = Path(
    "dataset/raw/esc50/ESC-50-master"
)

AUDIO_DIR = ESC50_ROOT / "audio"
META_FILE = ESC50_ROOT / "meta" / "esc50.csv"

OUTPUT_ROOT = Path(
    "dataset/processed"
)

MANIFEST_FILE = Path(
    "dataset/esc50_manifest.csv"
)


# ============================================================
# MAPPING ESC-50 -> KELAS PENELITIAN
# ============================================================

CLASS_MAPPING = {
    "siren": "siren",
    "car_horn": "vehicle_horn",
    "glass_breaking": "glass_breaking",
}


# ============================================================
# KELAS BACKGROUND
# ============================================================

BACKGROUND_CLASSES = {
    "rain",
    "sea_waves",
    "wind",
    "footsteps",
    "keyboard_typing",
    "vacuum_cleaner",
    "clock_tick",
    "washing_machine",
}


# ============================================================
# BUAT FOLDER
# ============================================================

def create_directories():

    folders = [
        "siren",
        "vehicle_horn",
        "glass_breaking",
        "fire_alarm",
        "scream",
        "background",
    ]

    for folder in folders:

        path = OUTPUT_ROOT / folder

        path.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# VALIDASI DATASET
# ============================================================

def validate_paths():

    if not ESC50_ROOT.exists():
        raise FileNotFoundError(
            f"Folder ESC-50 tidak ditemukan:\n"
            f"{ESC50_ROOT}"
        )

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Folder audio tidak ditemukan:\n"
            f"{AUDIO_DIR}"
        )

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"File metadata tidak ditemukan:\n"
            f"{META_FILE}"
        )


# ============================================================
# COPY AUDIO
# ============================================================

def copy_audio(
    source_filename,
    target_class
):

    source = (
        AUDIO_DIR
        / source_filename
    )

    destination_folder = (
        OUTPUT_ROOT
        / target_class
    )

    destination = (
        destination_folder
        / source_filename
    )

    if not source.exists():

        print(
            f"WARNING: file tidak ditemukan: "
            f"{source}"
        )

        return False

    shutil.copy2(
        source,
        destination
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 55)
    print("PREPARE ESC-50 DATASET")
    print("=" * 55)

    validate_paths()

    create_directories()

    selected_rows = []

    class_counts = {
        "siren": 0,
        "vehicle_horn": 0,
        "glass_breaking": 0,
        "background": 0,
    }

    with open(
        META_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            original_category = (
                row["category"]
            )

            target_class = None

            # =================================================
            # TARGET DANGER CLASSES
            # =================================================

            if original_category in CLASS_MAPPING:

                target_class = (
                    CLASS_MAPPING[
                        original_category
                    ]
                )

            # =================================================
            # BACKGROUND
            # =================================================

            elif original_category in BACKGROUND_CLASSES:

                target_class = (
                    "background"
                )

            # Tidak diperlukan
            else:
                continue


            copied = copy_audio(
                row["filename"],
                target_class
            )

            if not copied:
                continue


            class_counts[
                target_class
            ] += 1


            selected_rows.append({
                "filename":
                    row["filename"],

                "dataset":
                    "ESC-50",

                "original_category":
                    original_category,

                "target_class":
                    target_class,

                "fold":
                    row["fold"],

                "source_file":
                    row.get(
                        "src_file",
                        ""
                    ),

                "take":
                    row.get(
                        "take",
                        ""
                    ),
            })


    # ========================================================
    # SIMPAN MANIFEST
    # ========================================================

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "filename",
        "dataset",
        "original_category",
        "target_class",
        "fold",
        "source_file",
        "take",
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
            selected_rows
        )


    # ========================================================
    # HASIL
    # ========================================================

    print()
    print("=" * 55)
    print("HASIL ESC-50")
    print("=" * 55)

    total = 0

    for class_name, count in (
        class_counts.items()
    ):

        print(
            f"{class_name:<20} : "
            f"{count} file"
        )

        total += count

    print("-" * 55)

    print(
        f"{'TOTAL':<20} : "
        f"{total} file"
    )

    print()

    print(
        "Manifest disimpan di:"
    )

    print(
        MANIFEST_FILE
    )

    print()

    print(
        "Selesai."
    )


if __name__ == "__main__":
    main()
