from pathlib import Path
import csv
import shutil


# ============================================================
# PATH
# ============================================================

DATASET_ROOT = Path(
    "dataset/raw/nigens/NIGENS"
)

FEMALE_DIR = (
    DATASET_ROOT
    / "femaleScream"
)

MALE_DIR = (
    DATASET_ROOT
    / "maleScream"
)

OUTPUT_DIR = Path(
    "dataset/processed/scream"
)

MANIFEST_FILE = Path(
    "dataset/nigens_manifest.csv"
)


# ============================================================
# VALIDASI
# ============================================================

def validate_paths():

    if not FEMALE_DIR.exists():

        raise FileNotFoundError(
            f"Folder femaleScream tidak ditemukan:\n"
            f"{FEMALE_DIR}"
        )

    if not MALE_DIR.exists():

        raise FileNotFoundError(
            f"Folder maleScream tidak ditemukan:\n"
            f"{MALE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# AMBIL SEMUA WAV
# ============================================================

def get_wav_files(directory):

    return sorted([
        file
        for file in directory.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower() == ".wav"
        )
    ])


# ============================================================
# COPY SATU GROUP
# ============================================================

def process_group(
    source_directory,
    original_class,
    prefix,
    manifest_rows
):

    files = get_wav_files(
        source_directory
    )

    print()
    print(
        f"{original_class}: "
        f"{len(files)} file"
    )

    success = 0
    failed = 0

    for index, source_file in enumerate(
        files,
        start=1
    ):

        try:

            # =================================================
            # NAMA FILE BARU
            # =================================================

            new_filename = (
                f"nigens_"
                f"{prefix}_"
                f"{index:04d}_"
                f"{source_file.name}"
            )

            destination = (
                OUTPUT_DIR
                / new_filename
            )

            shutil.copy2(
                source_file,
                destination
            )


            # =================================================
            # SOURCE GROUP
            #
            # Kita simpan identitas file asli.
            # Nanti dipakai untuk mencegah leakage.
            # =================================================

            source_group = (
                f"NIGENS_"
                f"{original_class}_"
                f"{source_file.stem}"
            )


            manifest_rows.append({

                "filename":
                    new_filename,

                "dataset":
                    "NIGENS",

                "original_class":
                    original_class,

                "target_class":
                    "scream",

                "source_group":
                    source_group,

                "original_filename":
                    source_file.name,

                "original_path":
                    str(source_file),

            })

            success += 1

        except Exception as error:

            failed += 1

            print(
                f"ERROR {source_file.name}: "
                f"{error}"
            )


    return (
        success,
        failed
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("PREPARE NIGENS - SCREAM")
    print("=" * 65)

    validate_paths()

    manifest_rows = []


    # ========================================================
    # FEMALE SCREAM
    # ========================================================

    female_success, female_failed = (
        process_group(
            FEMALE_DIR,
            "femaleScream",
            "female",
            manifest_rows
        )
    )


    # ========================================================
    # MALE SCREAM
    # ========================================================

    male_success, male_failed = (
        process_group(
            MALE_DIR,
            "maleScream",
            "male",
            manifest_rows
        )
    )


    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    fieldnames = [
        "filename",
        "dataset",
        "original_class",
        "target_class",
        "source_group",
        "original_filename",
        "original_path",
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


    # ========================================================
    # SUMMARY
    # ========================================================

    total_success = (
        female_success
        + male_success
    )

    total_failed = (
        female_failed
        + male_failed
    )

    print()
    print("=" * 65)
    print("HASIL NIGENS")
    print("=" * 65)

    print(
        f"femaleScream       : "
        f"{female_success} file"
    )

    print(
        f"maleScream         : "
        f"{male_success} file"
    )

    print("-" * 65)

    print(
        f"scream total       : "
        f"{total_success} file"
    )

    print(
        f"gagal              : "
        f"{total_failed} file"
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