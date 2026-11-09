from pathlib import Path
import csv
import re

import soundfile as sf


# ============================================================
# PATH
# ============================================================

DATASET_ROOT = Path(
    "dataset/raw/tau_sebin"
)

AUDIO_ROOT = (
    DATASET_ROOT
    / "audio"
    / "bin_prox_dir_one"
)

METADATA_ROOT = (
    DATASET_ROOT
    / "metadata"
    / "bin_prox_dir_one"
)

OUTPUT_ROOT = Path(
    "dataset/processed"
)

MANIFEST_FILE = Path(
    "dataset/tau_sebin_manifest.csv"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {
    "scream": "scream",
    "glassbreak": "glass_breaking",
}


# ============================================================
# VALIDASI PATH
# ============================================================

def validate_paths():

    if not AUDIO_ROOT.exists():

        raise FileNotFoundError(
            f"Folder audio tidak ditemukan:\n"
            f"{AUDIO_ROOT}"
        )

    if not METADATA_ROOT.exists():

        raise FileNotFoundError(
            f"Folder metadata tidak ditemukan:\n"
            f"{METADATA_ROOT}"
        )


# ============================================================
# PARSE FOLD
# ============================================================

def get_fold(filename):

    match = re.search(
        r"split(\d+)",
        filename
    )

    if match:
        return int(
            match.group(1)
        )

    return ""


# ============================================================
# POTONG SATU EVENT
# ============================================================

def extract_event(
    source_audio,
    start_time,
    end_time,
    destination
):

    audio, sample_rate = sf.read(
        str(source_audio),
        dtype="float32",
        always_2d=True
    )

    start_sample = int(
        start_time
        * sample_rate
    )

    end_sample = int(
        end_time
        * sample_rate
    )

    # Safety
    start_sample = max(
        0,
        start_sample
    )

    end_sample = min(
        len(audio),
        end_sample
    )

    if end_sample <= start_sample:

        raise ValueError(
            "Durasi event tidak valid."
        )

    event_audio = audio[
        start_sample:end_sample
    ]

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        str(destination),
        event_audio,
        sample_rate,
        subtype="PCM_16"
    )

    duration = (
        len(event_audio)
        / sample_rate
    )

    return {
        "sample_rate":
            sample_rate,

        "duration":
            duration,

        "samples":
            len(event_audio),

        "channels":
            event_audio.shape[1],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("PREPARE TAU-SEBIN 2021")
    print("=" * 65)

    validate_paths()

    counts = {
        "scream": 0,
        "glass_breaking": 0,
    }

    manifest_rows = []

    metadata_files = sorted(
        METADATA_ROOT.glob(
            "*.csv"
        )
    )

    print()
    print(
        "Jumlah metadata:",
        len(metadata_files)
    )

    event_number = 0
    failed = 0


    # ========================================================
    # LOOP METADATA
    # ========================================================

    for metadata_file in metadata_files:

        recording_name = (
            metadata_file.stem
        )

        audio_file = (
            AUDIO_ROOT
            / f"{recording_name}.wav"
        )

        if not audio_file.exists():

            print(
                "WARNING audio tidak ditemukan:",
                audio_file
            )

            continue

        fold = get_fold(
            recording_name
        )

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            local_event_index = 0

            for row in reader:

                original_class = (
                    row[
                        "sound_event_recording"
                    ]
                    .strip()
                    .lower()
                )

                if (
                    original_class
                    not in CLASS_MAPPING
                ):
                    continue


                # =============================================
                # TARGET CLASS
                # =============================================

                target_class = (
                    CLASS_MAPPING[
                        original_class
                    ]
                )


                # =============================================
                # TIME
                # =============================================

                start_time = float(
                    row["start_time"]
                )

                end_time = float(
                    row["end_time"]
                )

                local_event_index += 1
                event_number += 1


                # =============================================
                # FILE NAME BARU
                # =============================================

                new_filename = (
                    f"tau_"
                    f"{recording_name}_"
                    f"{original_class}_"
                    f"{local_event_index:02d}.wav"
                )

                destination = (
                    OUTPUT_ROOT
                    / target_class
                    / new_filename
                )


                try:

                    info = extract_event(
                        audio_file,
                        start_time,
                        end_time,
                        destination
                    )

                    counts[
                        target_class
                    ] += 1


                    # =========================================
                    # SOURCE GROUP
                    #
                    # Semua event dari WAV yang sama
                    # harus tetap dalam split yang sama.
                    # =========================================

                    source_group = (
                        f"TAU_SEBIN_"
                        f"bin_prox_dir_one_"
                        f"{recording_name}"
                    )


                    manifest_rows.append({

                        "filename":
                            new_filename,

                        "dataset":
                            "TAU-SEBin-2021",

                        "subset":
                            "bin_prox_dir_one",

                        "source_recording":
                            f"{recording_name}.wav",

                        "source_group":
                            source_group,

                        "fold":
                            fold,

                        "original_class":
                            original_class,

                        "target_class":
                            target_class,

                        "start_time":
                            start_time,

                        "end_time":
                            end_time,

                        "duration_sec":
                            round(
                                info["duration"],
                                4
                            ),

                        "original_sample_rate":
                            info[
                                "sample_rate"
                            ],

                        "channels":
                            info[
                                "channels"
                            ],

                        "azi":
                            row.get(
                                "azi",
                                ""
                            ),

                        "ele":
                            row.get(
                                "ele",
                                ""
                            ),

                        "dist":
                            row.get(
                                "dist",
                                ""
                            ),
                    })


                except Exception as error:

                    failed += 1

                    print(
                        "ERROR:",
                        metadata_file.name,
                        original_class,
                        error
                    )


    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    fieldnames = [
        "filename",
        "dataset",
        "subset",
        "source_recording",
        "source_group",
        "fold",
        "original_class",
        "target_class",
        "start_time",
        "end_time",
        "duration_sec",
        "original_sample_rate",
        "channels",
        "azi",
        "ele",
        "dist",
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

    print()
    print("=" * 65)
    print("HASIL TAU-SEBIN")
    print("=" * 65)

    total = 0

    for class_name, count in (
        counts.items()
    ):

        print(
            f"{class_name:<20}: "
            f"{count} event"
        )

        total += count

    print("-" * 65)

    print(
        f"{'TOTAL':<20}: "
        f"{total} event"
    )

    print(
        f"{'GAGAL':<20}: "
        f"{failed}"
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