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
    "dataset/processed/alarm"
)

MANIFEST_FILE = Path(
    "dataset/tau_alarm_manifest.csv"
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
        return int(match.group(1))

    return ""


# ============================================================
# VALIDASI
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

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# EXTRACT EVENT
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
        start_time * sample_rate
    )

    end_sample = int(
        end_time * sample_rate
    )

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

    sf.write(
        str(destination),
        event_audio,
        sample_rate,
        subtype="PCM_16"
    )

    return {
        "sample_rate": sample_rate,
        "duration": len(event_audio) / sample_rate,
        "samples": len(event_audio),
        "channels": event_audio.shape[1],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("PREPARE TAU-SEBIN - ALARM")
    print("=" * 65)

    validate_paths()

    metadata_files = sorted(
        METADATA_ROOT.glob("*.csv")
    )

    print()
    print(
        "Jumlah metadata:",
        len(metadata_files)
    )

    manifest_rows = []

    success = 0
    failed = 0


    for metadata_file in metadata_files:

        recording_name = metadata_file.stem

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

            reader = csv.DictReader(file)

            alarm_index = 0

            for row in reader:

                original_class = (
                    row[
                        "sound_event_recording"
                    ]
                    .strip()
                    .lower()
                )

                if original_class != "alarm":
                    continue

                alarm_index += 1

                start_time = float(
                    row["start_time"]
                )

                end_time = float(
                    row["end_time"]
                )

                new_filename = (
                    f"tau_"
                    f"{recording_name}_"
                    f"alarm_"
                    f"{alarm_index:02d}.wav"
                )

                destination = (
                    OUTPUT_ROOT
                    / new_filename
                )

                try:

                    info = extract_event(
                        audio_file,
                        start_time,
                        end_time,
                        destination
                    )

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
                            "alarm",

                        "target_class":
                            "alarm",

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
                            info["sample_rate"],

                        "channels":
                            info["channels"],

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

                    success += 1

                except Exception as error:

                    failed += 1

                    print(
                        "ERROR:",
                        metadata_file.name,
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


    print()
    print("=" * 65)
    print("HASIL ALARM TAU-SEBIN")
    print("=" * 65)

    print(
        f"alarm               : "
        f"{success} event"
    )

    print(
        f"gagal               : "
        f"{failed} event"
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