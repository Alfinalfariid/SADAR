from pathlib import Path
import csv
import math
import shutil

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# KONFIGURASI
# ============================================================

INPUT_ROOT = Path(
    "dataset/processed"
)

OUTPUT_ROOT = Path(
    "dataset/standardized"
)

OUTPUT_MANIFEST = Path(
    "dataset/master_standardized_manifest.csv"
)

TARGET_SAMPLE_RATE = 16000


CLASSES = [
    "siren",
    "alarm",
    "vehicle_horn",
    "scream",
    "glass_breaking",
    "background",
]


# ============================================================
# MANIFEST SUMBER
# ============================================================

ESC50_MANIFEST = Path(
    "dataset/esc50_manifest.csv"
)

URBANSOUND_MANIFEST = Path(
    "dataset/urbansound8k_manifest.csv"
)

TAU_GLASS_MANIFEST = Path(
    "dataset/tau_sebin_manifest.csv"
)

TAU_ALARM_MANIFEST = Path(
    "dataset/tau_alarm_manifest.csv"
)

NIGENS_MANIFEST = Path(
    "dataset/nigens_manifest.csv"
)


# ============================================================
# BUAT OUTPUT BERSIH
# ============================================================

def prepare_output():

    # standardized adalah hasil turunan,
    # jadi aman dibuat ulang.
    if OUTPUT_ROOT.exists():

        print(
            "Menghapus standardized lama..."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for class_name in CLASSES:

        (
            OUTPUT_ROOT
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    metadata = {}

    # --------------------------------------------------------
    # ESC-50
    # --------------------------------------------------------

    if ESC50_MANIFEST.exists():

        with open(
            ESC50_MANIFEST,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                filename = row[
                    "filename"
                ]

                source_file = row.get(
                    "source_file",
                    ""
                )

                if source_file:

                    source_group = (
                        f"ESC50_{source_file}"
                    )

                else:

                    source_group = (
                        f"ESC50_{filename}"
                    )

                metadata[
                    filename
                ] = {

                    "dataset":
                        "ESC-50",

                    "original_class":
                        row.get(
                            "original_category",
                            ""
                        ),

                    "target_class":
                        row.get(
                            "target_class",
                            ""
                        ),

                    "fold":
                        row.get(
                            "fold",
                            ""
                        ),

                    "source_group":
                        source_group,
                }


    # --------------------------------------------------------
    # URBANSOUND8K
    # --------------------------------------------------------

    if URBANSOUND_MANIFEST.exists():

        with open(
            URBANSOUND_MANIFEST,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                filename = row[
                    "filename"
                ]

                fsid = row.get(
                    "fsID",
                    filename
                )

                metadata[
                    filename
                ] = {

                    "dataset":
                        "UrbanSound8K",

                    "original_class":
                        row.get(
                            "original_class",
                            ""
                        ),

                    "target_class":
                        row.get(
                            "target_class",
                            ""
                        ),

                    "fold":
                        row.get(
                            "fold",
                            ""
                        ),

                    "source_group":
                        f"US8K_{fsid}",
                }


    # --------------------------------------------------------
    # TAU-SEBIN GLASS BREAKING
    # --------------------------------------------------------

    if TAU_GLASS_MANIFEST.exists():

        with open(
            TAU_GLASS_MANIFEST,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                filename = row[
                    "filename"
                ]

                metadata[
                    filename
                ] = {

                    "dataset":
                        "TAU-SEBin-2021",

                    "original_class":
                        row.get(
                            "original_class",
                            ""
                        ),

                    "target_class":
                        row.get(
                            "target_class",
                            ""
                        ),

                    "fold":
                        row.get(
                            "fold",
                            ""
                        ),

                    "source_group":
                        row.get(
                            "source_group",
                            filename
                        ),
                }


    # --------------------------------------------------------
    # TAU-SEBIN ALARM
    # --------------------------------------------------------

    if TAU_ALARM_MANIFEST.exists():

        with open(
            TAU_ALARM_MANIFEST,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                filename = row[
                    "filename"
                ]

                metadata[
                    filename
                ] = {

                    "dataset":
                        "TAU-SEBin-2021",

                    "original_class":
                        "alarm",

                    "target_class":
                        "alarm",

                    "fold":
                        row.get(
                            "fold",
                            ""
                        ),

                    "source_group":
                        row.get(
                            "source_group",
                            filename
                        ),
                }


    # --------------------------------------------------------
    # NIGENS
    # --------------------------------------------------------

    if NIGENS_MANIFEST.exists():

        with open(
            NIGENS_MANIFEST,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                filename = row[
                    "filename"
                ]

                metadata[
                    filename
                ] = {

                    "dataset":
                        "NIGENS",

                    "original_class":
                        row.get(
                            "original_class",
                            ""
                        ),

                    "target_class":
                        "scream",

                    "fold":
                        "",

                    "source_group":
                        row.get(
                            "source_group",
                            filename
                        ),
                }

    return metadata


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False
    )

    # --------------------------------------------------------
    # STEREO / MULTICHANNEL -> MONO
    # --------------------------------------------------------

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1
        )

    elif audio.ndim != 1:

        raise ValueError(
            f"Dimensi audio tidak valid: "
            f"{audio.shape}"
        )

    return (
        audio.astype(
            np.float32
        ),
        sample_rate
    )


# ============================================================
# RESAMPLE
# ============================================================

def resample_audio(
    audio,
    original_sample_rate
):

    if (
        original_sample_rate
        == TARGET_SAMPLE_RATE
    ):

        return audio.astype(
            np.float32
        )

    divisor = math.gcd(
        original_sample_rate,
        TARGET_SAMPLE_RATE
    )

    up = (
        TARGET_SAMPLE_RATE
        // divisor
    )

    down = (
        original_sample_rate
        // divisor
    )

    result = resample_poly(
        audio,
        up,
        down
    )

    return result.astype(
        np.float32
    )


# ============================================================
# NORMALISASI SAFETY
# ============================================================

def normalize_audio(audio):

    if len(audio) == 0:

        raise ValueError(
            "Audio kosong."
        )

    peak = np.max(
        np.abs(audio)
    )

    # Hanya cegah clipping.
    # Tidak membuat semua audio memiliki volume sama.
    if peak > 1.0:

        audio = (
            audio
            / peak
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# PROCESS SATU FILE
# ============================================================

def process_file(
    source,
    destination
):

    audio, original_sr = (
        load_audio(
            source
        )
    )

    original_duration = (
        len(audio)
        / original_sr
    )

    audio = resample_audio(
        audio,
        original_sr
    )

    audio = normalize_audio(
        audio
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        str(destination),
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )

    return {

        "original_sample_rate":
            original_sr,

        "target_sample_rate":
            TARGET_SAMPLE_RATE,

        "duration_sec":
            len(audio)
            / TARGET_SAMPLE_RATE,

        "num_samples":
            len(audio),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FINAL DATASET PREPROCESSING")
    print("=" * 70)

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            f"Folder tidak ditemukan: "
            f"{INPUT_ROOT}"
        )

    prepare_output()

    metadata_lookup = (
        load_metadata()
    )

    print()
    print(
        "Metadata terdaftar:",
        len(metadata_lookup)
    )

    results = []

    class_counts = {
        class_name: 0
        for class_name in CLASSES
    }

    dataset_counts = {}

    unknown_files = []

    success = 0
    failed = 0


    # ========================================================
    # LOOP CLASS
    # ========================================================

    for class_name in CLASSES:

        source_directory = (
            INPUT_ROOT
            / class_name
        )

        if not source_directory.exists():

            print()
            print(
                f"{class_name}: "
                f"folder tidak ada"
            )

            continue

        files = sorted(
            source_directory.glob(
                "*.wav"
            )
        )

        print()
        print(
            f"{class_name:<20}: "
            f"{len(files)} file"
        )

        for source_file in files:

            destination = (
                OUTPUT_ROOT
                / class_name
                / source_file.name
            )

            try:

                info = process_file(
                    source_file,
                    destination
                )

                metadata = (
                    metadata_lookup.get(
                        source_file.name
                    )
                )

                if metadata is None:

                    dataset_name = (
                        "UNKNOWN"
                    )

                    original_class = ""

                    fold = ""

                    source_group = (
                        source_file.name
                    )

                    unknown_files.append(
                        source_file.name
                    )

                else:

                    dataset_name = (
                        metadata[
                            "dataset"
                        ]
                    )

                    original_class = (
                        metadata[
                            "original_class"
                        ]
                    )

                    fold = metadata[
                        "fold"
                    ]

                    source_group = (
                        metadata[
                            "source_group"
                        ]
                    )


                results.append({

                    "filename":
                        source_file.name,

                    "dataset":
                        dataset_name,

                    "original_class":
                        original_class,

                    "target_class":
                        class_name,

                    "fold":
                        fold,

                    "source_group":
                        source_group,

                    "original_sample_rate":
                        info[
                            "original_sample_rate"
                        ],

                    "target_sample_rate":
                        info[
                            "target_sample_rate"
                        ],

                    "duration_sec":
                        round(
                            info[
                                "duration_sec"
                            ],
                            4
                        ),

                    "num_samples":
                        info[
                            "num_samples"
                        ],

                    "path":
                        str(destination),
                })


                class_counts[
                    class_name
                ] += 1


                dataset_counts[
                    dataset_name
                ] = (
                    dataset_counts.get(
                        dataset_name,
                        0
                    )
                    + 1
                )

                success += 1


            except Exception as error:

                failed += 1

                print(
                    f"ERROR "
                    f"{source_file.name}: "
                    f"{error}"
                )


    # ========================================================
    # SAVE MASTER MANIFEST
    # ========================================================

    fieldnames = [
        "filename",
        "dataset",
        "original_class",
        "target_class",
        "fold",
        "source_group",
        "original_sample_rate",
        "target_sample_rate",
        "duration_sec",
        "num_samples",
        "path",
    ]

    with open(
        OUTPUT_MANIFEST,
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
            results
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("HASIL FINAL STANDARDISASI")
    print("=" * 70)

    print()
    print("JUMLAH PER KELAS")
    print("-" * 40)

    total = 0

    for class_name in CLASSES:

        count = class_counts[
            class_name
        ]

        print(
            f"{class_name:<20}: "
            f"{count}"
        )

        total += count

    print("-" * 40)

    print(
        f"{'TOTAL':<20}: "
        f"{total}"
    )


    print()
    print("JUMLAH PER DATASET")
    print("-" * 40)

    for dataset_name, count in sorted(
        dataset_counts.items()
    ):

        print(
            f"{dataset_name:<20}: "
            f"{count}"
        )


    print()
    print(
        f"Berhasil             : "
        f"{success}"
    )

    print(
        f"Gagal                : "
        f"{failed}"
    )

    print(
        f"Metadata UNKNOWN     : "
        f"{len(unknown_files)}"
    )

    print(
        f"Sample rate akhir    : "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    print(
        "Channel akhir        : mono"
    )

    print()

    print(
        "Master manifest:"
    )

    print(
        OUTPUT_MANIFEST
    )


    if unknown_files:

        print()
        print(
            "PERINGATAN:"
        )

        print(
            "Ada file yang tidak ditemukan "
            "di manifest."
        )

        for filename in (
            unknown_files[:20]
        ):

            print(
                " -",
                filename
            )

    print()


if __name__ == "__main__":
    main()