
from pathlib import Path
import csv
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub


# ======================================================
# KONFIGURASI
# ======================================================

INPUT_DIR = Path(
    "dataset/robustness_v1/demand_review"
)

OUTPUT_DIR = Path(
    "results/robustness_v1"
)

DETAIL_CSV = (
    OUTPUT_DIR / "demand_auto_screen.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "demand_auto_screen_summary.csv"
)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

# Threshold eksplorasi, bukan label final.
SCREEN_THRESHOLD = 0.08

# Hanya dicatat sebagai informasi.
# Sama dengan threshold RMS di Flutter.
REFERENCE_RMS = 0.008

SCENES = [
    "NPARK",
    "OMEETING",
    "OOFFICE",
    "PCAFETER",
    "PSTATION",
    "SPSQUARE",
    "STRAFFIC",
    "TBUS",
    "TCAR",
    "TMETRO",
]

# Kata kunci untuk menemukan kelas target
# pada class map YAMNet pretrained.
TARGET_KEYWORDS = {
    "alarm": [
        "alarm",
        "buzzer",
        "beep",
        "bell",
    ],
    "glass_breaking": [
        "glass",
        "shatter",
        "breaking",
        "smash",
    ],
    "scream": [
        "scream",
        "shout",
        "yell",
        "bellow",
    ],
    "siren": [
        "siren",
    ],
    "vehicle_horn": [
        "horn",
        "honking",
        "honk",
        "klaxon",
    ],
}


# ======================================================
# LOAD CLASS MAP
# ======================================================

def load_class_names(model):

    path = model.class_map_path().numpy()

    if isinstance(path, bytes):
        path = path.decode("utf-8")

    with tf.io.gfile.GFile(path, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    rows.sort(key=lambda row: int(row["index"]))

    names = [
        row["display_name"]
        for row in rows
    ]

    if len(names) != 521:
        raise ValueError(
            f"Jumlah kelas YAMNet tidak sesuai: {len(names)}"
        )

    return names


# ======================================================
# TARGET INDICES
# ======================================================

def build_target_indices(class_names):

    target_indices = {}

    print()
    print("PEMETAAN KELAS TARGET YAMNET")
    print("-" * 65)

    for target, keywords in TARGET_KEYWORDS.items():

        indices = [
            i
            for i, name in enumerate(class_names)
            if any(
                keyword in name.lower()
                for keyword in keywords
            )
        ]

        if not indices:
            raise ValueError(
                f"Kelas YAMNet untuk {target} tidak ditemukan."
            )

        target_indices[target] = indices

        matched_names = [
            class_names[i]
            for i in indices
        ]

        print(
            f"{target}: {', '.join(matched_names)}"
        )

    return target_indices


# ======================================================
# AUDIO RMS
# ======================================================

def calculate_rms(audio):

    x = audio.astype(np.float64)

    return float(
        np.sqrt(np.mean(x * x))
    )


# ======================================================
# PROCESS AUDIO SEGMENT
# ======================================================

def analyze_segment(
    model,
    audio,
    class_names,
    target_indices
):

    waveform = tf.convert_to_tensor(
        audio,
        dtype=tf.float32
    )

    scores, embeddings, spectrogram = model(
        waveform
    )

    frame_scores = scores.numpy()

    if frame_scores.ndim != 2:
        raise ValueError(
            f"Bentuk output YAMNet salah: {frame_scores.shape}"
        )

    # Skor tertinggi pada salah satu frame.
    group_scores = {}

    for target, indices in target_indices.items():

        group_scores[target] = float(
            np.max(
                frame_scores[:, indices]
            )
        )

    top_position = np.unravel_index(
        np.argmax(frame_scores),
        frame_scores.shape
    )

    top_index = int(top_position[1])

    top_event = class_names[top_index]

    top_score = float(
        frame_scores[top_position]
    )

    max_target_score = max(
        group_scores.values()
    )

    if max_target_score >= SCREEN_THRESHOLD:
        status = "REVIEW_TARGET_POSSIBLE"
    else:
        status = "CANDIDATE_BACKGROUND_UNVERIFIED"

    return {
        "status": status,
        "top_event": top_event,
        "top_event_score": top_score,
        "max_target_score": max_target_score,
        "target_scores": group_scores,
        "yamnet_frames": int(frame_scores.shape[0]),
    }


# ======================================================
# SAVE CSV
# ======================================================

def save_csv(path, rows):

    if not rows:
        raise ValueError(
            f"Tidak ada data untuk disimpan: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
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


# ======================================================
# MAIN
# ======================================================

def main():

    print("=" * 70)
    print("DANGER SOUND AI - DEMAND AUTOMATIC SCREENING")
    print("=" * 70)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {INPUT_DIR}"
        )

    for scene in SCENES:

        path = INPUT_DIR / f"{scene}_ch01.wav"

        if not path.exists():
            raise FileNotFoundError(
                f"Audio tidak ditemukan: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Loading pretrained YAMNet...")

    model = hub.load(
        YAMNET_URL
    )

    class_names = load_class_names(
        model
    )

    target_indices = build_target_indices(
        class_names
    )

    detail_rows = []
    summary_rows = []

    # ==================================================
    # PROCESS PER LINGKUNGAN
    # ==================================================

    for scene in SCENES:

        audio_path = (
            INPUT_DIR / f"{scene}_ch01.wav"
        )

        print()
        print("=" * 70)
        print(f"MEMPROSES: {scene}")
        print("=" * 70)

        scene_rows = []

        with sf.SoundFile(str(audio_path)) as audio_file:

            if audio_file.samplerate != SAMPLE_RATE:
                raise ValueError(
                    f"{scene}: sample rate tidak sesuai."
                )

            if audio_file.channels != 1:
                raise ValueError(
                    f"{scene}: audio harus mono."
                )

            # Sisa audio kurang dari 10 detik
            # tidak diproses pada tahap ini.
            total_segments = (
                len(audio_file) // SEGMENT_SAMPLES
            )

            for index in range(total_segments):

                audio = audio_file.read(
                    frames=SEGMENT_SAMPLES,
                    dtype="float32",
                    always_2d=False
                )

                if len(audio) != SEGMENT_SAMPLES:
                    raise ValueError(
                        f"{scene} segmen {index + 1} tidak lengkap."
                    )

                start_sec = index * SEGMENT_SECONDS
                end_sec = start_sec + SEGMENT_SECONDS

                rms = calculate_rms(
                    audio
                )

                prediction = analyze_segment(
                    model,
                    audio,
                    class_names,
                    target_indices
                )

                row = {
                    "segment_id": f"{scene}_{index + 1:03d}",
                    "scene": scene,
                    "source_group": f"DEMAND_{scene}",
                    "audio_path": str(audio_path),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "status": prediction["status"],
                    "rms": round(rms, 6),
                    "below_reference_rms": rms < REFERENCE_RMS,
                    "yamnet_frames": prediction["yamnet_frames"],
                    "top_yamnet_event": prediction["top_event"],
                    "top_yamnet_score": round(
                        prediction["top_event_score"], 5
                    ),
                    "max_target_score": round(
                        prediction["max_target_score"], 5
                    ),
                }

                for target, score in prediction[
                    "target_scores"
                ].items():

                    row[f"score_{target}"] = round(
                        score, 5
                    )

                detail_rows.append(row)
                scene_rows.append(row)

                print(
                    f"[{index + 1:02d}/{total_segments}] "
                    f"{prediction['status']} | "
                    f"max_target="
                    f"{prediction['max_target_score']:.3f}"
                )

        candidates = sum(
            row["status"] == "CANDIDATE_BACKGROUND_UNVERIFIED"
            for row in scene_rows
        )

        review_needed = sum(
            row["status"] == "REVIEW_TARGET_POSSIBLE"
            for row in scene_rows
        )

        summary_rows.append({
            "scene": scene,
            "source_group": f"DEMAND_{scene}",
            "segments": len(scene_rows),
            "candidate_background": candidates,
            "review_target_possible": review_needed,
            "screen_threshold": SCREEN_THRESHOLD,
        })

        print()
        print(
            f"{scene}: kandidat={candidates}, "
            f"perlu_review={review_needed}"
        )

    # ==================================================
    # SAVE RESULTS
    # ==================================================

    save_csv(
        DETAIL_CSV,
        detail_rows
    )

    save_csv(
        SUMMARY_CSV,
        summary_rows
    )

    total_candidates = sum(
        row["candidate_background"]
        for row in summary_rows
    )

    total_review = sum(
        row["review_target_possible"]
        for row in summary_rows
    )

    print()
    print("=" * 70)
    print("RINGKASAN PENYARINGAN DEMAND")
    print("=" * 70)

    print("Lingkungan          :", len(summary_rows))
    print("Total segmen        :", len(detail_rows))
    print("Kandidat background :", total_candidates)
    print("Perlu pemeriksaan   :", total_review)
    print("Threshold screening :", SCREEN_THRESHOLD)

    print()
    print("Detail:")
    print(DETAIL_CSV)

    print()
    print("Ringkasan:")
    print(SUMMARY_CSV)

    print()
    print(
        "CATATAN: Kandidat background belum "
        "merupakan label ground truth."
    )

    print(
        "Dataset dan model baseline tidak diubah."
    )


if __name__ == "__main__":
    main()
