
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import math
import re

import numpy as np
import sounddevice as sd
import soundfile as sf

from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / (
    "dataset/robustness_v2/raw_hard_negative"
)

MANIFEST = ROOT / (
    "dataset/robustness_v2/"
    "hard_negative_recordings.csv"
)

TARGET_SR = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680
RMS_THRESHOLD = 0.008


def safe_name(text):

    cleaned = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        text
    )

    if not cleaned.strip("_"):
        raise ValueError("Nama tidak valid.")

    return cleaned


def check_session(session, split):

    if not MANIFEST.exists():
        return

    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        for row in csv.DictReader(file):

            if (
                row["session_id"] == session
                and row["split"] != split
            ):
                raise ValueError(
                    "Session yang sama tidak boleh "
                    "berada di training dan test."
                )


def calculate_activity(waveform):

    total = 0
    active = 0

    if len(waveform) < WINDOW_SIZE:
        return 0, 0, 0.0

    for start in range(
        0,
        len(waveform) - WINDOW_SIZE + 1,
        HOP_SIZE
    ):

        window = waveform[
            start:start + WINDOW_SIZE
        ].astype(np.float64)

        rms = float(
            np.sqrt(
                np.mean(window * window)
            )
        )

        total += 1

        if rms >= RMS_THRESHOLD:
            active += 1

    percent = (
        100 * active / total
        if total
        else 0
    )

    return total, active, percent


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=str
    )

    parser.add_argument(
        "--session",
        type=str
    )

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train"
    )

    parser.add_argument(
        "--minutes",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None
    )

    parser.add_argument(
        "--list-devices",
        action="store_true"
    )

    args = parser.parse_args()

    if args.list_devices:

        print(sd.query_devices())
        return

    if not args.scene or not args.session:

        parser.error(
            "--scene dan --session wajib diisi."
        )

    if not (0.1 <= args.minutes <= 15):

        parser.error(
            "--minutes harus 0.1 sampai 15."
        )

    scene = safe_name(args.scene)
    session = safe_name(args.session)

    check_session(
        session,
        args.split
    )

    device_info = sd.query_devices(
        args.device,
        "input"
    )

    native_sr = int(
        round(
            device_info["default_samplerate"]
        )
    )

    if native_sr <= 0:
        raise ValueError(
            "Sample rate mikrofon tidak valid."
        )

    sd.check_input_settings(
        device=args.device,
        channels=1,
        samplerate=native_sr
    )

    seconds = args.minutes * 60
    frames = int(
        round(
            seconds * native_sr
        )
    )

    print("=" * 65)
    print("DANGER SOUND AI - HARD NEGATIVE RECORDER")
    print("=" * 65)

    print("Scene       :", scene)
    print("Session     :", session)
    print("Split       :", args.split)
    print("Durasi      :", seconds, "detik")
    print("Mikrofon    :", device_info["name"])
    print("Sample rate :", native_sr)

    print()
    print(
        "Pastikan rekaman hanya berisi "
        "suara background."
    )
    print(
        "Jangan sengaja memasukkan "
        "klakson, sirene, alarm, "
        "teriakan bahaya, atau kaca pecah."
    )

    input(
        "\nTekan ENTER untuk mulai merekam..."
    )

    print()
    print("PEREKAMAN DIMULAI")
    print("Jangan tutup terminal.")

    recording = sd.rec(
        frames,
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        device=args.device,
        blocking=True
    )

    waveform = recording[:, 0]

    print()
    print("PEREKAMAN SELESAI")

    # Samakan sample rate dengan Flutter.
    if native_sr != TARGET_SR:

        divisor = math.gcd(
            native_sr,
            TARGET_SR
        )

        waveform = resample_poly(
            waveform,
            TARGET_SR // divisor,
            native_sr // divisor
        )

    waveform = np.asarray(
        waveform,
        dtype=np.float32
    )

    peak = float(
        np.max(np.abs(waveform))
    )

    overall_rms = float(
        np.sqrt(
            np.mean(
                waveform.astype(np.float64) ** 2
            )
        )
    )

    total, active, percent = (
        calculate_activity(waveform)
    )

    print()
    print("=" * 65)
    print("AUDIT HASIL REKAMAN")
    print("=" * 65)

    print("Peak             :", round(peak, 6))
    print("RMS              :", round(overall_rms, 6))
    print("Total window     :", total)
    print("Window aktif     :", active)
    print("Persentase aktif :", round(percent, 2), "%")

    if percent < 10:

        print()
        print(
            "PERINGATAN: Sedikit window yang "
            "melewati RMS threshold."
        )
        print(
            "Periksa volume speaker, jarak "
            "mikrofon, dan perangkat input."
        )

    print()
    print(
        "Apakah selama perekaman benar-benar "
        "tidak terdengar suara bahaya?"
    )

    print(
        "Ketik y jika yakin hanya background."
    )
    print(
        "Ketik n jika mengandung bahaya "
        "atau masih ragu."
    )

    answer = input(
        "Konfirmasi (y/n): "
    ).strip().lower()

    verified = answer == "y"

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    recording_id = (
        f"{scene}_{session}_{timestamp}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    wav_path = (
        OUTPUT_DIR / f"{recording_id}.wav"
    )

    json_path = (
        OUTPUT_DIR / f"{recording_id}.json"
    )

    sf.write(
        str(wav_path),
        waveform,
        TARGET_SR,
        subtype="PCM_16"
    )

    label_status = (
        "background_confirmed_by_recorder"
        if verified
        else "needs_review"
    )

    metadata = {
        "recording_id": recording_id,
        "scene": scene,
        "session_id": session,
        "source_group": f"MIC_{session}",
        "split": args.split,
        "sample_rate": TARGET_SR,
        "duration_sec": len(waveform) / TARGET_SR,
        "total_windows": total,
        "active_windows": active,
        "active_percent": percent,
        "peak": peak,
        "rms": overall_rms,
        "label_status": label_status,
        "verified_background": verified,
        "audio_path": str(
            wav_path.relative_to(ROOT)
        ).replace("\\", "/"),
        "device_name": device_info["name"]
    }

    json_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = list(
        metadata.keys()
    )

    file_exists = MANIFEST.exists()

    with MANIFEST.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(metadata)

    print()
    print("=" * 65)
    print("REKAMAN BERHASIL DISIMPAN")
    print("=" * 65)

    print("Audio    :", wav_path)
    print("Metadata :", json_path)
    print("Manifest :", MANIFEST)
    print("Status   :", label_status)

    if not verified:
        print(
            "Rekaman ini harus ditinjau "
            "sebelum digunakan untuk training."
        )

    print()
    print(
        "Model lama dan Robustness V1 "
        "tidak diubah."
    )


if __name__ == "__main__":
    main()
