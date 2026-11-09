
from pathlib import Path
from datetime import datetime
from math import gcd
from tempfile import TemporaryDirectory
import hashlib
import re
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    ROOT / "dataset/robustness_v3/raw_alarm_candidates"
)

OUTPUT_ROOT = (
    ROOT / "dataset/robustness_v3/prepared_alarm_candidates"
)

REPORT_ROOT = (
    ROOT / "results/robustness_v3/alarm_preparation"
)

TARGET_SR = 16000

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".ogg",
    ".oga",
    ".flac",
}


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def safe_filename(name):
    result = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name
    ).strip("_")

    return result[:60] or "audio"


def calculate_rms(audio):
    values = np.asarray(
        audio,
        dtype=np.float64
    )

    return float(
        np.sqrt(np.mean(values ** 2))
    )


def read_audio_with_fallback(path):
    try:
        audio, sr = sf.read(
            path,
            dtype="float32",
            always_2d=True
        )

        return {
            "audio": audio,
            "sample_rate": sr,
            "original_sample_rate": sr,
            "original_channels": audio.shape[1],
            "decoder": "soundfile",
        }

    except Exception as original_error:
        print(
            "SoundFile tidak mendukung audio ini."
        )
        print("Mencoba FFmpeg...")

        try:
            import imageio_ffmpeg

            ffmpeg_exe = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )

            with TemporaryDirectory() as temp_dir:
                temp_wav = (
                    Path(temp_dir) / "decoded.wav"
                )

                command = [
                    ffmpeg_exe,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(TARGET_SR),
                    "-c:a",
                    "pcm_s16le",
                    str(temp_wav),
                ]

                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False
                )

                if process.returncode != 0:
                    raise RuntimeError(
                        process.stderr.strip()
                        or "FFmpeg gagal membaca audio."
                    )

                audio, sr = sf.read(
                    temp_wav,
                    dtype="float32",
                    always_2d=True
                )

            return {
                "audio": audio,
                "sample_rate": sr,
                # Metadata sumber tidak ditebak.
                # FFmpeg sudah mengonversi ke
                # mono 16 kHz sebelum dibaca.
                "original_sample_rate": None,
                "original_channels": None,
                "decoder": "ffmpeg_fallback",
            }

        except Exception as ffmpeg_error:
            raise RuntimeError(
                "SoundFile gagal: "
                f"{original_error}; "
                "FFmpeg gagal: "
                f"{ffmpeg_error}"
            ) from ffmpeg_error


def prepare_audio(input_path, output_path):
    decoded = read_audio_with_fallback(
        input_path
    )

    audio = decoded["audio"]
    sr = decoded["sample_rate"]

    if audio.size == 0:
        raise ValueError("Audio kosong.")

    if sr <= 0:
        raise ValueError(
            "Sample rate tidak valid."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Audio mengandung NaN atau Infinity."
        )

    mono = audio.mean(axis=1)

    if sr != TARGET_SR:
        factor = gcd(
            sr,
            TARGET_SR
        )

        mono = resample_poly(
            mono,
            TARGET_SR // factor,
            sr // factor
        )

    mono = np.asarray(
        mono,
        dtype=np.float32
    )

    duration = len(mono) / TARGET_SR

    peak_before = float(
        np.max(np.abs(mono))
    )

    peak_adjusted = False

    if peak_before > 1.0:
        mono *= 0.99 / peak_before
        peak_adjusted = True

    rms = calculate_rms(mono)

    peak = float(
        np.max(np.abs(mono))
    )

    sf.write(
        output_path,
        mono,
        TARGET_SR,
        subtype="PCM_16"
    )

    warnings = []

    if duration < 1.455:
        warnings.append("SHORT_AUDIO")

    if rms < 0.008:
        warnings.append("LOW_RMS")

    if peak >= 0.99:
        warnings.append("POSSIBLE_CLIPPING")

    if peak_adjusted:
        warnings.append("PEAK_ADJUSTED")

    status = (
        "TECHNICAL_OK"
        if not warnings
        else "REVIEW_" + "_".join(warnings)
    )

    return {
        "original_sample_rate": decoded[
            "original_sample_rate"
        ],
        "original_channels": decoded[
            "original_channels"
        ],
        "decoder": decoded["decoder"],
        "prepared_sample_rate": TARGET_SR,
        "prepared_channels": 1,
        "duration_sec": round(
            duration, 3
        ),
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "status": status,
    }


def main():
    print("=" * 60)
    print("ROBUSTNESS V3 - ALARM PREPARATION")
    print("=" * 60)

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(INPUT_DIR)

    files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not files:
        print("Tidak ada file audio.")
        return

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )

    prepared_dir = OUTPUT_ROOT / run_id
    report_dir = REPORT_ROOT / run_id

    prepared_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    records = []
    failures = []

    for index, path in enumerate(
        files,
        start=1
    ):
        print(
            f"\n[{index}/{len(files)}]"
        )
        print("File:", path.name)

        try:
            sha256 = file_sha256(path)

            output_name = (
                f"alarm_{index:02d}_"
                f"{safe_filename(path.stem)}.wav"
            )

            output_path = (
                prepared_dir / output_name
            )

            info = prepare_audio(
                path,
                output_path
            )

            records.append({
                "path": str(output_path),
                "target_class": "alarm",
                "dataset_role": "candidate",
                "source_group": (
                    f"v3_alarm_{sha256[:16]}"
                ),
                "original_file": path.name,
                "original_path": str(path),
                "original_sha256": sha256,
                "source_url": "",
                "creator": "",
                "license": "",
                "attribution": "",
                "heard_sound_type": "",
                "suitable_for_target_alarm": "",
                "manual_review_notes": "",
                **info,
            })

            print(
                "Decoder:",
                info["decoder"]
            )
            print(
                "Durasi:",
                info["duration_sec"],
                "detik"
            )
            print(
                "RMS:",
                info["rms"]
            )
            print(
                "Peak:",
                info["peak"]
            )
            print(
                "Status:",
                info["status"]
            )

        except Exception as error:
            failures.append({
                "original_file": path.name,
                "error": str(error),
            })

            print("GAGAL:", error)

    report_path = (
        report_dir
        / "alarm_preparation_report.csv"
    )

    if records:
        pd.DataFrame(records).to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig"
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            report_dir / "failed_files.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("\n" + "=" * 60)
    print("ALARM PREPARATION SELESAI")
    print("=" * 60)

    print(
        "Total file ditemukan:",
        len(files)
    )
    print(
        "Berhasil dikonversi:",
        len(records)
    )
    print(
        "Gagal dikonversi:",
        len(failures)
    )
    print(
        "Output:",
        prepared_dir
    )

    if records:
        print(
            "Report:",
            report_path
        )

    print("\nCATATAN:")
    print("- Semua audio masih kandidat")
    print("- Belum ada train/test split")
    print("- Lisensi dan isi audio belum diaudit")
    print("- File asli tetap aman")
    print("- Model lama tidak diubah")
    print("- Belum dilakukan training")


if __name__ == "__main__":
    main()
