
from pathlib import Path
import csv
import wave
import zipfile

# ==================================================
# KONFIGURASI
# ==================================================

INPUT_ROOT = Path("dataset/raw/DEMAND_16k")

OUTPUT_ROOT = Path("results/robustness_v1")

DETAIL_CSV = OUTPUT_ROOT / "demand_audio_metadata.csv"

SUMMARY_CSV = OUTPUT_ROOT / "demand_scene_summary.csv"

TARGET_SR = 16000


# ==================================================
# MAIN
# ==================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {INPUT_ROOT}"
        )

    archives = sorted(
        INPUT_ROOT.glob("*_16k.zip")
    )

    details = []
    summaries = []

    print("=" * 70)
    print("AUDIT METADATA AUDIO DEMAND")
    print("=" * 70)

    for archive_path in archives:

        scene = archive_path.stem.replace(
            "_16k", ""
        )

        scene_rows = []

        with zipfile.ZipFile(archive_path, "r") as archive:

            wav_names = sorted(
                name for name in archive.namelist()
                if name.lower().endswith(".wav")
                and not name.startswith("__MACOSX/")
            )

            for name in wav_names:

                row = {
                    "scene": scene,
                    "archive": archive_path.name,
                    "wav_member": name,
                    "sample_rate": "",
                    "channels": "",
                    "bit_depth": "",
                    "duration_sec": "",
                    "num_frames": "",
                    "status": "ERROR"
                }

                try:

                    with archive.open(name) as audio_file:

                        with wave.open(audio_file, "rb") as wav:

                            sr = wav.getframerate()
                            channels = wav.getnchannels()
                            bit_depth = wav.getsampwidth() * 8
                            frames = wav.getnframes()

                    duration = frames / sr if sr > 0 else 0

                    row.update({
                        "sample_rate": sr,
                        "channels": channels,
                        "bit_depth": bit_depth,
                        "duration_sec": round(duration, 3),
                        "num_frames": frames,
                        "status": (
                            "OK"
                            if sr == TARGET_SR
                            and channels == 1
                            and frames > 0
                            else "REVIEW"
                        )
                    })

                except Exception as error:

                    row["status"] = f"ERROR: {error}"

                details.append(row)
                scene_rows.append(row)

        valid_rows = [
            row for row in scene_rows
            if row["status"] == "OK"
        ]

        durations = [
            row["duration_sec"]
            for row in valid_rows
        ]

        summary = {
            "scene": scene,
            "total_wav": len(scene_rows),
            "valid_wav": len(valid_rows),
            "duration_min_sec": (
                min(durations) if durations else 0
            ),
            "duration_max_sec": (
                max(durations) if durations else 0
            ),
            "example_member": (
                scene_rows[0]["wav_member"]
                if scene_rows else ""
            )
        }

        summaries.append(summary)

        print()
        print(f"LINGKUNGAN: {scene}")
        print(f"Jumlah WAV : {len(scene_rows)}")
        print(f"Valid      : {len(valid_rows)}")

        if valid_rows:
            first = valid_rows[0]

            print(f"Sample rate: {first['sample_rate']} Hz")
            print(f"Channel    : {first['channels']}")
            print(f"Bit depth  : {first['bit_depth']}")
            print(f"Durasi     : {first['duration_sec']} detik")
            print(f"Contoh file: {first['wav_member']}")

    # ==================================================
    # SAVE CSV
    # ==================================================

    if details:

        with open(
            DETAIL_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(details[0].keys())
            )

            writer.writeheader()
            writer.writerows(details)

    if summaries:

        with open(
            SUMMARY_CSV,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(summaries[0].keys())
            )

            writer.writeheader()
            writer.writerows(summaries)

    print()
    print("=" * 70)
    print("RINGKASAN")
    print("=" * 70)

    print("Jumlah lingkungan :", len(summaries))
    print("Total WAV         :", len(details))
    print(
        "WAV valid         :",
        sum(row["status"] == "OK" for row in details)
    )

    print()
    print("Metadata :", DETAIL_CSV)
    print("Ringkasan:", SUMMARY_CSV)

    print()
    print("Dataset lama tidak diubah.")


if __name__ == "__main__":
    main()
