
from pathlib import Path
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent

MANIFEST = ROOT / "dataset/dataset_splits.csv"

OUTPUT = ROOT / (
    "results/robustness_v2/alarm_dataset_audit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

required = {
    "path",
    "target_class",
    "split",
    "source_group"
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Kolom tidak ditemukan: "
        f"{required - set(df.columns)}"
    )

alarm = df[
    df["target_class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("alarm")
].copy()

rows = []

for record in alarm.itertuples(index=False):

    path = Path(str(record.path))

    if not path.is_absolute():
        path = ROOT / path

    exists = path.is_file()

    duration = None
    sample_rate = None
    error = ""

    if exists:
        try:
            info = sf.info(str(path))
            duration = info.duration
            sample_rate = info.samplerate

        except Exception as exc:
            error = str(exc)

    rows.append({
        "split": record.split,
        "source_group": record.source_group,
        "audio_path": str(path),
        "file_exists": exists,
        "duration_sec": duration,
        "sample_rate": sample_rate,
        "shorter_than_two_windows": (
            duration < 1.455
            if duration is not None
            else None
        ),
        "read_error": error
    })

report = pd.DataFrame(rows)

report.to_csv(
    OUTPUT / "alarm_dataset_audit.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 65)
print("AUDIT DATASET KELAS ALARM")
print("=" * 65)

print("\nTOTAL AUDIO ALARM PER SPLIT:")

print(
    report.groupby("split")
    .size()
    .to_string()
)

print("\nSUMBER AUDIO ALARM:")

print(
    report.groupby(["split", "source_group"])
    .size()
    .to_string()
)

print("\nSTATISTIK DURASI:")

print(
    report.groupby("split")["duration_sec"]
    .agg(["count", "min", "median", "max"])
    .to_string()
)

print("\nAUDIO TERLALU PENDEK UNTUK 2 WINDOW:")

print(
    report.groupby("split")[
        "shorter_than_two_windows"
    ].sum().to_string()
)

print("\nCONTOH FILE UNTUK DIPERIKSA:")

samples = (
    report.sort_values(
        ["split", "source_group", "audio_path"]
    )
    .groupby("split")
    .head(5)
)

for row in samples.itertuples(index=False):
    print()
    print("Split :", row.split)
    print("Sumber:", row.source_group)
    print("File  :", row.audio_path)

print("\nFile CSV disimpan di:")
print(OUTPUT / "alarm_dataset_audit.csv")
print("\nTidak ada dataset atau model yang diubah.")
