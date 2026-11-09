import pandas as pd


FILE = "dataset/master_standardized_manifest.csv"


df = pd.read_csv(FILE)


print()
print("=" * 60)
print("JUMLAH FILE PER KELAS")
print("=" * 60)

print(
    df.groupby(
        "target_class"
    ).size()
)


print()
print("=" * 60)
print("JUMLAH SOURCE UNIK PER KELAS")
print("=" * 60)

print(
    df.groupby(
        "target_class"
    )["source_group"]
    .nunique()
)


print()
print("=" * 60)
print("DATASET × KELAS")
print("=" * 60)

print(
    df.groupby(
        [
            "dataset",
            "target_class"
        ]
    ).size()
)