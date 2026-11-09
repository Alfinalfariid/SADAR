from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIG
# ============================================================

INPUT_MANIFEST = Path(
    "dataset/master_standardized_manifest.csv"
)

OUTPUT_MANIFEST = Path(
    "dataset/dataset_splits.csv"
)

SUMMARY_FILE = Path(
    "dataset/split_summary.csv"
)

RANDOM_STATE = 42

# 20 fold:
#
# 14 fold = train      = 70%
# 3 fold  = validation = 15%
# 3 fold  = test       = 15%

N_SPLITS = 20


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest tidak ditemukan:\n"
            f"{INPUT_MANIFEST}"
        )

    df = pd.read_csv(
        INPUT_MANIFEST
    )

    required_columns = [
        "filename",
        "target_class",
        "source_group",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Kolom tidak ditemukan: "
                f"{column}"
            )

    if df[
        "source_group"
    ].isna().any():

        raise ValueError(
            "Ada source_group kosong."
        )

    return df


# ============================================================
# BUAT 20 STRATIFIED GROUP FOLD
# ============================================================

def create_cv_folds(df):

    print()
    print(
        "Membuat StratifiedGroupKFold..."
    )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    df["cv_fold"] = -1

    x_dummy = df.index

    y = df[
        "target_class"
    ]

    groups = df[
        "source_group"
    ]

    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        splitter.split(
            x_dummy,
            y,
            groups
        )
    ):

        df.loc[
            test_index,
            "cv_fold"
        ] = fold_number


    if (
        df["cv_fold"] == -1
    ).any():

        raise RuntimeError(
            "Ada data yang tidak mendapat fold."
        )

    return df


# ============================================================
# 20 FOLD -> TRAIN / VAL / TEST
# ============================================================

def assign_final_split(df):

    def fold_to_split(
        fold
    ):

        # Fold 0 - 13
        if 0 <= fold <= 13:

            return "train"

        # Fold 14 - 16
        elif 14 <= fold <= 16:

            return "validation"

        # Fold 17 - 19
        else:

            return "test"


    df["split"] = (
        df["cv_fold"]
        .apply(
            fold_to_split
        )
    )

    return df


# ============================================================
# CHECK DATA LEAKAGE
# ============================================================

def check_leakage(df):

    split_per_group = (
        df.groupby(
            "source_group"
        )["split"]
        .nunique()
    )

    leaked = (
        split_per_group[
            split_per_group > 1
        ]
    )

    print()
    print("=" * 72)
    print("DATA LEAKAGE CHECK")
    print("=" * 72)

    if len(leaked) > 0:

        print(
            "DATA LEAKAGE TERDETEKSI!"
        )

        print(
            leaked
        )

        raise RuntimeError(
            "Source group masuk "
            "lebih dari satu split."
        )

    print(
        "AMAN - tidak ada source_group "
        "yang muncul di lebih dari satu split."
    )


# ============================================================
# CEK SETIAP KELAS ADA DI SEMUA SPLIT
# ============================================================

def validate_classes(df):

    classes = sorted(
        df[
            "target_class"
        ].unique()
    )

    splits = [
        "train",
        "validation",
        "test",
    ]

    for class_name in classes:

        for split in splits:

            count = len(
                df[
                    (
                        df[
                            "target_class"
                        ]
                        == class_name
                    )
                    &
                    (
                        df[
                            "split"
                        ]
                        == split
                    )
                ]
            )

            if count == 0:

                raise RuntimeError(
                    f"Kelas {class_name} "
                    f"tidak memiliki data "
                    f"di {split}."
                )


# ============================================================
# SUMMARY PER KELAS
# ============================================================

def show_class_distribution(
    df
):

    classes = sorted(
        df[
            "target_class"
        ].unique()
    )

    print()
    print("=" * 90)
    print("DISTRIBUSI FILE PER KELAS")
    print("=" * 90)

    print(
        f"{'Class':<20}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
        f"{'Total':>10}"
        f"{'Train %':>10}"
        f"{'Val %':>10}"
        f"{'Test %':>10}"
    )

    print(
        "-" * 90
    )

    summary_rows = []

    for class_name in classes:

        class_df = df[
            df[
                "target_class"
            ]
            == class_name
        ]

        total = len(
            class_df
        )

        train_count = len(
            class_df[
                class_df[
                    "split"
                ]
                == "train"
            ]
        )

        val_count = len(
            class_df[
                class_df[
                    "split"
                ]
                == "validation"
            ]
        )

        test_count = len(
            class_df[
                class_df[
                    "split"
                ]
                == "test"
            ]
        )

        train_pct = (
            train_count
            / total
            * 100
        )

        val_pct = (
            val_count
            / total
            * 100
        )

        test_pct = (
            test_count
            / total
            * 100
        )

        print(
            f"{class_name:<20}"
            f"{train_count:>10}"
            f"{val_count:>10}"
            f"{test_count:>10}"
            f"{total:>10}"
            f"{train_pct:>9.2f}%"
            f"{val_pct:>9.2f}%"
            f"{test_pct:>9.2f}%"
        )

        summary_rows.append({

            "target_class":
                class_name,

            "train":
                train_count,

            "validation":
                val_count,

            "test":
                test_count,

            "total":
                total,

            "train_percent":
                round(
                    train_pct,
                    2
                ),

            "validation_percent":
                round(
                    val_pct,
                    2
                ),

            "test_percent":
                round(
                    test_pct,
                    2
                ),
        })


    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def show_global_summary(df):

    print()
    print("=" * 72)
    print("TOTAL FILE PER SPLIT")
    print("=" * 72)

    total = len(
        df
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        count = len(
            df[
                df["split"]
                == split
            ]
        )

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"{split:<15}: "
            f"{count:4} "
            f"({percentage:.2f}%)"
        )


    print()
    print("=" * 72)
    print("SOURCE GROUP PER SPLIT")
    print("=" * 72)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        group_count = (
            df[
                df["split"]
                == split
            ][
                "source_group"
            ]
            .nunique()
        )

        print(
            f"{split:<15}: "
            f"{group_count}"
        )


# ============================================================
# SAVE
# ============================================================

def save_result(df):

    df.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    print()
    print("=" * 72)
    print("SPLIT SELESAI")
    print("=" * 72)

    print()
    print(
        "Dataset split:"
    )

    print(
        OUTPUT_MANIFEST
    )

    print()

    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("STRATIFIED GROUP DATASET SPLIT")
    print("=" * 72)

    print()
    print(
        "Target:"
    )

    print(
        "Train      : 70%"
    )

    print(
        "Validation : 15%"
    )

    print(
        "Test       : 15%"
    )

    print(
        f"Random seed: "
        f"{RANDOM_STATE}"
    )


    df = load_data()

    print()
    print(
        "Jumlah file:",
        len(df)
    )

    print(
        "Source group:",
        df[
            "source_group"
        ].nunique()
    )

    print(
        "Jumlah kelas:",
        df[
            "target_class"
        ].nunique()
    )


    df = create_cv_folds(
        df
    )

    df = assign_final_split(
        df
    )


    check_leakage(
        df
    )

    validate_classes(
        df
    )

    show_class_distribution(
        df
    )

    show_global_summary(
        df
    )

    save_result(
        df
    )


if __name__ == "__main__":
    main()