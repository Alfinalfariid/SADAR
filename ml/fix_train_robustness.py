
from pathlib import Path
from datetime import datetime
import shutil


# ==================================================
# KONFIGURASI
# ==================================================

ROOT = Path(__file__).resolve().parent

TARGET_FILE = ROOT / "train_robustness_v1.py"

PATCH_MARKER = "FIX_SOURCE_GROUP_MERGE_V1"


# ==================================================
# KODE YANG BERMASALAH
# ==================================================

OLD_CODE = '''    active = active.merge(
        train_segments,
        on=["segment_id", "scene"],
        how="inner",
        validate="many_to_one"
    )
'''


# ==================================================
# KODE PERBAIKAN
# ==================================================

NEW_CODE = '''    # FIX_SOURCE_GROUP_MERGE_V1
    #
    # source_group dari dataset window dan
    # manifest split memiliki nama yang sama.
    #
    # Gunakan source_group dari manifest split
    # sebagai metadata utama.

    active = active.drop(
        columns=["source_group"],
        errors="ignore"
    )

    active = active.merge(
        train_segments,
        on=["segment_id", "scene"],
        how="inner",
        validate="many_to_one"
    )

    # Pastikan metadata source_group tetap tersedia.

    if "source_group" not in active.columns:
        raise ValueError(
            "source_group tidak ditemukan setelah merge."
        )

    if active["source_group"].isna().any():
        raise ValueError(
            "Ada source_group kosong setelah merge."
        )
'''


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 65)
    print("PERBAIKAN TRAIN ROBUSTNESS V1")
    print("=" * 65)

    if not TARGET_FILE.is_file():
        raise FileNotFoundError(
            f"File tidak ditemukan: {TARGET_FILE}"
        )

    original = TARGET_FILE.read_text(
        encoding="utf-8"
    )

    if PATCH_MARKER in original:
        print()
        print("Perbaikan sebelumnya sudah terpasang.")
        print("Tidak ada perubahan tambahan.")
        return

    matches = original.count(OLD_CODE)

    if matches != 1:
        raise ValueError(
            f"Blok kode ditemukan {matches} kali. "
            "File tidak diubah karena format "
            "tidak sesuai dengan versi yang diperiksa."
        )

    # ==============================================
    # BACKUP SCRIPT SEBELUM DIUBAH
    # ==============================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_file = ROOT / (
        f"train_robustness_v1_backup_{timestamp}.py"
    )

    shutil.copy2(
        TARGET_FILE,
        backup_file
    )

    # ==============================================
    # PERBAIKI BLOK MERGE
    # ==============================================

    updated = original.replace(
        OLD_CODE,
        NEW_CODE,
        1
    )

    compile(
        updated,
        str(TARGET_FILE),
        "exec"
    )

    TARGET_FILE.write_text(
        updated,
        encoding="utf-8"
    )

    print()
    print("PERBAIKAN BERHASIL")
    print("-" * 65)

    print("Backup disimpan:")
    print(backup_file)

    print()
    print("Script diperbaiki:")
    print(TARGET_FILE)

    print()
    print("Masalah source_group_x/source_group_y")
    print("sudah ditangani.")

    print()
    print("Dataset, model, dan hasil baseline")
    print("tidak diubah.")


if __name__ == "__main__":
    main()
