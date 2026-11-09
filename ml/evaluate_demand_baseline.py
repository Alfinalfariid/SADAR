
from pathlib import Path
import hashlib
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf


# ============================================================
# KONFIGURASI
# ============================================================

AUDIO_ROOT = Path(
    "dataset/robustness_v1/demand_review"
)

WINDOWS_CSV = Path(
    "results/robustness_v1/demand_window_rms.csv"
)

OUTPUT_DIR = Path(
    "results/robustness_v1/baseline_demand"
)

SAMPLE_RATE = 16000
WINDOW_SIZE = 15600
HOP_SIZE = 7680
RMS_THRESHOLD = 0.008

CLASSES = [
    "alarm",
    "background",
    "glass_breaking",
    "scream",
    "siren",
    "vehicle_horn",
]

THRESHOLDS = {
    "alarm": 0.70,
    "glass_breaking": 0.80,
    "scream": 0.70,
    "siren": 0.70,
    "vehicle_horn": 0.70,
}

CONFIRMATIONS = {
    "alarm": 2,
    "glass_breaking": 1,
    "scream": 2,
    "siren": 2,
    "vehicle_horn": 2,
}


# ============================================================
# CARI MODEL TFLITE
# ============================================================

def find_models():

    folders = [
        Path("../mobile/assets/models"),
        Path("models"),
    ]

    for folder in folders:

        yamnet = (
            folder /
            "yamnet_embedding_dynamic_int8.tflite"
        )

        classifier = (
            folder /
            "danger_classifier_fp32.tflite"
        )

        if yamnet.is_file() and classifier.is_file():

            print()
            print("Folder model:", folder)

            return yamnet, classifier

    raise FileNotFoundError(
        "Pasangan model YAMNet dan classifier "
        "tidak ditemukan di folder yang sama."
    )


# ============================================================
# HASH MODEL
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# LOAD TFLITE DENGAN RESIZE INPUT DINAMIS
# ============================================================

def load_interpreter(
    path,
    expected_input_shape,
    expected_output_size
):

    print()
    print("-" * 65)
    print("LOAD MODEL:", path.name)
    print("-" * 65)

    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=2
    )

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    if len(inputs) != 1:
        raise ValueError(
            f"{path.name}: jumlah input bukan 1."
        )

    if len(outputs) != 1:
        raise ValueError(
            f"{path.name}: jumlah output bukan 1."
        )

    input_detail = inputs[0]

    original_shape = list(
        input_detail["shape"]
    )

    print("Input awal     :", original_shape)

    print(
        "Shape signature:",
        list(
            input_detail["shape_signature"]
        )
    )

    print(
        "Input target   :",
        expected_input_shape
    )

    # ========================================================
    # PERBAIKAN UTAMA
    #
    # YAMNet memiliki input dinamis.
    # Bentuk awal bisa [1].
    #
    # Ubah menjadi [15600] SEBELUM
    # melakukan allocate_tensors().
    # ========================================================

    if original_shape != expected_input_shape:

        interpreter.resize_tensor_input(
            input_detail["index"],
            expected_input_shape,
            strict=False
        )

        print(
            "Input berhasil di-resize."
        )

    interpreter.allocate_tensors()

    # Ambil ulang detail setelah alokasi.
    input_detail = (
        interpreter.get_input_details()[0]
    )

    output_detail = (
        interpreter.get_output_details()[0]
    )

    actual_input_shape = list(
        input_detail["shape"]
    )

    actual_output_shape = list(
        output_detail["shape"]
    )

    print(
        "Input final    :",
        actual_input_shape
    )

    print(
        "Output final   :",
        actual_output_shape
    )

    print(
        "Input dtype    :",
        input_detail["dtype"]
    )

    print(
        "Output dtype   :",
        output_detail["dtype"]
    )

    if actual_input_shape != expected_input_shape:

        raise ValueError(
            f"{path.name}: input masih salah. "
            f"{actual_input_shape}"
        )

    if input_detail["dtype"] != np.float32:

        raise ValueError(
            f"{path.name}: input bukan float32."
        )

    if output_detail["dtype"] != np.float32:

        raise ValueError(
            f"{path.name}: output bukan float32."
        )

    total_output = int(
        np.prod(actual_output_shape)
    )

    if total_output != expected_output_size:

        raise ValueError(
            f"{path.name}: output memiliki "
            f"{total_output} elemen, diharapkan "
            f"{expected_output_size}."
        )

    print(
        "MODEL SIAP DIGUNAKAN."
    )

    return interpreter


# ============================================================
# JALANKAN MODEL
# ============================================================

def run_model(
    interpreter,
    values,
    expected_output_size
):

    input_detail = (
        interpreter.get_input_details()[0]
    )

    output_detail = (
        interpreter.get_output_details()[0]
    )

    input_shape = tuple(
        int(value)
        for value in input_detail["shape"]
    )

    data = np.asarray(
        values,
        dtype=np.float32
    )

    if data.size != int(np.prod(input_shape)):

        raise ValueError(
            f"Input tidak sesuai: "
            f"{data.size} elemen untuk "
            f"tensor {input_shape}."
        )

    data = data.reshape(
        input_shape
    )

    interpreter.set_tensor(
        input_detail["index"],
        data
    )

    interpreter.invoke()

    result = interpreter.get_tensor(
        output_detail["index"]
    )

    result = np.asarray(
        result,
        dtype=np.float32
    ).reshape(-1)

    if result.size != expected_output_size:

        raise ValueError(
            "Jumlah output tidak sesuai: "
            f"{result.size}"
        )

    if not np.all(np.isfinite(result)):

        raise ValueError(
            "Output model mengandung NaN atau Inf."
        )

    return result


# ============================================================
# SIMULASI KONFIRMASI
#
# Dilakukan per segmen 10 detik.
# Belum menyimulasikan latch 2,5 detik,
# maupun kontinuitas antarsegmen.
# ============================================================

def calculate_confirmations(df):

    df["confirmation_candidate"] = False

    for _, part in df.groupby(
        "segment_id",
        sort=False
    ):

        last_label = None
        consecutive = 0

        for idx, row in part.iterrows():

            if not row["passes_threshold"]:

                last_label = None
                consecutive = 0

                continue

            label = row["top1_label"]

            if label == last_label:

                consecutive += 1

            else:

                last_label = label
                consecutive = 1

            required = CONFIRMATIONS.get(
                label,
                2
            )

            if consecutive == required:

                df.at[
                    idx,
                    "confirmation_candidate"
                ] = True

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DANGER SOUND AI")
    print("BASELINE TFLITE TEST ON DEMAND")
    print("=" * 70)

    if not WINDOWS_CSV.exists():

        raise FileNotFoundError(
            WINDOWS_CSV
        )

    df = pd.read_csv(
        WINDOWS_CSV
    )

    required_columns = [
        "scene",
        "source_group",
        "segment_id",
        "start_sec",
        "rms",
        "activity",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Kolom CSV hilang: {missing}"
        )

    df = df.sort_values(
        ["scene", "start_sec"]
    ).reset_index(drop=True)

    # ========================================================
    # LOAD MODEL
    # ========================================================

    yamnet_path, classifier_path = (
        find_models()
    )

    yamnet = load_interpreter(
        yamnet_path,
        expected_input_shape=[WINDOW_SIZE],
        expected_output_size=1024
    )

    classifier = load_interpreter(
        classifier_path,
        expected_input_shape=[1, 1024],
        expected_output_size=len(CLASSES)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    # ========================================================
    # PROSES PER LINGKUNGAN
    # ========================================================

    for scene, group in df.groupby(
        "scene",
        sort=True
    ):

        audio_path = (
            AUDIO_ROOT /
            f"{scene}_ch01.wav"
        )

        if not audio_path.is_file():

            raise FileNotFoundError(
                audio_path
            )

        active_count = 0
        below_count = 0

        print()
        print("-" * 65)
        print("PROSES:", scene)
        print("-" * 65)

        with sf.SoundFile(
            str(audio_path)
        ) as audio_file:

            if audio_file.samplerate != SAMPLE_RATE:

                raise ValueError(
                    f"Sample rate salah: {scene}"
                )

            if audio_file.channels != 1:

                raise ValueError(
                    f"Audio tidak mono: {scene}"
                )

            for item in group.itertuples(
                index=False
            ):

                rms = float(
                    item.rms
                )

                active = (
                    item.activity == "ACTIVE"
                )

                probabilities = np.zeros(
                    len(CLASSES),
                    dtype=np.float32
                )

                if not active:

                    # Sama seperti silence filter:
                    # tidak perlu inference model.
                    below_count += 1

                    top1_label = "background"
                    top1_score = 1.0

                    probabilities[
                        CLASSES.index("background")
                    ] = 1.0

                else:

                    active_count += 1

                    start_sample = int(
                        round(
                            float(item.start_sec)
                            * SAMPLE_RATE
                        )
                    )

                    audio_file.seek(
                        start_sample
                    )

                    waveform = audio_file.read(
                        frames=WINDOW_SIZE,
                        dtype="float32",
                        always_2d=False
                    )

                    if len(waveform) != WINDOW_SIZE:

                        raise ValueError(
                            f"Window tidak lengkap: "
                            f"{scene}, {item.start_sec}"
                        )

                    # ====================================
                    # YAMNET DYNAMIC INT8
                    # ====================================

                    embedding = run_model(
                        yamnet,
                        waveform,
                        expected_output_size=1024
                    )

                    # ====================================
                    # CLASSIFIER FP32
                    # ====================================

                    probabilities = run_model(
                        classifier,
                        embedding,
                        expected_output_size=len(CLASSES)
                    )

                    winner = int(
                        np.argmax(probabilities)
                    )

                    top1_label = CLASSES[winner]

                    top1_score = float(
                        probabilities[winner]
                    )

                # ========================================
                # THRESHOLD PER KELAS
                # ========================================

                threshold = THRESHOLDS.get(
                    top1_label
                )

                passes_threshold = (
                    active
                    and threshold is not None
                    and top1_score >= threshold
                )

                result_row = {
                    "scene": scene,
                    "source_group": item.source_group,
                    "segment_id": item.segment_id,
                    "start_sec": float(item.start_sec),
                    "rms": rms,
                    "activity": item.activity,
                    "inference_performed": active,
                    "top1_label": top1_label,
                    "top1_score": round(
                        top1_score, 6
                    ),
                    "passes_threshold": bool(
                        passes_threshold
                    ),
                }

                for i, name in enumerate(CLASSES):

                    result_row[
                        f"prob_{name}"
                    ] = round(
                        float(probabilities[i]),
                        6
                    )

                rows.append(
                    result_row
                )

        print("Window aktif  :", active_count)
        print("Window rendah :", below_count)

    # ========================================================
    # HASIL PREDIKSI
    # ========================================================

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        raise ValueError(
            "Tidak ada hasil pengujian."
        )

    result = calculate_confirmations(
        result
    )

    # ========================================================
    # REKAP PER LINGKUNGAN
    # ========================================================

    summary_rows = []

    for scene, part in result.groupby(
        "scene",
        sort=True
    ):

        horn = (
            part["top1_label"] == "vehicle_horn"
        )

        summary_rows.append({
            "scene": scene,
            "total_windows": len(part),
            "active_windows": int(
                part["inference_performed"].sum()
            ),
            "background_windows": int(
                (
                    part["top1_label"] == "background"
                ).sum()
            ),
            "horn_top1": int(
                horn.sum()
            ),
            "horn_above_threshold": int(
                (
                    horn &
                    part["passes_threshold"]
                ).sum()
            ),
            "all_danger_above_threshold": int(
                part["passes_threshold"].sum()
            ),
            "confirmation_candidates": int(
                part["confirmation_candidate"].sum()
            ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # SIMPAN HASIL
    # ========================================================

    predictions_file = (
        OUTPUT_DIR / "window_predictions.csv"
    )

    summary_file = (
        OUTPUT_DIR / "scene_summary.csv"
    )

    config_file = (
        OUTPUT_DIR / "experiment_config.json"
    )

    result.to_csv(
        predictions_file,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "experiment": "baseline_demand_v1",
        "purpose": (
            "Diagnostic inference on DEMAND "
            "pseudo-background candidates"
        ),
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "rms_threshold": RMS_THRESHOLD,
        "classes": CLASSES,
        "thresholds": THRESHOLDS,
        "confirmations": CONFIRMATIONS,
        "yamnet_model": str(
            yamnet_path.resolve()
        ),
        "classifier_model": str(
            classifier_path.resolve()
        ),
        "yamnet_sha256": sha256_file(
            yamnet_path
        ),
        "classifier_sha256": sha256_file(
            classifier_path
        ),
        "notes": [
            "DEMAND candidates are not verified ground truth.",
            "Input YAMNet resized to 15600 samples.",
            "Non-active windows included as background.",
            "Confirmation resets between 10-second segments.",
            "No full 2.5-second latch simulation.",
            "Selected windows are overlapping and correlated.",
            "Dataset and baseline model are unchanged."
        ]
    }

    with open(
        config_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    # ========================================================
    # RINGKASAN
    # ========================================================

    print()
    print("=" * 70)
    print("HASIL MODEL BASELINE DI DEMAND")
    print("=" * 70)

    print()
    print(
        summary_df.to_string(index=False)
    )

    print()
    print("Total window:", len(result))

    print(
        "Window inference:",
        int(
            result["inference_performed"].sum()
        )
    )

    print(
        "Window tanpa inference:",
        int(
            (~result["inference_performed"]).sum()
        )
    )

    print(
        "Prediksi bahaya melewati threshold:",
        int(
            result["passes_threshold"].sum()
        )
    )

    print(
        "Kandidat konfirmasi bahaya:",
        int(
            result["confirmation_candidate"].sum()
        )
    )

    print()
    print("HASIL DISIMPAN:")
    print(predictions_file)
    print(summary_file)
    print(config_file)

    print()
    print(
        "CATATAN: Hasil ini bukan false positive "
        "terverifikasi tanpa ground truth."
    )

    print(
        "Model, TFLite dan dataset baseline "
        "tidak dimodifikasi."
    )


if __name__ == "__main__":
    main()
