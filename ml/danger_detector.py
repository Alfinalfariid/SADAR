from pathlib import Path
import time

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter


# ============================================================
# PATH
# ============================================================

MODEL_PATH = Path("models/yamnet.tflite")
LABEL_PATH = Path("models/yamnet_521_labels.txt")
AUDIO_PATH = Path("audio/siren.wav")


# ============================================================
# KONFIGURASI AUDIO YAMNET
# ============================================================

TARGET_SR = 16000

# 0.975 detik x 16000 Hz
FRAME_SIZE = 15600

# Geser sekitar 0.5 detik
HOP_SIZE = 8000


# ============================================================
# THRESHOLD SEMENTARA
# ============================================================

# Ini hanya untuk development.
# Jangan dianggap threshold final penelitian.
DANGER_THRESHOLD = 0.30


# ============================================================
# MAPPING LABEL YAMNET -> KATEGORI BAHAYA
# ============================================================

DANGER_CLASSES = {
    "Siren": [
        "Siren",
        "Civil defense siren",
        "Police car (siren)",
        "Ambulance (siren)",
        "Fire engine, fire truck (siren)",
        "Emergency vehicle",
    ],

    "Fire Alarm": [
        "Fire alarm",
        "Smoke detector, smoke alarm",
    ],

    "Vehicle Horn": [
        "Vehicle horn, car horn, honking",
        "Air horn, truck horn",
    ],

    "Scream": [
        "Screaming",
        "Scream",
        "Shout",
    ],

    "Glass Breaking": [
        "Glass",
        "Shatter",
    ],
}


# ============================================================
# LOAD LABEL
# ============================================================

def load_labels():

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"File label tidak ditemukan: {LABEL_PATH}"
        )

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        labels = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return labels


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File audio tidak ditemukan: {path}"
        )

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32"
    )

    print()
    print("Informasi audio")
    print("-----------------------------")
    print("Sample rate asli :", sample_rate)
    print("Audio shape      :", audio.shape)

    # --------------------------------------------------------
    # Stereo / multi-channel -> mono
    # --------------------------------------------------------

    if audio.ndim == 2:

        print(
            "Mengubah audio stereo/multi-channel -> mono"
        )

        audio = np.mean(
            audio,
            axis=1
        )

    elif audio.ndim != 1:

        raise ValueError(
            "Format channel audio tidak didukung."
        )

    audio = audio.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Resample ke 16 kHz
    # --------------------------------------------------------

    if sample_rate != TARGET_SR:

        print(
            f"Resampling {sample_rate} Hz "
            f"-> {TARGET_SR} Hz"
        )

        old_length = len(audio)

        new_length = int(
            old_length
            * TARGET_SR
            / sample_rate
        )

        old_positions = np.linspace(
            0,
            1,
            old_length,
            endpoint=False
        )

        new_positions = np.linspace(
            0,
            1,
            new_length,
            endpoint=False
        )

        audio = np.interp(
            new_positions,
            old_positions,
            audio
        ).astype(
            np.float32
        )

    print("Jumlah sample    :", len(audio))

    return audio


# ============================================================
# POTONG AUDIO MENJADI FRAME
# ============================================================

def create_frames(audio):

    # Kalau audio lebih pendek dari 15600 sample,
    # tambahkan silence.
    if len(audio) < FRAME_SIZE:

        audio = np.pad(
            audio,
            (
                0,
                FRAME_SIZE - len(audio)
            )
        )

    frames = []

    start = 0

    while start + FRAME_SIZE <= len(audio):

        frame = audio[
            start:start + FRAME_SIZE
        ]

        frames.append(
            frame.astype(
                np.float32
            )
        )

        start += HOP_SIZE

    # Safety
    if len(frames) == 0:

        frames.append(
            np.zeros(
                FRAME_SIZE,
                dtype=np.float32
            )
        )

    return frames


# ============================================================
# SETUP YAMNET
# ============================================================

def load_yamnet():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    print("Loading YAMNet...")

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        num_threads=2
    )

    # --------------------------------------------------------
    # Ambil info input SEBELUM resize
    # --------------------------------------------------------

    input_info = (
        interpreter
        .get_input_details()[0]
    )

    print()
    print("Model sebelum resize")
    print("-----------------------------")

    print(
        "Input shape          :",
        input_info["shape"]
    )

    print(
        "Input shape signature:",
        input_info["shape_signature"]
    )

    print(
        "Input dtype          :",
        input_info["dtype"]
    )

    # --------------------------------------------------------
    # Model ini menggunakan waveform 1D dinamis.
    #
    # Awalnya bisa terlihat:
    # [1]
    #
    # Dengan signature:
    # [-1]
    #
    # Resize menjadi:
    # [15600]
    # --------------------------------------------------------

    interpreter.resize_tensor_input(
        input_info["index"],
        [FRAME_SIZE],
        strict=False
    )

    # Wajib allocate setelah resize
    interpreter.allocate_tensors()

    # Ambil info tensor lagi
    input_info = (
        interpreter
        .get_input_details()[0]
    )

    output_info = (
        interpreter
        .get_output_details()[0]
    )

    print()
    print("Model setelah resize")
    print("-----------------------------")

    print(
        "Input shape :",
        input_info["shape"]
    )

    print(
        "Input dtype :",
        input_info["dtype"]
    )

    print(
        "Output shape:",
        output_info["shape"]
    )

    print(
        "Output dtype:",
        output_info["dtype"]
    )

    return (
        interpreter,
        input_info,
        output_info
    )


# ============================================================
# INFERENCE YAMNET
# ============================================================

def get_yamnet_scores(
    interpreter,
    input_info,
    output_info,
    frames
):

    all_scores = []

    total_inference_time = 0.0

    for frame_number, frame in enumerate(
        frames,
        start=1
    ):

        # ----------------------------------------------------
        # YAMNet model ini meminta input rank-1:
        #
        # [15600]
        #
        # JANGAN pakai expand_dims.
        # ----------------------------------------------------

        input_frame = frame.astype(
            input_info["dtype"]
        )

        start_time = time.perf_counter()

        interpreter.set_tensor(
            input_info["index"],
            input_frame
        )

        interpreter.invoke()

        end_time = time.perf_counter()

        inference_time = (
            end_time - start_time
        ) * 1000

        total_inference_time += (
            inference_time
        )

        output = interpreter.get_tensor(
            output_info["index"]
        )

        # ----------------------------------------------------
        # Output biasanya [1, 521]
        # ----------------------------------------------------

        if output.ndim == 2:
            scores = output[0]

        else:
            scores = output

        all_scores.append(
            scores.astype(
                np.float32
            )
        )

    all_scores = np.array(
        all_scores
    )

    # Ambil skor tertinggi setiap kelas
    # dari seluruh frame.
    final_scores = np.max(
        all_scores,
        axis=0
    )

    average_latency = (
        total_inference_time
        / len(frames)
    )

    return (
        final_scores,
        average_latency
    )


# ============================================================
# TOP PREDICTION
# ============================================================

def show_top_predictions(
    labels,
    scores,
    top_k=10
):

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    print()
    print("==============================================")
    print("TOP PREDICTION YAMNET")
    print("==============================================")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        label = labels[index]

        score = float(
            scores[index]
        )

        print(
            f"{rank:2}. "
            f"{label:<40} "
            f"{score:.4f}"
        )


# ============================================================
# DETEKSI BAHAYA
# ============================================================

def detect_danger(
    labels,
    scores
):

    best_category = None
    best_yamnet_label = None
    best_score = 0.0

    for index, label in enumerate(
        labels
    ):

        score = float(
            scores[index]
        )

        for category, yamnet_labels in (
            DANGER_CLASSES.items()
        ):

            if label in yamnet_labels:

                if score > best_score:

                    best_score = score
                    best_category = category
                    best_yamnet_label = label

    if (
        best_category is not None
        and best_score >= DANGER_THRESHOLD
    ):

        return {
            "danger": True,
            "category": best_category,
            "yamnet_label": best_yamnet_label,
            "confidence": best_score,
        }

    return {
        "danger": False,
        "category": "Normal",
        "yamnet_label": best_yamnet_label,
        "confidence": best_score,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print("DANGER SOUND AI - YAMNET TEST")
    print("==============================================")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        interpreter,
        input_info,
        output_info
    ) = load_yamnet()

    # --------------------------------------------------------
    # Load label
    # --------------------------------------------------------

    labels = load_labels()

    print()
    print(
        "Jumlah label:",
        len(labels)
    )

    if len(labels) != 521:

        print(
            "WARNING: Jumlah label bukan 521!"
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    print()
    print(
        "Loading audio:",
        AUDIO_PATH
    )

    audio = load_audio(
        AUDIO_PATH
    )

    # --------------------------------------------------------
    # Create frame
    # --------------------------------------------------------

    frames = create_frames(
        audio
    )

    print()
    print(
        "Jumlah frame:",
        len(frames)
    )

    print(
        "Frame size  :",
        FRAME_SIZE
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print(
        "Menjalankan inference..."
    )

    (
        scores,
        latency_ms
    ) = get_yamnet_scores(
        interpreter,
        input_info,
        output_info,
        frames
    )

    # --------------------------------------------------------
    # Top prediction
    # --------------------------------------------------------

    show_top_predictions(
        labels,
        scores,
        top_k=10
    )

    # --------------------------------------------------------
    # Danger filter
    # --------------------------------------------------------

    result = detect_danger(
        labels,
        scores
    )

    # --------------------------------------------------------
    # Output final
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("HASIL DETEKSI BAHAYA")
    print("==============================================")

    if result["danger"]:

        print(
            "STATUS       : BAHAYA"
        )

        print(
            "CATEGORY     :",
            result["category"]
        )

        print(
            "YAMNET LABEL :",
            result["yamnet_label"]
        )

        print(
            "SCORE        :",
            f'{result["confidence"]:.4f}'
        )

    else:

        print(
            "STATUS       : NORMAL"
        )

        print(
            "BEST LABEL   :",
            result["yamnet_label"]
        )

        print(
            "BEST SCORE   :",
            f'{result["confidence"]:.4f}'
        )

        print(
            "THRESHOLD    :",
            DANGER_THRESHOLD
        )

    print(
        "AVG LATENCY  :",
        f"{latency_ms:.2f} ms/frame"
    )

    print()
    print("==============================================")
    print("SELESAI")
    print("==============================================")
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()