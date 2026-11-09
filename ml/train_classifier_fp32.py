from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIG
# ============================================================

SEED = 42

EMBEDDING_FILE = Path(
    "dataset/embeddings/yamnet_embeddings.npz"
)

MODEL_DIR = Path(
    "models"
)

RESULT_DIR = Path(
    "results"
)

BEST_MODEL_FILE = (
    MODEL_DIR
    / "danger_classifier_fp32_best.keras"
)

TFLITE_FP32_FILE = (
    MODEL_DIR
    / "danger_classifier_fp32.tflite"
)

METRICS_FILE = (
    RESULT_DIR
    / "fp32_metrics.json"
)

REPORT_FILE = (
    RESULT_DIR
    / "fp32_classification_report.csv"
)

CONFUSION_FILE = (
    RESULT_DIR
    / "fp32_confusion_matrix.csv"
)

HISTORY_FILE = (
    RESULT_DIR
    / "fp32_training_history.csv"
)

EPOCHS = 100
BATCH_SIZE = 32


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not EMBEDDING_FILE.exists():

        raise FileNotFoundError(
            f"Embedding tidak ditemukan:\n"
            f"{EMBEDDING_FILE}"
        )

    data = np.load(
        EMBEDDING_FILE,
        allow_pickle=True
    )

    X = data[
        "embeddings"
    ].astype(np.float32)

    y = data[
        "labels"
    ].astype(np.int64)

    splits = data[
        "splits"
    ].astype(str)

    class_names = data[
        "class_names"
    ].astype(str)

    return (
        X,
        y,
        splits,
        class_names,
    )


# ============================================================
# SPLIT DATA
# ============================================================

def split_dataset(
    X,
    y,
    splits
):

    train_mask = (
        splits == "train"
    )

    val_mask = (
        splits == "validation"
    )

    test_mask = (
        splits == "test"
    )

    X_train = X[
        train_mask
    ]

    y_train = y[
        train_mask
    ]

    X_val = X[
        val_mask
    ]

    y_val = y[
        val_mask
    ]

    X_test = X[
        test_mask
    ]

    y_test = y[
        test_mask
    ]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

def calculate_class_weights(
    y_train
):

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(class_id):
            float(weight)

        for class_id, weight
        in zip(
            classes,
            weights
        )
    }

    return class_weights


# ============================================================
# MODEL
# ============================================================

def build_model(
    input_dim,
    num_classes
):

    inputs = tf.keras.Input(
        shape=(input_dim,),
        name="yamnet_embedding",
    )

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=
            tf.keras.regularizers.l2(
                1e-4
            ),
        name="dense_128",
    )(inputs)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="sound_class",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="danger_sound_classifier",
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ],
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    class_names
):

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(
            len(class_names)
        ),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(
            len(class_names)
        ),
    )

    metrics = {

        "accuracy":
            float(accuracy),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "test_samples":
            int(len(y_test)),
    }

    return (
        metrics,
        report,
        matrix,
        predictions,
    )


# ============================================================
# EXPORT TFLITE FP32
# ============================================================

def export_tflite(
    model
):

    print()
    print(
        "Export FP32 TFLite..."
    )

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(
            model
        )
    )

    tflite_model = (
        converter.convert()
    )

    with open(
        TFLITE_FP32_FILE,
        "wb"
    ) as file:

        file.write(
            tflite_model
        )

    return len(
        tflite_model
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("TRAIN FP32 DANGER SOUND CLASSIFIER")
    print("=" * 75)


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD
    # ========================================================

    (
        X,
        y,
        splits,
        class_names,
    ) = load_dataset()


    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = split_dataset(
        X,
        y,
        splits
    )


    print()
    print(
        "Embedding dim :",
        X.shape[1]
    )

    print(
        "Jumlah kelas  :",
        len(class_names)
    )

    print()
    print(
        "Train         :",
        len(X_train)
    )

    print(
        "Validation    :",
        len(X_val)
    )

    print(
        "Test          :",
        len(X_test)
    )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print()
    print("=" * 75)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 75)

    for class_id, class_name in enumerate(
        class_names
    ):

        count = np.sum(
            y_train == class_id
        )

        print(
            f"{class_id} "
            f"{class_name:<20}: "
            f"{count}"
        )


    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    class_weights = (
        calculate_class_weights(
            y_train
        )
    )

    print()
    print("=" * 75)
    print("CLASS WEIGHT")
    print("=" * 75)

    for class_id, weight in (
        class_weights.items()
    ):

        print(
            f"{class_id} "
            f"{class_names[class_id]:<20}: "
            f"{weight:.4f}"
        )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = build_model(
        input_dim=X.shape[1],
        num_classes=len(
            class_names
        ),
    )

    print()
    print("=" * 75)
    print("MODEL")
    print("=" * 75)

    model.summary()

    print()
    print(
        "Total parameter:",
        model.count_params()
    )


    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=12,

            restore_best_weights=True,

            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=1e-6,

            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(
                BEST_MODEL_FILE
            ),

            monitor="val_loss",

            save_best_only=True,

            verbose=1,
        ),
    ]


    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 75)
    print("TRAINING")
    print("=" * 75)

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=
            class_weights,

        callbacks=
            callbacks,

        verbose=1,
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history.history
    )

    history_df.index.name = (
        "epoch"
    )

    history_df.to_csv(
        HISTORY_FILE
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model = tf.keras.models.load_model(
        BEST_MODEL_FILE
    )


    # ========================================================
    # TEST
    # ========================================================

    print()
    print("=" * 75)
    print("TEST SET EVALUATION")
    print("=" * 75)

    (
        metrics,
        report,
        matrix,
        predictions,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
    )


    print()
    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1     : "
        f"{metrics['weighted_f1']:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 75)
    print("CLASSIFICATION REPORT")
    print("=" * 75)

    text_report = classification_report(

        y_test,
        predictions,

        labels=np.arange(
            len(class_names)
        ),

        target_names=
            class_names,

        zero_division=0,
    )

    print()
    print(
        text_report
    )


    report_df = (
        pd.DataFrame(
            report
        )
        .transpose()
    )

    report_df.to_csv(
        REPORT_FILE
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix_df = pd.DataFrame(

        matrix,

        index=[
            f"actual_{name}"
            for name in class_names
        ],

        columns=[
            f"pred_{name}"
            for name in class_names
        ],
    )

    matrix_df.to_csv(
        CONFUSION_FILE
    )

    print()
    print("=" * 75)
    print("CONFUSION MATRIX")
    print("=" * 75)

    print()
    print(
        matrix_df
    )


    # ========================================================
    # EXPORT FP32 TFLITE
    # ========================================================

    tflite_size_bytes = (
        export_tflite(
            model
        )
    )

    tflite_size_kb = (
        tflite_size_bytes
        / 1024
    )

    tflite_size_mb = (
        tflite_size_bytes
        / (
            1024 * 1024
        )
    )


    metrics[
        "model_parameters"
    ] = int(
        model.count_params()
    )

    metrics[
        "tflite_fp32_size_bytes"
    ] = int(
        tflite_size_bytes
    )

    metrics[
        "tflite_fp32_size_kb"
    ] = float(
        tflite_size_kb
    )

    metrics[
        "tflite_fp32_size_mb"
    ] = float(
        tflite_size_mb
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("FP32 BASELINE SELESAI")
    print("=" * 75)

    print()
    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.2%}"
    )

    print(
        f"Macro Precision : "
        f"{metrics['macro_precision']:.2%}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['macro_recall']:.2%}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.2%}"
    )

    print(
        f"Weighted F1     : "
        f"{metrics['weighted_f1']:.2%}"
    )

    print()

    print(
        "Parameter model:",
        model.count_params()
    )

    print(
        f"TFLite FP32 size: "
        f"{tflite_size_kb:.2f} KB "
        f"({tflite_size_mb:.3f} MB)"
    )

    print()

    print(
        "Model Keras:"
    )

    print(
        BEST_MODEL_FILE
    )

    print()

    print(
        "Model TFLite FP32:"
    )

    print(
        TFLITE_FP32_FILE
    )

    print()

    print(
        "Metrics:"
    )

    print(
        METRICS_FILE
    )

    print()

    print(
        "Classification report:"
    )

    print(
        REPORT_FILE
    )

    print()

    print(
        "Confusion matrix:"
    )

    print(
        CONFUSION_FILE
    )

    print()


if __name__ == "__main__":
    main()