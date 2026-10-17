"""
RampSense — P1 PSO Full-Split Validation
========================================

Evaluates the saved PSO parameter vector on:
    1. Training split
    2. Validation split
    3. Test split

Metrics:
    - Combined objective
    - Scaled MSE
    - MAE and RMSE in watts
    - Precision, recall and F1
    - TP, FP, FN and TN
    - Actual and predicted ramp rates

Configuration matches fitness.py:
    Input size: 6
    Hidden size: 16
    Output size: 8
    Input history: 60 minutes
    Ramp threshold: 12.25 W/min
    Objective: 0.75 * MSE_scaled + 0.25 * (1 - F1)

Run:
    python validate_pso.py
"""

from pathlib import Path
import importlib.util
import json
import time

import numpy as np
import torch
import joblib


# ============================================================
# 1. CONFIGURATION AND PATHS
# ============================================================

OPTIMIZER_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = OPTIMIZER_DIR / "outputs"

DATA_DIR = (
    OPTIMIZER_DIR.parent
    / "multistep_8"
    / "split_scaled"
)

BEST_VECTOR_PATH = (
    OUTPUT_DIR / "stage_p1_pso_best.npy"
)

REPORT_PATH = (
    OUTPUT_DIR / "stage_p1_pso_validation_report.json"
)

INPUT_SIZE = 6
HIDDEN_SIZE = 16
OUTPUT_SIZE = 8

BATCH_SIZE = 512

THRESHOLD_W_PER_MIN = 12.25

W1 = 0.75
W2 = 0.25

DEVICE = torch.device("cpu")


# ============================================================
# 2. LOAD PYTHON MODULES
# ============================================================

def load_module(module_name, file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Python file not found:\n{file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module:\n{file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# Load model.py
model_module = load_module(
    "rampsense_validation_model",
    OPTIMIZER_DIR / "model.py"
)

PVLSTM = model_module.PVLSTM


# Load param_vector.py
vector_module = load_module(
    "rampsense_validation_param_vector",
    OPTIMIZER_DIR / "param_vector.py"
)

set_vector = vector_module.set_vector


# ============================================================
# 3. CLASSIFICATION METRICS
# ============================================================

def calculate_classification_metrics(
    actual_flags,
    predicted_flags
):

    tp = int(
        np.sum(actual_flags & predicted_flags)
    )

    fp = int(
        np.sum(~actual_flags & predicted_flags)
    )

    fn = int(
        np.sum(actual_flags & ~predicted_flags)
    )

    tn = int(
        np.sum(~actual_flags & ~predicted_flags)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        "actual_ramps": int(
            np.sum(actual_flags)
        ),

        "predicted_ramps": int(
            np.sum(predicted_flags)
        ),

        "actual_ramp_rate": float(
            np.mean(actual_flags)
        ),

        "predicted_ramp_rate": float(
            np.mean(predicted_flags)
        )
    }


# ============================================================
# 4. LOAD PSO PARAMETERS
# ============================================================

def load_model():

    print("Loading optimized PSO parameters...")

    if not BEST_VECTOR_PATH.exists():
        raise FileNotFoundError(
            f"PSO parameter vector not found:\n"
            f"{BEST_VECTOR_PATH}"
        )

    best_vector = np.asarray(
        np.load(BEST_VECTOR_PATH),
        dtype=np.float64
    )

    if best_vector.shape != (1608,):
        raise ValueError(
            f"Expected parameter vector shape (1608,), "
            f"got {best_vector.shape}"
        )

    if not np.all(np.isfinite(best_vector)):
        raise ValueError(
            "PSO vector contains NaN or Inf."
        )

    model = PVLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    set_vector(
        model,
        best_vector
    )

    model.eval()

    print("PSO parameters loaded successfully.")
    print("Vector shape:", best_vector.shape)
    print("Vector file:", BEST_VECTOR_PATH)

    return model


# ============================================================
# 5. LOAD DATA
# ============================================================

def load_data():

    print()
    print("Loading dataset splits...")

    required_files = [
        "X_train.npy",
        "Y_train.npy",
        "X_val.npy",
        "Y_val.npy",
        "X_test.npy",
        "Y_test.npy",
        "target_scaler.pkl"
    ]

    for filename in required_files:

        filepath = DATA_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Required data file not found:\n"
                f"{filepath}"
            )

    # Memory-map arrays to reduce unnecessary memory usage.
    X_train = np.load(
        DATA_DIR / "X_train.npy",
        mmap_mode="r"
    )

    Y_train = np.load(
        DATA_DIR / "Y_train.npy",
        mmap_mode="r"
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy",
        mmap_mode="r"
    )

    Y_val = np.load(
        DATA_DIR / "Y_val.npy",
        mmap_mode="r"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy",
        mmap_mode="r"
    )

    Y_test = np.load(
        DATA_DIR / "Y_test.npy",
        mmap_mode="r"
    )

    target_scaler = joblib.load(
        DATA_DIR / "target_scaler.pkl"
    )

    splits = {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test)
    }

    for split_name, (X, Y) in splits.items():

        if X.ndim != 3 or Y.ndim != 2:
            raise ValueError(
                f"Invalid array dimensions for {split_name}: "
                f"X={X.shape}, Y={Y.shape}"
            )

        if len(X) != len(Y):
            raise ValueError(
                f"Sample count mismatch in {split_name}."
            )

        if X.shape[1:] != (60, INPUT_SIZE):
            raise ValueError(
                f"Unexpected X shape for {split_name}: "
                f"{X.shape}"
            )

        if Y.shape[1] != OUTPUT_SIZE:
            raise ValueError(
                f"Unexpected Y shape for {split_name}: "
                f"{Y.shape}"
            )

        print(
            f"{split_name.upper():5s}: "
            f"X={X.shape}, Y={Y.shape}"
        )

    print(
        "Target scaler:",
        type(target_scaler).__name__
    )

    return splits, target_scaler


# ============================================================
# 6. PREDICT IN BATCHES
# ============================================================

def predict_in_batches(model, X):

    predictions = []

    with torch.inference_mode():

        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X)
            )

            batch = np.asarray(
                X[start:end],
                dtype=np.float32
            )

            batch_tensor = torch.from_numpy(
                batch
            ).to(DEVICE)

            output = model(
                batch_tensor
            )

            predictions.append(
                output.cpu().numpy()
            )

    return np.concatenate(
        predictions,
        axis=0
    )


# ============================================================
# 7. EVALUATE ONE DATASET SPLIT
# ============================================================

def evaluate_split(
    model,
    X,
    Y,
    target_scaler,
    split_name
):

    print()
    print("=" * 70)
    print(f"EVALUATING {split_name.upper()} SPLIT")
    print("=" * 70)

    started = time.perf_counter()

    # --------------------------------------------------------
    # Predictions in scaled space
    # --------------------------------------------------------

    predictions_scaled = predict_in_batches(
        model,
        X
    )

    actual_scaled = np.asarray(
        Y,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Scaled-space MSE
    # --------------------------------------------------------

    mse_scaled = float(
        np.mean(
            (
                predictions_scaled.astype(np.float64)
                - actual_scaled
            ) ** 2
        )
    )

    # --------------------------------------------------------
    # Inverse-transform to watts
    # --------------------------------------------------------

    predictions_w = target_scaler.inverse_transform(
        predictions_scaled
    )

    actual_w = target_scaler.inverse_transform(
        actual_scaled
    )

    # --------------------------------------------------------
    # Regression metrics
    # --------------------------------------------------------

    mae_w = float(
        np.mean(
            np.abs(
                predictions_w - actual_w
            )
        )
    )

    mse_w = float(
        np.mean(
            (
                predictions_w - actual_w
            ) ** 2
        )
    )

    rmse_w = float(
        np.sqrt(mse_w)
    )

    # --------------------------------------------------------
    # Ramp-event detection
    #
    # Target vector: t+6 ... t+13
    # Calculate seven consecutive power differences.
    # --------------------------------------------------------

    actual_differences = np.diff(
        actual_w,
        axis=1
    )

    predicted_differences = np.diff(
        predictions_w,
        axis=1
    )

    actual_max_ramp = np.max(
        np.abs(actual_differences),
        axis=1
    )

    predicted_max_ramp = np.max(
        np.abs(predicted_differences),
        axis=1
    )

    actual_flags = (
        actual_max_ramp >= THRESHOLD_W_PER_MIN
    )

    predicted_flags = (
        predicted_max_ramp >= THRESHOLD_W_PER_MIN
    )

    classification = calculate_classification_metrics(
        actual_flags,
        predicted_flags
    )

    # --------------------------------------------------------
    # Combined objective
    # --------------------------------------------------------

    objective = float(
        W1 * mse_scaled
        + W2 * (1.0 - classification["f1"])
    )

    # --------------------------------------------------------
    # Collect metrics
    # --------------------------------------------------------

    metrics = {
        "split": split_name,
        "samples": int(len(X)),

        "objective": objective,
        "mse_scaled": mse_scaled,

        "mae_w": mae_w,
        "mse_w": mse_w,
        "rmse_w": rmse_w,

        **classification,

        "actual_max_ramp_mean": float(
            np.mean(actual_max_ramp)
        ),

        "actual_max_ramp_median": float(
            np.median(actual_max_ramp)
        ),

        "actual_max_ramp_max": float(
            np.max(actual_max_ramp)
        ),

        "predicted_max_ramp_mean": float(
            np.mean(predicted_max_ramp)
        ),

        "predicted_max_ramp_median": float(
            np.median(predicted_max_ramp)
        ),

        "predicted_max_ramp_max": float(
            np.max(predicted_max_ramp)
        ),

        "elapsed_seconds": float(
            time.perf_counter() - started
        )
    }

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    for key, value in metrics.items():

        print(
            f"{key:32s}: {value}"
        )

    return metrics


# ============================================================
# 8. MAIN
# ============================================================

def main():

    total_started = time.perf_counter()

    print("=" * 70)
    print("RampSense — PSO Full-Split Validation")
    print("=" * 70)

    print("Optimizer directory:", OPTIMIZER_DIR)
    print("Data directory:", DATA_DIR)
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Ramp threshold:", THRESHOLD_W_PER_MIN, "W/min")

    model = load_model()

    splits, target_scaler = load_data()

    results = {}

    for split_name in (
        "train",
        "val",
        "test"
    ):

        X, Y = splits[split_name]

        results[split_name] = evaluate_split(
            model=model,
            X=X,
            Y=Y,
            target_scaler=target_scaler,
            split_name=split_name
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "stage": "P1_PSO_FULL_SPLIT_VALIDATION",

        "best_vector_path": str(
            BEST_VECTOR_PATH
        ),

        "data_directory": str(
            DATA_DIR
        ),

        "threshold_w_per_min": THRESHOLD_W_PER_MIN,

        "objective_weights": {
            "mse": W1,
            "one_minus_f1": W2
        },

        "batch_size": BATCH_SIZE,

        "splits": results,

        "total_elapsed_seconds": float(
            time.perf_counter() - total_started
        )
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False
        )

    # --------------------------------------------------------
    # Final comparison table
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("PSO VALIDATION SUMMARY")
    print("=" * 90)

    print(
        f"{'Split':<10}"
        f"{'Objective':>12}"
        f"{'MSE scaled':>14}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Pred ramps':>12}"
    )

    for split_name in (
        "train",
        "val",
        "test"
    ):

        metrics = results[split_name]

        print(
            f"{split_name:<10}"
            f"{metrics['objective']:>12.6f}"
            f"{metrics['mse_scaled']:>14.6f}"
            f"{metrics['precision']:>12.6f}"
            f"{metrics['recall']:>12.6f}"
            f"{metrics['f1']:>12.6f}"
            f"{metrics['predicted_ramp_rate'] * 100:>11.2f}%"
        )

    print()
    print("Validation report saved to:")
    print(REPORT_PATH)

    print(
        "Total runtime:",
        f"{report['total_elapsed_seconds']:.2f} seconds"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()