import sys
import time
from pathlib import Path

import numpy as np
import torch
import joblib


# ============================================================
# PATHS
# ============================================================

OPTIMIZER_DIR = Path(__file__).resolve().parent

PROJECT_DIR = (
    OPTIMIZER_DIR
    .parents[2]
)

DATA_DIR = (
    PROJECT_DIR
    / "PV dataset"
    / "sequences"
    / "multistep_8"
    / "split_scaled"
)


# ============================================================
# IMPORT OUR MODEL / PARAMETER VECTOR
# ============================================================

MODEL_PATH = OPTIMIZER_DIR / "model.py"
PARAM_VECTOR_PATH = OPTIMIZER_DIR / "param_vector.py"


# Load model.py directly
import importlib.util


spec_model = importlib.util.spec_from_file_location(
    "rampsense_model",
    MODEL_PATH
)

model_module = importlib.util.module_from_spec(
    spec_model
)

spec_model.loader.exec_module(
    model_module
)

PVLSTM = model_module.PVLSTM


# Load param_vector.py directly
spec_vector = importlib.util.spec_from_file_location(
    "rampsense_param_vector",
    PARAM_VECTOR_PATH
)

param_vector_module = (
    importlib.util.module_from_spec(
        spec_vector
    )
)

spec_vector.loader.exec_module(
    param_vector_module
)

get_vector = (
    param_vector_module.get_vector
)

set_vector = (
    param_vector_module.set_vector
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_SIZE = 6
HIDDEN_SIZE = 16
OUTPUT_SIZE = 8

INPUT_LENGTH = 60

TARGET_START = 6
TARGET_END = 13

THRESHOLD_W_PER_MIN = 12.25

W1 = 0.75
W2 = 0.25

FITNESS_SAMPLES = 10_000

RANDOM_SEED = 42

BATCH_SIZE = 512

DEVICE = torch.device("cpu")


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("RampSense - Stage B")
print("Combined MSE + F1 Fitness Validation")
print("=" * 70)

print()
print("Optimizer directory:")
print(OPTIMIZER_DIR)

print()
print("Data directory:")
print(DATA_DIR)

print()
print("Configuration:")
print(f"Input size              : {INPUT_SIZE}")
print(f"Hidden size             : {HIDDEN_SIZE}")
print(f"Output size             : {OUTPUT_SIZE}")
print(f"Input history           : {INPUT_LENGTH} minutes")
print(f"Future target           : t+{TARGET_START} ... t+{TARGET_END}")
print(f"Ramp threshold          : {THRESHOLD_W_PER_MIN} W/min")
print(f"MSE weight              : {W1}")
print(f"F1 weight               : {W2}")
print(f"Fitness samples         : {FITNESS_SAMPLES}")
print(f"Random seed             : {RANDOM_SEED}")
print(f"Batch size              : {BATCH_SIZE}")
print(f"Device                  : {DEVICE}")


# ============================================================
# VERIFY DATA FILES
# ============================================================

required_files = [
    r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\multistep_8\split_scaled\X_train.npy",
    r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\multistep_8\split_scaled\Y_train.npy",
    r"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\multistep_8\split_scaled\target_scaler.pkl"
]

print()
print("Checking required files...")

for filename in required_files:

    path = DATA_DIR / filename

    print(
        f"{filename}: {path.exists()}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("Loading training data...")

X_train = np.load(
     rb"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\multistep_8\split_scaled\X_train.npy"
    
)

Y_train = np.load(
     rb"c:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences\multistep_8\split_scaled\Y_train.npy"
)

print()
print("Full training shapes:")
print(
    "X_train:",
    X_train.shape,
    X_train.dtype
)

print(
    "Y_train:",
    Y_train.shape,
    Y_train.dtype
)


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

if X_train.ndim != 3:

    raise ValueError(
        f"X_train must be 3-D, got {X_train.shape}"
    )


if Y_train.ndim != 2:

    raise ValueError(
        f"Y_train must be 2-D, got {Y_train.shape}"
    )


if X_train.shape[1:] != (
    INPUT_LENGTH,
    INPUT_SIZE
):

    raise ValueError(
        "Unexpected X_train shape: "
        f"{X_train.shape}"
    )


if Y_train.shape[1] != OUTPUT_SIZE:

    raise ValueError(
        "Unexpected Y_train output dimension: "
        f"{Y_train.shape}"
    )


if len(X_train) != len(Y_train):

    raise ValueError(
        "X_train and Y_train have different "
        "numbers of samples."
    )


# ============================================================
# LOAD TARGET SCALER
# ============================================================

print()
print("Loading target scaler...")

target_scaler = joblib.load(
    DATA_DIR / "target_scaler.pkl"
)

print(
    "Target scaler type:",
    type(target_scaler).__name__
)

print(
    "Target scaler mean shape:",
    np.asarray(
        target_scaler.mean_
    ).shape
)

print(
    "Target scaler scale shape:",
    np.asarray(
        target_scaler.scale_
    ).shape
)


# ============================================================
# FIXED FITNESS SUBSET
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

fitness_count = min(
    FITNESS_SAMPLES,
    len(X_train)
)

fitness_indices = rng.choice(
    len(X_train),
    size=fitness_count,
    replace=False
)

fitness_indices.sort()

X_fit = X_train[
    fitness_indices
]

Y_fit = Y_train[
    fitness_indices
]


print()
print("=" * 70)
print("Fixed fitness subset")
print("=" * 70)

print(
    "Number of samples:",
    len(X_fit)
)

print(
    "X_fit shape:",
    X_fit.shape
)

print(
    "Y_fit shape:",
    Y_fit.shape
)

print(
    "First indices:",
    fitness_indices[:10]
)

print(
    "Last indices:",
    fitness_indices[-10:]
)


# ============================================================
# MODEL
# ============================================================

model = PVLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE
).to(DEVICE)

model.eval()


# ============================================================
# MODEL PARAMETER CHECK
# ============================================================

optimizer_parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    "Optimizer parameter count:",
    optimizer_parameter_count
)

if optimizer_parameter_count != 1608:

    raise RuntimeError(
        "Expected exactly 1608 optimizer "
        f"parameters, got {optimizer_parameter_count}"
    )


# ============================================================
# HELPER: F1
# ============================================================

def f1_from_flags(
    actual_flags,
    predicted_flags
):

    tp = int(
        np.sum(
            actual_flags
            &
            predicted_flags
        )
    )

    fp = int(
        np.sum(
            (~actual_flags)
            &
            predicted_flags
        )
    )

    fn = int(
        np.sum(
            actual_flags
            &
            (~predicted_flags)
        )
    )

    tn = int(
        np.sum(
            (~actual_flags)
            &
            (~predicted_flags)
        )
    )

    denominator = (
        2 * tp
        + fp
        + fn
    )

    if denominator == 0:

        f1 = 0.0

    else:

        f1 = (
            2.0 * tp
            /
            denominator
        )

    return (
        float(f1),
        tp,
        fp,
        fn,
        tn
    )


# ============================================================
# HELPER: PREDICTIONS
# ============================================================

def predict_dataset(
    parameter_vector
):

    """
    Put candidate parameter vector into the model
    and predict the complete fixed fitness subset.
    """

    set_vector(
        model,
        parameter_vector
    )

    model.eval()

    predictions = []

    start_time = (
        time.perf_counter()
    )

    with torch.inference_mode():

        for start in range(
            0,
            len(X_fit),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X_fit)
            )

            batch = torch.from_numpy(
                X_fit[start:end]
            ).to(
                DEVICE,
                dtype=torch.float32
            )

            output = model(
                batch
            )

            predictions.append(
                output.cpu().numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    return (
        predictions,
        elapsed
    )


# ============================================================
# OBJECTIVE EVALUATION
# ============================================================

def evaluate_fitness(
    parameter_vector
):

    """
    Evaluate one 1608-dimensional candidate.

    Returns a dictionary containing:

        MSE
        F1
        objective
        confusion matrix
        ramp statistics
    """

    parameter_vector = np.asarray(
        parameter_vector,
        dtype=np.float64
    )

    if parameter_vector.shape != (
        1608,
    ):

        raise ValueError(
            "Expected parameter vector "
            f"shape (1608,), got "
            f"{parameter_vector.shape}"
        )

    if not np.all(
        np.isfinite(parameter_vector)
    ):

        raise ValueError(
            "Parameter vector contains "
            "NaN or Inf."
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    predictions_scaled, elapsed = (
        predict_dataset(
            parameter_vector
        )
    )

    # --------------------------------------------------------
    # Scaled-space MSE
    # --------------------------------------------------------

    mse_scaled = np.mean(
        (
            predictions_scaled
            -
            Y_fit
        ) ** 2
    )

    # --------------------------------------------------------
    # Convert predictions and targets
    # back to physical watts.
    # --------------------------------------------------------

    predictions_w = (
        target_scaler.inverse_transform(
            predictions_scaled
        )
    )

    actual_w = (
        target_scaler.inverse_transform(
            Y_fit
        )
    )

    # --------------------------------------------------------
    # 8-point future target:
    #
    # t+6, t+7, ..., t+13
    #
    # Seven consecutive differences.
    # --------------------------------------------------------

    actual_differences = np.diff(
        actual_w,
        axis=1
    )

    predicted_differences = np.diff(
        predictions_w,
        axis=1
    )

    # --------------------------------------------------------
    # Maximum absolute ramp in
    # each future window.
    # --------------------------------------------------------

    actual_max_ramp = np.max(
        np.abs(
            actual_differences
        ),
        axis=1
    )

    predicted_max_ramp = np.max(
        np.abs(
            predicted_differences
        ),
        axis=1
    )

    # --------------------------------------------------------
    # ORE / PRE flags
    # --------------------------------------------------------

    actual_flags = (
        actual_max_ramp
        >= THRESHOLD_W_PER_MIN
    )

    predicted_flags = (
        predicted_max_ramp
        >= THRESHOLD_W_PER_MIN
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    (
        f1,
        tp,
        fp,
        fn,
        tn
    ) = f1_from_flags(
        actual_flags,
        predicted_flags
    )

    # --------------------------------------------------------
    # Combined objective
    #
    # Phi = 0.75*MSE + 0.25*(1-F1)
    # --------------------------------------------------------

    objective = (
        W1 * mse_scaled
        +
        W2 * (1.0 - f1)
    )

    # --------------------------------------------------------
    # Regression metrics in physical units
    # --------------------------------------------------------

    mae_w = np.mean(
        np.abs(
            predictions_w
            -
            actual_w
        )
    )

    mse_w = np.mean(
        (
            predictions_w
            -
            actual_w
        ) ** 2
    )

    rmse_w = np.sqrt(
        mse_w
    )

    # --------------------------------------------------------
    # Return all information
    # --------------------------------------------------------

    return {
        "objective": float(
            objective
        ),

        "mse_scaled": float(
            mse_scaled
        ),

        "f1": float(
            f1
        ),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),

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
        ),

        "mae_w": float(
            mae_w
        ),

        "mse_w": float(
            mse_w
        ),

        "rmse_w": float(
            rmse_w
        ),

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

        "forward_seconds": float(
            elapsed
        )
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("Testing fixed fitness function")
    print("=" * 70)

    # --------------------------------------------------------
    # Obtain current random model vector.
    # --------------------------------------------------------

    initial_vector = get_vector(
        model
    )

    print()
    print(
        "Initial vector shape:",
        initial_vector.shape
    )

    print(
        "Initial vector dtype:",
        initial_vector.dtype
    )

    # --------------------------------------------------------
    # Evaluate the candidate.
    # --------------------------------------------------------

    result = evaluate_vector(
        initial_vector
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FITNESS RESULT")
    print("=" * 70)

    for key, value in result.items():

        print(
            f"{key:35s}: {value}"
        )

    # --------------------------------------------------------
    # Objective consistency check.
    # --------------------------------------------------------

    expected_objective = (
        W1 * result["mse_scaled"]
        +
        W2 * (
            1.0
            -
            result["f1"]
        )
    )

    objective_difference = abs(
        result["objective"]
        -
        expected_objective
    )

    print()
    print("=" * 70)
    print("OBJECTIVE CONSISTENCY CHECK")
    print("=" * 70)

    print(
        "Reported objective :",
        result["objective"]
    )

    print(
        "Recalculated       :",
        expected_objective
    )

    print(
        "Absolute difference:",
        objective_difference
    )

    if not np.isclose(
        result["objective"],
        expected_objective,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Objective consistency check failed."
        )

    print()
    print(
        "OBJECTIVE CHECK PASSED"
    )

    # --------------------------------------------------------
    # Verify expected F1 range.
    # --------------------------------------------------------

    if not (
        0.0
        <= result["f1"]
        <= 1.0
    ):

        raise RuntimeError(
            "F1 is outside [0,1]."
        )

    # --------------------------------------------------------
    # Verify confusion matrix.
    # --------------------------------------------------------

    confusion_total = (
        result["tp"]
        +
        result["fp"]
        +
        result["fn"]
        +
        result["tn"]
    )

    if confusion_total != len(
        X_fit
    ):

        raise RuntimeError(
            "Confusion matrix does not "
            "sum to fitness subset size."
        )

    print()
    print(
        "Confusion matrix total:",
        confusion_total
    )

    print(
        "Fitness subset size:",
        len(X_fit)
    )

    print()
    print(
        "CONFUSION MATRIX CHECK PASSED"
    )

    # --------------------------------------------------------
    # Verify objective components.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OBJECTIVE COMPONENTS")
    print("=" * 70)

    print(
        f"0.75 × MSE = "
        f"{W1 * result['mse_scaled']:.10f}"
    )

    print(
        f"0.25 × (1-F1) = "
        f"{W2 * (1.0 - result['f1']):.10f}"
    )

    print(
        f"Phi = "
        f"{result['objective']:.10f}"
    )

    # --------------------------------------------------------
    # Final message.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE B FITNESS TEST PASSED")
    print("=" * 70)
  
   
