"""
RampSense - SAMURAI C10
=======================

C10 controlled SAMURAI experiment.

Changes relative to C9:
    - covariance adaptation DISABLED
    - covariance remains fixed diagonal
    - calibrated initial temperature retained
    - fixed proposal move scale retained
    - adaptive temperature retained

Configuration:
    Model:
        input_size  = 6
        hidden_size = 16
        output_size = 8

    Optimizer vector:
        1608 parameters

    Fitness:
        fixed 10,000 training samples
        MSE + F1 objective
        w1 = 0.75
        w2 = 0.25
        ramp threshold = 12.25 W/min

    SAMURAI:
        T0 = 1.2885171556e-5
        fixed move scale = 0.0010019844
        fixed diagonal covariance
        adaptive temperature
        500 evaluations

IMPORTANT:
    This is a SAMURAI-style engineering implementation.
    The paper does not disclose all numerical constants required
    for exact reproduction.
"""

from pathlib import Path
import json
import pickle
import time

import numpy as np
import torch

from model import PVLSTM
from param_vector import set_vector, num_params


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\archa\Downloads\rampsense dataset\PV dataset\sequences"
)

OPT_DIR = BASE_DIR / "optimizer"

OUTPUT_DIR = OPT_DIR / "outputs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEQ_DIR = (
    BASE_DIR
    / "multistep_8"
    / "split_scaled"
)

X_TRAIN_PATH = (
    SEQ_DIR / "X_train.npy"
)

Y_TRAIN_PATH = (
    SEQ_DIR / "Y_train.npy"
)

TARGET_SCALER_PATH = (
    SEQ_DIR / "target_scaler.pkl"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(8)

DEVICE = torch.device("cpu")


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

INPUT_SIZE = 6
HIDDEN_SIZE = 16
OUTPUT_SIZE = 8

EXPECTED_OPTIMIZER_DIM = 1608


# ============================================================
# 4. FITNESS CONFIGURATION
# ============================================================

FITNESS_SUBSET_SIZE = 10_000

BATCH_SIZE = 512

THRESHOLD_W_PER_MIN = 12.25

W1 = 0.75
W2 = 0.25


# ============================================================
# 5. SAMURAI C10 CONFIGURATION
# ============================================================

MAX_EVALS = 500

# Calibrated during C8 from actual C6 proposal deltas.
T0 = 1.2885171556e-5

T_MIN = 1e-8

# Initial candidate distribution.
INITIAL_STD = 0.10

# C10 fixed move range.
FIXED_MOVE_SCALE = 0.0010019844

# Adaptive temperature.
ALPHA_MIN = 0.85
ALPHA_MAX = 0.97

TARGET_ACCEPT_LOW = 0.20
TARGET_ACCEPT_HIGH = 0.40

TEMP_ADAPT_WINDOW = 20


# ============================================================
# 6. FILE CHECKS
# ============================================================

print("=" * 70)
print("RampSense SAMURAI C10")
print("=" * 70)

print("\nChecking required files...")

required_files = {
    "X_train": X_TRAIN_PATH,
    "Y_train": Y_TRAIN_PATH,
    "target_scaler": TARGET_SCALER_PATH,
}

for name, path in required_files.items():

    exists = path.exists()

    print(
        f"{name}: {path} -> {exists}"
    )

    if not exists:
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# 7. LOAD TRAINING DATA
# ============================================================

print("\nLoading data...")

X_train = np.load(
    X_TRAIN_PATH,
    mmap_mode="r",
)

Y_train = np.load(
    Y_TRAIN_PATH,
    mmap_mode="r",
)

print(
    f"X_train: {X_train.shape}"
)

print(
    f"Y_train: {Y_train.shape}"
)


# ============================================================
# 8. FIXED FITNESS SUBSET
# ============================================================

rng = np.random.default_rng(SEED)

n_train = len(X_train)

if FITNESS_SUBSET_SIZE > n_train:
    raise ValueError(
        "Fitness subset is larger than training set."
    )

fitness_indices = rng.choice(
    n_train,
    size=FITNESS_SUBSET_SIZE,
    replace=False,
)

# Keep deterministic ordering.
fitness_indices = np.sort(
    fitness_indices
)

X_fit = np.asarray(
    X_train[fitness_indices],
    dtype=np.float32,
)

Y_fit = np.asarray(
    Y_train[fitness_indices],
    dtype=np.float32,
)

print(
    f"Fitness subset: {X_fit.shape}"
)

print(
    f"Fitness targets: {Y_fit.shape}"
)


# ============================================================
# 9. LOAD TARGET SCALER
# ============================================================

def load_target_scaler(path):
    """
    Load the existing StandardScaler.

    First tries joblib because sklearn objects saved through
    joblib are commonly loaded most reliably that way.

    Falls back to standard pickle.
    """

    # --------------------------------------------------------
    # Attempt 1: joblib
    # --------------------------------------------------------

    try:

        import joblib

        print(
            "\nTrying joblib.load()..."
        )

        scaler = joblib.load(path)

        print(
            "Target scaler loaded successfully "
            "with joblib."
        )

        return scaler

    except Exception as joblib_error:

        print(
            "joblib.load() failed:"
        )

        print(
            repr(joblib_error)
        )

    # --------------------------------------------------------
    # Attempt 2: pickle
    # --------------------------------------------------------

    try:

        print(
            "\nTrying pickle.load()..."
        )

        with open(
            path,
            "rb",
        ) as f:

            scaler = pickle.load(f)

        print(
            "Target scaler loaded successfully "
            "with pickle."
        )

        return scaler

    except Exception as pickle_error:

        print(
            "pickle.load() failed:"
        )

        print(
            repr(pickle_error)
        )

        raise RuntimeError(
            "\nCould not load the existing target scaler.\n"
            "\n"
            "The file exists and appears to contain a "
            "scikit-learn StandardScaler, but the current "
            "Python/NumPy environment cannot deserialize it.\n"
            "\n"
            "Do not regenerate the scaler because the "
            "optimizer must use the same target scaling "
            "as Stage B."
        ) from pickle_error


target_scaler = load_target_scaler(
    TARGET_SCALER_PATH
)


# ============================================================
# 10. VERIFY TARGET SCALER
# ============================================================

if not hasattr(
    target_scaler,
    "mean_",
):

    raise TypeError(
        "Loaded target scaler does not have mean_."
    )

if not hasattr(
    target_scaler,
    "scale_",
):

    raise TypeError(
        "Loaded target scaler does not have scale_."
    )

print(
    "\nTarget scaler:"
)

print(
    f"  Type: {type(target_scaler)}"
)

print(
    f"  mean_ shape: {np.shape(target_scaler.mean_)}"
)

print(
    f"  scale_ shape: {np.shape(target_scaler.scale_)}"
)

print(
    f"  n_features: "
    f"{getattr(target_scaler, 'n_features_in_', 'unknown')}"
)

if len(target_scaler.mean_) != OUTPUT_SIZE:

    raise ValueError(
        f"Target scaler has "
        f"{len(target_scaler.mean_)} features, "
        f"but model has {OUTPUT_SIZE} outputs."
    )


# ============================================================
# 11. CREATE MODEL
# ============================================================

model = PVLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
).to(DEVICE)

model.eval()

torch_parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

optimizer_dimension = num_params(
    model
)

print("\nModel:")

print(
    f"  Input size:       {INPUT_SIZE}"
)

print(
    f"  Hidden size:      {HIDDEN_SIZE}"
)

print(
    f"  Output size:      {OUTPUT_SIZE}"
)

print(
    f"  Torch parameters: {torch_parameter_count}"
)

print(
    f"  Optimizer dims:   {optimizer_dimension}"
)

if optimizer_dimension != EXPECTED_OPTIMIZER_DIM:

    raise RuntimeError(
        f"Expected optimizer dimension "
        f"{EXPECTED_OPTIMIZER_DIM}, "
        f"got {optimizer_dimension}."
    )


# ============================================================
# 12. FITNESS FUNCTION
# ============================================================

def evaluate(vector):
    """
    Evaluate one 1608-dimensional parameter vector.

    Objective:

        phi =
            0.75 * MSE_scaled
            + 0.25 * (1 - F1)

    MSE is calculated in scaled space.

    F1 is calculated after inverse-transforming the
    8-point prediction vectors back into watts.
    """

    # --------------------------------------------------------
    # Put candidate parameters into model.
    # --------------------------------------------------------

    set_vector(
        model,
        vector,
    )

    total_squared_error = 0.0
    total_elements = 0

    predictions = []

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        for start_idx in range(
            0,
            len(X_fit),
            BATCH_SIZE,
        ):

            end_idx = min(
                start_idx + BATCH_SIZE,
                len(X_fit),
            )

            xb = torch.from_numpy(
                X_fit[start_idx:end_idx]
            )

            yb = torch.from_numpy(
                Y_fit[start_idx:end_idx]
            )

            pred = model(xb)

            diff = (
                pred - yb
            )

            total_squared_error += (
                diff.pow(2)
                .sum()
                .item()
            )

            total_elements += (
                diff.numel()
            )

            predictions.append(
                pred.numpy()
            )

    # --------------------------------------------------------
    # Scaled MSE
    # --------------------------------------------------------

    mse_scaled = (
        total_squared_error
        / total_elements
    )

    pred_scaled = np.concatenate(
        predictions,
        axis=0,
    )

    # --------------------------------------------------------
    # Inverse transform to watts
    # --------------------------------------------------------

    pred_watts = (
        target_scaler
        .inverse_transform(
            pred_scaled
        )
    )

    true_watts = (
        target_scaler
        .inverse_transform(
            Y_fit
        )
    )

    # --------------------------------------------------------
    # Maximum absolute consecutive ramp
    #
    # 8 target points -> 7 consecutive differences.
    # --------------------------------------------------------

    pred_ramp = np.max(
        np.abs(
            np.diff(
                pred_watts,
                axis=1,
            )
        ),
        axis=1,
    )

    true_ramp = np.max(
        np.abs(
            np.diff(
                true_watts,
                axis=1,
            )
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Event classification
    # --------------------------------------------------------

    pred_event = (
        pred_ramp
        >= THRESHOLD_W_PER_MIN
    )

    true_event = (
        true_ramp
        >= THRESHOLD_W_PER_MIN
    )

    tp = int(
        np.sum(
            pred_event
            & true_event
        )
    )

    fp = int(
        np.sum(
            pred_event
            & ~true_event
        )
    )

    fn = int(
        np.sum(
            ~pred_event
            & true_event
        )
    )

    tn = int(
        np.sum(
            ~pred_event
            & ~true_event
        )
    )

    # --------------------------------------------------------
    # Precision / recall / F1
    # --------------------------------------------------------

    precision_den = tp + fp

    if precision_den > 0:

        precision = (
            tp
            / precision_den
        )

    else:

        precision = 0.0

    recall_den = tp + fn

    if recall_den > 0:

        recall = (
            tp
            / recall_den
        )

    else:

        recall = 0.0

    if (
        precision + recall
        > 0.0
    ):

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    # --------------------------------------------------------
    # Combined objective
    # --------------------------------------------------------

    objective = (
        W1 * mse_scaled
        + W2 * (1.0 - f1)
    )

    # --------------------------------------------------------
    # Additional regression metrics
    # --------------------------------------------------------

    error_w = (
        pred_watts
        - true_watts
    )

    mae_w = float(
        np.mean(
            np.abs(error_w)
        )
    )

    mse_w = float(
        np.mean(
            error_w ** 2
        )
    )

    rmse_w = float(
        np.sqrt(mse_w)
    )

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

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "mae_w": mae_w,
        "mse_w": mse_w,
        "rmse_w": rmse_w,

        "actual_ramp_rate": float(
            np.mean(
                true_event
            )
        ),

        "predicted_ramp_rate": float(
            np.mean(
                pred_event
            )
        ),

        "actual_max_ramp_mean": float(
            np.mean(
                true_ramp
            )
        ),

        "actual_max_ramp_median": float(
            np.median(
                true_ramp
            )
        ),

        "actual_max_ramp_max": float(
            np.max(
                true_ramp
            )
        ),

        "predicted_max_ramp_mean": float(
            np.mean(
                pred_ramp
            )
        ),

        "predicted_max_ramp_median": float(
            np.median(
                pred_ramp
            )
        ),

        "predicted_max_ramp_max": float(
            np.max(
                pred_ramp
            )
        ),
    }


# ============================================================
# 13. INITIAL PARAMETER VECTOR
# ============================================================

initial_vector = rng.normal(
    loc=0.0,
    scale=INITIAL_STD,
    size=EXPECTED_OPTIMIZER_DIM,
).astype(
    np.float64
)

current_vector = (
    initial_vector.copy()
)


# ============================================================
# 14. FIXED DIAGONAL COVARIANCE
# ============================================================

# C10:
#
#     covariance adaptation = OFF
#
# Initial covariance:
#
#     C = diag(INITIAL_STD^2)
#
# Cholesky:
#
#     L = diag(INITIAL_STD)
#
# This NEVER changes during C10.

chol_diag = np.full(
    EXPECTED_OPTIMIZER_DIM,
    INITIAL_STD,
    dtype=np.float64,
)


# ============================================================
# 15. CONFIGURATION PRINT
# ============================================================

print("\nSAMURAI C10 configuration:")

print(
    f"  Max evaluations:       {MAX_EVALS}"
)

print(
    f"  T0:                    {T0:.13e}"
)

print(
    f"  T_MIN:                 {T_MIN:.13e}"
)

print(
    f"  Initial parameter std: {INITIAL_STD}"
)

print(
    f"  Fixed move scale:      {FIXED_MOVE_SCALE:.10f}"
)

print(
    "  Effective coordinate:  "
    f"{FIXED_MOVE_SCALE * INITIAL_STD:.13e}"
)

print(
    "  Covariance:            FIXED DIAGONAL"
)

print(
    "  Covariance updates:    0"
)

print(
    "  Temperature adaptation: ENABLED"
)


# ============================================================
# 16. INITIAL FITNESS
# ============================================================

print(
    "\nEvaluating initial candidate..."
)

initial_start = time.perf_counter()

initial_metrics = evaluate(
    initial_vector
)

initial_eval_time = (
    time.perf_counter()
    - initial_start
)

current_metrics = (
    initial_metrics.copy()
)

current_objective = (
    initial_metrics["objective"]
)

best_vector = (
    initial_vector.copy()
)

best_metrics = (
    initial_metrics.copy()
)

print(
    f"Initial objective: "
    f"{current_objective:.12f}"
)

print(
    f"Initial MSE: "
    f"{initial_metrics['mse_scaled']:.12f}"
)

print(
    f"Initial F1: "
    f"{initial_metrics['f1']:.12f}"
)

print(
    f"Initial evaluation time: "
    f"{initial_eval_time:.3f} s"
)


# ============================================================
# 17. SAMURAI STATE
# ============================================================

temperature = T0

evaluation_count = 1

accepted_total = 0
rejected_total = 0

uphill_attempts_total = 0
uphill_accepts_total = 0

# Rolling 20-evaluation window.
window_accepted = 0
window_rejected = 0

window_uphill_attempts = 0
window_uphill_accepts = 0

temperature_history = []


# ============================================================
# 18. MAIN SAMURAI LOOP
# ============================================================

run_start = time.perf_counter()

while evaluation_count < MAX_EVALS:

    # --------------------------------------------------------
    # Generate standard normal direction.
    # --------------------------------------------------------

    z = rng.normal(
        loc=0.0,
        scale=1.0,
        size=EXPECTED_OPTIMIZER_DIM,
    )

    # --------------------------------------------------------
    # Fixed covariance proposal.
    #
    # step =
    #     FIXED_MOVE_SCALE * L * z
    #
    # Since L is diagonal:
    #
    # step_i =
    #     FIXED_MOVE_SCALE
    #     * INITIAL_STD
    #     * z_i
    # --------------------------------------------------------

    step = (
        FIXED_MOVE_SCALE
        * chol_diag
        * z
    )

    proposal_vector = (
        current_vector
        + step
    )

    # --------------------------------------------------------
    # Evaluate candidate.
    # --------------------------------------------------------

    proposal_metrics = evaluate(
        proposal_vector
    )

    proposal_objective = (
        proposal_metrics["objective"]
    )

    evaluation_count += 1

    delta = (
        proposal_objective
        - current_objective
    )

    # --------------------------------------------------------
    # Metropolis acceptance.
    # --------------------------------------------------------

    if delta <= 0.0:

        # Improvement.
        accept = True

    else:

        # Uphill move.
        uphill_attempts_total += 1
        window_uphill_attempts += 1

        acceptance_probability = np.exp(
            -delta
            / max(
                temperature,
                T_MIN,
            )
        )

        random_number = rng.random()

        accept = (
            random_number
            < acceptance_probability
        )

        if accept:

            uphill_accepts_total += 1
            window_uphill_accepts += 1

    # --------------------------------------------------------
    # Apply decision.
    # --------------------------------------------------------

    if accept:

        current_vector = (
            proposal_vector
        )

        current_metrics = (
            proposal_metrics
        )

        current_objective = (
            proposal_objective
        )

        accepted_total += 1
        window_accepted += 1

        # ----------------------------------------------------
        # Best-so-far.
        # ----------------------------------------------------

        if (
            proposal_objective
            < best_metrics["objective"]
        ):

            best_vector = (
                proposal_vector.copy()
            )

            best_metrics = (
                proposal_metrics.copy()
            )

    else:

        rejected_total += 1
        window_rejected += 1

    # --------------------------------------------------------
    # Adaptive temperature every 20 evaluations.
    # --------------------------------------------------------

    if (
        evaluation_count
        % TEMP_ADAPT_WINDOW
        == 0
    ):

        window_total = (
            window_accepted
            + window_rejected
        )

        if window_total > 0:

            window_acceptance = (
                window_accepted
                / window_total
            )

        else:

            window_acceptance = 0.0

        if (
            window_uphill_attempts
            > 0
        ):

            window_uphill_acceptance = (
                window_uphill_accepts
                / window_uphill_attempts
            )

        else:

            window_uphill_acceptance = 0.0

        # ----------------------------------------------------
        # Temperature adaptation.
        #
        # High acceptance:
        #     cool faster.
        #
        # Low acceptance:
        #     cool slower.
        # ----------------------------------------------------

        if (
            window_acceptance
            > TARGET_ACCEPT_HIGH
        ):

            alpha = ALPHA_MIN

        elif (
            window_acceptance
            < TARGET_ACCEPT_LOW
        ):

            alpha = ALPHA_MAX

        else:

            fraction = (
                window_acceptance
                - TARGET_ACCEPT_LOW
            ) / (
                TARGET_ACCEPT_HIGH
                - TARGET_ACCEPT_LOW
            )

            alpha = (
                ALPHA_MAX
                - fraction
                * (
                    ALPHA_MAX
                    - ALPHA_MIN
                )
            )

        temperature *= alpha

        temperature = max(
            temperature,
            T_MIN,
        )

        temperature_history.append(
            {
                "evaluation": int(
                    evaluation_count
                ),

                "temperature": float(
                    temperature
                ),

                "alpha": float(
                    alpha
                ),

                "window_acceptance": float(
                    window_acceptance
                ),

                "window_uphill_acceptance": float(
                    window_uphill_acceptance
                ),

                "best_objective": float(
                    best_metrics[
                        "objective"
                    ]
                ),
            }
        )

        print(
            f"Eval {evaluation_count:4d} | "
            f"acc {window_acceptance:.4f} | "
            f"uphill "
            f"{window_uphill_acceptance:.4f} | "
            f"alpha {alpha:.4f} | "
            f"T {temperature:.4e} | "
            f"best "
            f"{best_metrics['objective']:.9f}"
        )

        # Reset rolling window.
        window_accepted = 0
        window_rejected = 0

        window_uphill_attempts = 0
        window_uphill_accepts = 0


# ============================================================
# 19. FINAL FITNESS
# ============================================================

run_time = (
    time.perf_counter()
    - run_start
)

# The best vector has already been evaluated.
# We simply restore it to the model.
set_vector(
    model,
    best_vector
)


# ============================================================
# 20. SUMMARY METRICS
# ============================================================

decision_count = (
    accepted_total
    + rejected_total
)

overall_acceptance = (
    accepted_total
    / decision_count
)

uphill_acceptance = (
    uphill_accepts_total
    / uphill_attempts_total
    if uphill_attempts_total > 0
    else 0.0
)

objective_improvement = (
    initial_metrics["objective"]
    - best_metrics["objective"]
)


# ============================================================
# 21. VALIDATION
# ============================================================

print(
    "\nRunning validation checks..."
)

assert optimizer_dimension == (
    EXPECTED_OPTIMIZER_DIM
)

assert evaluation_count == (
    MAX_EVALS
)

assert decision_count == (
    MAX_EVALS - 1
)

assert (
    best_metrics["objective"]
    <= initial_metrics["objective"]
    + 1e-12
)

assert (
    uphill_accepts_total
    <= uphill_attempts_total
)

assert np.isfinite(
    best_metrics["objective"]
)

assert np.isfinite(
    best_vector
).all()

assert best_vector.shape == (
    EXPECTED_OPTIMIZER_DIM,
)

# C10 must have zero covariance updates.
covariance_updates = 0

assert covariance_updates == 0

# Check objective reconstruction.
reconstructed_objective = (
    W1
    * best_metrics["mse_scaled"]
    + W2
    * (
        1.0
        - best_metrics["f1"]
    )
)

assert np.isclose(
    reconstructed_objective,
    best_metrics["objective"],
    atol=1e-10,
)


# ============================================================
# 22. SAVE BEST VECTOR
# ============================================================

best_vector_path = (
    OUTPUT_DIR
    / "stage_c10_samurai_best.npy"
)

np.save(
    best_vector_path,
    best_vector,
)


# ============================================================
# 23. REPORT
# ============================================================

report = {

    "stage": "C10",

    "description": (
        "SAMURAI with calibrated initial "
        "temperature, fixed diagonal covariance, "
        "fixed move scale, and adaptive temperature"
    ),

    "seed": SEED,

    "model": {
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "torch_parameter_count": int(
            torch_parameter_count
        ),
        "optimizer_dimension": int(
            optimizer_dimension
        ),
    },

    "fitness": {
        "subset_size": FITNESS_SUBSET_SIZE,
        "batch_size": BATCH_SIZE,
        "threshold_w_per_min": (
            THRESHOLD_W_PER_MIN
        ),
        "w1": W1,
        "w2": W2,
    },

    "samurai": {
        "max_evaluations": MAX_EVALS,
        "T0": T0,
        "T_MIN": T_MIN,
        "initial_std": INITIAL_STD,
        "fixed_move_scale": (
            FIXED_MOVE_SCALE
        ),
        "effective_coordinate_scale": (
            FIXED_MOVE_SCALE
            * INITIAL_STD
        ),
        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,
        "target_accept_low": (
            TARGET_ACCEPT_LOW
        ),
        "target_accept_high": (
            TARGET_ACCEPT_HIGH
        ),
        "temperature_adaptation_window": (
            TEMP_ADAPT_WINDOW
        ),
        "covariance_adaptation": False,
        "covariance_updates": 0,
        "covariance_type": (
            "fixed_diagonal"
        ),
    },

    "results": {

        "initial_objective": float(
            initial_metrics["objective"]
        ),

        "best_objective": float(
            best_metrics["objective"]
        ),

        "improvement": float(
            objective_improvement
        ),

        "evaluations": int(
            evaluation_count
        ),

        "accepted": int(
            accepted_total
        ),

        "rejected": int(
            rejected_total
        ),

        "overall_acceptance": float(
            overall_acceptance
        ),

        "uphill_attempts": int(
            uphill_attempts_total
        ),

        "uphill_accepts": int(
            uphill_accepts_total
        ),

        "uphill_acceptance": float(
            uphill_acceptance
        ),

        "final_temperature": float(
            temperature
        ),

        "runtime_seconds": float(
            run_time
        ),
    },

    "initial_metrics": (
        initial_metrics
    ),

    "best_metrics": (
        best_metrics
    ),

    "temperature_history": (
        temperature_history
    ),
}


# ============================================================
# 24. SAVE REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "stage_c10_samurai_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "C10 COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Initial objective : "
    f"{initial_metrics['objective']:.12f}"
)

print(
    f"Best objective    : "
    f"{best_metrics['objective']:.12f}"
)

print(
    f"Improvement       : "
    f"{objective_improvement:.12f}"
)

print(
    f"Evaluations       : "
    f"{evaluation_count}"
)

print(
    f"Accepted          : "
    f"{accepted_total}"
)

print(
    f"Rejected          : "
    f"{rejected_total}"
)

print(
    f"Overall acceptance: "
    f"{overall_acceptance:.6f}"
)

print(
    f"Uphill attempts   : "
    f"{uphill_attempts_total}"
)

print(
    f"Uphill accepts    : "
    f"{uphill_accepts_total}"
)

print(
    f"Uphill acceptance : "
    f"{uphill_acceptance:.6f}"
)

print(
    f"Final temperature : "
    f"{temperature:.12e}"
)

print(
    f"Runtime           : "
    f"{run_time:.3f} s"
)

print(
    "\nBest vector:"
)

print(
    best_vector_path
)

print(
    "\nReport:"
)

print(
    report_path
)

print(
    "\nSTAGE C10 PASSED"
)
