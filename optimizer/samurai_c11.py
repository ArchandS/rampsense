"""
RampSense - SAMURAI C11
=======================

C11 = C10 + adaptive move range.

Controlled changes:
    C10:
        fixed diagonal covariance
        fixed move scale
        adaptive temperature

    C11:
        fixed diagonal covariance
        ADAPTIVE move scale
        adaptive temperature

Everything else remains unchanged.

Model:
    input_size  = 6
    hidden_size = 16
    output_size = 8

Optimizer dimension:
    1608

Fitness:
    fixed 10,000 training samples
    threshold = 12.25 W/min
    objective = 0.75*MSE_scaled + 0.25*(1-F1)

SAMURAI:
    T0 = 1.2885171556e-5
    initial move scale = 0.0010019844
    adaptive move scale
    fixed diagonal covariance
    adaptive temperature
    500 evaluations
"""

from pathlib import Path
import json
import time

import numpy as np
import torch
import joblib

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

X_TRAIN_PATH = SEQ_DIR / "X_train.npy"
Y_TRAIN_PATH = SEQ_DIR / "Y_train.npy"
TARGET_SCALER_PATH = SEQ_DIR / "target_scaler.pkl"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(8)

DEVICE = torch.device("cpu")


# ============================================================
# 3. MODEL
# ============================================================

INPUT_SIZE = 6
HIDDEN_SIZE = 16
OUTPUT_SIZE = 8

EXPECTED_DIM = 1608


# ============================================================
# 4. FITNESS
# ============================================================

FITNESS_SUBSET_SIZE = 10_000

BATCH_SIZE = 512

THRESHOLD_W_PER_MIN = 12.25

W1 = 0.75
W2 = 0.25


# ============================================================
# 5. SAMURAI C11
# ============================================================

MAX_EVALS = 500

# C8 calibrated temperature.
T0 = 1.2885171556e-5

T_MIN = 1e-8

# Same initial parameter distribution as C10.
INITIAL_STD = 0.10

# Same initial move scale as C10.
INITIAL_MOVE_SCALE = 0.0010019844


# ------------------------------------------------------------
# Adaptive move range
# ------------------------------------------------------------

MOVE_SCALE_MIN = 0.0001
MOVE_SCALE_MAX = 0.01

MOVE_TARGET_LOW = 0.20
MOVE_TARGET_HIGH = 0.40

MOVE_ADAPT_WINDOW = 20

MOVE_SCALE_UP = 1.25
MOVE_SCALE_DOWN = 0.80


# ------------------------------------------------------------
# Adaptive temperature
# ------------------------------------------------------------

ALPHA_MIN = 0.85
ALPHA_MAX = 0.97

TARGET_ACCEPT_LOW = 0.20
TARGET_ACCEPT_HIGH = 0.40

TEMP_ADAPT_WINDOW = 20


# ============================================================
# 6. FILE CHECKS
# ============================================================

print("=" * 70)
print("RampSense SAMURAI C11")
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
# 7. LOAD DATA
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

# IMPORTANT:
# Separate RNG stream for dataset selection.
#
# This makes the optimizer RNG independent from the
# fitness-subset RNG.

rng_data = np.random.default_rng(
    SEED
)

n_train = len(X_train)

if FITNESS_SUBSET_SIZE > n_train:
    raise ValueError(
        "Fitness subset is larger than training set."
    )

fitness_indices = rng_data.choice(
    n_train,
    size=FITNESS_SUBSET_SIZE,
    replace=False,
)

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
# 9. OPTIMIZER RNG
# ============================================================

# This RNG is used ONLY for:
#   - initial parameter vector
#   - proposal directions
#   - Metropolis random numbers

rng = np.random.default_rng(
    SEED
)


# ============================================================
# 10. LOAD TARGET SCALER
# ============================================================

print(
    "\nLoading target scaler..."
)

try:

    target_scaler = joblib.load(
        TARGET_SCALER_PATH
    )

except Exception as e:

    raise RuntimeError(
        "Could not load target_scaler.pkl "
        "with joblib."
    ) from e


print(
    "Target scaler loaded:"
)

print(
    f"  Type: {type(target_scaler)}"
)

print(
    f"  mean_ shape: "
    f"{np.shape(target_scaler.mean_)}"
)

print(
    f"  scale_ shape: "
    f"{np.shape(target_scaler.scale_)}"
)

if len(target_scaler.mean_) != OUTPUT_SIZE:
    raise ValueError(
        "Target scaler does not have "
        f"{OUTPUT_SIZE} output features."
    )


# ============================================================
# 11. MODEL
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

if optimizer_dimension != EXPECTED_DIM:

    raise RuntimeError(
        f"Expected {EXPECTED_DIM} "
        f"optimizer parameters, got "
        f"{optimizer_dimension}."
    )


# ============================================================
# 12. FITNESS FUNCTION
# ============================================================

def evaluate(vector):

    set_vector(
        model,
        vector,
    )

    total_squared_error = 0.0
    total_elements = 0

    predictions = []

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

    mse_scaled = (
        total_squared_error
        / total_elements
    )

    pred_scaled = np.concatenate(
        predictions,
        axis=0,
    )

    # --------------------------------------------------------
    # Convert back to watts.
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
    # Maximum absolute consecutive ramp.
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

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    objective = (
        W1 * mse_scaled
        + W2 * (1.0 - f1)
    )

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
            np.mean(true_event)
        ),

        "predicted_ramp_rate": float(
            np.mean(pred_event)
        ),

        "actual_max_ramp_mean": float(
            np.mean(true_ramp)
        ),

        "actual_max_ramp_median": float(
            np.median(true_ramp)
        ),

        "actual_max_ramp_max": float(
            np.max(true_ramp)
        ),

        "predicted_max_ramp_mean": float(
            np.mean(pred_ramp)
        ),

        "predicted_max_ramp_median": float(
            np.median(pred_ramp)
        ),

        "predicted_max_ramp_max": float(
            np.max(pred_ramp)
        ),
    }


# ============================================================
# 13. INITIAL PARAMETER VECTOR
# ============================================================

initial_vector = rng.normal(
    loc=0.0,
    scale=INITIAL_STD,
    size=EXPECTED_DIM,
).astype(
    np.float64
)

current_vector = (
    initial_vector.copy()
)


# ============================================================
# 14. FIXED DIAGONAL COVARIANCE
# ============================================================

# IMPORTANT:
#
# C11 DOES NOT adapt covariance.
#
# The covariance stays:
#
#     C = diag(INITIAL_STD^2)
#
# throughout the complete experiment.

chol_diag = np.full(
    EXPECTED_DIM,
    INITIAL_STD,
    dtype=np.float64,
)


# ============================================================
# 15. INITIAL MOVE SCALE
# ============================================================

move_scale = (
    INITIAL_MOVE_SCALE
)


# ============================================================
# 16. CONFIGURATION
# ============================================================

print(
    "\nSAMURAI C11 configuration:"
)

print(
    f"  Max evaluations: "
    f"{MAX_EVALS}"
)

print(
    f"  T0: "
    f"{T0:.13e}"
)

print(
    f"  Initial std: "
    f"{INITIAL_STD}"
)

print(
    f"  Initial move scale: "
    f"{move_scale:.10f}"
)

print(
    f"  Move scale min: "
    f"{MOVE_SCALE_MIN:.10f}"
)

print(
    f"  Move scale max: "
    f"{MOVE_SCALE_MAX:.10f}"
)

print(
    "  Covariance: FIXED DIAGONAL"
)

print(
    "  Covariance adaptation: DISABLED"
)

print(
    "  Temperature adaptation: ENABLED"
)

print(
    "  Move-range adaptation: ENABLED"
)


# ============================================================
# 17. INITIAL FITNESS
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
# 18. SAMURAI STATE
# ============================================================

temperature = T0

evaluation_count = 1

accepted_total = 0
rejected_total = 0

uphill_attempts_total = 0
uphill_accepts_total = 0

window_accepted = 0
window_rejected = 0

window_uphill_attempts = 0
window_uphill_accepts = 0

temperature_history = []
move_scale_history = []


# ============================================================
# 19. MAIN LOOP
# ============================================================

run_start = time.perf_counter()

while evaluation_count < MAX_EVALS:

    # --------------------------------------------------------
    # Gaussian proposal direction.
    # --------------------------------------------------------

    z = rng.normal(
        loc=0.0,
        scale=1.0,
        size=EXPECTED_DIM,
    )

    # --------------------------------------------------------
    # Fixed covariance + CURRENT move scale.
    #
    # The covariance does not change.
    #
    # Only move_scale changes.
    # --------------------------------------------------------

    step = (
        move_scale
        * chol_diag
        * z
    )

    proposal_vector = (
        current_vector
        + step
    )

    # --------------------------------------------------------
    # Evaluate proposal.
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

        accept = True

    else:

        uphill_attempts_total += 1
        window_uphill_attempts += 1

        acceptance_probability = np.exp(
            -delta
            / max(
                temperature,
                T_MIN,
            )
        )

        accept = (
            rng.random()
            < acceptance_probability
        )

        if accept:

            uphill_accepts_total += 1
            window_uphill_accepts += 1

    # --------------------------------------------------------
    # Apply acceptance decision.
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
        # Best candidate.
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


    # ========================================================
    # ADAPT EVERY 20 EVALUATIONS
    # ========================================================

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


        # ====================================================
        # TEMPERATURE ADAPTATION
        # ====================================================

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


        # ====================================================
        # MOVE RANGE ADAPTATION
        # ====================================================
        #
        # If acceptance is too high:
        #     proposals are too small
        #     -> increase move range.
        #
        # If acceptance is too low:
        #     proposals are too large
        #     -> decrease move range.
        #
        # This is the ONLY new mechanism compared with C10.
        # ====================================================

        old_move_scale = (
            move_scale
        )

        if (
            window_acceptance
            > MOVE_TARGET_HIGH
        ):

            move_scale *= (
                MOVE_SCALE_UP
            )

        elif (
            window_acceptance
            < MOVE_TARGET_LOW
        ):

            move_scale *= (
                MOVE_SCALE_DOWN
            )

        move_scale = np.clip(
            move_scale,
            MOVE_SCALE_MIN,
            MOVE_SCALE_MAX,
        )


        # ----------------------------------------------------
        # Save histories.
        # ----------------------------------------------------

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

        move_scale_history.append(
            {
                "evaluation": int(
                    evaluation_count
                ),

                "old_move_scale": float(
                    old_move_scale
                ),

                "new_move_scale": float(
                    move_scale
                ),

                "window_acceptance": float(
                    window_acceptance
                ),

                "best_objective": float(
                    best_metrics[
                        "objective"
                    ]
                ),
            }
        )


        # ----------------------------------------------------
        # Progress.
        # ----------------------------------------------------

        print(
            f"Eval {evaluation_count:4d} | "
            f"acc {window_acceptance:.4f} | "
            f"uphill "
            f"{window_uphill_acceptance:.4f} | "
            f"alpha {alpha:.4f} | "
            f"T {temperature:.4e} | "
            f"move {move_scale:.7f} | "
            f"best "
            f"{best_metrics['objective']:.9f}"
        )


        # ----------------------------------------------------
        # Reset window.
        # ----------------------------------------------------

        window_accepted = 0
        window_rejected = 0

        window_uphill_attempts = 0
        window_uphill_accepts = 0


# ============================================================
# 20. FINAL
# ============================================================

runtime = (
    time.perf_counter()
    - run_start
)

set_vector(
    model,
    best_vector,
)


# ============================================================
# 21. SUMMARY
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

improvement = (
    initial_metrics["objective"]
    - best_metrics["objective"]
)


# ============================================================
# 22. VALIDATION
# ============================================================

print(
    "\nRunning validation checks..."
)

assert optimizer_dimension == (
    EXPECTED_DIM
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

assert np.all(
    np.isfinite(best_vector)
)

assert best_vector.shape == (
    EXPECTED_DIM,
)

# C11 still has zero covariance updates.
covariance_updates = 0

assert covariance_updates == 0

# Objective reconstruction.
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
# 23. SAVE BEST VECTOR
# ============================================================

best_vector_path = (
    OUTPUT_DIR
    / "stage_c11_samurai_best.npy"
)

np.save(
    best_vector_path,
    best_vector,
)


# ============================================================
# 24. REPORT
# ============================================================

report = {

    "stage": "C11",

    "description": (
        "C10 with adaptive move range added. "
        "Covariance remains fixed."
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

        "initial_move_scale": (
            INITIAL_MOVE_SCALE
        ),

        "final_move_scale": (
            move_scale
        ),

        "move_scale_min": (
            MOVE_SCALE_MIN
        ),

        "move_scale_max": (
            MOVE_SCALE_MAX
        ),

        "move_target_low": (
            MOVE_TARGET_LOW
        ),

        "move_target_high": (
            MOVE_TARGET_HIGH
        ),

        "move_scale_up": (
            MOVE_SCALE_UP
        ),

        "move_scale_down": (
            MOVE_SCALE_DOWN
        ),

        "covariance_adaptation": False,

        "covariance_updates": 0,

        "covariance_type": (
            "fixed_diagonal"
        ),

        "temperature_adaptation": True,

        "move_range_adaptation": True,
    },

    "results": {

        "initial_objective": float(
            initial_metrics[
                "objective"
            ]
        ),

        "best_objective": float(
            best_metrics[
                "objective"
            ]
        ),

        "improvement": float(
            improvement
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

        "final_move_scale": float(
            move_scale
        ),

        "runtime_seconds": float(
            runtime
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

    "move_scale_history": (
        move_scale_history
    ),
}


# ============================================================
# 25. SAVE REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "stage_c11_samurai_report.json"
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
# 26. FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "C11 COMPLETE"
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
    f"{improvement:.12f}"
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
    f"Initial move scale: "
    f"{INITIAL_MOVE_SCALE:.10e}"
)

print(
    f"Final move scale  : "
    f"{move_scale:.10e}"
)

print(
    f"Runtime           : "
    f"{runtime:.3f} s"
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
    "\nSTAGE C11 PASSED"
)
