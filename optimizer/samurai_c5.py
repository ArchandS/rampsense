"""
Stage C5 — Calibrated SAMURAI prototype

Purpose
-------
Direct optimization of the 1608-dimensional LSTM parameter vector using
a SAMURAI-style simulated annealing phase.

C5 changes from C3:
    - Calibrated initial temperature:
        T0 = 0.0002658506

The optimizer:
    1. Loads the fixed Stage-B fitness function.
    2. Initializes a 1608-dimensional parameter vector.
    3. Uses covariance-adapted proposals.
    4. Uses Metropolis acceptance.
    5. Adapts move scale according to acceptance behavior.
    6. Uses adaptive cooling.
    7. Stops after a fixed evaluation budget.
    8. Saves the best vector, covariance matrix, and report.

IMPORTANT
---------
This is a project implementation/prototype of the SAMURAI structure.
The uploaded paper does not disclose every numerical SAMURAI constant,
so constants such as T0, cooling bounds, and move-scale settings are
engineering/calibration choices rather than claims of exact reproduction.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

OPTIMIZER_DIR = Path(
    r"C:\Users\archa\Downloads\rampsense dataset"
    r"\\PV dataset\sequences\optimizer"
)

OUTPUT_DIR = OPTIMIZER_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make optimizer directory importable.
if str(OPTIMIZER_DIR) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_DIR))


# ============================================================
# IMPORT FITNESS
# ============================================================

import fitness as fitness_module


# ============================================================
# SAMURAI CONFIGURATION
# ============================================================

DIM = 1608

# ------------------------------------------------------------
# C5 CALIBRATED INITIAL TEMPERATURE
# ------------------------------------------------------------

T0 = 0.0002658506

# Temperature termination.
T_MIN = 1.0e-8

# Adaptive cooling bounds.
ALPHA_MIN = 0.85
ALPHA_MAX = 0.97

# ------------------------------------------------------------
# Proposal scale
# ------------------------------------------------------------

# C3/C5 uses parameter-standard-deviation scaling.
INITIAL_SCALE_FRACTION = 0.01

# Keep proposal scale within reasonable bounds.
MOVE_SCALE_MIN = 0.001
MOVE_SCALE_MAX = 0.10

# ------------------------------------------------------------
# Covariance
# ------------------------------------------------------------

COV_WINDOW = 50

# Small diagonal regularization before Cholesky.
COV_REG = 1.0e-6

# ------------------------------------------------------------
# Evaluation budget
# ------------------------------------------------------------

MAX_EVALUATIONS = 500

# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

SEED = 42

# ------------------------------------------------------------
# Initial vector
# ------------------------------------------------------------

# Parameter initialization scale.
INITIAL_PARAMETER_STD = 0.10

# ------------------------------------------------------------
# Acceptance adaptation
# ------------------------------------------------------------

# Target uphill acceptance.
TARGET_ACCEPTANCE_LOW = 0.20
TARGET_ACCEPTANCE_HIGH = 0.40

# Number of proposals between adaptation checks.
ADAPTATION_WINDOW = 20

# Move-scale multipliers.
SCALE_INCREASE = 1.20
SCALE_DECREASE = 0.80

# ------------------------------------------------------------
# Covariance adaptation
# ------------------------------------------------------------

COV_UPDATE_EVERY = 50

# ------------------------------------------------------------
# Safety
# ------------------------------------------------------------

MAX_ABS_PARAMETER = 10.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clip_vector(vector: np.ndarray) -> np.ndarray:
    """
    Keep parameters within a broad numerical safety range.
    """
    return np.clip(
        vector,
        -MAX_ABS_PARAMETER,
        MAX_ABS_PARAMETER,
    )


def safe_cholesky(
    covariance: np.ndarray,
    regularization: float = COV_REG,
) -> np.ndarray:
    """
    Robust Cholesky decomposition.

    If the covariance is not positive definite, progressively increase
    diagonal regularization.
    """

    covariance = np.asarray(covariance, dtype=np.float64)

    covariance = 0.5 * (covariance + covariance.T)

    identity = np.eye(covariance.shape[0], dtype=np.float64)

    reg = regularization

    for _ in range(12):
        try:
            return np.linalg.cholesky(
                covariance + reg * identity
            )
        except np.linalg.LinAlgError:
            reg *= 10.0

    # Last resort: eigenvalue correction.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    eigenvalues = np.maximum(
        eigenvalues,
        regularization,
    )

    covariance_fixed = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    covariance_fixed = 0.5 * (
        covariance_fixed + covariance_fixed.T
    )

    return np.linalg.cholesky(
        covariance_fixed + regularization * identity
    )


def evaluate(vector: np.ndarray) -> float:
    """
    Evaluate a candidate parameter vector using the Stage-B fitness
    implementation.
    """

    result = fitness_module.evaluate_fitness(vector)

    return float(result["objective"])


def summarize_acceptance(
    uphill_attempts: int,
    uphill_accepts: int,
) -> float:

    if uphill_attempts == 0:
        return 0.0

    return float(uphill_accepts / uphill_attempts)


# ============================================================
# INITIAL PARAMETER VECTOR
# ============================================================

def initialize_vector(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Initialize a 1608-dimensional parameter vector.

    The exact model parameterization is handled by Stage A's
    parameter-vector utilities. Here we only create the starting
    numerical vector.
    """

    vector = rng.normal(
        loc=0.0,
        scale=INITIAL_PARAMETER_STD,
        size=DIM,
    )

    return vector.astype(np.float64)


# ============================================================
# COVARIANCE INITIALIZATION
# ============================================================

def initialize_covariance(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Initialize covariance using the parameter-scale information.

    C5 follows the C3 idea of scaling proposals relative to the
    parameter standard deviation.
    """

    parameter_std = float(np.std(vector))

    if not np.isfinite(parameter_std):
        parameter_std = INITIAL_PARAMETER_STD

    if parameter_std <= 0.0:
        parameter_std = INITIAL_PARAMETER_STD

    # Base covariance.
    variance = parameter_std ** 2

    covariance = np.eye(
        DIM,
        dtype=np.float64,
    ) * variance

    return covariance


# ============================================================
# PROPOSAL GENERATION
# ============================================================

def propose(
    current: np.ndarray,
    chol: np.ndarray,
    move_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a covariance-adapted proposal.

    Proposal:

        x_new = x + scale * L z

    where:

        z ~ N(0, I)
        LL^T = covariance
    """

    z = rng.normal(
        loc=0.0,
        scale=1.0,
        size=DIM,
    )

    step = move_scale * (chol @ z)

    proposal = current + step

    proposal = clip_vector(proposal)

    return proposal.astype(np.float64)


# ============================================================
# METROPOLIS ACCEPTANCE
# ============================================================

def metropolis_accept(
    current_objective: float,
    proposed_objective: float,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[bool, bool]:
    """
    Metropolis acceptance.

    Returns:
        accepted
        is_uphill
    """

    delta = proposed_objective - current_objective

    # Improvement.
    if delta <= 0.0:
        return True, False

    # Uphill move.
    if temperature <= 0.0:
        return False, True

    exponent = -delta / temperature

    # Underflow-safe acceptance probability.
    if exponent < -745.0:
        probability = 0.0
    else:
        probability = math.exp(exponent)

    accepted = bool(
        rng.random() < probability
    )

    return accepted, True


# ============================================================
# COVARIANCE UPDATE
# ============================================================

def update_covariance(
    history: list[np.ndarray],
    fallback_covariance: np.ndarray,
) -> np.ndarray:
    """
    Estimate covariance from recent accepted/current states.

    To keep the matrix numerically symmetric, explicitly symmetrize
    the result.
    """

    if len(history) < 2:
        return fallback_covariance.copy()

    data = np.asarray(
        history,
        dtype=np.float64,
    )

    # Safety against accidental huge arrays.
    if data.shape[0] > COV_WINDOW:
        data = data[-COV_WINDOW:]

    # Need enough observations to estimate covariance.
    if data.shape[0] < 2:
        return fallback_covariance.copy()

    covariance = np.cov(
        data,
        rowvar=False,
        dtype=np.float64,
    )

    # np.cov returns scalar in degenerate cases.
    if covariance.ndim != 2:
        return fallback_covariance.copy()

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    covariance = 0.5 * (
        covariance + covariance.T
    )

    # Regularization.
    covariance += (
        COV_REG * np.eye(
            covariance.shape[0],
            dtype=np.float64,
        )
    )

    return covariance


# ============================================================
# ADAPTIVE MOVE SCALE
# ============================================================

def adapt_move_scale(
    move_scale: float,
    uphill_acceptance: float,
) -> float:
    """
    Adapt proposal magnitude according to observed uphill acceptance.

    High acceptance:
        proposals are too small -> increase scale.

    Low acceptance:
        proposals are too large -> decrease scale.
    """

    if uphill_acceptance > TARGET_ACCEPTANCE_HIGH:

        move_scale *= SCALE_INCREASE

    elif uphill_acceptance < TARGET_ACCEPTANCE_LOW:

        move_scale *= SCALE_DECREASE

    move_scale = float(
        np.clip(
            move_scale,
            MOVE_SCALE_MIN,
            MOVE_SCALE_MAX,
        )
    )

    return move_scale


# ============================================================
# ADAPTIVE TEMPERATURE
# ============================================================

def adapt_temperature(
    temperature: float,
    uphill_acceptance: float,
) -> tuple[float, float]:
    """
    Adapt cooling rate.

    If acceptance is high:
        cool more aggressively.

    If acceptance is low:
        cool more slowly.

    Returns:
        new_temperature
        alpha_used
    """

    if uphill_acceptance > TARGET_ACCEPTANCE_HIGH:

        # Faster cooling.
        alpha = ALPHA_MIN

    elif uphill_acceptance < TARGET_ACCEPTANCE_LOW:

        # Slower cooling.
        alpha = ALPHA_MAX

    else:

        # Interpolate between slow and fast cooling.
        ratio = (
            uphill_acceptance
            - TARGET_ACCEPTANCE_LOW
        ) / (
            TARGET_ACCEPTANCE_HIGH
            - TARGET_ACCEPTANCE_LOW
        )

        alpha = (
            ALPHA_MAX
            - ratio
            * (
                ALPHA_MAX
                - ALPHA_MIN
            )
        )

    new_temperature = temperature * alpha

    return float(new_temperature), float(alpha)


# ============================================================
# MAIN SAMURAI OPTIMIZATION
# ============================================================

def run_samurai() -> dict:
    """
    Execute the complete C5 SAMURAI phase.
    """

    rng = np.random.default_rng(SEED)

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Initial vector
    # --------------------------------------------------------

    current = initialize_vector(rng)

    current = clip_vector(current)

    initial_std = float(
        np.std(current)
    )

    # --------------------------------------------------------
    # Initial objective
    # --------------------------------------------------------

    current_objective = evaluate(current)

    evaluations = 1

    best = current.copy()
    best_objective = current_objective

    # --------------------------------------------------------
    # Initial covariance
    # --------------------------------------------------------

    covariance = initialize_covariance(
        current
    )

    chol = safe_cholesky(
        covariance
    )

    # --------------------------------------------------------
    # Initial move scale
    # --------------------------------------------------------

    move_scale = (
        initial_std
        * INITIAL_SCALE_FRACTION
    )

    move_scale = float(
        np.clip(
            move_scale,
            MOVE_SCALE_MIN,
            MOVE_SCALE_MAX,
        )
    )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temperature = T0

    # --------------------------------------------------------
    # Histories
    # --------------------------------------------------------

    state_history: list[np.ndarray] = [
        current.copy()
    ]

    objective_history = [
        float(current_objective)
    ]

    temperature_history = [
        float(temperature)
    ]

    move_scale_history = [
        float(move_scale)
    ]

    acceptance_history = []

    alpha_history = []

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    accepted_total = 0

    rejected_total = 0

    uphill_attempts_total = 0

    uphill_accepts_total = 0

    covariance_updates = 0

    uphill_attempts_window = 0

    uphill_accepts_window = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while (
        evaluations < MAX_EVALUATIONS
        and temperature > T_MIN
    ):

        # ----------------------------------------------------
        # Proposal
        # ----------------------------------------------------

        proposal = propose(
            current=current,
            chol=chol,
            move_scale=move_scale,
            rng=rng,
        )

        # ----------------------------------------------------
        # Fitness
        # ----------------------------------------------------

        proposed_objective = evaluate(
            proposal
        )

        evaluations += 1

        # ----------------------------------------------------
        # Acceptance
        # ----------------------------------------------------

        accepted, uphill = metropolis_accept(
            current_objective=current_objective,
            proposed_objective=proposed_objective,
            temperature=temperature,
            rng=rng,
        )

        if uphill:

            uphill_attempts_total += 1
            uphill_attempts_window += 1

        if accepted:

            accepted_total += 1

            current = proposal
            current_objective = proposed_objective

            if uphill:

                uphill_accepts_total += 1
                uphill_accepts_window += 1

            # Record accepted state.
            state_history.append(
                current.copy()
            )

            # Keep only recent states.
            if len(state_history) > COV_WINDOW:

                state_history = (
                    state_history[-COV_WINDOW:]
                )

            # ------------------------------------------------
            # Best solution
            # ------------------------------------------------

            if current_objective < best_objective:

                best_objective = (
                    current_objective
                )

                best = current.copy()

        else:

            rejected_total += 1

        # ----------------------------------------------------
        # Objective history
        # ----------------------------------------------------

        objective_history.append(
            float(current_objective)
        )

        # ----------------------------------------------------
        # Best-effort covariance update
        # ----------------------------------------------------

        if (
            evaluations > 1
            and evaluations % COV_UPDATE_EVERY == 0
        ):

            covariance = update_covariance(
                state_history,
                covariance,
            )

            chol = safe_cholesky(
                covariance
            )

            covariance_updates += 1

        # ----------------------------------------------------
        # Adaptation window
        # ----------------------------------------------------

        if (
            evaluations > 1
            and evaluations % ADAPTATION_WINDOW == 0
        ):

            uphill_acceptance = (
                summarize_acceptance(
                    uphill_attempts_window,
                    uphill_accepts_window,
                )
            )

            acceptance_history.append(
                float(uphill_acceptance)
            )

            # ----------------------------------------------
            # Adapt move scale
            # ----------------------------------------------

            move_scale = adapt_move_scale(
                move_scale,
                uphill_acceptance,
            )

            # ----------------------------------------------
            # Adapt temperature
            # ----------------------------------------------

            temperature, alpha = (
                adapt_temperature(
                    temperature,
                    uphill_acceptance,
                )
            )

            alpha_history.append(
                float(alpha)
            )

            uphill_attempts_window = 0
            uphill_accepts_window = 0

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        temperature_history.append(
            float(temperature)
        )

        move_scale_history.append(
            float(move_scale)
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    overall_uphill_acceptance = (
        summarize_acceptance(
            uphill_attempts_total,
            uphill_accepts_total,
        )
    )

    overall_acceptance = (
        accepted_total
        / max(evaluations - 1, 1)
    )

    final_parameter_std = float(
        np.std(current)
    )

    best_parameter_std = float(
        np.std(best)
    )

    # ========================================================
    # SAVE BEST VECTOR
    # ========================================================

    best_path = (
        OUTPUT_DIR
        / "stage_c5_samurai_best.npy"
    )

    np.save(
        best_path,
        best,
    )

    # ========================================================
    # SAVE COVARIANCE
    # ========================================================

    covariance_path = (
        OUTPUT_DIR
        / "stage_c5_samurai_covariance.npy"
    )

    np.save(
        covariance_path,
        covariance,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage": "C5",
        "algorithm": "SAMURAI-style simulated annealing",
        "dimension": DIM,

        "seed": SEED,

        "initial_temperature": T0,
        "final_temperature": float(
            temperature
        ),
        "temperature_min": T_MIN,

        "alpha_min": ALPHA_MIN,
        "alpha_max": ALPHA_MAX,

        "initial_parameter_std": initial_std,
        "final_parameter_std": final_parameter_std,
        "best_parameter_std": best_parameter_std,

        "initial_scale_fraction": (
            INITIAL_SCALE_FRACTION
        ),

        "initial_move_scale": (
            initial_std
            * INITIAL_SCALE_FRACTION
        ),

        "final_move_scale": float(
            move_scale
        ),

        "move_scale_min": MOVE_SCALE_MIN,
        "move_scale_max": MOVE_SCALE_MAX,

        "covariance_window": COV_WINDOW,
        "covariance_regularization": COV_REG,
        "covariance_updates": covariance_updates,

        "max_evaluations": MAX_EVALUATIONS,
        "evaluations": evaluations,

        "initial_objective": float(
            objective_history[0]
        ),

        "best_objective": float(
            best_objective
        ),

        "final_objective": float(
            current_objective
        ),

        "accepted_moves": accepted_total,
        "rejected_moves": rejected_total,

        "overall_acceptance": float(
            overall_acceptance
        ),

        "uphill_attempts": (
            uphill_attempts_total
        ),

        "uphill_accepts": (
            uphill_accepts_total
        ),

        "uphill_acceptance": float(
            overall_uphill_acceptance
        ),

        "runtime_seconds": float(
            runtime
        ),

        "best_vector_path": str(
            best_path
        ),

        "covariance_path": str(
            covariance_path
        ),

        "objective_history": [
            float(x)
            for x in objective_history
        ],

        "temperature_history": [
            float(x)
            for x in temperature_history
        ],

        "move_scale_history": [
            float(x)
            for x in move_scale_history
        ],

        "uphill_acceptance_history": [
            float(x)
            for x in acceptance_history
        ],

        "alpha_history": [
            float(x)
            for x in alpha_history
        ],

        "notes": [
            "C5 uses the calibrated initial temperature "
            "from Stage C4.",
            "T0 = 0.0002658506 corresponds to the "
            "C4 1-percent perturbation calibration.",
            "The optimizer directly modifies the 1608-dimensional "
            "model parameter vector.",
            "The Stage-B fixed fitness subset is used.",
            "SAMURAI numerical constants not disclosed by the "
            "paper are treated as engineering/calibration choices.",
        ],
    }

    report_path = (
        OUTPUT_DIR
        / "stage_c5_samurai_report.json"
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

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE C5 — CALIBRATED SAMURAI")
    print("=" * 70)

    print(f"Dimension              : {DIM}")
    print(f"T0                     : {T0:.10f}")
    print(f"Initial parameter std  : {initial_std:.8f}")
    print(
        f"Initial move scale     : "
        f"{initial_std * INITIAL_SCALE_FRACTION:.8f}"
    )

    print()
    print(
        f"Initial objective      : "
        f"{objective_history[0]:.12f}"
    )

    print(
        f"Best objective         : "
        f"{best_objective:.12f}"
    )

    print(
        f"Final objective        : "
        f"{current_objective:.12f}"
    )

    print()
    print(
        f"Evaluations            : "
        f"{evaluations}"
    )

    print(
        f"Accepted moves        : "
        f"{accepted_total}"
    )

    print(
        f"Rejected moves        : "
        f"{rejected_total}"
    )

    print(
        f"Overall acceptance    : "
        f"{overall_acceptance:.6f}"
    )

    print(
        f"Uphill attempts       : "
        f"{uphill_attempts_total}"
    )

    print(
        f"Uphill accepts        : "
        f"{uphill_accepts_total}"
    )

    print(
        f"Uphill acceptance     : "
        f"{overall_uphill_acceptance:.6f}"
    )

    print()
    print(
        f"Final temperature     : "
        f"{temperature:.10e}"
    )

    print(
        f"Final move scale      : "
        f"{move_scale:.8f}"
    )

    print(
        f"Covariance updates    : "
        f"{covariance_updates}"
    )

    print(
        f"Runtime                : "
        f"{runtime:.3f} s"
    )

    print()
    print(
        f"Best vector saved     : "
        f"{best_path}"
    )

    print(
        f"Covariance saved      : "
        f"{covariance_path}"
    )

    print(
        f"Report saved          : "
        f"{report_path}"
    )

    print("=" * 70)

    # ========================================================
    # BASIC PASS CONDITIONS
    # ========================================================

    checks = []

    checks.append(
        best.shape == (DIM,)
    )

    checks.append(
        np.all(np.isfinite(best))
    )

    checks.append(
        np.isfinite(best_objective)
    )

    checks.append(
        evaluations <= MAX_EVALUATIONS
    )

    checks.append(
        covariance.shape == (DIM, DIM)
    )

    checks.append(
        np.all(np.isfinite(covariance))
    )

    checks.append(
        best_objective
        <= objective_history[0]
        + 1e-12
    )

    print()
    print("C5 validation checks:")

    check_names = [
        "best vector shape",
        "best vector finite",
        "best objective finite",
        "evaluation budget",
        "covariance shape",
        "covariance finite",
        "best <= initial objective",
    ]

    for name, passed in zip(
        check_names,
        checks,
    ):

        print(
            f"  {'PASS' if passed else 'FAIL'}  {name}"
        )

    if all(checks):

        print()
        print("STAGE C5 PASSED.")

    else:

        print()
        print("STAGE C5 FAILED.")

    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_samurai()
