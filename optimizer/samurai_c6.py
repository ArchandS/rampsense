"""
RampSense - Stage C6
====================

Calibrated SAMURAI temperature experiment.

Purpose
-------
C6 isolates the temperature behavior of the SAMURAI-style simulated
annealing phase.

Compared with C5:
    - T0 remains the calibrated C4 value.
    - Proposal move scale is FIXED.
    - Move-scale adaptation is disabled.
    - Temperature remains adaptive.
    - Covariance adaptation remains enabled.
    - 500 fitness evaluations are used.

This experiment is intended to determine whether the C4 temperature
calibration works when the proposal scale is held constant.

Important
---------
This is a SAMURAI-style engineering implementation. The paper does not
disclose every numerical SAMURAI constant, so the numerical values used
here should not be described as an exact reproduction of every internal
optimizer setting.
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
    r"\PV dataset\sequences\optimizer"
)

OUTPUT_DIR = OPTIMIZER_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(OPTIMIZER_DIR) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_DIR))


# ============================================================
# IMPORT FITNESS
# ============================================================

import fitness as fitness_module


# ============================================================
# MODEL / OPTIMIZER DIMENSION
# ============================================================

DIM = 1608


# ============================================================
# C6 TEMPERATURE CONFIGURATION
# ============================================================

# Calibrated from Stage C4.
T0 = 0.0002658506

# Temperature termination.
T_MIN = 1.0e-8


# ============================================================
# FIXED PROPOSAL SCALE
# ============================================================

# C5 initial parameter standard deviation:
#
#     0.10019844
#
# C5 initial move scale:
#
#     0.10019844 * 0.01
#     = 0.00100198
#
# C6 deliberately freezes this value.

INITIAL_PARAMETER_STD = 0.10019844

INITIAL_SCALE_FRACTION = 0.01

FIXED_MOVE_SCALE = (
    INITIAL_PARAMETER_STD
    * INITIAL_SCALE_FRACTION
)

# Explicit numerical value used by C6.
FIXED_MOVE_SCALE = 0.0010019844


# ============================================================
# TEMPERATURE ADAPTATION
# ============================================================

ALPHA_MIN = 0.85
ALPHA_MAX = 0.97

TARGET_ACCEPTANCE_LOW = 0.20
TARGET_ACCEPTANCE_HIGH = 0.40


# ============================================================
# ADAPTATION WINDOW
# ============================================================

ADAPTATION_WINDOW = 20


# ============================================================
# COVARIANCE
# ============================================================

COV_WINDOW = 50

COV_UPDATE_EVERY = 50

COV_REG = 1.0e-6


# ============================================================
# EVALUATION BUDGET
# ============================================================

MAX_EVALUATIONS = 500


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# INITIALIZATION
# ============================================================

INITIAL_VECTOR_STD = 0.10

MAX_ABS_PARAMETER = 10.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clip_vector(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Apply a broad numerical safety bound to the optimizer vector.
    """

    return np.clip(
        vector,
        -MAX_ABS_PARAMETER,
        MAX_ABS_PARAMETER,
    )


# ============================================================
# FITNESS
# ============================================================

def evaluate(
    vector: np.ndarray,
) -> float:
    """
    Evaluate one candidate using the Stage-B fitness implementation.
    """

    result = fitness_module.evaluate_fitness(
        vector
    )

    return float(
        result["objective"]
    )


# ============================================================
# INITIAL VECTOR
# ============================================================

def initialize_vector(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create the initial 1608-dimensional parameter vector.
    """

    vector = rng.normal(
        loc=0.0,
        scale=INITIAL_VECTOR_STD,
        size=DIM,
    )

    return vector.astype(
        np.float64
    )


# ============================================================
# COVARIANCE
# ============================================================

def initialize_covariance(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Initialize diagonal covariance using the parameter standard
    deviation.
    """

    parameter_std = float(
        np.std(vector)
    )

    if (
        not np.isfinite(parameter_std)
        or parameter_std <= 0.0
    ):
        parameter_std = INITIAL_VECTOR_STD

    variance = parameter_std ** 2

    covariance = (
        np.eye(
            DIM,
            dtype=np.float64,
        )
        * variance
    )

    return covariance


def safe_cholesky(
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Robust Cholesky decomposition with increasing diagonal
    regularization.
    """

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    covariance = 0.5 * (
        covariance
        + covariance.T
    )

    identity = np.eye(
        covariance.shape[0],
        dtype=np.float64,
    )

    regularization = COV_REG

    for _ in range(12):

        try:

            return np.linalg.cholesky(
                covariance
                + regularization * identity
            )

        except np.linalg.LinAlgError:

            regularization *= 10.0

    # Eigenvalue fallback.
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

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
        covariance_fixed
        + covariance_fixed.T
    )

    return np.linalg.cholesky(
        covariance_fixed
        + regularization * identity
    )


def update_covariance(
    history: list[np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    """
    Estimate covariance from recent accepted states.
    """

    if len(history) < 2:

        return fallback.copy()

    data = np.asarray(
        history,
        dtype=np.float64,
    )

    if data.shape[0] > COV_WINDOW:

        data = data[-COV_WINDOW:]

    if data.shape[0] < 2:

        return fallback.copy()

    covariance = np.cov(
        data,
        rowvar=False,
        dtype=np.float64,
    )

    if covariance.ndim != 2:

        return fallback.copy()

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    covariance = 0.5 * (
        covariance
        + covariance.T
    )

    covariance += (
        COV_REG
        * np.eye(
            DIM,
            dtype=np.float64,
        )
    )

    return covariance


# ============================================================
# PROPOSAL
# ============================================================

def propose(
    current: np.ndarray,
    chol: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one covariance-adapted Gaussian proposal.

    C6 deliberately uses a FIXED move scale.
    """

    z = rng.normal(
        loc=0.0,
        scale=1.0,
        size=DIM,
    )

    step = (
        FIXED_MOVE_SCALE
        * (chol @ z)
    )

    proposal = (
        current
        + step
    )

    proposal = clip_vector(
        proposal
    )

    return proposal.astype(
        np.float64
    )


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
    Metropolis acceptance rule.

    Returns:
        accepted
        uphill_move
    """

    delta = (
        proposed_objective
        - current_objective
    )

    # --------------------------------------------------------
    # Downhill / equal move
    # --------------------------------------------------------

    if delta <= 0.0:

        return True, False

    # --------------------------------------------------------
    # Uphill move
    # --------------------------------------------------------

    if temperature <= 0.0:

        return False, True

    exponent = (
        -delta
        / temperature
    )

    if exponent < -745.0:

        probability = 0.0

    else:

        probability = math.exp(
            exponent
        )

    accepted = (
        rng.random()
        < probability
    )

    return bool(accepted), True


# ============================================================
# ACCEPTANCE CALCULATION
# ============================================================

def calculate_acceptance(
    attempts: int,
    accepts: int,
) -> float:
    """
    Calculate uphill acceptance rate.
    """

    if attempts == 0:

        return 0.0

    return float(
        accepts / attempts
    )


# ============================================================
# TEMPERATURE ADAPTATION
# ============================================================

def adapt_temperature(
    temperature: float,
    uphill_acceptance: float,
) -> tuple[float, float]:
    """
    Adaptive cooling rule.

    High acceptance:
        cool faster.

    Low acceptance:
        cool more slowly.

    Target range:
        20% - 40%
    """

    if uphill_acceptance > TARGET_ACCEPTANCE_HIGH:

        alpha = ALPHA_MIN

    elif uphill_acceptance < TARGET_ACCEPTANCE_LOW:

        alpha = ALPHA_MAX

    else:

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

    new_temperature = (
        temperature
        * alpha
    )

    return (
        float(new_temperature),
        float(alpha),
    )


# ============================================================
# MAIN
# ============================================================

def run_c6() -> dict:

    rng = np.random.default_rng(
        SEED
    )

    start_time = time.perf_counter()

    # ========================================================
    # INITIAL STATE
    # ========================================================

    current = initialize_vector(
        rng
    )

    current = clip_vector(
        current
    )

    initial_std = float(
        np.std(current)
    )

    # ========================================================
    # INITIAL FITNESS
    # ========================================================

    current_objective = evaluate(
        current
    )

    evaluations = 1

    best = current.copy()

    best_objective = (
        current_objective
    )

    # ========================================================
    # COVARIANCE
    # ========================================================

    covariance = initialize_covariance(
        current
    )

    chol = safe_cholesky(
        covariance
    )

    # ========================================================
    # TEMPERATURE
    # ========================================================

    temperature = T0

    # ========================================================
    # STATE HISTORY
    # ========================================================

    state_history = [
        current.copy()
    ]

    objective_history = [
        float(current_objective)
    ]

    temperature_history = [
        float(temperature)
    ]

    acceptance_history = []

    alpha_history = []

    # ========================================================
    # COUNTERS
    # ========================================================

    accepted_total = 0

    rejected_total = 0

    uphill_attempts_total = 0

    uphill_accepts_total = 0

    uphill_attempts_window = 0

    uphill_accepts_window = 0

    covariance_updates = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while (
        evaluations < MAX_EVALUATIONS
        and temperature > T_MIN
    ):

        # ----------------------------------------------------
        # Generate proposal
        # ----------------------------------------------------

        proposal = propose(
            current=current,
            chol=chol,
            rng=rng,
        )

        # ----------------------------------------------------
        # Evaluate proposal
        # ----------------------------------------------------

        proposed_objective = evaluate(
            proposal
        )

        evaluations += 1

        # ----------------------------------------------------
        # Metropolis decision
        # ----------------------------------------------------

        accepted, uphill = (
            metropolis_accept(
                current_objective=current_objective,
                proposed_objective=proposed_objective,
                temperature=temperature,
                rng=rng,
            )
        )

        # ----------------------------------------------------
        # Uphill bookkeeping
        # ----------------------------------------------------

        if uphill:

            uphill_attempts_total += 1

            uphill_attempts_window += 1

        # ----------------------------------------------------
        # Accepted
        # ----------------------------------------------------

        if accepted:

            accepted_total += 1

            current = proposal

            current_objective = (
                proposed_objective
            )

            if uphill:

                uphill_accepts_total += 1

                uphill_accepts_window += 1

            state_history.append(
                current.copy()
            )

            if (
                len(state_history)
                > COV_WINDOW
            ):

                state_history = (
                    state_history[
                        -COV_WINDOW:
                    ]
                )

            # ------------------------------------------------
            # Best solution
            # ------------------------------------------------

            if (
                current_objective
                < best_objective
            ):

                best_objective = (
                    current_objective
                )

                best = current.copy()

        # ----------------------------------------------------
        # Rejected
        # ----------------------------------------------------

        else:

            rejected_total += 1

        # ----------------------------------------------------
        # Record objective
        # ----------------------------------------------------

        objective_history.append(
            float(current_objective)
        )

        # ====================================================
        # COVARIANCE UPDATE
        # ====================================================

        if (
            evaluations > 1
            and evaluations
            % COV_UPDATE_EVERY
            == 0
        ):

            covariance = update_covariance(
                state_history,
                covariance,
            )

            chol = safe_cholesky(
                covariance
            )

            covariance_updates += 1

        # ====================================================
        # TEMPERATURE ADAPTATION
        # ====================================================

        if (
            evaluations > 1
            and evaluations
            % ADAPTATION_WINDOW
            == 0
        ):

            uphill_acceptance = (
                calculate_acceptance(
                    uphill_attempts_window,
                    uphill_accepts_window,
                )
            )

            acceptance_history.append(
                float(
                    uphill_acceptance
                )
            )

            # ------------------------------------------------
            # C6:
            # NO MOVE-SCALE ADAPTATION
            # ------------------------------------------------

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
        # Record temperature
        # ----------------------------------------------------

        temperature_history.append(
            float(temperature)
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    overall_acceptance = (
        accepted_total
        / max(
            evaluations - 1,
            1,
        )
    )

    uphill_acceptance = (
        calculate_acceptance(
            uphill_attempts_total,
            uphill_accepts_total,
        )
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
        / "stage_c6_samurai_best.npy"
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
        / "stage_c6_samurai_covariance.npy"
    )

    np.save(
        covariance_path,
        covariance,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "stage": "C6",

        "algorithm": (
            "SAMURAI-style simulated annealing "
            "with fixed proposal scale"
        ),

        "dimension": DIM,

        "seed": SEED,

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        "initial_temperature": T0,

        "final_temperature": float(
            temperature
        ),

        "temperature_min": T_MIN,

        "alpha_min": ALPHA_MIN,

        "alpha_max": ALPHA_MAX,

        # ----------------------------------------------------
        # Proposal
        # ----------------------------------------------------

        "initial_parameter_std": (
            initial_std
        ),

        "initial_scale_fraction": (
            INITIAL_SCALE_FRACTION
        ),

        "fixed_move_scale": (
            FIXED_MOVE_SCALE
        ),

        "move_scale_adaptation": False,

        # ----------------------------------------------------
        # Covariance
        # ----------------------------------------------------

        "covariance_window": (
            COV_WINDOW
        ),

        "covariance_update_every": (
            COV_UPDATE_EVERY
        ),

        "covariance_regularization": (
            COV_REG
        ),

        "covariance_updates": (
            covariance_updates
        ),

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        "max_evaluations": (
            MAX_EVALUATIONS
        ),

        "evaluations": (
            evaluations
        ),

        # ----------------------------------------------------
        # Objective
        # ----------------------------------------------------

        "initial_objective": float(
            objective_history[0]
        ),

        "best_objective": float(
            best_objective
        ),

        "final_objective": float(
            current_objective
        ),

        # ----------------------------------------------------
        # Acceptance
        # ----------------------------------------------------

        "accepted_moves": (
            accepted_total
        ),

        "rejected_moves": (
            rejected_total
        ),

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
            uphill_acceptance
        ),

        # ----------------------------------------------------
        # Runtime
        # ----------------------------------------------------

        "runtime_seconds": float(
            runtime
        ),

        # ----------------------------------------------------
        # Files
        # ----------------------------------------------------

        "best_vector_path": str(
            best_path
        ),

        "covariance_path": str(
            covariance_path
        ),

        # ----------------------------------------------------
        # Histories
        # ----------------------------------------------------

        "objective_history": [
            float(x)
            for x in objective_history
        ],

        "temperature_history": [
            float(x)
            for x in temperature_history
        ],

        "uphill_acceptance_history": [
            float(x)
            for x in acceptance_history
        ],

        "alpha_history": [
            float(x)
            for x in alpha_history
        ],

        # ----------------------------------------------------
        # Method note
        # ----------------------------------------------------

        "notes": [
            (
                "C6 is a calibration experiment intended "
                "to isolate temperature behavior."
            ),
            (
                "The C4-calibrated T0 of 0.0002658506 "
                "is retained."
            ),
            (
                "Proposal move scale is fixed at "
                "0.0010019844."
            ),
            (
                "Move-scale adaptation is deliberately "
                "disabled."
            ),
            (
                "Temperature adaptation remains enabled."
            ),
            (
                "Covariance adaptation remains enabled."
            ),
            (
                "The Stage-B fixed fitness subset is used."
            ),
            (
                "Undisclosed numerical constants are "
                "engineering choices rather than claims "
                "of exact reproduction."
            ),
        ],
    }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "stage_c6_samurai_report.json"
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
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("STAGE C6 — FIXED-SCALE SAMURAI CALIBRATION")
    print("=" * 70)

    print(
        f"Dimension              : {DIM}"
    )

    print(
        f"T0                     : "
        f"{T0:.10f}"
    )

    print(
        f"Initial parameter std  : "
        f"{initial_std:.8f}"
    )

    print(
        f"Fixed move scale       : "
        f"{FIXED_MOVE_SCALE:.10f}"
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
        f"{uphill_acceptance:.6f}"
    )

    print()
    print(
        f"Final temperature     : "
        f"{temperature:.10e}"
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
    # VALIDATION CHECKS
    # ========================================================

    checks = [

        (
            best.shape
            == (DIM,)
        ),

        np.all(
            np.isfinite(best)
        ),

        np.isfinite(
            best_objective
        ),

        (
            evaluations
            <= MAX_EVALUATIONS
        ),

        (
            covariance.shape
            == (DIM, DIM)
        ),

        np.all(
            np.isfinite(
                covariance
            )
        ),

        (
            best_objective
            <= objective_history[0]
            + 1e-12
        ),

        (
            FIXED_MOVE_SCALE
            == 0.0010019844
        ),

    ]

    check_names = [

        "best vector shape",

        "best vector finite",

        "best objective finite",

        "evaluation budget",

        "covariance shape",

        "covariance finite",

        "best <= initial objective",

        "fixed move scale",

    ]

    print()
    print("C6 validation checks:")

    for name, passed in zip(
        check_names,
        checks,
    ):

        print(
            f"  "
            f"{'PASS' if passed else 'FAIL'}"
            f"  {name}"
        )

    if all(checks):

        print()
        print(
            "STAGE C6 PASSED."
        )

    else:

        print()
        print(
            "STAGE C6 FAILED."
        )

    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_c6()
