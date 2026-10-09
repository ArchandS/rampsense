"""
RampSense - Stage C9
====================

SAMURAI temperature recalibration experiment.

Purpose
-------
C9 is identical to Stage C6 except for the initial temperature.

C6:
    T0 = 0.0002658506

C9:
    T0 = 0.000012885171556

The C9 temperature was calibrated in Stage C8 using the
ACTUAL C6 proposal mechanism.

Everything else is intentionally kept unchanged so that
C6 vs C9 is a controlled experiment.

C9 configuration:
    - 1608-dimensional direct parameter optimization
    - fixed proposal move scale
    - adaptive temperature
    - adaptive covariance
    - 500 fitness evaluations
    - same seed
    - same fixed Stage-B fitness subset
    - same covariance settings
    - same temperature adaptation rule
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
# C9 TEMPERATURE
# ============================================================

# Stage C8 calibration:
#
# Target uphill acceptance = 30%
# Recommended temperature:
#
#     1.2885171556e-05
#
# This is the ONLY major numerical change from C6.

T0 = 1.2885171556e-05

T_MIN = 1.0e-8


# ============================================================
# FIXED PROPOSAL SCALE
# ============================================================

# EXACTLY the same as C6.

INITIAL_PARAMETER_STD = 0.10019844

INITIAL_SCALE_FRACTION = 0.01

FIXED_MOVE_SCALE = (
    INITIAL_PARAMETER_STD
    * INITIAL_SCALE_FRACTION
)

# Explicit C6 value.
FIXED_MOVE_SCALE = 0.0010019844


# ============================================================
# TEMPERATURE ADAPTATION
# ============================================================

# EXACTLY the same as C6.

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

# EXACTLY the same as C6.

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
# UTILITY
# ============================================================

def clip_vector(
    vector: np.ndarray,
) -> np.ndarray:
    """
    Apply the same numerical safety bound as C6.
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
    Evaluate one candidate using Stage-B fitness.
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
    EXACTLY the same initialization as C6.
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
# COVARIANCE INITIALIZATION
# ============================================================

def initialize_covariance(
    vector: np.ndarray,
) -> np.ndarray:
    """
    EXACTLY the same covariance initialization as C6.

        Sigma = parameter_std^2 * I
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


# ============================================================
# SAFE CHOLESKY
# ============================================================

def safe_cholesky(
    covariance: np.ndarray,
) -> np.ndarray:
    """
    EXACTLY the same Cholesky strategy as C6.
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

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
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


# ============================================================
# COVARIANCE UPDATE
# ============================================================

def update_covariance(
    history: list[np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    """
    EXACTLY the same covariance update as C6.
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
    EXACTLY the same proposal as C6.

        z ~ N(0,I)

        step = FIXED_MOVE_SCALE * (chol @ z)

        proposal = current + step
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
    EXACTLY the same Metropolis rule as C6.

    Returns:
        accepted
        uphill_move
    """

    delta = (
        proposed_objective
        - current_objective
    )

    # Downhill/equal move.

    if delta <= 0.0:

        return True, False

    # Uphill move.

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

    return (
        bool(accepted),
        True,
    )


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
    EXACTLY the same adaptive cooling rule as C6.

    High acceptance:
        cool faster.

    Low acceptance:
        cool more slowly.
    """

    if (
        uphill_acceptance
        > TARGET_ACCEPTANCE_HIGH
    ):

        alpha = ALPHA_MIN

    elif (
        uphill_acceptance
        < TARGET_ACCEPTANCE_LOW
    ):

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

def run_c9() -> dict:

    rng = np.random.default_rng(
        SEED
    )

    start_time = time.perf_counter()

    print()
    print("=" * 72)
    print("STAGE C9 — CALIBRATED-TEMPERATURE SAMURAI")
    print("=" * 72)

    print()
    print("C9 differs from C6 ONLY in initial temperature.")

    print(
        f"C6 T0                 : "
        f"0.0002658506"
    )

    print(
        f"C9 T0                 : "
        f"{T0:.10f}"
    )

    print(
        f"Fixed move scale      : "
        f"{FIXED_MOVE_SCALE:.10f}"
    )

    print(
        f"Covariance window     : "
        f"{COV_WINDOW}"
    )

    print(
        f"Covariance update     : "
        f"every {COV_UPDATE_EVERY}"
    )

    print(
        f"Evaluation budget     : "
        f"{MAX_EVALUATIONS}"
    )

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

    print()
    print(
        f"Initial parameter std : "
        f"{initial_std:.10f}"
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

    print(
        f"Initial objective     : "
        f"{current_objective:.12f}"
    )

    # ========================================================
    # COVARIANCE
    # ========================================================

    covariance = (
        initialize_covariance(
            current
        )
    )

    chol = safe_cholesky(
        covariance
    )

    print(
        f"Initial covariance variance: "
        f"{initial_std ** 2:.10f}"
    )

    print(
        f"Initial Cholesky diag mean : "
        f"{np.mean(np.diag(chol)):.10f}"
    )

    # ========================================================
    # TEMPERATURE
    # ========================================================

    temperature = T0

    # ========================================================
    # HISTORY
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
        # Proposal
        # ----------------------------------------------------

        proposal = propose(
            current=current,
            chol=chol,
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
        # Metropolis
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

            covariance = (
                update_covariance(
                    state_history,
                    covariance,
                )
            )

            chol = safe_cholesky(
                covariance
            )

            covariance_updates += 1

            print(
                f"[Covariance update] "
                f"eval={evaluations:3d} "
                f"best={best_objective:.9f} "
                f"current={current_objective:.9f}"
            )

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

            temperature, alpha = (
                adapt_temperature(
                    temperature,
                    uphill_acceptance,
                )
            )

            alpha_history.append(
                float(alpha)
            )

            print(
                f"[Temperature] "
                f"eval={evaluations:3d} "
                f"uphill_acc={uphill_acceptance:.4f} "
                f"alpha={alpha:.4f} "
                f"T={temperature:.10e} "
                f"best={best_objective:.9f}"
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
    # SAVE BEST
    # ========================================================

    best_path = (
        OUTPUT_DIR
        / "stage_c9_samurai_best.npy"
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
        / "stage_c9_samurai_covariance.npy"
    )

    np.save(
        covariance_path,
        covariance,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "stage": "C9",

        "algorithm": (
            "SAMURAI-style simulated annealing "
            "with C8-calibrated initial temperature"
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

        "target_acceptance_low": (
            TARGET_ACCEPTANCE_LOW
        ),

        "target_acceptance_high": (
            TARGET_ACCEPTANCE_HIGH
        ),

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
        # Parameter statistics
        # ----------------------------------------------------

        "final_parameter_std": (
            final_parameter_std
        ),

        "best_parameter_std": (
            best_parameter_std
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
        # Experimental note
        # ----------------------------------------------------

        "notes": [

            (
                "C9 is a controlled comparison against C6."
            ),

            (
                "The only intended numerical change is "
                "the initial temperature."
            ),

            (
                "C8 calibrated T0 using the actual C6 "
                "initial proposal distribution."
            ),

            (
                "C8 recommended approximately 30% "
                "initial uphill acceptance."
            ),

            (
                "C9 retains covariance adaptation."
            ),

            (
                "C9 retains the C6 temperature adaptation "
                "rule after each 20-evaluation window."
            ),

            (
                "The calibrated temperature is an engineering "
                "choice, not a value disclosed by the paper."
            ),
        ],
    }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "stage_c9_samurai_report.json"
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
    print("=" * 72)
    print("STAGE C9 — FINAL RESULTS")
    print("=" * 72)

    print(
        f"Dimension              : {DIM}"
    )

    print(
        f"Initial temperature    : "
        f"{T0:.10e}"
    )

    print(
        f"Final temperature      : "
        f"{temperature:.10e}"
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
        f"Accepted moves         : "
        f"{accepted_total}"
    )

    print(
        f"Rejected moves         : "
        f"{rejected_total}"
    )

    print(
        f"Overall acceptance     : "
        f"{overall_acceptance:.6f}"
    )

    print(
        f"Uphill attempts        : "
        f"{uphill_attempts_total}"
    )

    print(
        f"Uphill accepts         : "
        f"{uphill_accepts_total}"
    )

    print(
        f"Uphill acceptance      : "
        f"{uphill_acceptance:.6f}"
    )

    print()

    print(
        f"Final parameter std    : "
        f"{final_parameter_std:.8f}"
    )

    print(
        f"Best parameter std     : "
        f"{best_parameter_std:.8f}"
    )

    print(
        f"Covariance updates     : "
        f"{covariance_updates}"
    )

    print(
        f"Runtime                : "
        f"{runtime:.3f} s"
    )

    print()

    print(
        f"Best vector saved      : "
        f"{best_path}"
    )

    print(
        f"Covariance saved       : "
        f"{covariance_path}"
    )

    print(
        f"Report saved           : "
        f"{report_path}"
    )

    print("=" * 72)

    # ========================================================
    # VALIDATION
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
            + 1.0e-12
        ),

        (
            FIXED_MOVE_SCALE
            == 0.0010019844
        ),

        (
            T0
            == 1.2885171556e-05
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

        "C8 calibrated T0",

    ]

    print()
    print("C9 validation checks:")

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
            "STAGE C9 PASSED."
        )

    else:

        print()
        print(
            "STAGE C9 FAILED."
        )

    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_c9()
