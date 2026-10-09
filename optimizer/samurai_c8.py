"""
RampSense - Stage C8
====================

Actual-proposal temperature calibration for SAMURAI.

Purpose
-------
Calibrate the initial SAMURAI temperature using the EXACT initial
proposal mechanism used by Stage C6.

C6 proposal:

    x' = x + FIXED_MOVE_SCALE * (L @ z)

where:

    z ~ N(0, I)

and the initial covariance is:

    Sigma = parameter_std^2 * I

Therefore:

    L = parameter_std * I

This stage:
    1. Recreates the exact C6 initial vector.
    2. Recreates the exact C6 initial covariance.
    3. Generates a fixed set of proposals.
    4. Evaluates their actual fitness deltas.
    5. Measures positive-delta statistics.
    6. Estimates temperatures producing approximately
       20%, 30%, 40%, and 50% uphill acceptance.
    7. Reports the expected acceptance at the original C4 T0.

IMPORTANT
---------
This is a calibration experiment, not a SAMURAI optimization run.

No:
    - covariance adaptation
    - temperature adaptation
    - move-scale adaptation
    - state-history updates

are performed here.
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
# CONFIGURATION
# ============================================================

DIM = 1608

SEED = 42

# Must match C6 exactly.
INITIAL_VECTOR_STD = 0.10

# Must match C6 exactly.
FIXED_MOVE_SCALE = 0.0010019844

# C6's original temperature.
C4_T0 = 0.0002658506

# Number of actual proposals to evaluate.
N_PROPOSALS = 500

# Parameter safety bound.
MAX_ABS_PARAMETER = 10.0

# Target uphill acceptance levels.
TARGET_ACCEPTANCES = (
    0.20,
    0.30,
    0.40,
    0.50,
)

# Numerical tolerance.
EPS = 1.0e-15


# ============================================================
# UTILITY
# ============================================================

def clip_vector(vector: np.ndarray) -> np.ndarray:
    """
    Apply the same broad numerical safety bound used by C6.
    """

    return np.clip(
        vector,
        -MAX_ABS_PARAMETER,
        MAX_ABS_PARAMETER,
    )


# ============================================================
# FITNESS
# ============================================================

def evaluate(vector: np.ndarray) -> float:
    """
    Evaluate one candidate using the Stage-B fitness function.
    """

    result = fitness_module.evaluate_fitness(vector)

    return float(result["objective"])


# ============================================================
# INITIAL VECTOR
# ============================================================

def initialize_vector(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Reproduce the exact C6 initial vector.
    """

    vector = rng.normal(
        loc=0.0,
        scale=INITIAL_VECTOR_STD,
        size=DIM,
    )

    return vector.astype(np.float64)


# ============================================================
# C6 INITIAL COVARIANCE
# ============================================================

def initialize_covariance(
    vector: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Reproduce C6 initial covariance.

        covariance = parameter_std^2 * I

    Returns:
        covariance
        parameter_std
    """

    parameter_std = float(np.std(vector))

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

    return covariance, parameter_std


# ============================================================
# CHOLESKY
# ============================================================

def safe_cholesky(
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Same Cholesky approach as C6.
    """

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    covariance = 0.5 * (
        covariance + covariance.T
    )

    identity = np.eye(
        covariance.shape[0],
        dtype=np.float64,
    )

    regularization = 1.0e-6

    for _ in range(12):

        try:

            return np.linalg.cholesky(
                covariance
                + regularization * identity
            )

        except np.linalg.LinAlgError:

            regularization *= 10.0

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


# ============================================================
# PROPOSAL
# ============================================================

def propose(
    current: np.ndarray,
    chol: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the exact C6 initial proposal:

        z = N(0, I)

        step = FIXED_MOVE_SCALE * (chol @ z)

        proposal = current + step

    Returns:
        proposal
        step
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

    proposal = current + step

    proposal = clip_vector(
        proposal
    )

    return (
        proposal.astype(np.float64),
        step.astype(np.float64),
    )


# ============================================================
# EXPECTED METROPOLIS ACCEPTANCE
# ============================================================

def expected_uphill_acceptance(
    positive_deltas: np.ndarray,
    temperature: float,
) -> float:
    """
    Expected Metropolis acceptance for positive deltas.

        p = exp(-delta / T)

    Returns the mean acceptance probability across
    observed uphill moves.
    """

    if positive_deltas.size == 0:
        return 0.0

    if temperature <= 0.0:
        return 0.0

    exponent = (
        -positive_deltas
        / temperature
    )

    exponent = np.maximum(
        exponent,
        -745.0,
    )

    probabilities = np.exp(
        exponent
    )

    return float(
        np.mean(probabilities)
    )


# ============================================================
# SOLVE TEMPERATURE
# ============================================================

def solve_temperature(
    positive_deltas: np.ndarray,
    target_acceptance: float,
) -> float:
    """
    Find T such that:

        mean(exp(-delta/T)) ~= target_acceptance

    using bisection in log-temperature space.
    """

    if positive_deltas.size == 0:
        return float("nan")

    target_acceptance = float(
        target_acceptance
    )

    if not (
        0.0 < target_acceptance < 1.0
    ):
        raise ValueError(
            "target_acceptance must be between 0 and 1."
        )

    # Very small starting temperature.
    log_low = math.log(1.0e-12)

    # Large starting temperature.
    log_high = math.log(1.0)

    # Make sure the upper bound gives acceptance above target.
    while (
        expected_uphill_acceptance(
            positive_deltas,
            math.exp(log_high),
        )
        < target_acceptance
    ):
        log_high += math.log(10.0)

        if log_high > math.log(1.0e6):
            raise RuntimeError(
                "Could not bracket temperature."
            )

    # Make sure the lower bound gives acceptance below target.
    while (
        expected_uphill_acceptance(
            positive_deltas,
            math.exp(log_low),
        )
        > target_acceptance
    ):
        log_low -= math.log(10.0)

        if log_low < math.log(1.0e-30):
            raise RuntimeError(
                "Could not bracket lower temperature."
            )

    for _ in range(100):

        log_mid = (
            0.5
            * (
                log_low
                + log_high
            )
        )

        temperature = math.exp(
            log_mid
        )

        acceptance = (
            expected_uphill_acceptance(
                positive_deltas,
                temperature,
            )
        )

        if acceptance < target_acceptance:

            log_low = log_mid

        else:

            log_high = log_mid

    return float(
        math.exp(
            0.5
            * (
                log_low
                + log_high
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def run_c8() -> dict:

    start_time = time.perf_counter()

    print()
    print("=" * 72)
    print("STAGE C8 — ACTUAL C6 PROPOSAL TEMPERATURE CALIBRATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    # --------------------------------------------------------
    # Recreate exact C6 initial vector
    # --------------------------------------------------------

    current = initialize_vector(
        rng
    )

    current = clip_vector(
        current
    )

    parameter_std = float(
        np.std(current)
    )

    print(
        f"Dimension              : {DIM}"
    )

    print(
        f"Seed                   : {SEED}"
    )

    print(
        f"Initial vector std     : "
        f"{parameter_std:.10f}"
    )

    # --------------------------------------------------------
    # Initial covariance
    # --------------------------------------------------------

    covariance, parameter_std = (
        initialize_covariance(
            current
        )
    )

    chol = safe_cholesky(
        covariance
    )

    print(
        f"Covariance variance    : "
        f"{parameter_std ** 2:.10f}"
    )

    print(
        f"Mean Cholesky diagonal : "
        f"{np.mean(np.diag(chol)):.10f}"
    )

    print(
        f"Fixed move scale       : "
        f"{FIXED_MOVE_SCALE:.10f}"
    )

    effective_coordinate_scale = (
        FIXED_MOVE_SCALE
        * parameter_std
    )

    print(
        f"Effective coord scale  : "
        f"{effective_coordinate_scale:.10e}"
    )

    # --------------------------------------------------------
    # Initial objective
    # --------------------------------------------------------

    current_objective = evaluate(
        current
    )

    print(
        f"Initial objective      : "
        f"{current_objective:.12f}"
    )

    print()
    print(
        f"Generating {N_PROPOSALS} "
        f"actual C6 proposals..."
    )

    # --------------------------------------------------------
    # Proposal measurements
    # --------------------------------------------------------

    deltas = []

    step_norms = []

    mean_abs_steps = []

    proposal_objectives = []

    positive_deltas = []

    negative_deltas = []

    zero_deltas = []

    # --------------------------------------------------------
    # Generate proposals
    # --------------------------------------------------------

    for i in range(
        N_PROPOSALS
    ):

        proposal, step = propose(
            current=current,
            chol=chol,
            rng=rng,
        )

        proposed_objective = evaluate(
            proposal
        )

        delta = (
            proposed_objective
            - current_objective
        )

        deltas.append(
            float(delta)
        )

        proposal_objectives.append(
            float(proposed_objective)
        )

        step_norms.append(
            float(np.linalg.norm(step))
        )

        mean_abs_steps.append(
            float(np.mean(np.abs(step)))
        )

        if delta > EPS:

            positive_deltas.append(
                float(delta)
            )

        elif delta < -EPS:

            negative_deltas.append(
                float(delta)
            )

        else:

            zero_deltas.append(
                float(delta)
            )

    deltas = np.asarray(
        deltas,
        dtype=np.float64,
    )

    positive_deltas = np.asarray(
        positive_deltas,
        dtype=np.float64,
    )

    negative_deltas = np.asarray(
        negative_deltas,
        dtype=np.float64,
    )

    zero_deltas = np.asarray(
        zero_deltas,
        dtype=np.float64,
    )

    step_norms = np.asarray(
        step_norms,
        dtype=np.float64,
    )

    mean_abs_steps = np.asarray(
        mean_abs_steps,
        dtype=np.float64,
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    positive_fraction = (
        positive_deltas.size
        / N_PROPOSALS
    )

    negative_fraction = (
        negative_deltas.size
        / N_PROPOSALS
    )

    zero_fraction = (
        zero_deltas.size
        / N_PROPOSALS
    )

    print()
    print("-" * 72)
    print("PROPOSAL STATISTICS")
    print("-" * 72)

    print(
        f"Mean delta             : "
        f"{np.mean(deltas):.10e}"
    )

    print(
        f"Median delta           : "
        f"{np.median(deltas):.10e}"
    )

    print(
        f"Std delta              : "
        f"{np.std(deltas):.10e}"
    )

    print(
        f"Min delta              : "
        f"{np.min(deltas):.10e}"
    )

    print(
        f"Max delta              : "
        f"{np.max(deltas):.10e}"
    )

    print(
        f"Positive fraction      : "
        f"{positive_fraction:.6f}"
    )

    print(
        f"Negative fraction      : "
        f"{negative_fraction:.6f}"
    )

    print(
        f"Zero fraction          : "
        f"{zero_fraction:.6f}"
    )

    print(
        f"Mean step L2           : "
        f"{np.mean(step_norms):.10e}"
    )

    print(
        f"Mean abs parameter Δ   : "
        f"{np.mean(mean_abs_steps):.10e}"
    )

    # ========================================================
    # POSITIVE DELTA DISTRIBUTION
    # ========================================================

    positive_stats = {}

    if positive_deltas.size > 0:

        positive_stats = {
            "count": int(
                positive_deltas.size
            ),
            "mean": float(
                np.mean(
                    positive_deltas
                )
            ),
            "median": float(
                np.median(
                    positive_deltas
                )
            ),
            "p75": float(
                np.percentile(
                    positive_deltas,
                    75,
                )
            ),
            "p90": float(
                np.percentile(
                    positive_deltas,
                    90,
                )
            ),
            "p95": float(
                np.percentile(
                    positive_deltas,
                    95,
                )
            ),
            "p99": float(
                np.percentile(
                    positive_deltas,
                    99,
            )
            ),
            "max": float(
                np.max(
                    positive_deltas
                )
            ),
        }

        print()
        print("-" * 72)
        print("POSITIVE DELTA STATISTICS")
        print("-" * 72)

        print(
            f"Count                  : "
            f"{positive_stats['count']}"
        )

        print(
            f"Mean                   : "
            f"{positive_stats['mean']:.10e}"
        )

        print(
            f"Median                 : "
            f"{positive_stats['median']:.10e}"
        )

        print(
            f"P75                    : "
            f"{positive_stats['p75']:.10e}"
        )

        print(
            f"P90                    : "
            f"{positive_stats['p90']:.10e}"
        )

        print(
            f"P95                    : "
            f"{positive_stats['p95']:.10e}"
        )

        print(
            f"P99                    : "
            f"{positive_stats['p99']:.10e}"
        )

        print(
            f"Max                    : "
            f"{positive_stats['max']:.10e}"
        )

    # ========================================================
    # TEMPERATURE CALIBRATION
    # ========================================================

    print()
    print("-" * 72)
    print("TEMPERATURE CALIBRATION")
    print("-" * 72)

    temperature_results = {}

    for target in TARGET_ACCEPTANCES:

        if positive_deltas.size == 0:

            temperature = float("nan")

            expected = 0.0

        else:

            temperature = (
                solve_temperature(
                    positive_deltas,
                    target,
                )
            )

            expected = (
                expected_uphill_acceptance(
                    positive_deltas,
                    temperature,
                )
            )

        key = f"{target:.2f}"

        temperature_results[key] = {
            "target_acceptance": float(
                target
            ),
            "temperature": float(
                temperature
            ),
            "expected_uphill_acceptance": float(
                expected
            ),
        }

        print(
            f"Target {target * 100:5.1f}%"
            f"  -> T = "
            f"{temperature:.10e}"
            f"  -> expected = "
            f"{expected:.6f}"
        )

    # ========================================================
    # ORIGINAL C4 TEMPERATURE
    # ========================================================

    c4_expected = (
        expected_uphill_acceptance(
            positive_deltas,
            C4_T0,
        )
    )

    print()
    print("-" * 72)
    print("ORIGINAL C4 TEMPERATURE")
    print("-" * 72)

    print(
        f"C4 T0                 : "
        f"{C4_T0:.10e}"
    )

    print(
        f"Expected uphill acc.  : "
        f"{c4_expected:.6f}"
    )

    # ========================================================
    # RECOMMENDED TEMPERATURE
    # ========================================================

    recommended_temperature = (
        temperature_results["0.30"][
            "temperature"
        ]
    )

    print()
    print("-" * 72)
    print("RECOMMENDATION")
    print("-" * 72)

    print(
        f"Recommended T0 "
        f"(30% uphill target): "
        f"{recommended_temperature:.10e}"
    )

    # ========================================================
    # RUNTIME
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        f"Runtime                : "
        f"{runtime:.3f} s"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report = {
        "stage": "C8",

        "algorithm": (
            "Actual-C6-proposal "
            "temperature calibration"
        ),

        "dimension": DIM,

        "seed": SEED,

        "initial_vector_std_config": (
            INITIAL_VECTOR_STD
        ),

        "initial_parameter_std": (
            parameter_std
        ),

        "initial_covariance_variance": (
            parameter_std ** 2
        ),

        "initial_covariance_type": (
            "diagonal_parameter_std_squared_I"
        ),

        "mean_cholesky_diagonal": float(
            np.mean(
                np.diag(chol)
            )
        ),

        "fixed_move_scale": (
            FIXED_MOVE_SCALE
        ),

        "effective_coordinate_scale": (
            effective_coordinate_scale
        ),

        "n_proposals": N_PROPOSALS,

        "initial_objective": (
            current_objective
        ),

        "delta_mean": float(
            np.mean(deltas)
        ),

        "delta_median": float(
            np.median(deltas)
        ),

        "delta_std": float(
            np.std(deltas)
        ),

        "delta_min": float(
            np.min(deltas)
        ),

        "delta_max": float(
            np.max(deltas)
        ),

        "positive_fraction": (
            positive_fraction
        ),

        "negative_fraction": (
            negative_fraction
        ),

        "zero_fraction": (
            zero_fraction
        ),

        "mean_step_l2": float(
            np.mean(step_norms)
        ),

        "mean_abs_parameter_change": float(
            np.mean(mean_abs_steps)
        ),

        "positive_delta_statistics": (
            positive_stats
        ),

        "temperature_results": (
            temperature_results
        ),

        "original_c4_temperature": (
            C4_T0
        ),

        "original_c4_expected_uphill_acceptance": (
            c4_expected
        ),

        "recommended_temperature_30pct": (
            recommended_temperature
        ),

        "runtime_seconds": (
            runtime
        ),

        "notes": [
            (
                "C8 reproduces the actual initial "
                "proposal mechanism used by C6."
            ),
            (
                "Initial covariance is parameter_std^2 * I."
            ),
            (
                "No covariance adaptation is performed "
                "during calibration."
            ),
            (
                "No temperature adaptation is performed "
                "during calibration."
            ),
            (
                "No move-scale adaptation is performed."
            ),
            (
                "Temperature is calibrated from actual "
                "objective deltas."
            ),
            (
                "The 30% target is an engineering "
                "calibration target, not a value disclosed "
                "by the original paper."
            ),
        ],
    }

    report_path = (
        OUTPUT_DIR
        / "stage_c8_temperature_calibration.json"
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
    # SAVE DELTAS
    # ========================================================

    deltas_path = (
        OUTPUT_DIR
        / "stage_c8_actual_deltas.npy"
    )

    np.save(
        deltas_path,
        deltas,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    checks = [

        (
            current.shape
            == (DIM,)
        ),

        np.all(
            np.isfinite(current)
        ),

        covariance.shape
        == (DIM, DIM),

        chol.shape
        == (DIM, DIM),

        np.all(
            np.isfinite(
                deltas
            )
        ),

        deltas.size
        == N_PROPOSALS,

        np.isfinite(
            current_objective
        ),

        positive_deltas.size
        + negative_deltas.size
        + zero_deltas.size
        == N_PROPOSALS,

        np.isfinite(
            recommended_temperature
        ),

        recommended_temperature > 0.0,

    ]

    check_names = [

        "initial vector shape",

        "initial vector finite",

        "covariance shape",

        "Cholesky shape",

        "deltas finite",

        "proposal count",

        "initial objective finite",

        "delta classification",

        "recommended temperature finite",

        "recommended temperature positive",

    ]

    print()
    print("=" * 72)
    print("C8 VALIDATION CHECKS")
    print("=" * 72)

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
            "STAGE C8 PASSED."
        )

    else:

        print()
        print(
            "STAGE C8 FAILED."
        )

    print()
    print(
        f"Report saved          : "
        f"{report_path}"
    )

    print(
        f"Deltas saved          : "
        f"{deltas_path}"
    )

    print("=" * 72)

    return report


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_c8()
