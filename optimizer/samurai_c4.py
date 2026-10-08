"""
RampSense - Stage C4
SAMURAI temperature / fitness-landscape calibration.

Purpose:
Estimate the typical objective change caused by perturbing the
1608-dimensional parameter vector.

The results are used to choose a sensible SAMURAI initial
temperature T0 instead of assuming T0 = 1.

No model files or fitness files are modified.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

OPTIMIZER_DIR = Path(__file__).resolve().parent

FITNESS_FILE = OPTIMIZER_DIR / "fitness.py"
PARAM_VECTOR_FILE = OPTIMIZER_DIR / "param_vector.py"
MODEL_FILE = OPTIMIZER_DIR / "model.py"

OUTPUT_DIR = OPTIMIZER_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(name: str, path: Path):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


fitness_module = load_module(
    "fitness_module_c4",
    FITNESS_FILE,
)

param_module = load_module(
    "param_vector_module_c4",
    PARAM_VECTOR_FILE,
)

model_module = load_module(
    "model_module_c4",
    MODEL_FILE,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

# Number of random perturbations per scale.
N_SAMPLES = 30

# Perturbation scale expressed as a fraction of the
# standard deviation of the initial parameter vector.
PERTURBATION_FRACTIONS = [
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050,
]

# Desired uphill acceptance probabilities.
TARGET_ACCEPTANCES = [
    0.50,
    0.40,
    0.30,
    0.20,
]


# ============================================================
# FITNESS
# ============================================================

def evaluate(vector):

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    if vector.shape != (1608,):

        raise ValueError(
            f"Expected vector shape (1608,), "
            f"got {vector.shape}"
        )

    if not np.all(
        np.isfinite(vector)
    ):

        return float("inf")

    result = (
        fitness_module
        .evaluate_fitness(vector)
    )

    return float(
        result["objective"]
    )


# ============================================================
# TEMPERATURE FROM DELTA
# ============================================================

def temperature_for_acceptance(
    delta,
    target,
):

    """
    Metropolis:

        p = exp(-delta / T)

    Therefore:

        T = -delta / ln(p)
    """

    if delta <= 0:
        return np.nan

    if target <= 0 or target >= 1:
        return np.nan

    return (
        -delta
        / math.log(target)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rng = np.random.default_rng(
        SEED
    )

    print()
    print("=" * 70)
    print("RampSense - Stage C4")
    print("SAMURAI Temperature Calibration")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = model_module.PVLSTM(
        input_size=6,
        hidden_size=16,
        output_size=8,
    )

    x0 = param_module.get_vector(
        model
    )

    if x0.shape != (1608,):

        raise RuntimeError(
            f"Expected 1608 parameters, "
            f"got {x0.shape}"
        )

    parameter_std = float(
        np.std(x0)
    )

    parameter_mean = float(
        np.mean(x0)
    )

    print(
        f"Parameter count      : "
        f"{x0.size}"
    )

    print(
        f"Parameter mean       : "
        f"{parameter_mean:.10f}"
    )

    print(
        f"Parameter std        : "
        f"{parameter_std:.10f}"
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print()
    print("Evaluating baseline...")

    start = time.perf_counter()

    baseline = evaluate(x0)

    baseline_time = (
        time.perf_counter()
        - start
    )

    print(
        f"Baseline objective   : "
        f"{baseline:.10f}"
    )

    print(
        f"Evaluation time      : "
        f"{baseline_time:.4f} s"
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    all_results = []

    for fraction in (
        PERTURBATION_FRACTIONS
    ):

        step_std = (
            parameter_std
            * fraction
        )

        print()
        print("-" * 70)

        print(
            f"Perturbation fraction : "
            f"{fraction}"
        )

        print(
            f"Step standard dev     : "
            f"{step_std:.10f}"
        )

        deltas = []
        positive_deltas = []

        accepted_at_t1 = 0
        total_uphill = 0

        for sample in range(
            N_SAMPLES
        ):

            noise = rng.normal(
                loc=0.0,
                scale=step_std,
                size=x0.size,
            )

            candidate = (
                x0 + noise
            )

            objective = evaluate(
                candidate
            )

            delta = (
                objective
                - baseline
            )

            deltas.append(
                float(delta)
            )

            if delta > 0:

                positive_deltas.append(
                    float(delta)
                )

                total_uphill += 1

                # What happens at T = 1?
                probability = math.exp(
                    max(
                        -delta,
                        -700.0,
                    )
                )

                if (
                    rng.random()
                    < probability
                ):
                    accepted_at_t1 += 1

        deltas = np.asarray(
            deltas,
            dtype=np.float64,
        )

        positive_deltas = np.asarray(
            positive_deltas,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        result = {

            "perturbation_fraction":
                float(fraction),

            "step_std":
                float(step_std),

            "samples":
                int(N_SAMPLES),

            "mean_delta":
                float(
                    np.mean(deltas)
                ),

            "median_delta":
                float(
                    np.median(deltas)
                ),

            "min_delta":
                float(
                    np.min(deltas)
                ),

            "max_delta":
                float(
                    np.max(deltas)
                ),

            "p75_delta":
                float(
                    np.percentile(
                        deltas,
                        75,
                    )
                ),

            "p90_delta":
                float(
                    np.percentile(
                        deltas,
                        90,
                    )
                ),

            "p95_delta":
                float(
                    np.percentile(
                        deltas,
                        95,
                    )
                ),

            "positive_delta_count":
                int(
                    positive_deltas.size
                ),

            "positive_delta_fraction":
                float(
                    positive_deltas.size
                    / len(deltas)
                ),
        }

        # ----------------------------------------------------
        # Temperature estimates
        # ----------------------------------------------------

        for target in (
            TARGET_ACCEPTANCES
        ):

            key = (
                f"T_for_acceptance_"
                f"{int(target * 100)}"
            )

            if positive_deltas.size > 0:

                median_positive = float(
                    np.median(
                        positive_deltas
                    )
                )

                temperature = (
                    temperature_for_acceptance(
                        median_positive,
                        target,
                    )
                )

            else:

                temperature = np.nan

            result[key] = (
                None
                if not np.isfinite(
                    temperature
                )
                else float(temperature)
            )

        # ----------------------------------------------------
        # T=1 empirical acceptance
        # ----------------------------------------------------

        if total_uphill > 0:

            result[
                "empirical_uphill_acceptance_T1"
            ] = float(
                accepted_at_t1
                / total_uphill
            )

        else:

            result[
                "empirical_uphill_acceptance_T1"
            ] = None

        all_results.append(
            result
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Mean Δφ              : "
            f"{result['mean_delta']:.10f}"
        )

        print(
            f"Median Δφ            : "
            f"{result['median_delta']:.10f}"
        )

        print(
            f"95th percentile Δφ   : "
            f"{result['p95_delta']:.10f}"
        )

        print(
            f"Positive Δφ fraction : "
            f"{result['positive_delta_fraction']:.4f}"
        )

        print(
            f"Acceptance at T=1    : "
            f"{result['empirical_uphill_acceptance_T1']}"
        )

        print(
            f"T for 50% acceptance : "
            f"{result['T_for_acceptance_50']}"
        )

        print(
            f"T for 40% acceptance : "
            f"{result['T_for_acceptance_40']}"
        )

        print(
            f"T for 30% acceptance : "
            f"{result['T_for_acceptance_30']}"
        )

        print(
            f"T for 20% acceptance : "
            f"{result['T_for_acceptance_20']}"
        )

    # ========================================================
    # Recommendation
    # ========================================================

    print()
    print("=" * 70)
    print("TEMPERATURE RECOMMENDATION")
    print("=" * 70)

    # Use the 1% perturbation scale as the reference.
    reference = None

    for result in all_results:

        if abs(
            result[
                "perturbation_fraction"
            ] - 0.010
        ) < 1e-12:

            reference = result
            break

    if reference is None:

        raise RuntimeError(
            "1% calibration result not found."
        )

    recommended_T0 = (
        reference[
            "T_for_acceptance_30"
        ]
    )

    if recommended_T0 is None:

        recommended_T0 = (
            reference[
                "T_for_acceptance_20"
            ]
        )

    if recommended_T0 is None:

        recommended_T0 = 0.01

    print()
    print(
        "Reference perturbation : 1% of parameter std"
    )

    print(
        f"Recommended T0         : "
        f"{recommended_T0:.10f}"
    )

    print()
    print(
        "This is an empirical calibration "
        "for the current fitness landscape."
    )

    print(
        "It is NOT claimed to be an exact "
        "paper-disclosed SAMURAI constant."
    )

    # ========================================================
    # Save
    # ========================================================

    report = {

        "seed": SEED,

        "parameter_count":
            int(x0.size),

        "parameter_mean":
            parameter_mean,

        "parameter_std":
            parameter_std,

        "baseline_objective":
            baseline,

        "baseline_evaluation_seconds":
            baseline_time,

        "samples_per_scale":
            N_SAMPLES,

        "perturbation_fractions":
            PERTURBATION_FRACTIONS,

        "target_acceptances":
            TARGET_ACCEPTANCES,

        "recommended_T0":
            float(recommended_T0),

        "results":
            all_results,
    }

    output_file = (
        OUTPUT_DIR
        / "stage_c4_temperature_calibration.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        "Report saved:"
    )

    print(output_file)

    print()
    print("=" * 70)
    print("STAGE C4 CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
