"""
RampSense - Stage C3
SAMURAI acceptance-calibration test.

Changes from C2:
- Proposal scale is relative to parameter standard deviation.
- Smaller initial proposal scale.
- Explicit uphill acceptance tracking.
- Adaptive cooling based on observed acceptance.
- Adaptive move scale is bounded.
- Covariance is normalized to correlation structure.
- 500-evaluation validation budget.

This is a calibration/engineering stage, not a claim of exact
reproduction of undisclosed SAMURAI constants.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import dataclass
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

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


fitness_module = load_module(
    "fitness_module",
    FITNESS_FILE,
)

param_module = load_module(
    "param_vector_module",
    PARAM_VECTOR_FILE,
)

model_module = load_module(
    "model_module",
    MODEL_FILE,
)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:

    seed: int = 42

    # Temperature
    T0: float = 1.0
    Tmin: float = 1e-4

    # Cooling
    alpha_min: float = 0.85
    alpha_max: float = 0.97

    # Inner loop
    inner_min: int = 20
    inner_max: int = 100

    # Proposal scale as fraction of parameter std
    initial_scale_fraction: float = 0.01

    # Bounds for adaptive scale
    scale_min: float = 0.001
    scale_max: float = 0.10

    # Target uphill acceptance
    target_acceptance: float = 0.20

    # Covariance
    covariance_window: int = 50
    covariance_update_frequency: int = 10
    covariance_regularization: float = 1e-6

    # Evaluation budget
    max_temperature_steps: int = 5
    max_evaluations: int = 500


# ============================================================
# SAMURAI C3
# ============================================================

class SAMURAIC3:

    def __init__(
        self,
        objective_function,
        x0,
        config: Config,
    ):

        self.objective_function = objective_function
        self.config = config

        self.rng = np.random.default_rng(
            config.seed
        )

        self.x = np.asarray(
            x0,
            dtype=np.float64,
        ).copy()

        if self.x.ndim != 1:
            raise ValueError(
                "x0 must be 1-D"
            )

        if not np.all(
            np.isfinite(self.x)
        ):
            raise ValueError(
                "x0 contains NaN/Inf"
            )

        self.dimension = self.x.size

        # ----------------------------------------------------
        # Parameter scale
        # ----------------------------------------------------

        self.parameter_std = float(
            np.std(self.x)
        )

        if (
            not np.isfinite(
                self.parameter_std
            )
            or self.parameter_std <= 0
        ):
            raise ValueError(
                "Invalid parameter standard deviation."
            )

        self.base_step = (
            self.parameter_std
            * config.initial_scale_fraction
        )

        self.move_scale = 1.0

        # ----------------------------------------------------
        # Initial objective
        # ----------------------------------------------------

        self.fx = float(
            self.objective_function(
                self.x
            )
        )

        self.best_x = self.x.copy()
        self.best_fx = self.fx

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        self.temperature = config.T0

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        self.evaluations = 1
        self.temperature_steps = 0

        self.attempted_moves = 0
        self.accepted_moves = 0

        self.uphill_attempts = 0
        self.accepted_uphill = 0

        self.downhill_moves = 0

        # ----------------------------------------------------
        # State history
        # ----------------------------------------------------

        self.state_history = [
            self.x.copy()
        ]

        # ----------------------------------------------------
        # Covariance
        # ----------------------------------------------------

        self.cholesky = np.eye(
            self.dimension,
            dtype=np.float64,
        )

        self.covariance_updates = 0

        # ----------------------------------------------------
        # Records
        # ----------------------------------------------------

        self.history = []

    # ========================================================
    # COVARIANCE
    # ========================================================

    def update_covariance(self):

        if len(
            self.state_history
        ) < 3:

            return

        window = min(
            len(self.state_history),
            self.config.covariance_window,
        )

        samples = np.asarray(
            self.state_history[-window:],
            dtype=np.float64,
        )

        centered = (
            samples
            - np.mean(
                samples,
                axis=0,
                keepdims=True,
            )
        )

        covariance = (
            centered.T @ centered
        ) / max(
            samples.shape[0] - 1,
            1,
        )

        # Convert covariance into correlation-like structure.
        diagonal = np.sqrt(
            np.maximum(
                np.diag(covariance),
                1e-12,
            )
        )

        normalized = covariance / (
            np.outer(
                diagonal,
                diagonal,
            )
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        normalized = (
            normalized
            + normalized.T
        ) * 0.5

        normalized += (
            self.config.covariance_regularization
            * np.eye(
                self.dimension,
                dtype=np.float64,
            )
        )

        try:

            self.cholesky = np.linalg.cholesky(
                normalized
            )

        except np.linalg.LinAlgError:

            normalized += (
                1e-4
                * np.eye(
                    self.dimension,
                    dtype=np.float64,
                )
            )

            self.cholesky = np.linalg.cholesky(
                normalized
            )

        self.covariance_updates += 1

    # ========================================================
    # PROPOSAL
    # ========================================================

    def propose(self):

        z = self.rng.normal(
            0.0,
            1.0,
            self.dimension,
        )

        correlated_step = (
            self.cholesky @ z
        )

        step = (
            self.base_step
            * self.move_scale
            * correlated_step
        )

        return self.x + step

    # ========================================================
    # METROPOLIS
    # ========================================================

    def acceptance_probability(
        self,
        delta,
    ):

        if delta <= 0:

            return 1.0

        exponent = (
            -delta
            / max(
                self.temperature,
                1e-12,
            )
        )

        exponent = max(
            exponent,
            -700.0,
        )

        return math.exp(
            exponent
        )

    # ========================================================
    # ONE MOVE
    # ========================================================

    def step(self):

        candidate = self.propose()

        candidate_fx = float(
            self.objective_function(
                candidate
            )
        )

        self.evaluations += 1
        self.attempted_moves += 1

        delta = (
            candidate_fx
            - self.fx
        )

        probability = (
            self.acceptance_probability(
                delta
            )
        )

        accepted = (
            self.rng.random()
            < probability
        )

        if delta > 0:

            self.uphill_attempts += 1

            if accepted:
                self.accepted_uphill += 1

        else:

            self.downhill_moves += 1

        if accepted:

            self.x = candidate
            self.fx = candidate_fx

            self.accepted_moves += 1

            self.state_history.append(
                self.x.copy()
            )

            if self.fx < self.best_fx:

                self.best_fx = self.fx
                self.best_x = self.x.copy()

        return accepted

    # ========================================================
    # ADAPTATION
    # ========================================================

    def adapt_scale(
        self,
        uphill_rate,
    ):

        target = (
            self.config.target_acceptance
        )

        if uphill_rate < (
            target * 0.5
        ):

            self.move_scale *= 0.75

        elif uphill_rate > (
            target * 1.5
        ):

            self.move_scale *= 1.10

        self.move_scale = float(
            np.clip(
                self.move_scale,
                self.config.scale_min,
                self.config.scale_max,
            )
        )

    def adapt_temperature(
        self,
        uphill_rate,
    ):

        # High acceptance -> cool more slowly.
        if uphill_rate > 0.30:

            alpha = (
                self.config.alpha_max
            )

        # Very low acceptance -> cool faster.
        elif uphill_rate < 0.05:

            alpha = (
                self.config.alpha_min
            )

        else:

            fraction = (
                uphill_rate - 0.05
            ) / 0.25

            alpha = (
                self.config.alpha_min
                + fraction
                * (
                    self.config.alpha_max
                    - self.config.alpha_min
                )
            )

        self.temperature *= alpha

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        print()
        print("=" * 70)
        print("RampSense - Stage C3")
        print("SAMURAI Acceptance Calibration")
        print("=" * 70)

        print()
        print(
            f"Dimension             : "
            f"{self.dimension}"
        )

        print(
            f"Parameter std         : "
            f"{self.parameter_std:.8f}"
        )

        print(
            f"Base proposal step    : "
            f"{self.base_step:.8f}"
        )

        print(
            f"Initial objective     : "
            f"{self.fx:.10f}"
        )

        print(
            f"Initial temperature   : "
            f"{self.temperature:.6f}"
        )

        print(
            f"Initial move scale    : "
            f"{self.move_scale:.6f}"
        )

        start_time = time.perf_counter()

        while (

            self.temperature > self.config.Tmin

            and self.temperature_steps
            < self.config.max_temperature_steps

            and self.evaluations
            < self.config.max_evaluations
        ):

            old_uphill_attempts = (
                self.uphill_attempts
            )

            old_uphill_accepts = (
                self.accepted_uphill
            )

            start_evaluations = (
                self.evaluations
            )

            inner_limit = int(
                np.clip(
                    self.config.inner_min
                    + (
                        self.config.inner_max
                        - self.config.inner_min
                    )
                    * min(
                        1.0,
                        self.temperature
                        / self.config.T0,
                    ),
                    self.config.inner_min,
                    self.config.inner_max,
                )
            )

            for _ in range(
                inner_limit
            ):

                if (
                    self.evaluations
                    >= self.config.max_evaluations
                ):
                    break

                self.step()

                if (
                    self.accepted_moves > 0
                    and self.accepted_moves
                    % self.config.covariance_update_frequency
                    == 0
                ):

                    self.update_covariance()

            local_uphill_attempts = (
                self.uphill_attempts
                - old_uphill_attempts
            )

            local_uphill_accepts = (
                self.accepted_uphill
                - old_uphill_accepts
            )

            if local_uphill_attempts > 0:

                uphill_rate = (
                    local_uphill_accepts
                    / local_uphill_attempts
                )

            else:

                uphill_rate = 0.0

            self.adapt_scale(
                uphill_rate
            )

            self.adapt_temperature(
                uphill_rate
            )

            self.temperature_steps += 1

            record = {

                "temperature_step":
                    self.temperature_steps,

                "temperature":
                    float(self.temperature),

                "current_objective":
                    float(self.fx),

                "best_objective":
                    float(self.best_fx),

                "move_scale":
                    float(self.move_scale),

                "base_step":
                    float(self.base_step),

                "uphill_attempts":
                    int(local_uphill_attempts),

                "uphill_accepts":
                    int(local_uphill_accepts),

                "uphill_acceptance":
                    float(uphill_rate),

                "evaluations":
                    int(self.evaluations),

                "inner_evaluations":
                    int(
                        self.evaluations
                        - start_evaluations
                    ),

                "covariance_updates":
                    int(
                        self.covariance_updates
                    ),
            }

            self.history.append(
                record
            )

            print()
            print(
                f"Temperature step      : "
                f"{self.temperature_steps}"
            )

            print(
                f"Temperature            : "
                f"{self.temperature:.8f}"
            )

            print(
                f"Best objective         : "
                f"{self.best_fx:.10f}"
            )

            print(
                f"Current objective      : "
                f"{self.fx:.10f}"
            )

            print(
                f"Move scale             : "
                f"{self.move_scale:.8f}"
            )

            print(
                f"Uphill attempts        : "
                f"{local_uphill_attempts}"
            )

            print(
                f"Uphill accepts         : "
                f"{local_uphill_accepts}"
            )

            print(
                f"Uphill acceptance      : "
                f"{uphill_rate:.6f}"
            )

            print(
                f"Covariance updates     : "
                f"{self.covariance_updates}"
            )

            print(
                f"Evaluations            : "
                f"{self.evaluations}"
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result = {

            "initial_objective":
                float(
                    self.history[0]["current_objective"]
                    if self.history
                    else self.fx
                ),

            "best_objective":
                float(self.best_fx),

            "final_objective":
                float(self.fx),

            "dimension":
                int(self.dimension),

            "parameter_std":
                float(self.parameter_std),

            "base_step":
                float(self.base_step),

            "evaluations":
                int(self.evaluations),

            "temperature_steps":
                int(self.temperature_steps),

            "final_temperature":
                float(self.temperature),

            "final_move_scale":
                float(self.move_scale),

            "covariance_updates":
                int(self.covariance_updates),

            "accepted_moves":
                int(self.accepted_moves),

            "uphill_attempts":
                int(self.uphill_attempts),

            "uphill_accepts":
                int(self.accepted_uphill),

            "elapsed_seconds":
                float(elapsed),

            "history":
                self.history,
        }

        return result


# ============================================================
# FITNESS
# ============================================================

def evaluate(
    vector: np.ndarray,
) -> float:

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
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RampSense - Stage C3")
    print("=" * 70)

    print()
    print("Loading model...")

    model = model_module.PVLSTM(
        input_size=6,
        hidden_size=16,
        output_size=8,
    )

    vector = param_module.get_vector(
        model
    )

    print(
        f"Optimizer dimension : "
        f"{vector.size}"
    )

    if vector.size != 1608:

        raise RuntimeError(
            f"Expected 1608 parameters, "
            f"got {vector.size}"
        )

    print()
    print("Testing initial fitness...")

    initial_objective = evaluate(
        vector
    )

    print(
        f"Initial objective   : "
        f"{initial_objective:.10f}"
    )

    config = Config()

    optimizer = SAMURAIC3(
        objective_function=evaluate,
        x0=vector,
        config=config,
    )

    result = optimizer.run()

    report_file = (
        OUTPUT_DIR
        / "stage_c3_samurai_report.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    best_vector_file = (
        OUTPUT_DIR
        / "stage_c3_samurai_best.npy"
    )

    np.save(
        best_vector_file,
        optimizer.best_x,
    )

    print()
    print("=" * 70)
    print("SAMURAI C3 COMPLETE")
    print("=" * 70)

    print(
        f"Initial objective : "
        f"{initial_objective:.10f}"
    )

    print(
        f"Best objective    : "
        f"{optimizer.best_fx:.10f}"
    )

    print(
        f"Evaluations       : "
        f"{optimizer.evaluations}"
    )

    print(
        f"Uphill attempts   : "
        f"{optimizer.uphill_attempts}"
    )

    print(
        f"Uphill accepts    : "
        f"{optimizer.accepted_uphill}"
    )

    print(
        f"Elapsed           : "
        f"{result['elapsed_seconds']:.3f} s"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if optimizer.best_x.shape != (
        1608,
    ):

        raise RuntimeError(
            "Best vector shape is incorrect."
        )

    if not np.all(
        np.isfinite(
            optimizer.best_x
        )
    ):

        raise RuntimeError(
            "Best vector contains NaN/Inf."
        )

    if not np.isfinite(
        optimizer.best_fx
    ):

        raise RuntimeError(
            "Best objective is not finite."
        )

    if optimizer.evaluations < 2:

        raise RuntimeError(
            "SAMURAI did not perform optimization."
        )

    if optimizer.uphill_attempts == 0:

        raise RuntimeError(
            "No uphill moves were attempted. "
            "Acceptance calibration cannot be validated."
        )

    print()
    print("=" * 70)
    print("STAGE C3 PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
