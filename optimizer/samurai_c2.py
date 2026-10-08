"""
RampSense - Stage C2
SAMURAI with adaptive covariance / Cholesky proposals.

Stage C2 goals:
1. Adaptive covariance estimation
2. Cholesky-based continuous proposals
3. Metropolis acceptance
4. Adaptive cooling
5. Adaptive move range
6. Reproducibility
7. Best-solution preservation

This is a research implementation/adaptation.
Undisclosed SAMURAI constants are kept configurable.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import dataclass, asdict
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
class SamuraiC2Config:

    seed: int = 42

    # Initial temperature
    T0: float = 1.0

    # Termination temperature
    Tmin: float = 1e-4

    # Cooling range
    alpha_min: float = 0.90
    alpha_max: float = 0.99

    # Inner loop
    inner_min: int = 20
    inner_max: int = 100

    # Proposal scale
    move_scale_initial: float = 0.05
    move_scale_min: float = 1e-4
    move_scale_max: float = 0.50

    # Desired uphill acceptance
    target_acceptance: float = 0.20

    # Covariance
    covariance_window: int = 50
    covariance_regularization: float = 1e-6

    # Covariance update frequency
    covariance_update_frequency: int = 10

    # Stage C2 budget
    max_temperature_steps: int = 5
    max_evaluations: int = 500

    # Numerical safety
    epsilon: float = 1e-12


# ============================================================
# SAMURAI C2
# ============================================================

class SAMURAIC2:

    def __init__(
        self,
        objective_function,
        x0: np.ndarray,
        config: SamuraiC2Config,
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
                "Initial vector must be 1-D."
            )

        if not np.all(
            np.isfinite(self.x)
        ):
            raise ValueError(
                "Initial vector contains NaN/Inf."
            )

        self.dimension = self.x.size

        self.fx = float(
            self.objective_function(self.x)
        )

        if not np.isfinite(self.fx):
            raise ValueError(
                "Initial objective is not finite."
            )

        self.best_x = self.x.copy()
        self.best_fx = self.fx

        self.temperature = config.T0

        self.move_scale = (
            config.move_scale_initial
        )

        self.evaluations = 1

        self.temperature_steps = 0

        self.attempted_moves = 0
        self.accepted_moves = 0

        self.uphill_attempts = 0
        self.accepted_uphill = 0

        # ----------------------------------------------------
        # History of accepted/current states
        # ----------------------------------------------------

        self.state_history = [
            self.x.copy()
        ]

        self.objective_history = [
            self.fx
        ]

        # ----------------------------------------------------
        # Initial covariance
        # ----------------------------------------------------

        self.covariance = np.eye(
            self.dimension,
            dtype=np.float64,
        )

        self.cholesky = np.eye(
            self.dimension,
            dtype=np.float64,
        )

        self.covariance_updates = 0

        self.history = []

    # ========================================================
    # COVARIANCE
    # ========================================================

    def update_covariance(self):

        n = len(
            self.state_history
        )

        if n < 3:
            return

        window = min(
            n,
            self.config.covariance_window,
        )

        samples = np.asarray(
            self.state_history[-window:],
            dtype=np.float64,
        )

        # Center samples
        centered = (
            samples
            - np.mean(
                samples,
                axis=0,
                keepdims=True,
            )
        )

        if samples.shape[0] < 3:
            return

        covariance = (
            centered.T @ centered
        ) / max(
            samples.shape[0] - 1,
            1,
        )

        # Regularization
        covariance += (
            self.config.covariance_regularization
            * np.eye(
                self.dimension,
                dtype=np.float64,
            )
        )

        # Numerical symmetry
        covariance = (
            covariance + covariance.T
        ) * 0.5

        try:

            chol = np.linalg.cholesky(
                covariance
            )

        except np.linalg.LinAlgError:

            # Increase diagonal regularization
            covariance += (
                1e-4
                * np.eye(
                    self.dimension,
                    dtype=np.float64,
                )
            )

            chol = np.linalg.cholesky(
                covariance
            )

        self.covariance = covariance
        self.cholesky = chol

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

        step = (
            self.cholesky @ z
        )

        candidate = (
            self.x
            + self.move_scale * step
        )

        return candidate

    # ========================================================
    # METROPOLIS
    # ========================================================

    def acceptance_probability(
        self,
        delta,
    ):

        if delta <= 0.0:
            return 1.0

        exponent = (
            -delta
            / max(
                self.temperature,
                self.config.epsilon,
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

        candidate_x = self.propose()

        candidate_fx = float(
            self.objective_function(
                candidate_x
            )
        )

        self.evaluations += 1
        self.attempted_moves += 1

        delta = (
            candidate_fx
            - self.fx
        )

        if delta <= 0.0:

            probability = 1.0
            accepted = True

        else:

            self.uphill_attempts += 1

            probability = (
                self.acceptance_probability(
                    delta
                )
            )

            accepted = (
                self.rng.random()
                < probability
            )

        if accepted:

            self.x = candidate_x
            self.fx = candidate_fx

            self.accepted_moves += 1

            self.state_history.append(
                self.x.copy()
            )

            self.objective_history.append(
                self.fx
            )

            if delta > 0.0:

                self.accepted_uphill += 1

            if self.fx < self.best_fx:

                self.best_fx = self.fx

                self.best_x = (
                    self.x.copy()
                )

        return {
            "accepted": accepted,
            "delta": float(delta),
            "candidate_objective": float(
                candidate_fx
            ),
            "acceptance_probability": float(
                probability
            ),
        }

    # ========================================================
    # ACCEPTANCE RATES
    # ========================================================

    def uphill_acceptance_rate(
        self,
        old_uphill_attempts,
        old_uphill_accepts,
    ):

        attempts = (
            self.uphill_attempts
            - old_uphill_attempts
        )

        accepts = (
            self.accepted_uphill
            - old_uphill_accepts
        )

        if attempts == 0:
            return 0.0

        return accepts / attempts

    def total_acceptance_rate(self):

        if self.attempted_moves == 0:
            return 0.0

        return (
            self.accepted_moves
            / self.attempted_moves
        )

    # ========================================================
    # MOVE RANGE ADAPTATION
    # ========================================================

    def adapt_move_scale(
        self,
        uphill_rate,
    ):

        target = (
            self.config.target_acceptance
        )

        if uphill_rate < (
            target * 0.5
        ):

            self.move_scale *= 0.80

        elif uphill_rate > (
            target * 1.5
        ):

            self.move_scale *= 1.20

        self.move_scale = float(
            np.clip(
                self.move_scale,
                self.config.move_scale_min,
                self.config.move_scale_max,
            )
        )

    # ========================================================
    # TEMPERATURE ADAPTATION
    # ========================================================

    def adapt_temperature(
        self,
        uphill_rate,
    ):

        if uphill_rate > 0.30:

            alpha = (
                self.config.alpha_max
            )

        elif uphill_rate < 0.05:

            alpha = (
                self.config.alpha_min
            )

        else:

            fraction = (
                (uphill_rate - 0.05)
                / (0.30 - 0.05)
            )

            alpha = (
                self.config.alpha_min
                + (
                    self.config.alpha_max
                    - self.config.alpha_min
                )
                * fraction
            )

        self.temperature *= alpha

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        print()
        print("=" * 70)
        print("RampSense - Stage C2")
        print("SAMURAI Adaptive Covariance Test")
        print("=" * 70)

        print()
        print(
            f"Dimension             : "
            f"{self.dimension}"
        )

        print(
            f"Initial objective     : "
            f"{self.fx:.10f}"
        )

        print(
            f"Initial temperature   : "
            f"{self.temperature:.6g}"
        )

        print(
            f"Initial move scale    : "
            f"{self.move_scale:.6g}"
        )

        print(
            f"Covariance window     : "
            f"{self.config.covariance_window}"
        )

        print(
            f"Maximum evaluations   : "
            f"{self.config.max_evaluations}"
        )

        start_time = time.perf_counter()

        while (

            self.temperature
            > self.config.Tmin

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

            step_start = (
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
                        / max(
                            self.config.T0,
                            self.config.epsilon,
                        ),
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

                # ------------------------------------------------
                # Update covariance periodically.
                # ------------------------------------------------

                if (
                    self.accepted_moves > 0
                    and self.accepted_moves
                    % self.config.covariance_update_frequency
                    == 0
                ):

                    self.update_covariance()

            uphill_rate = (
                self.uphill_acceptance_rate(
                    old_uphill_attempts,
                    old_uphill_accepts,
                )
            )

            self.adapt_move_scale(
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

                "uphill_acceptance_rate":
                    float(uphill_rate),

                "total_acceptance_rate":
                    float(
                        self.total_acceptance_rate()
                    ),

                "evaluations":
                    int(self.evaluations),

                "inner_evaluations":
                    int(
                        self.evaluations
                        - step_start
                    ),

                "covariance_updates":
                    int(
                        self.covariance_updates
                    ),

                "accepted_states":
                    int(
                        len(
                            self.state_history
                        )
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
                    self.objective_history[0]
                ),

            "best_objective":
                float(self.best_fx),

            "final_objective":
                float(self.fx),

            "dimension":
                int(self.dimension),

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

            "total_acceptance_rate":
                float(
                    self.total_acceptance_rate()
                ),

            "uphill_acceptance_rate":
                float(
                    self.uphill_acceptance_rate(
                        0,
                        0,
                    )
                ),

            "elapsed_seconds":
                float(elapsed),

            "history":
                self.history,
        }

        print()
        print("=" * 70)
        print("SAMURAI C2 TEST COMPLETE")
        print("=" * 70)

        print(
            f"Initial objective : "
            f"{result['initial_objective']:.10f}"
        )

        print(
            f"Best objective    : "
            f"{result['best_objective']:.10f}"
        )

        print(
            f"Evaluations       : "
            f"{result['evaluations']}"
        )

        print(
            f"Covariance updates: "
            f"{result['covariance_updates']}"
        )

        print(
            f"Elapsed           : "
            f"{elapsed:.3f} s"
        )

        return result


# ============================================================
# FITNESS WRAPPER
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
    print("RampSense - Stage C2")
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

    config = SamuraiC2Config()

    optimizer = SAMURAIC2(
        objective_function=evaluate,
        x0=vector,
        config=config,
    )

    result = optimizer.run()

    report_file = (
        OUTPUT_DIR
        / "stage_c2_samurai_report.json"
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
        / "stage_c2_samurai_best.npy"
    )

    np.save(
        best_vector_file,
        optimizer.best_x,
    )

    covariance_file = (
        OUTPUT_DIR
        / "stage_c2_samurai_covariance.npy"
    )

    np.save(
        covariance_file,
        optimizer.covariance,
    )

    print()
    print("Output files:")
    print(report_file)
    print(best_vector_file)
    print(covariance_file)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if optimizer.best_x.shape != (
        1608,
    ):

        raise RuntimeError(
            "Best vector has incorrect shape."
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
            "SAMURAI performed no optimization."
        )

    print()
    print("=" * 70)
    print("STAGE C2 PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
