"""
RampSense - Stage C1
SAMURAI optimizer for the 1608-dimensional LSTM parameter vector.

Important:
- Uses the verified Stage B fitness function.
- Optimizes LSTM weights/biases directly.
- No backpropagation.
- This is a research implementation/adaptation.
- Constants not explicitly recovered from the paper are exposed
  as configuration values rather than claimed as exact paper values.
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
# IMPORT FITNESS
# ============================================================

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fitness_module = load_module("fitness_module", FITNESS_FILE)
param_module = load_module("param_vector_module", PARAM_VECTOR_FILE)
model_module = load_module("model_module", MODEL_FILE)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SamuraiConfig:

    # Reproducibility
    seed: int = 42

    # Initial temperature.
    # Engineering value: expose it rather than claiming it is
    # explicitly specified in the RampSense paper.
    T0: float = 1.0

    # Termination temperature.
    Tmin: float = 1e-4

    # Adaptive cooling.
    alpha_min: float = 0.90
    alpha_max: float = 0.99

    # Inner-loop parameters.
    inner_min: int = 20
    inner_max: int = 100

    # Move-size adaptation.
    move_scale_min: float = 0.01
    move_scale_max: float = 0.50

    # Uphill acceptance target.
    target_acceptance: float = 0.20

    # Numerical safety.
    epsilon: float = 1e-12

    # For Stage C1 only.
    # Keep this small before attempting a full optimization.
    max_temperature_steps: int = 5

    # Maximum number of fitness evaluations.
    max_evaluations: int = 500


# ============================================================
# SAMURAI
# ============================================================

class SAMURAI:

    def __init__(
        self,
        objective_function,
        x0: np.ndarray,
        config: SamuraiConfig,
    ):
        self.objective_function = objective_function
        self.config = config

        self.rng = np.random.default_rng(config.seed)

        self.x = np.asarray(x0, dtype=np.float64).copy()

        if self.x.ndim != 1:
            raise ValueError("x0 must be a 1-D vector.")

        if not np.all(np.isfinite(self.x)):
            raise ValueError("x0 contains NaN or Inf.")

        self.dimension = self.x.size

        self.fx = float(self.objective_function(self.x))

        if not np.isfinite(self.fx):
            raise ValueError("Initial objective is not finite.")

        self.best_x = self.x.copy()
        self.best_fx = self.fx

        self.temperature = config.T0

        self.move_scale = 0.10

        self.evaluations = 1
        self.temperature_steps = 0

        self.accepted_moves = 0
        self.accepted_downhill = 0
        self.accepted_uphill = 0

        self.attempted_moves = 0
        self.uphill_attempts = 0

        self.history = []

    # --------------------------------------------------------
    # Proposal
    # --------------------------------------------------------

    def propose(self):
        """
        Continuous-space random proposal.

        The covariance is initially isotropic.
        """
        step = self.rng.normal(
            loc=0.0,
            scale=self.move_scale,
            size=self.dimension,
        )

        return self.x + step

    # --------------------------------------------------------
    # Metropolis acceptance
    # --------------------------------------------------------

    def acceptance_probability(self, current, candidate):

        delta = candidate - current

        if delta <= 0.0:
            return 1.0

        exponent = -delta / max(
            self.temperature,
            self.config.epsilon,
        )

        exponent = max(exponent, -700.0)

        return math.exp(exponent)

    # --------------------------------------------------------
    # One move
    # --------------------------------------------------------

    def step(self):

        candidate_x = self.propose()

        candidate_fx = float(
            self.objective_function(candidate_x)
        )

        self.evaluations += 1
        self.attempted_moves += 1

        delta = candidate_fx - self.fx

        if delta <= 0.0:

            accept_probability = 1.0
            accepted = True

            self.accepted_downhill += 1

        else:

            self.uphill_attempts += 1

            accept_probability = self.acceptance_probability(
                self.fx,
                candidate_fx,
            )

            accepted = (
                self.rng.random() < accept_probability
            )

            if accepted:
                self.accepted_uphill += 1

        if accepted:

            self.x = candidate_x
            self.fx = candidate_fx

            self.accepted_moves += 1

            if self.fx < self.best_fx:

                self.best_fx = self.fx
                self.best_x = self.x.copy()

        return {
            "candidate_fx": candidate_fx,
            "delta": delta,
            "accept_probability": accept_probability,
            "accepted": accepted,
        }

    # --------------------------------------------------------
    # Acceptance statistics
    # --------------------------------------------------------

    def uphill_acceptance_rate(self):

        if self.uphill_attempts == 0:
            return 0.0

        return (
            self.accepted_uphill
            / self.uphill_attempts
        )

    def total_acceptance_rate(self):

        if self.attempted_moves == 0:
            return 0.0

        return (
            self.accepted_moves
            / self.attempted_moves
        )

    # --------------------------------------------------------
    # Adaptive move range
    # --------------------------------------------------------

    def adapt_move_scale(self, uphill_rate):

        target = self.config.target_acceptance

        if uphill_rate < target * 0.5:

            self.move_scale *= 0.80

        elif uphill_rate > target * 1.5:

            self.move_scale *= 1.20

        self.move_scale = float(
            np.clip(
                self.move_scale,
                self.config.move_scale_min,
                self.config.move_scale_max,
            )
        )

    # --------------------------------------------------------
    # Adaptive cooling
    # --------------------------------------------------------

    def adapt_temperature(self, uphill_rate):

        """
        Adaptive cooling.

        Higher uphill acceptance -> cool more slowly.
        Lower uphill acceptance -> cool more aggressively.
        """

        if uphill_rate > 0.30:

            alpha = self.config.alpha_max

        elif uphill_rate < 0.05:

            alpha = self.config.alpha_min

        else:

            alpha = (
                self.config.alpha_min
                + (
                    self.config.alpha_max
                    - self.config.alpha_min
                )
                * (
                    (uphill_rate - 0.05)
                    / (0.30 - 0.05)
                )
            )

        self.temperature *= alpha

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self):

        print()
        print("=" * 70)
        print("RampSense - Stage C1")
        print("SAMURAI Optimization Test")
        print("=" * 70)

        print()
        print(f"Dimension              : {self.dimension}")
        print(f"Initial objective      : {self.fx:.10f}")
        print(f"Initial temperature    : {self.temperature:.6g}")
        print(f"Initial move scale     : {self.move_scale:.6g}")
        print(f"Maximum evaluations    : {self.config.max_evaluations}")
        print(
            f"Temperature steps      : "
            f"{self.config.max_temperature_steps}"
        )

        start_time = time.perf_counter()

        while (
            self.temperature > self.config.Tmin
            and self.temperature_steps
            < self.config.max_temperature_steps
            and self.evaluations
            < self.config.max_evaluations
        ):

            step_start_evals = self.evaluations

            start_uphill_attempts = self.uphill_attempts
            start_uphill_accepts = self.accepted_uphill

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
                        / max(self.config.T0, self.config.epsilon),
                    ),
                    self.config.inner_min,
                    self.config.inner_max,
                )
            )

            for _ in range(inner_limit):

                if self.evaluations >= self.config.max_evaluations:
                    break

                self.step()

            local_uphill_attempts = (
                self.uphill_attempts
                - start_uphill_attempts
            )

            local_uphill_accepts = (
                self.accepted_uphill
                - start_uphill_accepts
            )

            if local_uphill_attempts > 0:

                uphill_rate = (
                    local_uphill_accepts
                    / local_uphill_attempts
                )

            else:

                uphill_rate = 0.0

            self.adapt_move_scale(uphill_rate)

            self.adapt_temperature(uphill_rate)

            self.temperature_steps += 1

            record = {
                "temperature_step": self.temperature_steps,
                "temperature": self.temperature,
                "best_objective": self.best_fx,
                "current_objective": self.fx,
                "move_scale": self.move_scale,
                "uphill_acceptance_rate": uphill_rate,
                "total_acceptance_rate": self.total_acceptance_rate(),
                "evaluations": self.evaluations,
                "inner_evaluations": (
                    self.evaluations - step_start_evals
                ),
            }

            self.history.append(record)

            print()
            print(
                f"Temperature step       : "
                f"{self.temperature_steps}"
            )
            print(
                f"Temperature             : "
                f"{self.temperature:.8f}"
            )
            print(
                f"Best objective          : "
                f"{self.best_fx:.10f}"
            )
            print(
                f"Current objective       : "
                f"{self.fx:.10f}"
            )
            print(
                f"Move scale              : "
                f"{self.move_scale:.8f}"
            )
            print(
                f"Uphill acceptance       : "
                f"{uphill_rate:.6f}"
            )
            print(
                f"Total acceptance        : "
                f"{self.total_acceptance_rate():.6f}"
            )
            print(
                f"Evaluations             : "
                f"{self.evaluations}"
            )

        elapsed = time.perf_counter() - start_time

        result = {
            "best_objective": float(self.best_fx),
            "final_objective": float(self.fx),
            "best_vector_dimension": int(self.best_x.size),
            "evaluations": int(self.evaluations),
            "temperature_steps": int(self.temperature_steps),
            "final_temperature": float(self.temperature),
            "final_move_scale": float(self.move_scale),
            "total_acceptance_rate": float(
                self.total_acceptance_rate()
            ),
            "uphill_acceptance_rate": float(
                self.uphill_acceptance_rate()
            ),
            "elapsed_seconds": float(elapsed),
            "history": self.history,
        }

        print()
        print("=" * 70)
        print("SAMURAI TEST COMPLETE")
        print("=" * 70)

        print(f"Best objective : {self.best_fx:.10f}")
        print(f"Evaluations    : {self.evaluations}")
        print(f"Elapsed        : {elapsed:.3f} s")

        return result


# ============================================================
# FITNESS WRAPPER
# ============================================================

def evaluate(vector: np.ndarray) -> float:

    vector = np.asarray(vector, dtype=np.float64)

    if vector.shape != (1608,):
        raise ValueError(
            f"Expected vector shape (1608,), "
            f"got {vector.shape}"
        )

    if not np.all(np.isfinite(vector)):
        return float("inf")

    result = fitness_module.evaluate_fitness(vector)

    return float(result["objective"])

# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RampSense - SAMURAI Stage C1")
    print("=" * 70)

    print()
    print("Loading model...")

    model = model_module.PVLSTM(
        input_size=6,
        hidden_size=16,
        output_size=8,
    )

    vector = param_module.get_vector(model)

    print(
        f"Optimizer dimension : {vector.size}"
    )

    if vector.size != 1608:
        raise RuntimeError(
            f"Expected 1608 parameters, "
            f"got {vector.size}"
        )

    print()
    print("Testing initial fitness...")

    initial_objective = evaluate(vector)

    print(
        f"Initial objective   : "
        f"{initial_objective:.10f}"
    )

    config = SamuraiConfig()

    optimizer = SAMURAI(
        objective_function=evaluate,
        x0=vector,
        config=config,
    )

    result = optimizer.run()

    output_file = (
        OUTPUT_DIR
        / "stage_c1_samurai_report.json"
    )

    with open(
        output_file,
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
        / "stage_c1_samurai_best.npy"
    )

    np.save(
        best_vector_file,
        optimizer.best_x,
    )

    print()
    print("Output files:")
    print(output_file)
    print(best_vector_file)

    print()
    print("=" * 70)
    print("STAGE C1 PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
