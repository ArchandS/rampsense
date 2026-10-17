"""
RampSense — Stage P1: Particle Swarm Optimization (PSO)
======================================================

Compatible with the current fitness.py API.

Uses:
    fitness.model
    fitness.get_vector(model)
    fitness.evaluate_fitness(parameter_vector)

Default configuration:
    Swarm size: 20
    Iterations: 25
    Maximum evaluations: 500
    Parameter vector: 1608 dimensions

If outputs/stage_c11_samurai_best.npy exists, PSO uses the
SAMURAI C11 solution to initialize the swarm.

Run:
    python pso.py
"""

import json
import time
from pathlib import Path

import numpy as np
import fitness


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DIMENSION = 1608

SWARM_SIZE = 20
ITERATIONS = 25

# Standard PSO coefficients
INERTIA = 0.7298
COGNITIVE = 1.49618
SOCIAL = 1.49618

# Swarm initialization
INITIAL_POSITION_NOISE = 0.01
INITIAL_VELOCITY_NOISE = 0.001

# Maximum velocity for each parameter
VELOCITY_CLIP = 0.05


# ============================================================
# PATHS
# ============================================================

OPTIMIZER_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = OPTIMIZER_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C11_BEST_PATH = OUTPUT_DIR / "stage_c11_samurai_best.npy"

BEST_VECTOR_PATH = OUTPUT_DIR / "stage_p1_pso_best.npy"

REPORT_PATH = OUTPUT_DIR / "stage_p1_pso_report.json"

HISTORY_PATH = OUTPUT_DIR / "stage_p1_pso_history.json"

CHECKPOINT_PATH = OUTPUT_DIR / "stage_p1_pso_checkpoint.npz"


# ============================================================
# FITNESS API
# ============================================================

def evaluate_candidate(parameter_vector):
    """
    Evaluate one candidate using the existing fitness.py.

    fitness.evaluate_fitness() returns a dictionary containing:
        objective
        mse_scaled
        f1
        tp, fp, fn, tn
        mae_w
        rmse_w
        other metrics
    """

    parameter_vector = np.asarray(
        parameter_vector,
        dtype=np.float64,
    )

    if parameter_vector.shape != (DIMENSION,):
        raise ValueError(
            f"Expected vector shape ({DIMENSION},), "
            f"got {parameter_vector.shape}"
        )

    if not np.all(np.isfinite(parameter_vector)):
        raise ValueError(
            "Candidate contains NaN or infinite values."
        )

    # Correct API from your fitness.py
    result = fitness.evaluate_fitness(parameter_vector)

    if not isinstance(result, dict):
        raise TypeError(
            "fitness.evaluate_fitness() must return a dictionary."
        )

    if "objective" not in result:
        raise KeyError(
            "Fitness result does not contain 'objective'."
        )

    result = dict(result)

    result["objective"] = float(result["objective"])

    if not np.isfinite(result["objective"]):
        raise ValueError(
            f"Invalid fitness objective: {result['objective']}"
        )

    return result


# ============================================================
# JSON OUTPUT
# ============================================================

def save_json(path, data):
    """Save results as formatted JSON."""

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# INITIALIZE SWARM
# ============================================================

def initialize_swarm(center, rng):
    """
    Initialize the PSO population.

    Particle 0 is exactly the center vector.
    Other particles are small perturbations around it.
    """

    positions = center + rng.normal(
        loc=0.0,
        scale=INITIAL_POSITION_NOISE,
        size=(SWARM_SIZE, DIMENSION),
    )

    # Preserve the exact starting solution.
    positions[0] = center.copy()

    velocities = rng.normal(
        loc=0.0,
        scale=INITIAL_VELOCITY_NOISE,
        size=(SWARM_SIZE, DIMENSION),
    )

    # Start the center particle with zero velocity.
    velocities[0] = 0.0

    return positions, velocities


# ============================================================
# MAIN PSO
# ============================================================

def main():

    start_time = time.perf_counter()

    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("RampSense — Stage P1: Particle Swarm Optimization")
    print("=" * 70)

    print(f"Optimizer directory : {OPTIMIZER_DIR}")
    print(f"Output directory    : {OUTPUT_DIR}")
    print(f"Parameter dimension : {DIMENSION}")
    print(f"Swarm size          : {SWARM_SIZE}")
    print(f"Iterations          : {ITERATIONS}")
    print(f"Evaluation budget   : {SWARM_SIZE * ITERATIONS}")
    print(f"Random seed         : {SEED}")
    print()

    # --------------------------------------------------------
    # 1. Obtain initial model vector
    # --------------------------------------------------------

    if not hasattr(fitness, "model"):
        raise AttributeError(
            "fitness.py does not expose fitness.model."
        )

    if not hasattr(fitness, "get_vector"):
        raise AttributeError(
            "fitness.py does not expose fitness.get_vector."
        )

    center = np.asarray(
        fitness.get_vector(fitness.model),
        dtype=np.float64,
    )

    if center.shape != (DIMENSION,):
        raise ValueError(
            f"Expected initial vector shape ({DIMENSION},), "
            f"got {center.shape}"
        )

    if not np.all(np.isfinite(center)):
        raise ValueError(
            "Initial parameter vector contains NaN or Inf."
        )

    center_source = "current model initialization"

    # --------------------------------------------------------
    # 2. Prefer the C11 SAMURAI solution when available
    # --------------------------------------------------------

    if C11_BEST_PATH.exists():

        c11_vector = np.asarray(
            np.load(C11_BEST_PATH),
            dtype=np.float64,
        )

        if (
            c11_vector.shape == (DIMENSION,)
            and np.all(np.isfinite(c11_vector))
        ):

            center = c11_vector.copy()
            center_source = str(C11_BEST_PATH)

            print("Using C11 SAMURAI solution as swarm center.")

        else:

            print(
                "WARNING: C11 vector has an invalid shape "
                "or contains non-finite values."
            )

            print(
                "Using the current model initialization instead."
            )

    else:

        print(
            "C11 vector not found. "
            "Using the current model initialization."
        )

    print(f"Swarm center source: {center_source}")
    print()

    # --------------------------------------------------------
    # 3. Initialize particles
    # --------------------------------------------------------

    positions, velocities = initialize_swarm(
        center,
        rng,
    )

    personal_best_positions = positions.copy()

    personal_best_scores = np.full(
        SWARM_SIZE,
        np.inf,
        dtype=np.float64,
    )

    global_best_position = center.copy()

    global_best_score = np.inf

    global_best_result = None

    history = []

    evaluations = 0

    # --------------------------------------------------------
    # 4. PSO iterations
    # --------------------------------------------------------

    for iteration in range(1, ITERATIONS + 1):

        iteration_start = time.perf_counter()

        print()
        print("-" * 70)
        print(
            f"Iteration {iteration}/{ITERATIONS}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Evaluate all particles
        # ----------------------------------------------------

        for particle in range(SWARM_SIZE):

            result = evaluate_candidate(
                positions[particle]
            )

            score = result["objective"]

            evaluations += 1

            # Update personal best.
            if score < personal_best_scores[particle]:

                personal_best_scores[particle] = score

                personal_best_positions[particle] = (
                    positions[particle].copy()
                )

            # Update global best.
            if score < global_best_score:

                global_best_score = score

                global_best_position = (
                    positions[particle].copy()
                )

                global_best_result = result

            # Print progress periodically.
            if (
                particle == 0
                or (particle + 1) % 5 == 0
            ):

                print(
                    f"Particle {particle + 1:02d}/{SWARM_SIZE} | "
                    f"Objective: {score:.9f} | "
                    f"F1: {result.get('f1', float('nan')):.6f} | "
                    f"Global best: {global_best_score:.9f}"
                )

        # ----------------------------------------------------
        # Update particle velocities
        # ----------------------------------------------------

        r1 = rng.random(
            (SWARM_SIZE, DIMENSION)
        )

        r2 = rng.random(
            (SWARM_SIZE, DIMENSION)
        )

        cognitive_term = (
            COGNITIVE
            * r1
            * (
                personal_best_positions
                - positions
            )
        )

        social_term = (
            SOCIAL
            * r2
            * (
                global_best_position[np.newaxis, :]
                - positions
            )
        )

        velocities = (
            INERTIA * velocities
            + cognitive_term
            + social_term
        )

        # Prevent excessively large steps.
        velocities = np.clip(
            velocities,
            -VELOCITY_CLIP,
            VELOCITY_CLIP,
        )

        # Update positions.
        positions = positions + velocities

        iteration_seconds = (
            time.perf_counter() - iteration_start
        )

        # ----------------------------------------------------
        # Record iteration
        # ----------------------------------------------------

        row = {
            "iteration": iteration,
            "evaluations": evaluations,
            "global_best_objective": float(
                global_best_score
            ),
            "best_personal_objective": float(
                np.min(personal_best_scores)
            ),
            "iteration_seconds": float(
                iteration_seconds
            ),
            "global_best_metrics": global_best_result,
        }

        history.append(row)

        print()
        print(
            f"Iteration time: {iteration_seconds:.2f} seconds"
        )

        print(
            f"Evaluations completed: {evaluations}"
        )

        print(
            f"Global best objective: {global_best_score:.9f}"
        )

        if global_best_result is not None:

            print(
                "Best scaled MSE: "
                f"{global_best_result.get('mse_scaled', float('nan')):.9f}"
            )

            print(
                "Best F1: "
                f"{global_best_result.get('f1', float('nan')):.6f}"
            )

        # ----------------------------------------------------
        # Save checkpoint after every iteration
        # ----------------------------------------------------

        np.save(
            BEST_VECTOR_PATH,
            global_best_position,
        )

        np.savez_compressed(
            CHECKPOINT_PATH,
            positions=positions,
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            personal_best_scores=personal_best_scores,
            global_best_position=global_best_position,
            global_best_score=np.asarray(
                global_best_score
            ),
            iteration=np.asarray(iteration),
            evaluations=np.asarray(evaluations),
        )

        save_json(
            HISTORY_PATH,
            history,
        )

        print(f"Checkpoint saved: {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # 5. Save final report
    # --------------------------------------------------------

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    if global_best_result is None:
        raise RuntimeError(
            "PSO completed without a valid fitness result."
        )

    report = {
        "stage": "P1_PSO",
        "seed": SEED,
        "parameter_dimension": DIMENSION,
        "swarm_size": SWARM_SIZE,
        "iterations": ITERATIONS,
        "evaluation_budget": SWARM_SIZE * ITERATIONS,
        "actual_evaluations": evaluations,
        "pso_parameters": {
            "inertia": INERTIA,
            "cognitive": COGNITIVE,
            "social": SOCIAL,
            "initial_position_noise": INITIAL_POSITION_NOISE,
            "initial_velocity_noise": INITIAL_VELOCITY_NOISE,
            "velocity_clip": VELOCITY_CLIP,
        },
        "center_source": center_source,
        "best_objective": float(global_best_score),
        "best_metrics": global_best_result,
        "elapsed_seconds": float(elapsed_seconds),
        "best_vector_path": str(BEST_VECTOR_PATH),
        "history_path": str(HISTORY_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
    }

    save_json(
        REPORT_PATH,
        report,
    )

    # --------------------------------------------------------
    # 6. Print final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE P1 PSO COMPLETE")
    print("=" * 70)

    print(
        f"Best objective: {global_best_score:.9f}"
    )

    print(
        "Best scaled MSE: "
        f"{global_best_result.get('mse_scaled', float('nan')):.9f}"
    )

    print(
        f"Best F1: {global_best_result.get('f1', float('nan')):.6f}"
    )

    print(
        f"Best MAE (W): {global_best_result.get('mae_w', float('nan')):.6f}"
    )

    print(
        f"Best RMSE (W): {global_best_result.get('rmse_w', float('nan')):.6f}"
    )

    print(
        "TP / FP / FN / TN: "
        f"{global_best_result.get('tp', '?')} / "
        f"{global_best_result.get('fp', '?')} / "
        f"{global_best_result.get('fn', '?')} / "
        f"{global_best_result.get('tn', '?')}"
    )

    print(f"Evaluations: {evaluations}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")

    print()
    print(f"Best vector : {BEST_VECTOR_PATH}")
    print(f"Report      : {REPORT_PATH}")
    print(f"History     : {HISTORY_PATH}")
    print(f"Checkpoint  : {CHECKPOINT_PATH}")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()