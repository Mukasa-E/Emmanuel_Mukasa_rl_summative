"""Controlled A2C experiment with potential-based exploration shaping."""

import copy
import os

from main import SWEEPS
from training.pg_training import train_pg


def main():
    # Reuse the exact original A2C seed-0 hyperparameters.
    cfg = copy.deepcopy(SWEEPS["a2c"][0])

    # Change only the exploration shaping coefficient.
    cfg["seed"] = 0
    cfg["shaping_coef"] = 0.2

    # Keep evaluation separate from the original experiment.
    cfg["eval_seed"] = 20000
    cfg["task_eval_seed"] = 20000
    cfg["task_eval_episodes"] = 20

    os.makedirs("logs/shaping_test", exist_ok=True)
    os.makedirs("models/shaping_test", exist_ok=True)

    print("=" * 60)
    print("A2C WITH POTENTIAL-BASED EXPLORATION SHAPING")
    print("=" * 60)
    print(f"Seed: {cfg['seed']}")
    print(f"Frames: {cfg['total_frames']}")
    print(f"Learning rate: {cfg['lr']}")
    print(f"Gamma: {cfg['gamma']}")
    print(f"Shaping coefficient: {cfg['shaping_coef']}")
    print("=" * 60)

    metrics = train_pg(
        "a2c",
        cfg,
        env_seed=0,
        log_dir="logs/shaping_test",
        model_dir="models/shaping_test",
    )

    print("\n========== EXPERIMENT RESULTS ==========")
    print(f"Evaluation reward: {metrics['eval_greedy_reward']:.3f}")
    print(
        "Mission completion rate: "
        f"{metrics['eval_mission_completion_rate']:.3f}"
    )
    print(
        "Initial task resolution rate: "
        f"{metrics['eval_initial_task_resolution_rate']:.3f}"
    )
    print(
        "Successful interventions: "
        f"{metrics['eval_successful_interventions']:.2f}"
    )
    print(f"Model: {metrics['model_path']}")
    print(f"Log: {metrics['csv_path']}")


if __name__ == "__main__":
    main()
