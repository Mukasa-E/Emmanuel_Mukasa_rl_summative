
"""
Main training orchestrator for the RL summative assignment.

Trains DQN, REINFORCE, PPO and A2C on AgriFieldEnv using TorchRL.

Features:
- 10 hyperparameter configurations per algorithm (40 total).
- Shared environment settings and reproducible seeds.
- Checkpoint selection based on mission completion, task resolution,
  and evaluation reward.
- Generalization evaluation on 20 unseen environment seeds.
- Evaluation-only mode to reuse completed experiments.
- CSV reports and plots for the final report.

Usage:
    uv run main.py
    uv run main.py --evaluate-only

Outputs:
    logs/*_seed*.csv
    logs/summary.csv
    logs/generalization.csv
    models/*/best.pt
    assets/*.png
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from typing import Dict, List

import numpy as np

from training.dqn_training import train_dqn
from training.pg_training import train_pg


# ---------------------------------------------------------------------------
# Project paths and shared environment configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ASSET_DIR = os.path.join(PROJECT_ROOT, "assets")

GRID_SIZE = 6
MAX_STEPS = 80
TOTAL_FRAMES = 8192
FRAMES_PER_BATCH = 128

SEEDS = list(range(10))
GENERALIZATION_SEEDS = list(range(100, 120))


def ensure_directories():
    """Create the required project output directories."""
    for directory in (
        LOG_DIR,
        MODEL_DIR,
        ASSET_DIR,
        os.path.join(MODEL_DIR, "dqn"),
        os.path.join(MODEL_DIR, "reinforce"),
        os.path.join(MODEL_DIR, "ppo"),
        os.path.join(MODEL_DIR, "a2c"),
        os.path.join(MODEL_DIR, "pg"),
    ):
        os.makedirs(directory, exist_ok=True)


def _as_float(value, default=0.0):
    """Safely convert a metric to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_sweep(algo: str, configurations: List[Dict]) -> List[Dict]:
    """Combine method-specific settings with shared run settings."""
    sweep = []

    for seed, settings in zip(SEEDS, configurations):
        config = {
            "tag": algo,
            "grid_size": GRID_SIZE,
            "max_steps": MAX_STEPS,
            "total_frames": TOTAL_FRAMES,
            "frames_per_batch": FRAMES_PER_BATCH,
            "device": "cpu",
            "seed": seed,
            "eval_seed": 10000,
        }

        config.update(settings)
        sweep.append(config)

    if len(sweep) != len(SEEDS):
        raise ValueError(
            f"{algo} requires one configuration for each seed."
        )

    return sweep


# ---------------------------------------------------------------------------
# Hyperparameter sweeps: 10 configurations per algorithm
# ---------------------------------------------------------------------------

SWEEPS = {
    "dqn": _build_sweep("dqn", [
        {"lr": 1e-3, "gamma": 0.99, "buffer_size": 10000, "batch_size": 64, "eps_init": 1.0, "eps_end": 0.05, "eps_anneal": 2000, "target_update": 4, "double_dqn": False, "num_opt_steps": 4},
        {"lr": 5e-4, "gamma": 0.99, "buffer_size": 10000, "batch_size": 64, "eps_init": 1.0, "eps_end": 0.05, "eps_anneal": 4000, "target_update": 8, "double_dqn": True, "num_opt_steps": 4},
        {"lr": 1e-4, "gamma": 0.99, "buffer_size": 5000, "batch_size": 32, "eps_init": 1.0, "eps_end": 0.1, "eps_anneal": 8000, "target_update": 8, "double_dqn": False, "num_opt_steps": 4},
        {"lr": 1e-3, "gamma": 0.95, "buffer_size": 5000, "batch_size": 32, "eps_init": 1.0, "eps_end": 0.1, "eps_anneal": 4000, "target_update": 16, "double_dqn": True, "num_opt_steps": 4},
        {"lr": 5e-4, "gamma": 0.95, "buffer_size": 20000, "batch_size": 64, "eps_init": 1.0, "eps_end": 0.05, "eps_anneal": 8000, "target_update": 16, "double_dqn": False, "num_opt_steps": 4},
        {"lr": 1e-4, "gamma": 0.95, "buffer_size": 20000, "batch_size": 32, "eps_init": 1.0, "eps_end": 0.1, "eps_anneal": 2000, "target_update": 32, "double_dqn": True, "num_opt_steps": 4},
        {"lr": 1e-3, "gamma": 0.99, "buffer_size": 5000, "batch_size": 32, "eps_init": 1.0, "eps_end": 0.1, "eps_anneal": 2000, "target_update": 32, "double_dqn": False, "num_opt_steps": 4},
        {"lr": 5e-4, "gamma": 0.99, "buffer_size": 20000, "batch_size": 64, "eps_init": 1.0, "eps_end": 0.1, "eps_anneal": 4000, "target_update": 4, "double_dqn": True, "num_opt_steps": 8},
        {"lr": 1e-4, "gamma": 0.99, "buffer_size": 10000, "batch_size": 32, "eps_init": 1.0, "eps_end": 0.05, "eps_anneal": 8000, "target_update": 16, "double_dqn": False, "num_opt_steps": 8},
        {"lr": 1e-3, "gamma": 0.95, "buffer_size": 20000, "batch_size": 64, "eps_init": 1.0, "eps_end": 0.05, "eps_anneal": 4000, "target_update": 32, "double_dqn": True, "num_opt_steps": 8},
    ]),

    "reinforce": _build_sweep("reinforce", [
        {"lr": 1e-3, "gamma": 0.99, "entropy_coef": 0.005, "lmbda": 1.0, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128, "max_grad_norm": 0.5},
        {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.01, "lmbda": 0.9, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 128, "max_grad_norm": 0.5},
        {"lr": 1e-4, "gamma": 0.95, "entropy_coef": 0.0, "lmbda": 0.8, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128, "max_grad_norm": 1.0},
        {"lr": 5e-4, "gamma": 0.95, "entropy_coef": 0.005, "lmbda": 0.8, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 256, "max_grad_norm": 0.5},
        {"lr": 1e-3, "gamma": 0.95, "entropy_coef": 0.01, "lmbda": 0.9, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128, "max_grad_norm": 1.0},
        {"lr": 1e-4, "gamma": 0.99, "entropy_coef": 0.005, "lmbda": 1.0, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 256, "max_grad_norm": 1.0},
        {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.0, "lmbda": 1.0, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 256, "max_grad_norm": 0.5},
        {"lr": 1e-3, "gamma": 0.95, "entropy_coef": 0.0, "lmbda": 0.8, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 128, "max_grad_norm": 1.0},
        {"lr": 1e-4, "gamma": 0.95, "entropy_coef": 0.01, "lmbda": 0.9, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 256, "max_grad_norm": 0.5},
        {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.01, "lmbda": 0.8, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128, "max_grad_norm": 1.0},
    ]),

    "ppo": _build_sweep("ppo", [
        {"lr": 1e-3, "gamma": 0.99, "clip_epsilon": 0.2, "entropy_coef": 0.005, "lmbda": 0.9, "epochs": 4, "critic_coef": 1.0, "frames_per_batch": 128},
        {"lr": 5e-4, "gamma": 0.99, "clip_epsilon": 0.2, "entropy_coef": 0.01, "lmbda": 0.95, "epochs": 4, "critic_coef": 0.5, "frames_per_batch": 256},
        {"lr": 1e-4, "gamma": 0.95, "clip_epsilon": 0.1, "entropy_coef": 0.0, "lmbda": 0.9, "epochs": 2, "critic_coef": 1.0, "frames_per_batch": 128},
        {"lr": 5e-4, "gamma": 0.95, "clip_epsilon": 0.3, "entropy_coef": 0.005, "lmbda": 0.95, "epochs": 8, "critic_coef": 1.0, "frames_per_batch": 128},
        {"lr": 1e-3, "gamma": 0.95, "clip_epsilon": 0.1, "entropy_coef": 0.01, "lmbda": 0.95, "epochs": 4, "critic_coef": 0.5, "frames_per_batch": 256},
        {"lr": 1e-4, "gamma": 0.99, "clip_epsilon": 0.3, "entropy_coef": 0.0, "lmbda": 0.9, "epochs": 8, "critic_coef": 0.5, "frames_per_batch": 128},
        {"lr": 5e-4, "gamma": 0.99, "clip_epsilon": 0.1, "entropy_coef": 0.01, "lmbda": 0.9, "epochs": 2, "critic_coef": 1.0, "frames_per_batch": 256},
        {"lr": 1e-3, "gamma": 0.99, "clip_epsilon": 0.3, "entropy_coef": 0.005, "lmbda": 0.95, "epochs": 2, "critic_coef": 0.5, "frames_per_batch": 128},
        {"lr": 1e-4, "gamma": 0.95, "clip_epsilon": 0.2, "entropy_coef": 0.01, "lmbda": 0.9, "epochs": 4, "critic_coef": 1.0, "frames_per_batch": 256},
        {"lr": 5e-4, "gamma": 0.95, "clip_epsilon": 0.2, "entropy_coef": 0.0, "lmbda": 0.95, "epochs": 8, "critic_coef": 0.5, "frames_per_batch": 128},
    ]),

    "a2c": _build_sweep("a2c", [
        {"lr": 1e-3, "gamma": 0.99, "entropy_coef": 0.005, "lmbda": 0.9, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 128},
        {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.01, "lmbda": 0.95, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 256},
        {"lr": 1e-4, "gamma": 0.95, "entropy_coef": 0.0, "lmbda": 0.9, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 128},
        {"lr": 5e-4, "gamma": 0.95, "entropy_coef": 0.005, "lmbda": 0.95, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128},
        {"lr": 1e-3, "gamma": 0.95, "entropy_coef": 0.01, "lmbda": 1.0, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 256},
        {"lr": 1e-4, "gamma": 0.99, "entropy_coef": 0.005, "lmbda": 1.0, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128},
        {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.0, "lmbda": 1.0, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 256},
        {"lr": 1e-3, "gamma": 0.95, "entropy_coef": 0.0, "lmbda": 0.9, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128},
        {"lr": 1e-4, "gamma": 0.95, "entropy_coef": 0.01, "lmbda": 0.95, "epochs": 1, "critic_coef": 1.0, "frames_per_batch": 256},
        {"lr": 5e-4, "gamma": 0.99, "entropy_coef": 0.01, "lmbda": 0.9, "epochs": 1, "critic_coef": 0.5, "frames_per_batch": 128},
    ]),
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _read_episode_rewards(csv_path: str):
    """Return episode numbers and rewards from a training CSV."""
    episodes, rewards = [], []

    if not os.path.exists(csv_path):
        return episodes, rewards

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                episodes.append(
                    float(row.get("episode", len(episodes) + 1))
                )
                rewards.append(float(row["episode_reward"]))
            except (KeyError, TypeError, ValueError):
                continue

    return episodes, rewards


def _episodes_to_convergence(
    rewards: List[float],
    *,
    window: int = 10,
    minimum_mean_reward: float = 0.0,
    maximum_mean_change: float = 1.0,
    stable_windows: int = 3,
):
    """Find sustained non-negative rolling performance."""
    if len(rewards) < window + stable_windows:
        return None

    means = [
        float(np.mean(rewards[start:start + window]))
        for start in range(len(rewards) - window + 1)
    ]

    stable_count = 0

    for index in range(1, len(means)):
        is_stable = (
            means[index] >= minimum_mean_reward
            and means[index - 1] >= minimum_mean_reward
            and abs(means[index] - means[index - 1])
            <= maximum_mean_change
        )

        stable_count = stable_count + 1 if is_stable else 0

        if stable_count >= stable_windows:
            return index + window

    return None


def run_sweeps() -> List[Dict]:
    """Train all 40 configurations and save the summary."""
    ensure_directories()

    all_metrics = []

    # DQN experiments
    for cfg in SWEEPS["dqn"]:
        metrics = train_dqn(
            cfg,
            env_seed=int(cfg["seed"]),
            log_dir=LOG_DIR,
            model_dir=os.path.join(MODEL_DIR, "dqn"),
        )

        all_metrics.append(metrics)

        print(
            f"[DQN] seed={cfg['seed']} "
            f"lr={cfg['lr']} gamma={cfg['gamma']} "
            f"eval={metrics['eval_greedy_reward']:.2f}"
        )

    # REINFORCE, PPO and A2C experiments
    for algo in ("reinforce", "ppo", "a2c"):
        for cfg in SWEEPS[algo]:
            metrics = train_pg(
                algo,
                cfg,
                env_seed=int(cfg["seed"]),
                log_dir=LOG_DIR,
                model_dir=os.path.join(MODEL_DIR, "pg"),
            )

            all_metrics.append(metrics)

            print(
                f"[{algo.upper()}] seed={cfg['seed']} "
                f"lr={cfg['lr']} gamma={cfg['gamma']} "
                f"eval={metrics['eval_greedy_reward']:.2f}"
            )

    # Add convergence measurements.
    for metrics in all_metrics:
        _, rewards = _read_episode_rewards(metrics["csv_path"])
        metrics["episodes_to_converge"] = (
            _episodes_to_convergence(rewards)
        )

    # Save all experiment results.
    summary_path = os.path.join(LOG_DIR, "summary.csv")

    if all_metrics:
        keys = list(
            dict.fromkeys(
                key
                for row in all_metrics
                for key in row
            )
        )

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)

    print(
        f"\nSummary written to {summary_path} "
        f"({len(all_metrics)} runs)"
    )

    return all_metrics


# ---------------------------------------------------------------------------
# Best checkpoint selection
# ---------------------------------------------------------------------------

def best_per_algo(rows):
    """
    Select the best model for each algorithm.

    Priority:
    1. Mission completion rate.
    2. Initial task resolution rate.
    3. Greedy evaluation reward.
    """
    best = {}

    for algo in sorted({row["algo"] for row in rows}):
        candidates = [
            row for row in rows
            if row["algo"] == algo
        ]

        best[algo] = max(
            candidates,
            key=lambda row: (
                _as_float(
                    row.get("eval_mission_completion_rate")
                ),
                _as_float(
                    row.get("eval_initial_task_resolution_rate")
                ),
                _as_float(
                    row.get("eval_greedy_reward"),
                    float("-inf"),
                ),
            ),
        )

    return best


def copy_best_models(best: Dict[str, Dict]):
    """Copy each algorithm's selected model to its best.pt location."""
    for algo, metrics in best.items():
        source = metrics["model_path"]

        if not os.path.exists(source):
            raise FileNotFoundError(
                f"Selected {algo} checkpoint does not exist: {source}"
            )

        destination_dir = os.path.join(MODEL_DIR, algo)
        os.makedirs(destination_dir, exist_ok=True)

        destination = os.path.join(destination_dir, "best.pt")
        shutil.copy2(source, destination)

        print(f"Best {algo.upper()} model -> {destination}")


def load_existing_summary():
    """Load previously completed experiments from summary.csv."""
    summary_path = os.path.join(LOG_DIR, "summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Could not find {summary_path}. "
            "Run training first or restore your previous summary.csv."
        )

    rows = []

    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            converted = {}

            for key, value in row.items():
                if value == "":
                    converted[key] = None
                    continue

                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value

            rows.append(converted)

    if not rows:
        raise ValueError("summary.csv contains no experiment results.")

    required = {
        "algo",
        "model_path",
        "csv_path",
        "eval_greedy_reward",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"summary.csv is missing required columns: {sorted(missing)}"
        )

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(metrics: List[Dict], best: Dict[str, Dict]):
    """Generate the training and experiment comparison figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    except Exception as exc:
        print(f"Plotting skipped: {exc}")
        return

    os.makedirs(ASSET_DIR, exist_ok=True)

    algos = ["dqn", "reinforce", "ppo", "a2c"]

    titles = {
        "dqn": "DQN (Value-Based)",
        "reinforce": "REINFORCE",
        "ppo": "PPO",
        "a2c": "A2C",
    }

    # 1. Episode reward curves
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    fig.suptitle(
        "Training Episode Reward — Best Evaluation Run per Algorithm",
        fontsize=12,
        fontweight="bold",
    )

    for ax, algo in zip(axes.flat, algos):
        metrics_row = best.get(algo)

        episodes, rewards = (
            _read_episode_rewards(metrics_row["csv_path"])
            if metrics_row else ([], [])
        )

        if rewards:
            ax.plot(
                episodes,
                rewards,
                label=algo.upper(),
                color="#1e64c8",
            )

            if len(rewards) >= 10:
                trend = np.convolve(
                    rewards,
                    np.ones(10) / 10,
                    mode="valid",
                )

                ax.plot(
                    episodes[9:],
                    trend,
                    "--",
                    label="10-episode mean",
                    color="#c83c3c",
                )

        ax.set_title(titles[algo])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(
        os.path.join(ASSET_DIR, "cumulative_rewards.png"),
        dpi=150,
    )
    plt.close(fig)

    # 2. DQN loss and policy entropy
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    dqn_metrics = best.get("dqn")

    if (
        dqn_metrics
        and os.path.exists(dqn_metrics["csv_path"])
    ):
        frames, losses = [], []

        with open(dqn_metrics["csv_path"], newline="") as f:
            for row in csv.DictReader(f):
                try:
                    frames.append(float(row["frame"]))
                    losses.append(float(row["loss"]))
                except (KeyError, TypeError, ValueError):
                    continue

        axes[0].plot(frames, losses, color="#1e64c8")
        axes[0].set_title("DQN Objective / Loss (best run)")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

    else:
        axes[0].text(
            0.5, 0.5,
            "DQN log unavailable",
            ha="center",
            va="center",
        )

    axes[0].set_title("DQN Objective / Loss (best run)")

    for algo in ("reinforce", "ppo", "a2c"):
        metrics_row = best.get(algo)

        if (
            metrics_row
            and os.path.exists(metrics_row["csv_path"])
        ):
            episodes, entropies = [], []

            with open(metrics_row["csv_path"], newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        episodes.append(float(row["episode"]))
                        entropies.append(float(row["entropy"]))
                    except (KeyError, TypeError, ValueError):
                        continue

            if entropies:
                axes[1].plot(
                    episodes,
                    entropies,
                    label=algo.upper(),
                )

    axes[1].set_title("Policy Entropy (best evaluation runs)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Categorical entropy (nats)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ASSET_DIR, "stability.png"),
        dpi=150,
    )
    plt.close(fig)

    # 3. Evaluation reward comparison
    fig, ax = plt.subplots(figsize=(10, 5))

    labels, values, colors = [], [], []

    palette = {
        "dqn": "#1e64c8",
        "reinforce": "#3caa3c",
        "ppo": "#e6963c",
        "a2c": "#c83c3c",
    }

    for row in metrics:
        labels.append(
            f"{row['algo'].upper()} s{row['seed']}"
        )

        values.append(
            _as_float(row.get("eval_greedy_reward"))
        )

        colors.append(
            palette.get(row["algo"], "#888888")
        )

    order = np.argsort(values)

    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]

    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Deterministic Evaluation Reward")
    ax.set_title(
        "Hyperparameter Sweep — Evaluation Reward per Run"
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ASSET_DIR, "sweep_comparison.png"),
        dpi=150,
    )
    plt.close(fig)

    # 4. Convergence analysis
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    fig.suptitle(
        "Useful-Policy Convergence: "
        "10-Episode Mean ≥ 0, Stable for 3 Windows",
        fontsize=12,
        fontweight="bold",
    )

    for ax, algo in zip(axes.flat, algos):
        metrics_row = best.get(algo)

        episodes, rewards = (
            _read_episode_rewards(metrics_row["csv_path"])
            if metrics_row else ([], [])
        )

        if rewards:
            ax.plot(
                episodes,
                rewards,
                alpha=0.35,
                label="Episode reward",
            )

            if len(rewards) >= 10:
                trend = np.convolve(
                    rewards,
                    np.ones(10) / 10,
                    mode="valid",
                )

                ax.plot(
                    episodes[9:],
                    trend,
                    label="10-episode mean",
                )

            convergence = (
                metrics_row.get("episodes_to_converge")
                if metrics_row else None
            )

            if convergence is not None:
                ax.axvline(
                    float(convergence),
                    color="#c83c3c",
                    linestyle="--",
                    label=f"Converged: episode {convergence}",
                )

            else:
                ax.text(
                    0.98,
                    0.95,
                    "Criterion not reached",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )

        ax.axhline(
            0.0,
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

        ax.set_title(titles[algo])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(
        os.path.join(ASSET_DIR, "convergence.png"),
        dpi=150,
    )
    plt.close(fig)

    print(f"Training figures saved to {ASSET_DIR}")


# ---------------------------------------------------------------------------
# Model loading and generalization evaluation
# ---------------------------------------------------------------------------

def _load_dqn(model_path, grid_size, max_steps):
    """Load a trained DQN model from its checkpoint."""
    import torch

    from training.dqn_training import (
        build_dqn_model,
        make_env,
    )

    env = make_env(
        grid_size=grid_size,
        max_steps=max_steps,
        seed=0,
    )

    try:
        model = build_dqn_model(env, device="cpu")
    finally:
        env.close()

    state = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(state["model_state"])
    model.eval()

    return model


def generalization_test(
    best: Dict[str, Dict],
    n_episodes: int = 1,
):
    """
    Evaluate each selected model on held-out seeds 100–119.

    The existing evaluate_greedy functions return reward, so this
    evaluation reports reward statistics only.
    """
    import torch

    from training.dqn_training import (
        evaluate_greedy as dqn_eval,
    )

    from training.pg_training import (
        build_actor_critic,
        evaluate_greedy as pg_eval,
        make_env as make_pg_env,
    )

    results = {}

    for algo, metrics in best.items():
        model_path = metrics["model_path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing checkpoint for {algo}: {model_path}"
            )

        per_seed = []

        if algo == "dqn":
            model = _load_dqn(
                model_path,
                GRID_SIZE,
                MAX_STEPS,
            )

            for seed in GENERALIZATION_SEEDS:
                reward = dqn_eval(
                    model,
                    grid_size=GRID_SIZE,
                    max_steps=MAX_STEPS,
                    seed=seed,
                    episodes=n_episodes,
                )

                per_seed.append(float(reward))

        else:
            env_tmp = make_pg_env(
                grid_size=GRID_SIZE,
                max_steps=MAX_STEPS,
                seed=0,
            )

            try:
                actor, _ = build_actor_critic(env_tmp)
            finally:
                env_tmp.close()

            state = torch.load(
                model_path,
                map_location="cpu",
            )

            actor.load_state_dict(state["actor_state"])
            actor.eval()

            for seed in GENERALIZATION_SEEDS:
                reward = pg_eval(
                    actor,
                    grid_size=GRID_SIZE,
                    max_steps=MAX_STEPS,
                    seed=seed,
                    episodes=n_episodes,
                )

                per_seed.append(float(reward))

        results[algo] = {
            "seeds": list(GENERALIZATION_SEEDS),
            "rewards": per_seed,
            "mean": float(np.mean(per_seed)),
            "std": float(np.std(per_seed)),
            "episodes_per_seed": n_episodes,
        }

    return results


def write_generalization_report(gen_results: Dict):
    """Write per-seed generalization rewards to CSV."""
    path = os.path.join(LOG_DIR, "generalization.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "algorithm",
            "seed",
            "mean_reward",
            "episodes_per_seed",
            "policy_mode",
        ])

        for algo, result in gen_results.items():
            for seed, reward in zip(
                result["seeds"],
                result["rewards"],
            ):
                writer.writerow([
                    algo,
                    seed,
                    f"{reward:.4f}",
                    result["episodes_per_seed"],
                    "greedy",
                ])

    print(f"Generalization results saved to {path}")


def plot_generalization(results: Dict):
    """Plot mean reward and standard deviation on held-out seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = [
        algo for algo in ("dqn", "reinforce", "ppo", "a2c")
        if algo in results
    ]

    means = [
        results[algo]["mean"]
        for algo in algos
    ]

    deviations = [
        results[algo]["std"]
        for algo in algos
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        [algo.upper() for algo in algos],
        means,
        yerr=deviations,
        capsize=5,
        color=[
            "#1e64c8",
            "#3caa3c",
            "#e6963c",
            "#c83c3c",
        ][:len(algos)],
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel(
        "Mean episode reward (± SD across held-out seeds)"
    )

    ax.set_title(
        "Generalization Across 20 Unseen Environment Seeds"
    )

    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ASSET_DIR, "generalization.png"),
        dpi=150,
    )
    plt.close(fig)

    print(
        f"Generalization figure saved to "
        f"{os.path.join(ASSET_DIR, 'generalization.png')}"
    )


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main(evaluate_only: bool = False):
    """Run training or evaluate existing experiments."""
    print("=== AgriFieldEnv RL Training Orchestrator ===")

    print(
        "Algorithms: DQN, REINFORCE, PPO, A2C | "
        "Runs per algorithm: 10 | "
        f"Frames per run: {TOTAL_FRAMES}\n"
    )

    ensure_directories()

    if evaluate_only:
        print("EVALUATION-ONLY MODE")
        print("Loading existing training results...")
        metrics = load_existing_summary()

    else:
        print("TRAINING MODE")
        print("Running the 40 hyperparameter configurations...")
        metrics = run_sweeps()

    best = best_per_algo(metrics)

    print("\n=== Selected checkpoint per algorithm ===")
    print(
        "Priority: mission completion -> "
        "initial task resolution -> evaluation reward\n"
    )

    for algo, row in best.items():
        completion = _as_float(
            row.get("eval_mission_completion_rate")
        )

        resolution = _as_float(
            row.get("eval_initial_task_resolution_rate")
        )

        reward = _as_float(
            row.get("eval_greedy_reward"),
            float("nan"),
        )

        print(
            f"{algo.upper():10s} "
            f"seed={row['seed']} | "
            f"completion={completion:.3f} | "
            f"resolution={resolution:.3f} | "
            f"reward={reward:.2f}"
        )

        if not os.path.exists(row["model_path"]):
            raise FileNotFoundError(
                f"Checkpoint for {algo} was not found: "
                f"{row['model_path']}"
            )

    copy_best_models(best)

    print("\n=== Generating training figures ===")
    plot_results(metrics, best)

    print("\n=== Generalization evaluation ===")
    print("Held-out seeds: 100–119")
    print("Episodes per seed: 1\n")

    gen_results = generalization_test(
        best,
        n_episodes=1,
    )

    for algo, result in gen_results.items():
        print(
            f"{algo.upper():10s} "
            f"mean={result['mean']:.2f} | "
            f"std={result['std']:.2f} | "
            f"seeds={len(result['seeds'])}"
        )

    write_generalization_report(gen_results)
    plot_generalization(gen_results)

    print("\n=== Experiment pipeline completed ===")
    print(f"Summary CSV: {os.path.join(LOG_DIR, 'summary.csv')}")
    print(
        "Generalization CSV: "
        f"{os.path.join(LOG_DIR, 'generalization.csv')}"
    )
    print(f"Figures: {ASSET_DIR}")
    print(f"Checkpoints: {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgriFieldEnv RL experiment orchestrator"
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reuse existing experiments without retraining.",
    )

    args = parser.parse_args()

    main(evaluate_only=args.evaluate_only)
