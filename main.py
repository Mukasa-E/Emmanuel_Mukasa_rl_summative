"""
Main training orchestrator for the RL summative assignment.

Trains DQN, REINFORCE, PPO and A2C on the same AgriFieldEnv using TorchRL,
runs an extensive hyperparameter sweep (>= 10 runs per algorithm), aggregates
results, evaluates generalization on unseen initial states, and writes the
plots and summary CSV used by the report.

Usage (uv only):
    uv run main.py

Outputs:
    logs/*_seed*.csv          per-run training logs
    logs/summary.csv          one row per run (hyperparameters + metrics)
    models/*/best.pt          best model per algorithm
    assets/*.png              figures for the report
"""

from __future__ import annotations

import csv
import itertools
import os
import shutil
from typing import Dict, List

import numpy as np

from training.dqn_training import train_dqn
from training.pg_training import train_pg

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ASSET_DIR = os.path.join(PROJECT_ROOT, "assets")

# Shared environment settings (identical for every algorithm -> fair comparison).
GRID_SIZE = 6
MAX_STEPS = 80
TOTAL_FRAMES = 8000          # frames per training run
FRAMES_PER_BATCH = 128
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # distinct seed per run


def _combinations(param_grid: Dict[str, List], n: int) -> List[Dict]:
    """Return distinct hyperparameter combinations, assigning a seed to each."""
    seeds = param_grid.get("seed")
    keys = [key for key in param_grid if key != "seed"]
    vals = [param_grid[key] for key in keys]
    combos = []
    for combo in itertools.product(*vals):
        config = dict(zip(keys, combo))
        if seeds is not None:
            config["seed"] = seeds[len(combos) % len(seeds)]
        combos.append(config)
        if len(combos) >= n:
            break
    return combos


# ---------------------------------------------------------------------------
# Hyperparameter grids (>= 10 meaningfully different runs per algorithm)
# ---------------------------------------------------------------------------
SWEEPS = {
    "dqn": _combinations(
        {
            "tag": ["dqn"],
            "grid_size": [GRID_SIZE],
            "max_steps": [MAX_STEPS],
            "total_frames": [TOTAL_FRAMES],
            "frames_per_batch": [FRAMES_PER_BATCH],
            "lr": [1e-3, 5e-4, 1e-4],
            "gamma": [0.99, 0.95],
            "buffer_size": [10000, 5000, 20000],
            "batch_size": [64, 32],
            "eps_init": [1.0],
            "eps_end": [0.05, 0.1],
            "eps_anneal": [20000, 10000, 30000],
            "target_update": [4, 8, 16, 32],
            "double_dqn": [False, True],
            "num_opt_steps": [4],
            "seed": SEEDS,
            "device": ["cpu"],
        },
        10,
    ),
    "reinforce": _combinations(
        {
            "tag": ["reinforce"],
            "grid_size": [GRID_SIZE],
            "max_steps": [MAX_STEPS],
            "total_frames": [TOTAL_FRAMES],
            "frames_per_batch": [FRAMES_PER_BATCH],
            "lr": [1e-3, 5e-4, 1e-4],
            "gamma": [0.99, 0.95],
            "entropy_coef": [0.01, 0.001, 0.05],
            "lmbda": [1.0, 0.9],
            "epochs": [1],
            "critic_coef": [1.0, 0.5],
            "seed": SEEDS,
            "device": ["cpu"],
        },
        10,
    ),
    "ppo": _combinations(
        {
            "tag": ["ppo"],
            "grid_size": [GRID_SIZE],
            "max_steps": [MAX_STEPS],
            "total_frames": [TOTAL_FRAMES],
            "frames_per_batch": [FRAMES_PER_BATCH],
            "lr": [1e-3, 5e-4],
            "gamma": [0.99, 0.95],
            "clip_epsilon": [0.2, 0.1],
            "entropy_coef": [0.01, 0.005],
            "lmbda": [0.9, 0.95],
            "epochs": [4, 8],
            "critic_coef": [1.0],
            "seed": SEEDS,
            "device": ["cpu"],
        },
        10,
    ),
    "a2c": _combinations(
        {
            "tag": ["a2c"],
            "grid_size": [GRID_SIZE],
            "max_steps": [MAX_STEPS],
            "total_frames": [TOTAL_FRAMES],
            "frames_per_batch": [FRAMES_PER_BATCH],
            "lr": [1e-3, 5e-4],
            "gamma": [0.99, 0.95],
            "entropy_coef": [0.01, 0.005],
            "lmbda": [0.9, 1.0],
            "epochs": [1],
            "critic_coef": [1.0, 0.5],
            "seed": SEEDS,
            "device": ["cpu"],
        },
        10,
    ),
}


def run_sweeps() -> List[Dict]:
    """Train every sweep configuration and return the metrics list."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "dqn"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "pg"), exist_ok=True)

    all_metrics: List[Dict] = []

    # DQN (value-based)
    for cfg in SWEEPS["dqn"]:
        m = train_dqn(cfg, env_seed=int(cfg["seed"]), log_dir=LOG_DIR,
                      model_dir=os.path.join(MODEL_DIR, "dqn"))
        all_metrics.append(m)
        print(f"[DQN] seed={cfg['seed']} lr={cfg['lr']} gamma={cfg['gamma']} "
              f"eval={m['eval_greedy_reward']:.2f}")

    # Policy-gradient algorithms
    for algo in ("reinforce", "ppo", "a2c"):
        for cfg in SWEEPS[algo]:
            m = train_pg(algo, cfg, env_seed=int(cfg["seed"]), log_dir=LOG_DIR,
                         model_dir=os.path.join(MODEL_DIR, "pg"))
            all_metrics.append(m)
            print(f"[{algo.upper()}] seed={cfg['seed']} lr={cfg['lr']} "
                  f"gamma={cfg['gamma']} eval={m['eval_greedy_reward']:.2f}")

    # Summary CSV
    summary_path = os.path.join(LOG_DIR, "summary.csv")
    if all_metrics:
        keys = list(dict.fromkeys(key for row in all_metrics for key in row))
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in all_metrics:
                writer.writerow(row)
    print(f"\nSummary written to {summary_path} ({len(all_metrics)} runs)")
    return all_metrics


def best_per_algo(metrics: List[Dict]) -> Dict[str, Dict]:
    """Return the best run (by eval greedy reward) per algorithm."""
    best = {}
    for m in metrics:
        algo = m["algo"]
        if algo not in best or m["eval_greedy_reward"] > best[algo]["eval_greedy_reward"]:
            best[algo] = m
    return best


def copy_best_models(best: Dict[str, Dict]):
    """Copy each algorithm's best model to models/<algo>/best.pt."""
    for algo, m in best.items():
        src = m["model_path"]
        dst_dir = os.path.join(MODEL_DIR, algo)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "best.pt")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Best {algo.upper()} model -> {dst}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _read_episode_rewards(csv_path: str):
    """Return (frames, rewards) lists from a per-run CSV log."""
    frames, rewards = [], []
    if not os.path.exists(csv_path):
        return frames, rewards
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frames.append(float(row["frame"]))
                rewards.append(float(row["episode_reward"]))
            except (KeyError, ValueError):
                continue
    return frames, rewards


def plot_results(metrics: List[Dict], best: Dict[str, Dict]):
    """Generate the report figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Plotting skipped (matplotlib unavailable): {exc}")
        return

    os.makedirs(ASSET_DIR, exist_ok=True)

    # ---- 1. Comparative cumulative rewards (4 subplots, best model per algo) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Cumulative Reward over Episodes — Best Run per Algorithm",
                 fontsize=12, fontweight="bold")
    algos = ["dqn", "reinforce", "ppo", "a2c"]
    titles = {"dqn": "DQN (Value-Based)", "reinforce": "REINFORCE",
              "ppo": "PPO", "a2c": "A2C"}
    for ax, algo in zip(axes.flat, algos):
        m = best.get(algo)
        frames, rewards = _read_episode_rewards(m["csv_path"]) if m else ([], [])
        if rewards:
            ax.plot(frames, rewards, label=algo.upper(), color="#1e64c8")
            # running mean over last 5 episodes for trend
            if len(rewards) >= 5:
                win = 5
                trend = np.convolve(rewards, np.ones(win) / win, mode="valid")
                tf = frames[win - 1:]
                ax.plot(tf, trend, "--", label="5-ep moving avg", color="#c83c3c")
        ax.set_title(titles[algo])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Episode Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(ASSET_DIR, "cumulative_rewards.png"), dpi=150)
    plt.close(fig)

    # ---- 2. DQN loss curve + policy entropy proxy (best runs) ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dqn_m = best.get("dqn")
    if dqn_m and os.path.exists(dqn_m["csv_path"]):
        frames, losses = [], []
        with open(dqn_m["csv_path"]) as f:
            for row in csv.DictReader(f):
                try:
                    frames.append(float(row["frame"]))
                    losses.append(float(row["loss"]))
                except (KeyError, ValueError):
                    continue
        axes[0].plot(frames, losses, color="#1e64c8")
        axes[0].set_title("DQN Objective / Loss (best run)")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "DQN log unavailable", ha="center", va="center")
        axes[0].set_title("DQN Objective / Loss (best run)")

    # Entropy proxy: use per-run max reward as learning indicator; plot best-run
    # reward trajectory for all PG algorithms on shared axes.
    pg_algos = ["reinforce", "ppo", "a2c"]
    for algo in pg_algos:
        m = best.get(algo)
        frames, rewards = _read_episode_rewards(m["csv_path"]) if m else ([], [])
        if rewards:
            axes[1].plot(frames, rewards, label=algo.upper())
    axes[1].set_title("Policy-Gradient Episode Reward (best runs)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Episode Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "stability.png"), dpi=150)
    plt.close(fig)

    # ---- 3. Hyperparameter sweep comparison (bar chart of eval reward) ----
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, values, colors = [], [], []
    palette = {"dqn": "#1e64c8", "reinforce": "#3caa3c", "ppo": "#e6963c",
               "a2c": "#c83c3c"}
    for m in metrics:
        labels.append(f"{m['algo'].upper()} s{m['seed']}")
        values.append(m["eval_greedy_reward"])
        colors.append(palette.get(m["algo"], "#888888"))
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Evaluation Greedy Reward")
    ax.set_title("Hyperparameter Sweep — Evaluation Reward per Run (sorted)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, "sweep_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Figures saved to {ASSET_DIR}")


# ---------------------------------------------------------------------------
# Generalization: test each best model on unseen initial states (seeds)
# ---------------------------------------------------------------------------
def generalization_test(best: Dict[str, Dict], n_episodes: int = 10):
    """Evaluate each best model on several unseen seeds; return a dict."""
    import torch

    from training.dqn_training import evaluate_greedy as dqn_eval
    from training.pg_training import (
        build_actor_critic,
        evaluate_greedy as pg_eval,
        make_env as make_pg_env,
    )

    results = {}
    unseen_seeds = [100, 101, 102, 103, 104]
    for algo, m in best.items():
        per_seed = []
        for s in unseen_seeds:
            if algo == "dqn":
                model = _load_dqn(m["model_path"], GRID_SIZE, MAX_STEPS)
                r = dqn_eval(model, grid_size=GRID_SIZE, max_steps=MAX_STEPS,
                             seed=s, episodes=n_episodes)
            else:
                env_tmp = make_pg_env(grid_size=GRID_SIZE, max_steps=MAX_STEPS, seed=0)
                try:
                    actor, _ = build_actor_critic(env_tmp)
                finally:
                    env_tmp.close()
                state = torch.load(m["model_path"], map_location="cpu")
                actor.load_state_dict(state["actor_state"])
                r = pg_eval(actor, grid_size=GRID_SIZE, max_steps=MAX_STEPS,
                            seed=s, episodes=n_episodes)
            per_seed.append(r)
        results[algo] = {
            "seeds": unseen_seeds,
            "rewards": per_seed,
            "mean": float(np.mean(per_seed)),
            "std": float(np.std(per_seed)),
        }
    return results


def _load_dqn(model_path, grid_size, max_steps):
    import torch
    from training.dqn_training import build_dqn_model, make_env
    env = make_env(grid_size=grid_size, max_steps=max_steps, seed=0)
    try:
        model = build_dqn_model(env, device="cpu")
    finally:
        env.close()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    return model


def write_generalization_report(gen_results: Dict):
    path = os.path.join(LOG_DIR, "generalization.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "seed", "reward"])
        for algo, res in gen_results.items():
            for s, r in zip(res["seeds"], res["rewards"]):
                writer.writerow([algo, s, f"{r:.4f}"])
    print(f"Generalization results -> {path}")


def main():
    print("=== AgriFieldEnv RL Training Orchestrator ===")
    print(f"Algorithms: DQN, REINFORCE, PPO, A2C  |  Runs/algo: 10  |  "
          f"Frames/run: {TOTAL_FRAMES}\n")
    metrics = run_sweeps()
    best = best_per_algo(metrics)
    print("\n=== Best run per algorithm (by eval greedy reward) ===")
    for algo, m in best.items():
        print(f"  {algo.upper():10s} seed={m['seed']}  eval={m['eval_greedy_reward']:.2f}  "
              f"max={m['max_reward']:.2f}")
    copy_best_models(best)
    plot_results(metrics, best)
    print("\n=== Generalization test (unseen initial states) ===")
    gen = generalization_test(best, n_episodes=5)
    for algo, res in gen.items():
        print(f"  {algo.upper():10s} mean={res['mean']:.2f}  std={res['std']:.2f} "
              f"over seeds {res['seeds']}")
    write_generalization_report(gen)
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
