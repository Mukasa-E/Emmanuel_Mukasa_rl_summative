
"""
play.py — AgriFieldEnv interactive agent demonstration.

Examples:
    uv run play.py
    uv run play.py --agent task_aware --seed 100
    uv run play.py --agent a2c --seed 100
    uv run play.py --agent ppo --seed 100
    uv run play.py --agent dqn --seed 100
    uv run play.py --agent reinforce --seed 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from tensordict import TensorDict

from environment.custom_env import AgriFieldEnv, ACTION_NAMES
from environment.rendering import render_human
from environment.action_masks import legal_action_mask

from baseline import task_aware_action

from training.dqn_training import (
    build_dqn_model,
    make_env as make_dqn_env,
)

from training.pg_training import (
    build_actor_critic,
    make_env as make_pg_env,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

GRID_SIZE = 6
MAX_STEPS = 80
DEVICE = "cpu"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_checkpoint(algo=None):
    """Find the best checkpoint, optionally restricting to one algorithm."""

    summary_path = os.path.join(LOG_DIR, "summary.csv")
    candidates = []

    if os.path.exists(summary_path):
        with open(summary_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                name = str(row.get("algo", "")).lower()
                path = row.get("model_path")

                if not name or not path:
                    continue

                if algo is not None and name != algo:
                    continue

                if not os.path.isabs(path):
                    path = os.path.join(PROJECT_ROOT, path)

                if not os.path.isfile(path):
                    continue

                candidates.append((
                    safe_float(row.get("eval_mission_completion_rate")),
                    safe_float(row.get("eval_initial_task_resolution_rate")),
                    safe_float(row.get("eval_greedy_reward")),
                    name,
                    path,
                ))

    if candidates:
        candidates.sort(
            key=lambda item: (item[0], item[1], item[2]),
            reverse=True,
        )

        best = candidates[0]

        print("\nSelected checkpoint:")
        print(f"  Algorithm: {best[3].upper()}")
        print(f"  Completion rate: {best[0]:.3f}")
        print(f"  Task resolution: {best[1]:.3f}")
        print(f"  Evaluation reward: {best[2]:.3f}")

        return best[3], best[4]

    # Fallback to known checkpoint locations.
    algorithms = [algo] if algo else [
        "dqn", "a2c", "ppo", "reinforce"
    ]

    for name in algorithms:
        paths = [
            os.path.join(MODEL_DIR, name, "best.pt"),
            os.path.join(MODEL_DIR, "pg", f"{name}.pt"),
        ]

        for path in paths:
            if os.path.isfile(path):
                return name, path

    return None


def load_learned_model(algo, model_path):
    """Rebuild and load a trained agent."""

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if algo == "dqn":
        model_env = make_dqn_env(
            grid_size=GRID_SIZE,
            max_steps=MAX_STEPS,
            seed=0,
        )

        try:
            model = build_dqn_model(
                model_env,
                device=DEVICE,
            )
        finally:
            model_env.close()

        if "model_state" not in checkpoint:
            raise KeyError("Checkpoint is missing model_state.")

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        return "dqn", model

    model_env = make_pg_env(
        grid_size=GRID_SIZE,
        max_steps=MAX_STEPS,
        seed=0,
    )

    try:
        actor, _ = build_actor_critic(
            model_env,
            device=DEVICE,
        )
    finally:
        model_env.close()

    if "actor_state" not in checkpoint:
        raise KeyError("Checkpoint is missing actor_state.")

    actor.load_state_dict(checkpoint["actor_state"])
    actor.eval()

    return "pg", actor



def select_action(policy_kind, model, observation, grid_size=6):
    """Select a greedy action while enforcing the environment's legal-action mask."""

    obs_t = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=DEVICE,
    )

    # Accept either a single observation or a batch of observations.
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)

    td = TensorDict(
        {"observation": obs_t},
        batch_size=[obs_t.shape[0]],
    )

    mask = legal_action_mask(
        obs_t,
        grid_size=grid_size,
    ).bool()

    # Ensure every observation has at least one legal action.
    if not mask.any(dim=-1).all():
        raise RuntimeError("At least one observation has no legal actions.")

    with torch.no_grad():
        if policy_kind == "dqn":
            td.set("action_mask", mask)
            output = model(td)
            q_values = output.get("action_value")

            if q_values is None:
                raise RuntimeError("DQN output is missing action_value.")

            q_values = q_values.masked_fill(
                ~mask,
                torch.finfo(q_values.dtype).min,
            )
            actions = q_values.argmax(dim=-1)

        elif policy_kind in ("pg", "a2c", "ppo", "reinforce"):
            logits = model[0](td).get("logits")

            if logits is None:
                raise RuntimeError("Policy actor is missing logits.")

            logits = logits.masked_fill(
                ~mask,
                torch.finfo(logits.dtype).min,
            )
            actions = logits.argmax(dim=-1)

        else:
            raise ValueError(
                f"Unsupported policy kind: {policy_kind}"
            )

    # Playback and tests use one observation at a time.
    return int(actions[0].item())


def select_learned_action(policy_kind, model, observation):
    """Backward-compatible wrapper for the interactive demonstration."""
    return select_action(
        policy_kind,
        model,
        observation,
        grid_size=GRID_SIZE,
    )



def main():
    parser = argparse.ArgumentParser(
        description="AgriFieldEnv agent demonstration"
    )

    parser.add_argument(
        "--agent",
        choices=["task_aware", "dqn", "ppo", "a2c", "reinforce"],
        default="task_aware",
        help="Agent to demonstrate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Environment episode seed.",
    )

    args = parser.parse_args()

    algo = args.agent
    model = None
    policy_kind = None
    env = None

    if algo != "task_aware":
        choice = get_checkpoint(algo)

        if choice is None:
            print(f"\nERROR: No checkpoint found for {algo.upper()}.")
            sys.exit(1)

        algo, model_path = choice

        print(f"\nLoading {algo.upper()} checkpoint: {model_path}")

        policy_kind, model = load_learned_model(
            algo,
            model_path,
        )

    print("\n" + "=" * 72)
    print("              AGRIFIELDENV — LIVE AGENT DEMONSTRATION")
    print("=" * 72)

    print(f"Agent      : {algo.upper()}")
    print(f"Episode    : {args.seed}")
    print(f"Field      : {GRID_SIZE} x {GRID_SIZE}")
    print(f"Max steps  : {MAX_STEPS}")

    print("\nAvailable actions:")
    for action_id, action_name in ACTION_NAMES.items():
        print(f"  {action_id}: {action_name}")

    print("=" * 72)

    try:
        env = AgriFieldEnv(
            grid_size=GRID_SIZE,
            max_steps=MAX_STEPS,
            seed=args.seed,
            render_mode="human",
        )

        observation, info = env.reset(seed=args.seed)

        total_reward = 0.0
        step = 0
        done = False

        print("\nStarting demonstration...\n")

        print(
            f"{'Step':>4} | "
            f"{'Action':<20} | "
            f"{'Reward':>8} | "
            f"{'Battery':>7} | "
            f"{'Cell':<12} | "
            f"{'Crops':>9}"
        )

        print("-" * 82)

        render_human(env)

        while not done and step < MAX_STEPS:

            if algo == "task_aware":
                action = task_aware_action(
                    observation,
                    GRID_SIZE,
                )
            else:
                action = select_learned_action(
                    policy_kind,
                    model,
                    observation,
                )

            action_name = ACTION_NAMES.get(
                action,
                f"Unknown ({action})",
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_reward += float(reward)
            step += 1
            done = bool(terminated or truncated)

            battery = float(info.get("battery", 0.0))
            resolved = info.get("crops_resolved", 0)
            total = info.get("total_crops", 0)
            cell = str(info.get("current_cell", "Unknown"))

            print(
                f"{step:>4} | "
                f"{action_name:<20} | "
                f"{reward:>8.2f} | "
                f"{battery:>7.1f} | "
                f"{cell:<12} | "
                f"{resolved:>3}/{total:<5}"
            )

            render_human(env)

            if info.get("mission_completed", False):
                print("\nMISSION COMPLETED!")
                break

            if battery <= 0:
                print("\nBATTERY DEPLETED.")
                break

            if terminated or truncated:
                print("\nEPISODE TERMINATED.")
                break

        mission_completed = bool(
            info.get("mission_completed", False)
        )

        print("\n" + "=" * 72)
        print("                 FINAL DEMONSTRATION RESULTS")
        print("=" * 72)

        print(f"Agent                  : {algo.upper()}")
        print(f"Episode seed           : {args.seed}")
        print(f"Steps                  : {step}")
        print(f"Total reward           : {total_reward:.2f}")

        print(
            f"Crops resolved         : "
            f"{info.get('crops_resolved', 0)}/"
            f"{info.get('total_crops', 0)}"
        )

        print(
            f"Remaining battery      : "
            f"{info.get('battery', 0):.1f}"
        )

        print(
            f"Mission completed      : "
            f"{'YES' if mission_completed else 'NO'}"
        )

        print("=" * 72)

        if mission_completed:
            print("SUCCESS: The agent completed the agricultural mission!")
        else:
            print("The agent did not complete this episode.")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
