"""
play.py — Run the best-trained agent in the AgriFieldEnv GUI with verbose
terminal output.

Intended for the video demonstration: shows the agent navigating the field,
applying treatments, and harvesting, with the GUI window and a live terminal
log.

Usage (uv only):
    uv run play.py

The script picks the best-performing algorithm/model from logs/summary.csv
(falling back to models/dqn/best.pt if no summary exists) and runs it
greedily with the Pygame visualization enabled.
"""

from __future__ import annotations

import csv
import os
import sys

import torch
from tensordict import TensorDict

from environment.custom_env import AgriFieldEnv, ACTION_NAMES
from environment.rendering import render_human
from training.dqn_training import build_dqn_model, make_env as make_dqn_env
from training.pg_training import build_actor_critic, make_env as make_pg_env

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def _best_model_from_summary():
    """Return (algo, model_path) of the best run from summary.csv, or None."""
    summary = os.path.join(LOG_DIR, "summary.csv")
    if not os.path.exists(summary):
        return None
    best = None
    with open(summary) as f:
        for row in csv.DictReader(f):
            try:
                reward = float(row["eval_greedy_reward"])
            except (KeyError, ValueError):
                continue
            if best is None or reward > best[0]:
                best = (reward, row["algo"], row["model_path"])
    if best is None:
        return None
    return best[1], best[2]


def _default_model():
    """Fallback: models/dqn/best.pt or models/pg/<first>."""
    dqn_best = os.path.join(MODEL_DIR, "dqn", "best.pt")
    if os.path.exists(dqn_best):
        return "dqn", dqn_best
    pg_dir = os.path.join(MODEL_DIR, "pg")
    if os.path.isdir(pg_dir):
        for name in ("ppo.pt", "a2c.pt", "reinforce.pt", "best.pt"):
            p = os.path.join(pg_dir, name)
            if os.path.exists(p):
                return name.replace(".pt", ""), p
    return None


def load_model(algo, model_path, grid_size=8, max_steps=200, device="cpu"):
    env = AgriFieldEnv(grid_size=grid_size, max_steps=max_steps, seed=0)
    if algo == "dqn":
        model_env = make_dqn_env(grid_size=grid_size, max_steps=max_steps, seed=0)
        try:
            model = build_dqn_model(model_env, device=device)
        finally:
            model_env.close()
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model_state"])
        model.eval()
        return ("dqn", model, env)
    # Policy gradient: rebuild actor + load actor_state.
    model_env = make_pg_env(grid_size=grid_size, max_steps=max_steps, seed=0)
    try:
        actor, _ = build_actor_critic(model_env, device=device)
    finally:
        model_env.close()
    state = torch.load(model_path, map_location=device)
    actor.load_state_dict(state["actor_state"])
    actor.eval()
    return ("pg", actor, env)


def select_action(policy_kind, model, obs_t):
    device = next(model.parameters()).device
    td = TensorDict({"observation": obs_t}, batch_size=[])
    if policy_kind == "dqn":
        out = model(td)
    else:
        out = model(td)
    action = out.get("action", out.get("chosen_action"))
    action = action.detach().cpu()
    if action.numel() > 1:
        action = action.argmax()
    return int(action.item())


def main():
    grid_size = 6
    max_steps = 80

    choice = _best_model_from_summary() or _default_model()
    if choice is None:
        print("ERROR: No trained model found. Run `uv run main.py` first.")
        sys.exit(1)
    algo, model_path = choice

    print("=" * 64)
    print(" AgriFieldEnv — Agent Demonstration (play.py)")
    print(" Mission: autonomous precision-agriculture field agent")
    print("=" * 64)
    print(f"Model : {algo.upper()}  ({os.path.basename(model_path)})")
    print(f"Field : {grid_size}x{grid_size}  |  Max steps: {max_steps}")
    print("Actions: 0=N 1=S 2=E 3=W 4=Scan 5=Water 6=Fertilise "
          "7=Spray 8=Harvest 9=Wait")
    print("=" * 64)

    policy_kind, model, env = load_model(algo, model_path,
                                         grid_size=grid_size,
                                         max_steps=max_steps)
    env.reset()
    env.render_mode = "human"

    total_reward = 0.0
    step = 0
    done = False

    print(f"\n{'Step':>5} | {'Action':<16} | {'Reward':>7} | "
          f"{'Battery':>8} | {'Cell':<16} | {'Crops':<9}")
    print("-" * 70)

    while not done and step < max_steps:
        # Observation vector.
        x, y = int(env.agent_pos[0]), int(env.agent_pos[1])
        obs = env._get_obs()
        obs_t = torch.as_tensor(obs, device=next(model.parameters()).device)\
            .unsqueeze(0)

        action = select_action(policy_kind, model, obs_t)
        action_name = ACTION_NAMES.get(action, f"unk({action})")

        # Step.
        obs, reward, term, trunc, info = env.step(action)
        total_reward += float(reward)
        step += 1
        done = bool(term or trunc)

        cell = info["current_cell"]
        print(f"{step:>5} | {action_name:<16} | {reward:>7.2f} | "
              f"{info['battery']:>8.1f} | {cell:<16} | "
              f"{info['crops_resolved']}/{info['total_crops']}")

        # Render GUI frame.
        render_human(env)

        if info["battery"] <= 0:
            print(">>> BATTERY DEPLETED — terminal.")
            break
        if done:
            print(">>> TERMINAL CONDITION REACHED.")
            break

    print("-" * 70)
    print(f"Episodes steps: {step}   Total reward: {total_reward:.2f}   "
          f"Crops resolved: {info['crops_resolved']}/{info['total_crops']}")
    print("=" * 64)
    print("Demonstration complete. Close the GUI window to exit.")
    env.close()


if __name__ == "__main__":
    main()
