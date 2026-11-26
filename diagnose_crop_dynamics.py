"""Compare task-aware baseline performance with crop dynamics enabled/disabled."""

import numpy as np

from environment.custom_env import AgriFieldEnv
from baseline import task_aware_action


def evaluate(dynamics_enabled, seeds=range(100, 120)):
    results = []

    env = AgriFieldEnv(
        grid_size=6,
        max_steps=80,
        max_battery=60,
        crop_dynamics_enabled=dynamics_enabled,
    )

    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)

            total_reward = 0.0
            terminated = False
            truncated = False
            info = {}

            while not (terminated or truncated):
                action = task_aware_action(obs, 6)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

            results.append({
                "seed": seed,
                "completed": info["mission_completed"],
                "resolved": info["crops_resolved"],
                "total": info["total_crops"],
                "steps": info["step"],
                "reward": total_reward,
            })

    finally:
        env.close()

    completion = np.mean([r["completed"] for r in results])
    resolution = np.mean([
        r["resolved"] / max(r["total"], 1)
        for r in results
    ])
    reward = np.mean([r["reward"] for r in results])
    steps = np.mean([r["steps"] for r in results])

    print("\n" + "=" * 45)
    print("CROP DYNAMICS:", "ENABLED" if dynamics_enabled else "DISABLED")
    print("=" * 45)
    print(f"Episodes: {len(results)}")
    print(f"Mission completion: {completion:.1%}")
    print(f"Mean crop resolution: {resolution:.1%}")
    print(f"Mean reward: {reward:.2f}")
    print(f"Mean steps: {steps:.1f}")

    return results


if __name__ == "__main__":
    evaluate(True)
    evaluate(False)
