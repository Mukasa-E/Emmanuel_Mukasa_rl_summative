"""Compare random and observation-based baselines on seeded missions."""

from __future__ import annotations

import argparse
import json
from collections import deque
from typing import Callable

import numpy as np

from environment.custom_env import (
    ACTION_FERTILISE,
    ACTION_HARVEST,
    ACTION_MOVE_EAST,
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_WEST,
    ACTION_SCAN,
    ACTION_SPRAY_PESTICIDE,
    ACTION_WAIT,
    ACTION_WATER,
    AgriFieldEnv,
    CROP_DISEASED,
    CROP_HARVESTED,
    CROP_HEALTHY,
    CROP_MATURE,
    CROP_NEEDS_FERT,
    CROP_NEEDS_WATER,
)

Policy = Callable[[np.ndarray], int]
ACTION_FOR_CROP = {
    CROP_NEEDS_WATER: ACTION_WATER,
    CROP_NEEDS_FERT: ACTION_FERTILISE,
    CROP_DISEASED: ACTION_SPRAY_PESTICIDE,
    CROP_MATURE: ACTION_HARVEST,
}
TERMINAL_CROP_STATES = {CROP_HEALTHY, CROP_HARVESTED}
NEIGHBORS = (
    (-1, 0, ACTION_MOVE_NORTH),
    (1, 0, ACTION_MOVE_SOUTH),
    (0, 1, ACTION_MOVE_EAST),
    (0, -1, ACTION_MOVE_WEST),
)


def _decode_observation(observation: np.ndarray, grid_size: int):
    cell_count = grid_size**2
    expected_size = 4 + 3 * cell_count
    if observation.shape != (expected_size,):
        raise ValueError(
            f"Expected observation shape ({expected_size},), got {observation.shape}"
        )

    position = (
        int(round(float(observation[0]) * (grid_size - 1))),
        int(round(float(observation[1]) * (grid_size - 1))),
    )
    crop_start = 4
    visibility_start = crop_start + cell_count
    obstacle_start = visibility_start + cell_count
    crop_map = np.rint(
        observation[crop_start:visibility_start].reshape(grid_size, grid_size) * 6
    ).astype(np.int64)
    visible = (
        observation[visibility_start:obstacle_start].reshape(grid_size, grid_size)
        > 0.5
    )
    obstacles = (
        observation[obstacle_start:].reshape(grid_size, grid_size) > 0.5
    )
    return position, crop_map, visible, obstacles


def _shortest_path_action(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    obstacles: np.ndarray,
) -> int | None:
    grid_size = obstacles.shape[0]
    queue = deque([(start, None)])
    visited = {start}

    while queue:
        position, first_action = queue.popleft()
        if position in goals:
            return first_action
        x, y = position
        for dx, dy, action in NEIGHBORS:
            next_position = (x + dx, y + dy)
            nx, ny = next_position
            if (
                0 <= nx < grid_size
                and 0 <= ny < grid_size
                and not obstacles[nx, ny]
                and next_position not in visited
            ):
                visited.add(next_position)
                queue.append(
                    (next_position, action if first_action is None else first_action)
                )
    return None


def task_aware_action(observation: np.ndarray, grid_size: int) -> int:
    """Choose treatment, route to a known task, or explore using only observations."""
    position, crop_map, visible, obstacles = _decode_observation(
        observation, grid_size
    )
    x, y = position

    if visible[x, y] and crop_map[x, y] in ACTION_FOR_CROP:
        return ACTION_FOR_CROP[int(crop_map[x, y])]

    known_targets = {
        (int(tx), int(ty))
        for tx, ty in zip(*np.where(visible & np.isin(
            crop_map, list(ACTION_FOR_CROP)
        )))
    }
    if known_targets:
        action = _shortest_path_action(position, known_targets, obstacles)
        if action is not None:
            return action

    local_unknown = sum(
        not visible[nx, ny]
        for nx in range(max(0, x - 1), min(grid_size, x + 2))
        for ny in range(max(0, y - 1), min(grid_size, y + 2))
    )
    if local_unknown >= 2:
        return ACTION_SCAN

    unknown_cells = {
        (int(ux), int(uy)) for ux, uy in zip(*np.where(~visible & ~obstacles))
    }
    if unknown_cells:
        action = _shortest_path_action(position, unknown_cells, obstacles)
        if action is not None:
            return action
    return ACTION_WAIT


def evaluate_policy(
    policy: Policy,
    *,
    policy_name: str,
    grid_size: int = 6,
    max_steps: int = 80,
    seeds: tuple[int, ...] = tuple(range(100, 120)),
) -> dict:
    """Evaluate one policy on fixed episode seeds and return task-level metrics."""
    if not seeds:
        raise ValueError("At least one episode seed is required")

    env = AgriFieldEnv(grid_size=grid_size, max_steps=max_steps)
    episode_results = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            initial_field = env.field.copy()
            initially_actionable = np.isin(
                initial_field,
                [
                    CROP_NEEDS_WATER,
                    CROP_NEEDS_FERT,
                    CROP_DISEASED,
                    CROP_MATURE,
                ],
            )
            total_return = 0.0
            successful_interventions = 0
            terminated = truncated = False
            final_info = {}

            while not (terminated or truncated):
                position, crop_map, visible, _ = _decode_observation(
                    observation, grid_size
                )
                action = int(policy(observation))
                before_state = int(crop_map[position]) if visible[position] else None
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_return += reward
                if (
                    before_state in ACTION_FOR_CROP
                    and action == ACTION_FOR_CROP[before_state]
                ):
                    successful_interventions += 1

            resolved_initial = int(
                np.isin(
                    env.field[initially_actionable],
                    list(TERMINAL_CROP_STATES),
                ).sum()
            )
            total_initial = int(initially_actionable.sum())
            episode_results.append(
                {
                    "seed": seed,
                    "return": round(total_return, 3),
                    "steps": env.step_count,
                    "initial_tasks_resolved": resolved_initial,
                    "initial_tasks": total_initial,
                    "initial_task_resolution_rate": (
                        round(resolved_initial / total_initial, 3)
                        if total_initial
                        else 1.0
                    ),
                    "successful_interventions": successful_interventions,
                    "mission_completed": bool(
                        terminated
                        and env.battery > 0
                        and final_info["crops_resolved"] == final_info["total_crops"]
                    ),
                }
            )
    finally:
        env.close()

    return {
        "policy": policy_name,
        "episodes": len(episode_results),
        "mean_return": round(
            float(np.mean([result["return"] for result in episode_results])), 3
        ),
        "std_return": round(
            float(np.std([result["return"] for result in episode_results])), 3
        ),
        "mission_completion_rate": round(
            float(np.mean([result["mission_completed"] for result in episode_results])),
            3,
        ),
        "mean_initial_task_resolution_rate": round(
            float(
                np.mean(
                    [
                        result["initial_task_resolution_rate"]
                        for result in episode_results
                    ]
                )
            ),
            3,
        ),
        "mean_successful_interventions": round(
            float(
                np.mean(
                    [
                        result["successful_interventions"]
                        for result in episode_results
                    ]
                )
            ),
            3,
        ),
        "mean_steps": round(
            float(np.mean([result["steps"] for result in episode_results])), 3
        ),
        "episode_results": episode_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    seeds = tuple(range(args.seed, args.seed + args.episodes))
    random_rng = np.random.default_rng(args.seed + 100_000)
    random_policy = lambda _observation: int(random_rng.integers(10))

    results = [
        evaluate_policy(
            random_policy,
            policy_name="random",
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            seeds=seeds,
        ),
        evaluate_policy(
            lambda observation: task_aware_action(observation, args.grid_size),
            policy_name="task_aware",
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            seeds=seeds,
        ),
    ]
    summaries = [
        {key: value for key, value in result.items() if key != "episode_results"}
        for result in results
    ]
    print(json.dumps({"shared_episode_seeds": seeds, "results": summaries}, indent=2))


if __name__ == "__main__":
    main()
