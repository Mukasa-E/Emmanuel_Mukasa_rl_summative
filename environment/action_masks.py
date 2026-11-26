
"""Legality masks for AgriFieldEnv actions.

The mask is derived only from the agent's observation. Crop interventions are
legal only when the current cell is visible and its observed state matches
the intervention. The environment independently enforces this boundary.

Supports:
- Single observations: (observation_dim,)
- Batched observations: (batch_size, observation_dim)
- Arbitrary leading batch dimensions.
- CPU and GPU tensors.
"""

from __future__ import annotations

import torch

# Keep these values aligned with environment.custom_env.
ACTION_MOVE_NORTH = 0
ACTION_MOVE_SOUTH = 1
ACTION_MOVE_EAST = 2
ACTION_MOVE_WEST = 3
ACTION_SCAN = 4
ACTION_WATER = 5
ACTION_FERTILISE = 6
ACTION_SPRAY_PESTICIDE = 7
ACTION_HARVEST = 8
ACTION_WAIT = 9
NUM_ACTIONS = 10

CROP_NEEDS_WATER = 2
CROP_NEEDS_FERT = 3
CROP_DISEASED = 4
CROP_MATURE = 5


def legal_action_mask(
    observation: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Return a boolean action mask shaped ``(..., 10)``.

    Only information contained in the observation is used.

    Args:
        observation: One observation or a batch of observations.
        grid_size: Width and height of the square field.

    Returns:
        Boolean tensor with the same leading dimensions as observation
        and a final dimension of NUM_ACTIONS.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    obs = torch.as_tensor(observation)

    if not obs.is_floating_point():
        obs = obs.float()

    if obs.ndim < 1:
        raise ValueError("observation must have at least one dimension")

    expected_dim = 4 + 3 * grid_size * grid_size

    if obs.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected observation's last dimension to be {expected_dim}, "
            f"got {obs.shape[-1]}"
        )

    # Preserve all leading dimensions for the final output.
    batch_shape = obs.shape[:-1]

    # Flatten all leading batch dimensions into one dimension.
    # This makes cell selection consistent for single and batched inputs.
    obs_flat = obs.reshape(-1, expected_dim)

    batch_size = obs_flat.shape[0]
    n = grid_size * grid_size

    # Decode the agent's current position.
    x = torch.round(
        obs_flat[:, 0] * (grid_size - 1)
    ).long().clamp(0, grid_size - 1)

    y = torch.round(
        obs_flat[:, 1] * (grid_size - 1)
    ).long().clamp(0, grid_size - 1)

    # Decode crop states, visibility and obstacles.
    crop_map = torch.round(
        obs_flat[:, 4:4 + n] * 6.0
    ).long().reshape(batch_size, grid_size, grid_size)

    visibility = (
        obs_flat[:, 4 + n:4 + 2 * n] > 0.5
    ).reshape(batch_size, grid_size, grid_size)

    obstacles = (
        obs_flat[:, 4 + 2 * n:4 + 3 * n] > 0.5
    ).reshape(batch_size, grid_size, grid_size)

    # Flatten each grid so torch.gather can select one cell per observation.
    crop_flat = crop_map.reshape(batch_size, n)
    visibility_flat = visibility.reshape(batch_size, n)
    obstacles_flat = obstacles.reshape(batch_size, n)

    mask = torch.zeros(
        (batch_size, NUM_ACTIONS),
        dtype=torch.bool,
        device=obs.device,
    )

    def get_cell_values(
        grid: torch.Tensor,
        cell_x: torch.Tensor,
        cell_y: torch.Tensor,
    ) -> torch.Tensor:
        """Select corresponding grid cells independently per observation."""
        indices = cell_x * grid_size + cell_y

        return torch.gather(
            grid,
            dim=1,
            index=indices.unsqueeze(1),
        ).squeeze(1)

    # Movement: destination must be inside the grid and not an obstacle.
    directions = {
        ACTION_MOVE_NORTH: (-1, 0),
        ACTION_MOVE_SOUTH: (1, 0),
        ACTION_MOVE_EAST: (0, 1),
        ACTION_MOVE_WEST: (0, -1),
    }

    for action, (dx, dy) in directions.items():
        nx = x + dx
        ny = y + dy

        inside = (
            (nx >= 0)
            & (nx < grid_size)
            & (ny >= 0)
            & (ny < grid_size)
        )

        safe_x = nx.clamp(0, grid_size - 1)
        safe_y = ny.clamp(0, grid_size - 1)

        blocked = get_cell_values(
            obstacles_flat,
            safe_x,
            safe_y,
        )

        mask[:, action] = inside & ~blocked

    # Scanning is useful if any cell in the local 3x3 patch is unknown.
    local_unknown = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=obs.device,
    )

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx = x + dx
            ny = y + dy

            inside = (
                (nx >= 0)
                & (nx < grid_size)
                & (ny >= 0)
                & (ny < grid_size)
            )

            safe_x = nx.clamp(0, grid_size - 1)
            safe_y = ny.clamp(0, grid_size - 1)

            known = get_cell_values(
                visibility_flat,
                safe_x,
                safe_y,
            )

            local_unknown |= inside & ~known

    mask[:, ACTION_SCAN] = local_unknown

    # Treatments require a visible current cell with the matching crop state.
    current_visible = get_cell_values(
        visibility_flat,
        x,
        y,
    )

    current_state = get_cell_values(
        crop_flat,
        x,
        y,
    )

    mask[:, ACTION_WATER] = (
        current_visible & (current_state == CROP_NEEDS_WATER)
    )

    mask[:, ACTION_FERTILISE] = (
        current_visible & (current_state == CROP_NEEDS_FERT)
    )

    mask[:, ACTION_SPRAY_PESTICIDE] = (
        current_visible & (current_state == CROP_DISEASED)
    )

    mask[:, ACTION_HARVEST] = (
        current_visible & (current_state == CROP_MATURE)
    )

    # WAIT is always available as a fallback.
    mask[:, ACTION_WAIT] = True

    # Restore the original leading batch dimensions.
    return mask.reshape(*batch_shape, NUM_ACTIONS)
