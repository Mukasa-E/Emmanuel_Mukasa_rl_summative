"""
Custom Gymnasium environment: AgriFieldEnv.

Mission-based precision-agriculture environment derived from the AgriMind AI
capstone (agricultural decision-support for smallholder farmers in Kenya).

An autonomous field agent (robot / drone) navigates a farm plot, monitors crop
health and applies the right treatment (water, fertiliser, pesticide) or
harvests at the right time, while managing battery and time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Crop state encoding
# ---------------------------------------------------------------------------
CROP_EMPTY = 0
CROP_HEALTHY = 1
CROP_NEEDS_WATER = 2
CROP_NEEDS_FERT = 3
CROP_DISEASED = 4
CROP_MATURE = 5
CROP_HARVESTED = 6

CROP_STATE_NAMES = {
    CROP_EMPTY: "empty",
    CROP_HEALTHY: "healthy",
    CROP_NEEDS_WATER: "needs_water",
    CROP_NEEDS_FERT: "needs_fertiliser",
    CROP_DISEASED: "diseased",
    CROP_MATURE: "mature",
    CROP_HARVESTED: "harvested",
}

TERMINAL_CROP_STATES = {CROP_HEALTHY, CROP_HARVESTED}
MIN_ACTIONABLE_CROPS = 6
MAX_ACTIONABLE_CROPS = 8
HEALTHY_DEGRADATION_PROBABILITY = 0.002
NEEDY_DISEASE_PROBABILITY = 0.01

# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------
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

ACTION_NAMES = {
    ACTION_MOVE_NORTH: "move_north",
    ACTION_MOVE_SOUTH: "move_south",
    ACTION_MOVE_EAST: "move_east",
    ACTION_MOVE_WEST: "move_west",
    ACTION_SCAN: "scan_monitor",
    ACTION_WATER: "water_irrigate",
    ACTION_FERTILISE: "apply_fertiliser",
    ACTION_SPRAY_PESTICIDE: "spray_pesticide",
    ACTION_HARVEST: "harvest",
    ACTION_WAIT: "wait",
}

NUM_ACTIONS = len(ACTION_NAMES)


class AgriFieldEnv(gym.Env):
    """Precision-agriculture field navigation environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 200,
        max_battery: float = 60.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        shaping_coef: float = 0.0,
        shaping_gamma: float = 0.99,
        crop_dynamics_enabled: bool = True,
    ):
        super().__init__()

        if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 2:
            raise ValueError("grid_size must be an integer greater than or equal to 2")
        if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if not np.isfinite(max_battery) or max_battery <= 0:
            raise ValueError("max_battery must be finite and greater than 0")
        if not np.isfinite(shaping_coef):
            raise ValueError("shaping_coef must be finite")
        if not np.isfinite(shaping_gamma) or not 0.0 <= shaping_gamma <= 1.0:
            raise ValueError("shaping_gamma must be between 0 and 1")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if not isinstance(crop_dynamics_enabled, bool):
            raise ValueError("crop_dynamics_enabled must be a boolean")

        self.shaping_coef = float(shaping_coef)
        self.shaping_gamma = float(shaping_gamma)
        self.crop_dynamics_enabled = crop_dynamics_enabled

        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.max_battery = float(max_battery)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4 + 3 * self.grid_size**2,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.seed_value = seed

        self._screen = None
        self._clock = None
        self._cell_px = 60

        self.field = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.int64
        )
        self.agent_pos = np.zeros(2, dtype=int)
        self.crop_known = np.zeros(
            (self.grid_size, self.grid_size), dtype=bool
        )
        self.battery: float = 0.0
        self.step_count: int = 0

        self.obstacles: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=bool
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:
        self.seed_value = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed_value = seed
            self._rng = np.random.default_rng(seed)

        rng = self._rng

        cell_count = self.grid_size * self.grid_size
        min_targets = min(MIN_ACTIONABLE_CROPS, cell_count - 1)
        max_targets = min(MAX_ACTIONABLE_CROPS, cell_count - 1)

        target_count = int(rng.integers(min_targets, max_targets + 1))
        target_positions = rng.choice(
            cell_count, size=target_count, replace=False
        )

        self.field = np.full(
            (self.grid_size, self.grid_size),
            CROP_HEALTHY,
            dtype=np.int64,
        )

        target_states = rng.choice(
            [CROP_NEEDS_WATER, CROP_NEEDS_FERT, CROP_DISEASED, CROP_MATURE],
            size=target_count,
        )
        self.field.flat[target_positions] = target_states

        actionable_mask = np.zeros(cell_count, dtype=bool)
        actionable_mask[target_positions] = True
        actionable_mask = actionable_mask.reshape(
            self.grid_size, self.grid_size
        )

        non_target_positions = np.flatnonzero(~actionable_mask.ravel())
        empty_count = min(
            int(rng.integers(
                max(1, cell_count // 12),
                max(2, cell_count // 8) + 1,
            )),
            len(non_target_positions),
        )

        empty_positions = rng.choice(
            non_target_positions,
            size=empty_count,
            replace=False,
        )
        self.field.flat[empty_positions] = CROP_EMPTY

        self.obstacles = (
            rng.random((self.grid_size, self.grid_size)) < 0.08
        )
        self.obstacles[0, 0] = False
        self.obstacles[actionable_mask] = False

        self.agent_pos = np.array([0, 0], dtype=int)
        self.crop_known = np.zeros(
            (self.grid_size, self.grid_size), dtype=bool
        )
        self._reveal_crop(0, 0)

        self.battery = self.max_battery
        self.step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")

        action = int(action)

        self.step_count += 1
        self.battery -= 1.0

        phi_before = (
            self._frontier_potential() if self.shaping_coef else 0.0
        )

        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        reward = 0.0
        terminated = False
        truncated = False
        newly_revealed = 0

        # Action handling
        if action == ACTION_MOVE_NORTH:
            nx, ny = x - 1, y
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
            if moved:
                newly_revealed += self._reveal_crop(nx, ny)

        elif action == ACTION_MOVE_SOUTH:
            nx, ny = x + 1, y
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
            if moved:
                newly_revealed += self._reveal_crop(nx, ny)

        elif action == ACTION_MOVE_EAST:
            nx, ny = x, y + 1
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
            if moved:
                newly_revealed += self._reveal_crop(nx, ny)

        elif action == ACTION_MOVE_WEST:
            nx, ny = x, y - 1
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
            if moved:
                newly_revealed += self._reveal_crop(nx, ny)

        elif action == ACTION_SCAN:
            reward -= 0.1
            newly_revealed += self._reveal_local_crops(x, y)

        elif action == ACTION_WATER:
            reward += self._treat(CROP_NEEDS_WATER, CROP_HEALTHY, 5.0)

        elif action == ACTION_FERTILISE:
            reward += self._treat(CROP_NEEDS_FERT, CROP_HEALTHY, 5.0)

        elif action == ACTION_SPRAY_PESTICIDE:
            reward += self._treat(CROP_DISEASED, CROP_HEALTHY, 8.0)

        elif action == ACTION_HARVEST:
            reward += self._treat(CROP_MATURE, CROP_HARVESTED, 10.0)

        elif action == ACTION_WAIT:
            reward -= 0.5

        reward += 0.25 * newly_revealed

        # Crop dynamics can be disabled for controlled diagnostic experiments.
        # Default behaviour remains unchanged: dynamics are enabled.
        if self.crop_dynamics_enabled:
            self._advance_crop_dynamics()

        # Terminal / truncation semantics
        if self.battery <= 0:
            reward -= 20.0
            terminated = True

        crop_done_count = self._resolved_crop_count()
        total_crops = int((self.field != CROP_EMPTY).sum())

        if total_crops > 0 and crop_done_count == total_crops:
            reward += 50.0
            terminated = True

        if not terminated and self.step_count >= self.max_steps:
            truncated = True

        raw_reward = reward

        if self.shaping_coef:
            phi_after = (
                0.0 if terminated else self._frontier_potential()
            )
            reward += self.shaping_coef * (
                self.shaping_gamma * phi_after - phi_before
            )

        obs = self._get_obs()
        info = self._get_info()

        info["newly_revealed"] = int(newly_revealed)
        info["raw_reward"] = float(raw_reward)
        info["mission_completed"] = bool(
            total_crops > 0 and crop_done_count == total_crops
        )

        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reveal_crop(self, x: int, y: int) -> int:
        if self.crop_known[x, y]:
            return 0
        self.crop_known[x, y] = True
        return 1

    def _reveal_local_crops(self, x: int, y: int) -> int:
        newly_revealed = 0

        for nx in range(max(0, x - 1), min(self.grid_size, x + 2)):
            for ny in range(max(0, y - 1), min(self.grid_size, y + 2)):
                newly_revealed += self._reveal_crop(nx, ny)

        return newly_revealed

    def _frontier_potential(self) -> float:
        start = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        g = self.grid_size

        seen = {start}
        frontier = [start]
        dist = 0

        while frontier:
            nxt = []

            for cx, cy in frontier:
                if not self.crop_known[cx, cy]:
                    return -float(dist)

                for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy

                    if (
                        0 <= nx < g
                        and 0 <= ny < g
                        and not self.obstacles[nx, ny]
                        and (nx, ny) not in seen
                    ):
                        seen.add((nx, ny))
                        nxt.append((nx, ny))

            frontier = nxt
            dist += 1

        return 0.0

    def _resolved_crop_count(self) -> int:
        crop_cells = self.field != CROP_EMPTY
        resolved = np.isin(
            self.field, list(TERMINAL_CROP_STATES)
        )
        return int(np.logical_and(crop_cells, resolved).sum())

    def _try_move(self, nx: int, ny: int) -> bool:
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return False

        if self.obstacles[nx, ny]:
            return False

        self.agent_pos = np.array([nx, ny], dtype=int)
        return True

    def _treat(
        self,
        required_state: int,
        next_state: int,
        success_reward: float,
    ) -> float:
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])

        if not self.crop_known[x, y]:
            return -0.5

        if self.field[x, y] == required_state:
            self.field[x, y] = next_state
            return float(success_reward)

        if self.field[x, y] == CROP_HEALTHY:
            return -3.0

        return -0.1

    def _advance_crop_dynamics(self) -> None:
        self._degrade_crops()
        self._progress_disease()

    def _degrade_crops(self) -> None:
        mask = self.field == CROP_HEALTHY
        random_values = self._rng.random(self.field.shape)

        self.field[
            mask & (random_values < HEALTHY_DEGRADATION_PROBABILITY)
        ] = CROP_NEEDS_WATER

    def _progress_disease(self) -> None:
        needy = np.isin(
            self.field, (CROP_NEEDS_WATER, CROP_NEEDS_FERT)
        )
        random_values = self._rng.random(self.field.shape)

        self.field[
            needy & (random_values < NEEDY_DISEASE_PROBABILITY)
        ] = CROP_DISEASED

    def _get_obs(self) -> np.ndarray:
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])

        crop_map = np.where(
            self.crop_known, self.field / 6.0, 0.0
        ).astype(np.float32)

        visibility = self.crop_known.astype(np.float32)
        obstacle_map = self.obstacles.astype(np.float32)

        obs = np.array(
            [
                x / (self.grid_size - 1),
                y / (self.grid_size - 1),
                self.battery / self.max_battery,
                self.step_count / self.max_steps,
                *crop_map.flatten(),
                *visibility.flatten(),
                *obstacle_map.flatten(),
            ],
            dtype=np.float32,
        )

        return obs

    def _get_info(self) -> dict:
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        total_crops = int((self.field != CROP_EMPTY).sum())

        return {
            "agent_pos": (int(x), int(y)),
            "battery": float(self.battery),
            "step": int(self.step_count),
            "current_cell": (
                CROP_STATE_NAMES.get(int(self.field[x, y]), "unknown")
                if self.crop_known[x, y]
                else "unknown"
            ),
            "crops_resolved": self._resolved_crop_count(),
            "total_crops": total_crops,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            return self._render_human()

        if self.render_mode == "rgb_array":
            return self._render_rgb()

        return None

    def _render_human(self):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for human rendering"
            ) from exc

        if self._screen is None:
            pygame.init()
            size = self.grid_size * self._cell_px
            self._screen = pygame.display.set_mode((size, size + 40))
            pygame.display.set_caption(
                "AgriFieldEnv - Precision Agriculture RL"
            )
            self._clock = pygame.time.Clock()

        self._screen.fill((220, 230, 220))
        unknown_font = pygame.font.SysFont(None, 30)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self._cell_px,
                    x * self._cell_px,
                    self._cell_px,
                    self._cell_px,
                )

                if self.obstacles[x, y]:
                    color = (120, 120, 120)
                elif self.crop_known[x, y]:
                    color = self._crop_color(int(self.field[x, y]))
                else:
                    color = (120, 130, 135)

                pygame.draw.rect(self._screen, color, rect)
                pygame.draw.rect(self._screen, (180, 190, 180), rect, 1)

                if not self.obstacles[x, y] and not self.crop_known[x, y]:
                    marker = unknown_font.render(
                        "?", True, (245, 245, 245)
                    )
                    self._screen.blit(
                        marker, marker.get_rect(center=rect.center)
                    )

        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        agent_rect = pygame.Rect(
            ay * self._cell_px + 8,
            ax * self._cell_px + 8,
            self._cell_px - 16,
            self._cell_px - 16,
        )
        pygame.draw.ellipse(
            self._screen, (30, 100, 200), agent_rect
        )

        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _render_rgb(self) -> np.ndarray:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required for rgb_array rendering"
            ) from exc

        if self._screen is None:
            pygame.init()
            size = self.grid_size * self._cell_px
            self._screen = pygame.Surface((size, size + 40))

        self._screen.fill((220, 230, 220))
        unknown_font = pygame.font.SysFont(None, 30)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self._cell_px,
                    x * self._cell_px,
                    self._cell_px,
                    self._cell_px,
                )

                if self.obstacles[x, y]:
                    color = (120, 120, 120)
                elif self.crop_known[x, y]:
                    color = self._crop_color(int(self.field[x, y]))
                else:
                    color = (120, 130, 135)

                pygame.draw.rect(self._screen, color, rect)
                pygame.draw.rect(self._screen, (180, 190, 180), rect, 1)

                if not self.obstacles[x, y] and not self.crop_known[x, y]:
                    marker = unknown_font.render(
                        "?", True, (245, 245, 245)
                    )
                    self._screen.blit(
                        marker, marker.get_rect(center=rect.center)
                    )

        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        agent_rect = pygame.Rect(
            ay * self._cell_px + 8,
            ax * self._cell_px + 8,
            self._cell_px - 16,
            self._cell_px - 16,
        )
        pygame.draw.ellipse(
            self._screen, (30, 100, 200), agent_rect
        )

        arr = pygame.surfarray.array3d(self._screen)
        return np.transpose(arr, (1, 0, 2))

    def _crop_color(self, state: int):
        palette = {
            CROP_EMPTY: (210, 220, 200),
            CROP_HEALTHY: (60, 170, 60),
            CROP_NEEDS_WATER: (230, 200, 80),
            CROP_NEEDS_FERT: (230, 150, 60),
            CROP_DISEASED: (200, 60, 60),
            CROP_MATURE: (120, 80, 30),
            CROP_HARVESTED: (160, 160, 160),
        }
        return palette.get(state, (100, 100, 100))

    def _draw_hud(self):
        try:
            import pygame
        except ImportError:
            return

        font = pygame.font.SysFont(None, 24)
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        info = self._get_info()

        lines = [
            f"Step: {self.step_count}   Battery: {int(self.battery)}/{int(self.max_battery)}",
            f"Pos: ({x},{y})   Cell: {info['current_cell']}",
            f"Crops resolved: {info['crops_resolved']}/{info['total_crops']}",
        ]

        for i, line in enumerate(lines):
            surf = font.render(line, True, (0, 0, 0))
            self._screen.blit(
                surf,
                (10, self.grid_size * self._cell_px + 5 + i * 22),
            )

    def close(self):
        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass

            self._screen = None
            self._clock = None