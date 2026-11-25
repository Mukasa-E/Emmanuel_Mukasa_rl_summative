"""
Custom Gymnasium environment: AgriFieldEnv.

Mission-based precision-agriculture environment derived from the AgriMind AI
capstone (agricultural decision-support for smallholder farmers in Kenya).

An autonomous field agent (robot / drone) navigates a farm plot, monitors crop
health and applies the right treatment (water, fertiliser, pesticide) or
harvests at the right time, while managing battery and time.

The environment is sequential, action-dependent and non-trivial:
  * actions change persistent crop state across the field,
  * rewards are delayed (a diseased crop must be scouted before it can be
    treated, and treating too early wastes scarce inputs),
  * the agent must trade off battery consumption against coverage,
  * terminal conditions include full-field recovery/harvest, battery depletion
    and a step limit.

Real-world mapping: every action corresponds to a capability a physical
agricultural robot/drone can perform (drive to a plot, scan with a camera,
irrigate, spread fertiliser, spray pesticide, harvest, wait).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Crop state encoding (one value per field cell)
# ---------------------------------------------------------------------------
CROP_EMPTY = 0   # bare soil / no crop
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

# A crop is "done" (no further agent action required) when healthy, harvested
# or empty.
TERMINAL_CROP_STATES = {CROP_HEALTHY, CROP_HARVESTED, CROP_EMPTY}

# ---------------------------------------------------------------------------
# Action encoding (Discrete(10))
# ---------------------------------------------------------------------------
ACTION_MOVE_NORTH = 0
ACTION_MOVE_SOUTH = 1
ACTION_MOVE_EAST = 2
ACTION_MOVE_WEST = 3
ACTION_SCAN = 4          # monitor / sense current cell
ACTION_WATER = 5         # irrigate current cell
ACTION_FERTILISE = 6     # spread fertiliser on current cell
ACTION_SPRAY_PESTICIDE = 7  # treat disease on current cell
ACTION_HARVEST = 8       # harvest mature crop
ACTION_WAIT = 9          # pass one time step

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
    """
    Precision-agriculture field navigation environment.

    Observation (Box, float32, shape=(17,), all normalised to [0, 1]):
        [0]   agent_x / (grid_size - 1)
        [1]   agent_y / (grid_size - 1)
        [2]   battery / max_battery
        [3]   step / max_steps
        [4:13] 3x3 local crop grid around the agent (each crop_state / 6)
        [13]  current cell needs water        (0/1)
        [14]  current cell needs fertiliser   (0/1)
        [15]  current cell diseased           (0/1)
        [16]  current cell mature             (0/1)

    Action (Discrete(10)): see ACTION_NAMES above.

    Reward (per step):
        * step / wait / move penalty            : -0.1  (encourage efficiency)
        * move into boundary / obstacle         : -2.0
        * wrong treatment on a healthy cell     : -3.0  (wasted input)
        * successful water on needs-water crop  : +5.0
        * successful fertilise on needs-fert    : +5.0
        * successful pesticide on diseased crop : +8.0
        * successful harvest on mature crop     : +10.0
        * scan (information-gathering)          : +0.5
        * terminal bonus (all crops resolved)   : +50.0
        * battery depletion                     : -20.0

    Terminal conditions:
        * every crop cell is in a terminal state (healthy / harvested / empty),
        * battery <= 0,
        * step >= max_steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 200,
        max_battery: float = 100.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.max_battery = float(max_battery)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )

        # RNG (seeded via reset / seed).
        self.np_random: Optional[np.random.Generator] = None
        self.seed_value = seed

        # Pygame rendering surface (created lazily).
        self._screen = None
        self._clock = None
        self._cell_px = 60

        # Episode state (set in reset).
        self.field: Optional[np.ndarray] = None
        self.agent_pos: Optional[np.ndarray] = None
        self.battery: float = 0.0
        self.step_count: int = 0
        self._prev_crop_done_count: int = 0

        # Obstacle cells (immutable during an episode).
        self.obstacles: np.ndarray = np.zeros(
            (self.grid_size, self.grid_size), dtype=bool
        )

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def seed(self, seed: Optional[int] = None) -> None:
        self.seed_value = seed
        self.np_random = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed
        if self.np_random is None or self.seed_value != seed:
            self.np_random = np.random.default_rng(self.seed_value)
        rng = self.np_random

        # Random initial field: mostly crops, some empty, random states.
        self.field = rng.integers(0, 7, size=(self.grid_size, self.grid_size))

        # Ensure there is at least one actionable crop (needs treatment /
        # mature) so the episode is not trivially terminal.
        actionable_mask = np.isin(
            self.field,
            [CROP_NEEDS_WATER, CROP_NEEDS_FERT, CROP_DISEASED, CROP_MATURE],
        )
        if not actionable_mask.any():
            idx = rng.integers(0, self.grid_size * self.grid_size)
            self.field.flat[idx] = int(rng.choice(
                [CROP_NEEDS_WATER, CROP_NEEDS_FERT, CROP_DISEASED, CROP_MATURE]
            ))

        # Fixed obstacles (rocks / irrigation equipment / mud).
        self.obstacles = rng.random(
            (self.grid_size, self.grid_size)
        ) < 0.08
        # Never block the start cell or actionable cells.
        self.obstacles[0, 0] = False
        self.obstacles[actionable_mask] = False

        # Agent starts at top-left corner.
        self.agent_pos = np.array([0, 0], dtype=int)
        self.battery = self.max_battery
        self.step_count = 0
        self._prev_crop_done_count = int(
            np.isin(self.field, list(TERMINAL_CROP_STATES)).sum()
        )

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        action = int(action)
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.step_count += 1
        self.battery -= 1.0  # every action consumes battery

        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        reward = 0.0
        terminated = False
        truncated = False

        # ------------------------------------------------------------------ #
        # Action handling
        # ------------------------------------------------------------------ #
        if action == ACTION_MOVE_NORTH:
            nx, ny = max(x - 1, 0), y
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
        elif action == ACTION_MOVE_SOUTH:
            nx, ny = min(x + 1, self.grid_size - 1), y
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
        elif action == ACTION_MOVE_EAST:
            nx, ny = x, min(y + 1, self.grid_size - 1)
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
        elif action == ACTION_MOVE_WEST:
            nx, ny = x, max(y - 1, 0),  # type: ignore[assignment]
            moved = self._try_move(nx, ny)
            reward += -0.1 if moved else -2.0
        elif action == ACTION_SCAN:
            reward += 0.5  # information-gathering has a small intrinsic value
        elif action == ACTION_WATER:
            reward += self._treat(CROP_NEEDS_WATER, CROP_HEALTHY, 5.0)
        elif action == ACTION_FERTILISE:
            reward += self._treat(CROP_NEEDS_FERT, CROP_HEALTHY, 5.0)
        elif action == ACTION_SPRAY_PESTICIDE:
            reward += self._treat(CROP_DISEASED, CROP_HEALTHY, 8.0)
        elif action == ACTION_HARVEST:
            reward += self._treat(CROP_MATURE, CROP_HARVESTED, 10.0)
        elif action == ACTION_WAIT:
            reward += -0.5
            # Crops degrade over time: healthy crops may need water.
            self._degrade_crops()

        # Natural degradation: untreated needy crops worsen (disease spreads).
        if action != ACTION_WAIT:
            self._spread_disease()

        # ------------------------------------------------------------------ #
        # Termination / truncation
        # ------------------------------------------------------------------ #
        if self.battery <= 0:
            reward += -20.0
            terminated = True

        crop_done_count = int(
            np.isin(self.field, list(TERMINAL_CROP_STATES)).sum()
        )
        total_crops = int((self.field != CROP_EMPTY).sum())
        if total_crops > 0 and crop_done_count == total_crops:
            reward += 50.0
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        self._prev_crop_done_count = crop_done_count
        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _try_move(self, nx: int, ny: int) -> bool:
        """Attempt to move the agent; blocked by obstacles/boundaries."""
        if self.obstacles[nx, ny]:
            return False
        self.agent_pos = np.array([nx, ny], dtype=int)
        return True

    def _treat(self, required_state: int, next_state: int, success_reward: float) -> float:
        """Apply a treatment to the current cell. Returns the step reward."""
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if self.field[x, y] == required_state:
            self.field[x, y] = next_state
            return float(success_reward)
        # Wrong treatment or nothing to treat: wasted input / effort.
        if self.field[x, y] == CROP_HEALTHY:
            return -3.0
        return -0.1

    def _degrade_crops(self) -> None:
        """Wait-action degradation: healthy crops can become thirsty."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.field[x, y] == CROP_HEALTHY and self.np_random.random() < 0.05:
                    self.field[x, y] = CROP_NEEDS_WATER

    def _spread_disease(self) -> None:
        """Slow disease progression: needy crops can become diseased."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = self.field[x, y]
                if state in (CROP_NEEDS_WATER, CROP_NEEDS_FERT):
                    if self.np_random.random() < 0.03:
                        self.field[x, y] = CROP_DISEASED

    def _get_obs(self) -> np.ndarray:
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        # 3x3 local crop window around the agent (clipped at borders).
        local = np.zeros((3, 3), dtype=np.float32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    local[dx + 1, dy + 1] = self.field[nx, ny] / 6.0

        cur = float(self.field[x, y])
        obs = np.array(
            [
                x / (self.grid_size - 1),
                y / (self.grid_size - 1),
                self.battery / self.max_battery,
                self.step_count / self.max_steps,
                *local.flatten(),
                float(cur == CROP_NEEDS_WATER),
                float(cur == CROP_NEEDS_FERT),
                float(cur == CROP_DISEASED),
                float(cur == CROP_MATURE),
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict:
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        return {
            "agent_pos": (int(x), int(y)),
            "battery": float(self.battery),
            "step": int(self.step_count),
            "current_cell": CROP_STATE_NAMES.get(int(self.field[x, y]), "unknown"),
            "crops_resolved": int(
                np.isin(self.field, list(TERMINAL_CROP_STATES)).sum()
            ),
            "total_crops": int((self.field != CROP_EMPTY).sum()),
        }

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_human(self):
        try:
            import pygame  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pygame is required for human rendering") from exc

        if self._screen is None:
            pygame.init()
            size = self.grid_size * self._cell_px
            self._screen = pygame.display.set_mode((size, size + 40))
            pygame.display.set_caption("AgriFieldEnv - Precision Agriculture RL")
            self._clock = pygame.time.Clock()

        self._screen.fill((220, 230, 220))  # light soil background

        # Draw field.
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self._cell_px, x * self._cell_px, self._cell_px, self._cell_px
                )
                state = int(self.field[x, y])
                color = self._crop_color(state)
                if self.obstacles[x, y]:
                    color = (120, 120, 120)  # obstacle = grey
                pygame.draw.rect(self._screen, color, rect)
                pygame.draw.rect(self._screen, (180, 190, 180), rect, 1)

        # Draw agent.
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        agent_rect = pygame.Rect(
            ay * self._cell_px + 8,
            ax * self._cell_px + 8,
            self._cell_px - 16,
            self._cell_px - 16,
        )
        pygame.draw.ellipse(self._screen, (30, 100, 200), agent_rect)

        # HUD.
        self._draw_hud()

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _render_rgb(self) -> np.ndarray:
        try:
            import pygame  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pygame is required for rgb_array rendering") from exc

        if self._screen is None:
            pygame.init()
            size = self.grid_size * self._cell_px
            self._screen = pygame.Surface((size, size + 40))

        self._screen.fill((220, 230, 220))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * self._cell_px, x * self._cell_px, self._cell_px, self._cell_px
                )
                state = int(self.field[x, y])
                color = self._crop_color(state)
                if self.obstacles[x, y]:
                    color = (120, 120, 120)
                pygame.draw.rect(self._screen, color, rect)
                pygame.draw.rect(self._screen, (180, 190, 180), rect, 1)

        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        agent_rect = pygame.Rect(
            ay * self._cell_px + 8,
            ax * self._cell_px + 8,
            self._cell_px - 16,
            self._cell_px - 16,
        )
        pygame.draw.ellipse(self._screen, (30, 100, 200), agent_rect)

        arr = pygame.surfarray.array3d(self._screen)
        return np.transpose(arr, (1, 0, 2))  # (H, W, C)

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
            import pygame  # type: ignore
        except ImportError:
            return
        font = pygame.font.SysFont(None, 24)
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        lines = [
            f"Step: {self.step_count}   Battery: {int(self.battery)}/{int(self.max_battery)}",
            f"Pos: ({x},{y})   Cell: {CROP_STATE_NAMES.get(int(self.field[x, y]),'?')}",
            f"Crops resolved: {self._get_info()['crops_resolved']}/"
            f"{self._get_info()['total_crops']}",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, (0, 0, 0))
            self._screen.blit(surf, (10, self.grid_size * self._cell_px + 5 + i * 22))

    def close(self):
        if self._screen is not None:
            try:
                import pygame  # type: ignore

                pygame.quit()
            except ImportError:
                pass
            self._screen = None
            self._clock = None
