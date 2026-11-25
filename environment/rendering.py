"""
Rendering utilities for AgriFieldEnv.

Provides:
  * a thin wrapper around the environment's built-in Pygame render (GUI used
    by play.py and the video demonstration),
  * a matplotlib-based static figure of the environment (saved to assets/ and
    embedded in the report).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
except Exception:  # pragma: no cover - matplotlib optional at import time
    plt = None  # type: ignore

from .custom_env import (
    AgriFieldEnv,
    CROP_STATE_NAMES,
    CROP_EMPTY,
    CROP_HEALTHY,
    CROP_NEEDS_WATER,
    CROP_NEEDS_FERT,
    CROP_DISEASED,
    CROP_MATURE,
    CROP_HARVESTED,
)

# Colour map matching environment.custom_env.
CROP_COLOURS = {
    CROP_EMPTY: "#d2dcc8",
    CROP_HEALTHY: "#3caa3c",
    CROP_NEEDS_WATER: "#e6c850",
    CROP_NEEDS_FERT: "#e6963c",
    CROP_DISEASED: "#c83c3c",
    CROP_MATURE: "#78501e",
    CROP_HARVESTED: "#a0a0a0",
    "obstacle": "#787878",
}


def render_static(env: AgriFieldEnv, path: Optional[str] = None):
    """Draw the current environment state with matplotlib and optionally save."""
    if plt is None:
        raise RuntimeError("matplotlib is required for static rendering")

    field = env.field
    grid_size = env.grid_size
    obstacles = env.obstacles
    ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])

    fig, ax_plt = plt.subplots(figsize=(6, 5.2))
    for x in range(grid_size):
        for y in range(grid_size):
            if obstacles[x, y]:
                colour = CROP_COLOURS["obstacle"]
                hatch = "///"
            else:
                colour = CROP_COLOURS.get(int(field[x, y]), "#888888")
                hatch = None
            rect = Rectangle(
                (y, grid_size - 1 - x), 1, 1, facecolor=colour,
                edgecolor="#b4bec4", linewidth=0.8, hatch=hatch,
            )
            ax_plt.add_patch(rect)

    # Agent marker.
    circ = Circle(
        (ay + 0.5, grid_size - 1 - ax + 0.5), 0.3,
        facecolor="#1e64c8", edgecolor="black", linewidth=1.2, zorder=5,
    )
    ax_plt.add_patch(circ)

    ax_plt.set_xlim(-0.4, grid_size - 0.6)
    ax_plt.set_ylim(-0.4, grid_size - 0.6)
    ax_plt.set_aspect("equal")
    ax_plt.set_xticks(range(grid_size))
    ax_plt.set_yticks(range(grid_size))
    ax_plt.set_xticklabels([str(i) for i in range(grid_size)])
    ax_plt.set_yticklabels([str(grid_size - 1 - i) for i in range(grid_size)])
    ax_plt.set_xlabel("Field column (y)")
    ax_plt.set_ylabel("Field row (x)")
    ax_plt.set_title(
        "AgriFieldEnv — Precision Agriculture Field\n"
        f"Agent at ({ax},{ay})  |  Battery {int(env.battery)}/{int(env.max_battery)}  |  "
        f"Step {env.step_count}"
    )

    # Legend.
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=CROP_COLOURS[CROP_EMPTY], edgecolor="#b4bec4",
              label="Empty / bare soil"),
        Patch(facecolor=CROP_COLOURS[CROP_HEALTHY], edgecolor="#b4bec4",
              label="Healthy"),
        Patch(facecolor=CROP_COLOURS[CROP_NEEDS_WATER], edgecolor="#b4bec4",
              label="Needs water"),
        Patch(facecolor=CROP_COLOURS[CROP_NEEDS_FERT], edgecolor="#b4bec4",
              label="Needs fertiliser"),
        Patch(facecolor=CROP_COLOURS[CROP_DISEASED], edgecolor="#b4bec4",
              label="Diseased"),
        Patch(facecolor=CROP_COLOURS[CROP_MATURE], edgecolor="#b4bec4",
              label="Mature (ready to harvest)"),
        Patch(facecolor=CROP_COLOURS[CROP_HARVESTED], edgecolor="#b4bec4",
              label="Harvested"),
        Patch(facecolor=CROP_COLOURS["obstacle"], edgecolor="#b4bec4", hatch="///",
              label="Obstacle (rock / equipment / mud)"),
        Patch(facecolor="#1e64c8", edgecolor="black", label="Agent"),
    ]
    ax_plt.legend(
        handles=legend_patches, loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=8, frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.78, 1])

    if path:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig


def render_human(env: AgriFieldEnv):
    """Delegate to the environment's Pygame human render (one frame)."""
    return env.render()


def make_env_visual(path: str = "assets/environment_visual.png"):
    """Create a fresh environment, render it, and save the report visual."""
    env = AgriFieldEnv(grid_size=8, max_steps=200, seed=7, render_mode=None)
    env.reset()
    return render_static(env, path=path)
