"""Shared network backbones for every algorithm (DQN, REINFORCE, PPO, A2C).

The environment observation is a flat vector

    [x, y, battery, step | crop map (N) | visibility (N) | obstacles (N)]

with N = grid_size**2. A plain MLP on that vector has to infer "which cell am I
standing on?" from two scalar coordinates, which makes navigation very slow to
learn. The backbones below keep the *same information* but re-express it as
spatial features (agent-location map, one-hot crop states, visibility and
obstacle maps), so the learning algorithm - not the input format - is the thing
being compared. No information beyond the observation is used.

arch:
    "mlp"         raw flat vector -> MLP (the original baseline architecture)
    "grid_mlp"    spatial feature vector -> MLP
    "grid_cnn"    spatial feature maps -> small CNN -> MLP head
"""

from __future__ import annotations

import math

import torch
from torch import nn

NUM_CROP_CLASSES = 7  # empty, healthy, water, fert, diseased, mature, harvested


def grid_size_from_obs_dim(obs_dim: int) -> int:
    cells = (obs_dim - 4) // 3
    grid = int(round(math.sqrt(cells)))
    if 4 + 3 * grid * grid != obs_dim:
        raise ValueError(f"Observation size {obs_dim} is not 4 + 3 * grid_size**2")
    return grid


def decode_grid_channels(obs: torch.Tensor, grid_size: int):
    """Return (scalars[B,4], maps[B,C,G,G]) from flat observations [B, D]."""
    n = grid_size * grid_size
    scalars = obs[:, :4]
    crop = torch.round(obs[:, 4 : 4 + n] * 6).long().clamp(0, NUM_CROP_CLASSES - 1)
    visible = (obs[:, 4 + n : 4 + 2 * n] > 0.5).float()
    obstacles = (obs[:, 4 + 2 * n : 4 + 3 * n] > 0.5).float()

    x = torch.round(obs[:, 0] * (grid_size - 1)).long().clamp(0, grid_size - 1)
    y = torch.round(obs[:, 1] * (grid_size - 1)).long().clamp(0, grid_size - 1)
    position = torch.zeros_like(visible)
    position.scatter_(1, (x * grid_size + y).unsqueeze(1), 1.0)

    crop_one_hot = torch.nn.functional.one_hot(crop, NUM_CROP_CLASSES).float()
    crop_one_hot = crop_one_hot * visible.unsqueeze(-1)  # unknown cells -> all zero
    crop_one_hot = crop_one_hot.permute(0, 2, 1)  # [B, 7, N]

    maps = torch.cat(
        [
            position.unsqueeze(1),
            crop_one_hot,
            visible.unsqueeze(1),
            obstacles.unsqueeze(1),
        ],
        dim=1,
    )  # [B, 10, N]
    return scalars, maps.reshape(obs.shape[0], -1, grid_size, grid_size)


class _LeadingDims(nn.Module):
    """Flatten arbitrary leading batch dims, apply ``fn`` and restore them."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs.shape[:-1]
        out = self.fn(obs.reshape(-1, obs.shape[-1]))
        return out.reshape(*lead, out.shape[-1])


class _GridMLP(nn.Module):
    def __init__(self, grid_size: int, out_features: int, hidden: int):
        super().__init__()
        self.grid_size = grid_size
        in_features = 4 + (1 + NUM_CROP_CLASSES + 1 + 1) * grid_size**2
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scalars, maps = decode_grid_channels(obs, self.grid_size)
        return self.net(torch.cat([scalars, maps.flatten(1)], dim=1))


class _GridCNN(nn.Module):
    def __init__(self, grid_size: int, out_features: int, hidden: int, channels: int = 32):
        super().__init__()
        self.grid_size = grid_size
        in_channels = 1 + NUM_CROP_CLASSES + 1 + 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * grid_size**2 + 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scalars, maps = decode_grid_channels(obs, self.grid_size)
        features = self.conv(maps).flatten(1)
        return self.head(torch.cat([features, scalars], dim=1))


def build_backbone(
    arch: str,
    obs_dim: int,
    out_features: int,
    *,
    hidden: int = 128,
    device: str = "cpu",
) -> nn.Module:
    """Return a module mapping observations [..., obs_dim] -> [..., out_features]."""
    if arch == "mlp":
        from torchrl.modules import MLP

        return MLP(
            in_features=obs_dim,
            out_features=out_features,
            num_cells=[64, 64],
            device=device,
        )
    grid_size = grid_size_from_obs_dim(obs_dim)
    if arch == "grid_mlp":
        core: nn.Module = _GridMLP(grid_size, out_features, hidden)
    elif arch == "grid_cnn":
        core = _GridCNN(grid_size, out_features, hidden)
    else:
        raise ValueError(f"Unknown architecture {arch!r}")
    return _LeadingDims(core).to(device)
