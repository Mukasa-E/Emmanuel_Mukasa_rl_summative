"""
DQN (Deep Q-Network) training with TorchRL.

Implements a value-based agent for AgriFieldEnv using:
  * a Dueling MLP Q-network (torchrl.modules.models.DuelingMlpDQNet / MLP),
  * QValueModule + QValueActor for action selection,
  * EGreedyModule for epsilon-greedy exploration (annealed),
  * DQNLoss with a delayed target network (double DQN option),
  * ReplayBuffer with LazyTensorStorage (experience replay),
  * SyncDataCollector for environment interaction.

All runs share the same environment for an objective comparison.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional

import torch
from tensordict import TensorDict
from tensordict.nn import ProbabilisticTensorDictSequential
from torchrl.envs import GymWrapper, TransformedEnv, StepCounter, DoubleToFloat
from torchrl.modules import (
    QValueActor,
    QValueModule,
    EGreedyModule,
    MLP,
)
from torchrl.objectives import DQNLoss, ValueEstimators
from torchrl.objectives.utils import SoftUpdate
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.collectors import SyncDataCollector
from torch.optim import Adam

from environment.custom_env import AgriFieldEnv


def make_env(grid_size: int = 8, max_steps: int = 200, seed: int = 0):
    base = GymWrapper(AgriFieldEnv(grid_size=grid_size, max_steps=max_steps, seed=seed))
    env = TransformedEnv(
        base,
        StepCounter(),
        DoubleToFloat(),
    )
    env.set_seed(seed)
    return env


def build_dqn_model(env, device: str = "cpu"):
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    act_spec = env.action_spec
    num_actions = int(act_spec.n)

    qnet = MLP(
        in_features=obs_dim,
        out_features=num_actions,
        num_cells=[64, 64],
        device=device,
    )
    model = QValueActor(
        module=qnet,
        spec=act_spec,
        in_keys=["observation"],
    )
    return model.to(device)


def train_dqn(
    cfg: Dict,
    *,
    env_seed: int = 0,
    log_dir: str = "logs",
    model_dir: str = "models/dqn",
) -> Dict:
    """Train one DQN run. Returns a metrics dict."""
    import numpy as np

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = cfg.get("device", "cpu")
    grid_size = int(cfg.get("grid_size", 8))
    max_steps = int(cfg.get("max_steps", 200))
    total_frames = int(cfg.get("total_frames", 30000))
    frames_per_batch = int(cfg.get("frames_per_batch", 200))
    lr = float(cfg.get("lr", 1e-3))
    gamma = float(cfg.get("gamma", 0.99))
    buffer_size = int(cfg.get("buffer_size", 10000))
    batch_size = int(cfg.get("batch_size", 64))
    eps_init = float(cfg.get("eps_init", 1.0))
    eps_end = float(cfg.get("eps_end", 0.05))
    eps_anneal = int(cfg.get("eps_anneal", 20000))
    target_update = int(cfg.get("target_update", 8))
    double_dqn = bool(cfg.get("double_dqn", False))
    num_opt_steps = int(cfg.get("num_opt_steps", 4))
    seed = int(cfg.get("seed", env_seed))

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(grid_size=grid_size, max_steps=max_steps, seed=seed)
    model = build_dqn_model(env, device=device).to(device)

    exploration = EGreedyModule(
        env.action_spec,
        annealing_num_steps=eps_anneal,
        eps_init=eps_init,
        eps_end=eps_end,
    )
    explored_model = ProbabilisticTensorDictSequential(model, exploration)

    collector = SyncDataCollector(
        env,
        explored_model,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(buffer_size, device=device),
        batch_size=batch_size,
    )

    loss_module = DQNLoss(
        value_network=model,
        delay_value=True,
        double_dqn=double_dqn,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    loss_module.to(device)
    soft_updater = SoftUpdate(loss_module, eps=0.99)
    optimizer = Adam(loss_module.parameters(), lr=lr)

    # Logging
    os.makedirs(log_dir, exist_ok=True)
    run_tag = cfg.get("tag", "dqn")
    csv_path = os.path.join(log_dir, f"{run_tag}_seed{seed}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["frame", "episode_reward", "episode_length", "epsilon", "loss"]
    )

    frames = 0
    episode_rewards = []
    episode_return = 0.0
    episode_length = 0
    last_loss = 0.0
    num_updates = 0

    for batch in collector:
        batch = batch.to(device)
        # Expand to full tensordict for the replay buffer (add step counter etc.)
        replay_buffer.extend(batch)

        # Epsilon for logging
        epsilon = float(exploration.eps.item()) if hasattr(exploration, "eps") else eps_end

        for _ in range(num_opt_steps):
            sample = replay_buffer.sample()
            sample = sample.to(device)
            loss_out = loss_module(sample)
            loss_val = loss_out.get("loss", loss_out.get("loss_dqn"))
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            last_loss = float(loss_val.detach().item())
            num_updates += 1

            # DQNLoss creates the delayed network; SoftUpdate moves it toward
            # the online network at the configured optimizer-step interval.
            if num_updates % target_update == 0:
                soft_updater.step()

        frames += batch.numel()

        # Collect episode stats from the batch
        done = batch.get(("next", "done"), None)
        rewards = batch.get(("next", "reward"), None)
        if done is None or rewards is None:
            raise RuntimeError("Collector batch is missing episode completion data")
        for t_idx in range(batch.batch_size[0]):
            episode_return += rewards[t_idx].sum().item()
            episode_length += 1
            if done[t_idx].bool().any():
                episode_rewards.append(episode_return)
                writer.writerow(
                    [
                        frames - batch.batch_size[0] + t_idx + 1,
                        episode_return,
                        episode_length,
                        epsilon,
                        last_loss,
                    ]
                )
                episode_return = 0.0
                episode_length = 0

    collector.shutdown()
    env.close()
    csv_file.close()

    # Evaluate the trained policy (greedy) for a few episodes
    eval_reward = evaluate_greedy(model, grid_size=grid_size, max_steps=max_steps, seed=seed + 1000, episodes=10)

    # Save model
    model_path = os.path.join(model_dir, f"{run_tag}_seed{seed}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "env_seed": env_seed,
        },
        model_path,
    )

    recent = episode_rewards[-25:] if episode_rewards else [0.0]
    metrics = {
        "algo": "dqn",
        "tag": run_tag,
        "seed": seed,
        "frames": frames,
        "mean_reward_last25": float(np.mean(recent)),
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "eval_greedy_reward": float(eval_reward),
        "num_updates": num_updates,
        "model_path": model_path,
        "csv_path": csv_path,
        **cfg,
    }
    return metrics


def evaluate_greedy(model, *, grid_size, max_steps, seed, episodes=10):
    """Run the model greedily (no exploration) on the raw env and return mean reward."""
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_env = AgriFieldEnv(grid_size=grid_size, max_steps=max_steps, seed=seed)
    rewards = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = eval_env.reset()
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            done = False
            ep_r = 0.0
            while not done:
                td = TensorDict({"observation": obs_t}, batch_size=[])
                out = model(td)
                action = out.get("action", out.get("chosen_action"))
                if action is None:
                    q = out.get("action_value")
                    action = q.argmax(-1)
                action = action.detach().cpu()
                if action.numel() > 1:
                    action = action.argmax()
                action = int(action.item())
                obs, r, term, trunc, _ = eval_env.step(action)
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                ep_r += float(r)
                done = bool(term or trunc)
            rewards.append(ep_r)
    eval_env.close()
    return float(np.mean(rewards)) if rewards else 0.0
