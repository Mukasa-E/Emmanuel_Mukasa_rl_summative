"""
Policy-Gradient training with TorchRL.

Implements REINFORCE, PPO and A2C agents for AgriFieldEnv using:
  * ProbabilisticActor with a Categorical policy (MLP -> logits -> distribution),
  * ValueOperator critic (MLP -> state value),
  * ReinforceLoss / ClipPPOLoss / A2CLoss,
  * GAE advantage estimation (PPO / A2C),
  * SyncDataCollector for on-policy rollouts.

All runs share the same environment as DQN for an objective comparison.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule
from torch.distributions import Categorical
from torchrl.envs import GymWrapper, TransformedEnv, StepCounter, DoubleToFloat
from torchrl.modules import ValueOperator, MLP
from torchrl.objectives import ReinforceLoss, ClipPPOLoss, A2CLoss, ValueEstimators
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


def build_actor_critic(env, device: str = "cpu"):
    """Build a Categorical actor (ProbabilisticTensorDictSequential) + value critic."""
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    act_spec = env.action_spec
    num_actions = int(act_spec.n)

    # Policy network: observation -> action logits.
    actor_net = MLP(
        in_features=obs_dim,
        out_features=num_actions,
        num_cells=[64, 64],
        device=device,
    )
    actor_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    )
    # Convert logits -> Categorical distribution -> sample action + log_prob.
    prob_module = ProbabilisticTensorDictModule(
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        distribution_kwargs={},
        return_log_prob=True,
        log_prob_keys=["sample_log_prob"],
    )
    actor = ProbabilisticTensorDictSequential(actor_module, prob_module)

    # Value critic: observation -> state value.
    critic_net = MLP(
        in_features=obs_dim,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )
    critic = ValueOperator(
        module=critic_net, in_keys=["observation"], out_keys=["state_value"]
    )

    return actor, critic


def _make_loss(algo: str, actor, critic, gamma: float, cfg: Dict):
    lmbda = float(cfg.get("lmbda", 0.9))
    if algo == "reinforce":
        # REINFORCE with a value baseline for reduced variance.
        loss = ReinforceLoss(
            actor_network=actor,
            critic_network=critic,
            delay_value=False,
        )
    elif algo == "ppo":
        loss = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=float(cfg.get("clip_epsilon", 0.2)),
            entropy_coef=float(cfg.get("entropy_coef", 0.01)),
            critic_coef=float(cfg.get("critic_coef", 1.0)),
        )
    elif algo == "a2c":
        loss = A2CLoss(
            actor_network=actor,
            critic_network=critic,
            entropy_coef=float(cfg.get("entropy_coef", 0.01)),
            critic_coef=float(cfg.get("critic_coef", 1.0)),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    # TorchRL 0.6 deprecates gamma/lambda in the loss constructor; use a value
    # estimator instead.
    loss.make_value_estimator(ValueEstimators.TDLambda, gamma=gamma, lmbda=lmbda)
    return loss


def evaluate_greedy(actor, *, grid_size, max_steps, seed, episodes=10):
    """Run the actor greedily (deterministic) on the raw env and return mean reward."""
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_env = AgriFieldEnv(grid_size=grid_size, max_steps=max_steps, seed=seed)
    rewards = []
    actor.eval()
    device = next(actor.parameters()).device
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = eval_env.reset()
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            done = False
            ep_r = 0.0
            while not done:
                td = TensorDict({"observation": obs_t}, batch_size=[])
                out = actor(td)
                action = out.get("action", out.get("chosen_action"))
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


def train_pg(
    algo: str,
    cfg: Dict,
    *,
    env_seed: int = 0,
    log_dir: str = "logs",
    model_dir: str = "models/pg",
) -> Dict:
    """Train one policy-gradient run. Returns a metrics dict."""
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
    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    seed = int(cfg.get("seed", env_seed))
    epochs = int(cfg.get("epochs", 4))          # PPO multiple epochs
    clip_epsilon = float(cfg.get("clip_epsilon", 0.2))
    lmbda = float(cfg.get("lmbda", 0.9))        # GAE lambda
    critic_coef = float(cfg.get("critic_coef", 1.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(grid_size=grid_size, max_steps=max_steps, seed=seed)
    actor, critic = build_actor_critic(env, device=device)
    actor.to(device)
    critic.to(device)

    loss_module = _make_loss(algo, actor, critic, gamma, cfg)
    loss_module.to(device)

    # Shared parameters for the optimizer.
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = Adam(params, lr=lr)

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    run_tag = f"{algo}"
    csv_path = os.path.join(log_dir, f"{run_tag}_seed{seed}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "episode_reward", "episode_length", "loss"])

    frames = 0
    episode_rewards = []
    episode_return = 0.0
    episode_length = 0
    last_loss = 0.0

    for batch in collector:
        batch = batch.to(device)

        # Optimize (PPO: multiple epochs over the same on-policy batch).
        for _ in range(epochs):
            loss_out = loss_module(batch)
            # TorchRL 0.6 losses return component losses rather than a single
            # scalar (e.g. loss_objective/loss_critic/loss_entropy or
            # loss_actor/loss_value). Combine every "loss_*" entry.
            comps = [
                val for key, val in loss_out.items()
                if str(key).startswith("loss_") and val is not None
            ]
            if not comps:
                val = loss_out.get("loss", None)
                if val is None:
                    raise RuntimeError("loss module returned no usable loss key")
                comps = [val]
            loss_val = sum(comps)
            optimizer.zero_grad()
            loss_val.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            last_loss = float(loss_val.detach().item())

        frames += batch.numel()

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
                        last_loss,
                    ]
                )
                episode_return = 0.0
                episode_length = 0

    collector.shutdown()
    env.close()
    csv_file.close()

    eval_reward = evaluate_greedy(
        actor, grid_size=grid_size, max_steps=max_steps, seed=seed + 1000, episodes=10,
    )

    model_path = os.path.join(model_dir, f"{algo}_seed{seed}.pt")
    torch.save(
        {
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "cfg": cfg,
            "algo": algo,
            "env_seed": env_seed,
        },
        model_path,
    )

    recent = episode_rewards[-25:] if episode_rewards else [0.0]
    metrics = {
        "algo": algo,
        "tag": run_tag,
        "seed": seed,
        "frames": frames,
        "mean_reward_last25": float(np.mean(recent)),
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "eval_greedy_reward": float(eval_reward),
        "model_path": model_path,
        "csv_path": csv_path,
        **cfg,
    }
    return metrics
