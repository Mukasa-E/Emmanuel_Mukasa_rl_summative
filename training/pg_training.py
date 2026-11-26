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
from typing import Dict

import torch
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
    ProbabilisticTensorDictSequential,
    ProbabilisticTensorDictModule,
)
from torch.distributions import Categorical
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter
from torchrl.modules import ValueOperator, MLP
from torchrl.objectives import ReinforceLoss, ClipPPOLoss, A2CLoss, ValueEstimators
from torchrl.collectors import SyncDataCollector
from torch.optim import Adam

from environment.action_masks import legal_action_mask
from environment.custom_env import AgriFieldEnv
from training.networks import build_backbone


def make_env(
    grid_size: int = 8,
    max_steps: int = 200,
    seed: int = 0,
    shaping_coef: float = 0.0,
):
    base = GymWrapper(
        AgriFieldEnv(
            grid_size=grid_size,
            max_steps=max_steps,
            seed=seed,
            shaping_coef=shaping_coef,
        ),
        categorical_action_encoding=True,
    )
    env = TransformedEnv(
        base,
        Compose(StepCounter(), DoubleToFloat()),
    )
    env.set_seed(seed)
    return env


def build_actor_critic(env, device: str = "cpu", arch: str = "mlp"):
    """Build a Categorical actor (ProbabilisticTensorDictSequential) + value critic."""
    obs_dim = int(env.observation_spec["observation"].shape[-1])
    act_spec = env.action_spec
    num_actions = int(act_spec.n)

    # Policy network: observation -> action logits.
    actor_net = build_backbone(arch, obs_dim, num_actions, device=device)
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
    actor = ProbabilisticTensorDictSequential(
        [actor_module, _make_masked_logits_module(), prob_module]
    )

    # Value critic: observation -> state value.
    critic_net = build_backbone(arch, obs_dim, 1, device=device)
    critic = ValueOperator(
        module=critic_net, in_keys=["observation"], out_keys=["state_value"]
    )

    return actor, critic


class _MaskPolicyLogits(torch.nn.Module):
    def forward(
        self, logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)


def _make_masked_logits_module():
    return TensorDictModule(
        _MaskPolicyLogits(),
        in_keys=["logits", "action_mask"],
        out_keys=["logits"],
    )


def _make_action_mask_module(grid_size: int):
    class _ActionMask(torch.nn.Module):
        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            return legal_action_mask(observation, grid_size)

    return TensorDictModule(
        _ActionMask(),
        in_keys=["observation"],
        out_keys=["action_mask"],
    )


def _policy_logits(actor, tensordict: TensorDict):
    logits_tensordict = actor[0](tensordict.clone())
    logits_tensordict = actor[1](logits_tensordict)
    logits = logits_tensordict.get("logits")
    if logits is None:
        raise RuntimeError("Policy actor output is missing action logits")
    return logits


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
    """Run the highest-probability actor action on each state."""
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
                td.set("action_mask", legal_action_mask(obs_t, grid_size))
                logits = _policy_logits(actor, td)
                action = logits.argmax(dim=-1).detach().cpu()
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
    critic_coef = float(cfg.get("critic_coef", 1.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
    arch = str(cfg.get("arch", "mlp"))
    minibatch_size = int(cfg.get("minibatch_size", 0))   # PPO only; 0 = full batch

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(
        grid_size=grid_size,
        max_steps=max_steps,
        seed=seed,
        shaping_coef=float(cfg.get("shaping_coef", 0.0)),
    )
    actor, critic = build_actor_critic(env, device=device, arch=arch)
    actor.to(device)
    critic.to(device)

    loss_module = _make_loss(algo, actor, critic, gamma, cfg)
    loss_module.to(device)

    # Shared parameters for the optimizer.
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = Adam(params, lr=lr)

    gae = None
    if algo == "ppo" and minibatch_size > 0:
        from torchrl.objectives.value import GAE

        gae = GAE(
            gamma=gamma,
            lmbda=float(cfg.get("lmbda", 0.9)),
            value_network=critic,
            average_gae=True,
        )
    policy = TensorDictSequential(_make_action_mask_module(grid_size), actor)
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    run_tag = f"{algo}"
    csv_path = os.path.join(log_dir, f"{run_tag}_seed{seed}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["episode", "frame", "episode_reward", "episode_length", "loss", "entropy"]
    )

    frames = 0
    episode_rewards = []
    episode_entropies = []
    episode_return = 0.0
    episode_length = 0
    last_loss = 0.0
    last_entropy = 0.0

    for batch in collector:
        batch = batch.to(device)

        # Optimize (PPO: multiple epochs over the same on-policy batch).
        use_minibatches = algo == "ppo" and minibatch_size > 0
        if use_minibatches:
            # Advantages are computed once per rollout, in time order, so the
            # shuffled minibatches below do not break the GAE recursion.
            with torch.no_grad():
                gae(batch)

        for _ in range(epochs):
            if use_minibatches:
                order = torch.randperm(batch.batch_size[0], device=device)
                chunks = [
                    order[i : i + minibatch_size]
                    for i in range(0, len(order), minibatch_size)
                ]
            else:
                chunks = [None]
            for chunk in chunks:
                sub = batch if chunk is None else batch[chunk]
                loss_out = loss_module(sub)
                # TorchRL 0.6 losses return component losses rather than a
                # single scalar (e.g. loss_objective/loss_critic/loss_entropy
                # or loss_actor/loss_value). Combine every "loss_*" entry.
                comps = [
                    val for key, val in loss_out.items()
                    if str(key).startswith("loss_") and val is not None
                ]
                if not comps:
                    val = loss_out.get("loss", None)
                    if val is None:
                        raise RuntimeError("loss module returned no usable loss key")
                    comps = [val]
                if algo == "reinforce":
                    if "loss_actor" not in loss_out or "loss_value" not in loss_out:
                        raise RuntimeError(
                            "REINFORCE loss is missing actor or value loss components"
                        )
                    logits = _policy_logits(actor, sub)
                    entropy = Categorical(logits=logits).entropy().mean()
                    loss_val = (
                        loss_out["loss_actor"]
                        + critic_coef * loss_out["loss_value"]
                        - entropy_coef * entropy
                    )
                else:
                    loss_val = sum(comps[1:], comps[0])
                if not isinstance(loss_val, torch.Tensor):
                    raise RuntimeError("Policy-gradient loss must be a torch tensor")
                optimizer.zero_grad()
                loss_val.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                last_loss = float(loss_val.detach().item())

        with torch.no_grad():
            logits = _policy_logits(actor, batch)
            last_entropy = float(
                Categorical(logits=logits).entropy().mean().item()
            )

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
                episode_entropies.append(last_entropy)
                writer.writerow(
                    [
                        len(episode_rewards),
                        frames - batch.batch_size[0] + t_idx + 1,
                        episode_return,
                        episode_length,
                        last_loss,
                        last_entropy,
                    ]
                )
                episode_return = 0.0
                episode_length = 0

    collector.shutdown()
    env.close()
    csv_file.close()

    eval_seed = int(cfg.get("eval_seed", seed + 1000))
    eval_reward = evaluate_greedy(
        actor,
        grid_size=grid_size,
        max_steps=max_steps,
        seed=eval_seed,
        episodes=10,
    )
    from baseline import evaluate_policy as evaluate_task_policy

    def greedy_action(observation):
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        td = TensorDict({"observation": observation_tensor}, batch_size=[])
        td.set("action_mask", legal_action_mask(observation_tensor, grid_size))
        with torch.no_grad():
            return int(_policy_logits(actor, td).argmax(dim=-1).item())

    task_eval = evaluate_task_policy(
        greedy_action,
        policy_name=algo,
        grid_size=grid_size,
        max_steps=max_steps,
        seeds=tuple(
            range(
                int(cfg.get("task_eval_seed", eval_seed)),
                int(cfg.get("task_eval_seed", eval_seed))
                + int(cfg.get("task_eval_episodes", 20)),
            )
        ),
    )

    model_path = os.path.join(model_dir, f"{algo}_seed{seed}.pt")
    torch.save(
        {
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "cfg": {**cfg, "arch": arch},
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
        "mean_entropy_last25": float(np.mean(episode_entropies[-25:]))
        if episode_entropies else 0.0,
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "eval_greedy_reward": float(eval_reward),
        "eval_mission_completion_rate": task_eval["mission_completion_rate"],
        "eval_initial_task_resolution_rate": task_eval[
            "mean_initial_task_resolution_rate"
        ],
        "eval_successful_interventions": task_eval[
            "mean_successful_interventions"
        ],
        "model_path": model_path,
        "csv_path": csv_path,
        **cfg,
    }
    return metrics
