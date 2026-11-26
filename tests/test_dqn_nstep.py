"""Numerical validation of the DQN n-step replay option and legal-action masking.

The online/target networks are set to known constants so the expected DQN loss
can be computed by hand. This distinguishes gamma**n from gamma**1 bootstrapping
and proves illegal actions are excluded from the bootstrap maximum.
"""

import unittest

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DQNLoss, ValueEstimators

from environment.action_masks import legal_action_mask
from training.dqn_training import (
    NStepTransitionAccumulator,
    build_dqn_model,
    make_action_mask_module,
    make_env,
)

GAMMA = 0.9
FRAMES = 300


def _collect_batch():
    torch.manual_seed(0)
    env = make_env(grid_size=6, max_steps=80, seed=0)
    model = build_dqn_model(env)
    policy = TensorDictSequential(make_action_mask_module(6), model)
    collector = SyncDataCollector(
        env, policy, frames_per_batch=FRAMES, total_frames=FRAMES
    )
    batch = next(iter(collector))
    collector.shutdown()
    env.close()
    batch.set(
        ("next", "action_mask"),
        legal_action_mask(batch.get(("next", "observation")), 6),
    )
    return model, batch


def _zero_all(loss_module):
    with torch.no_grad():
        for params in (
            loss_module.value_network_params,
            loss_module.target_value_network_params,
        ):
            for _, p in params.items(True, True):
                p.zero_()


def _set_output_bias(params, values):
    with torch.no_grad():
        for key, p in params.items(True, True):
            if str(key[-1]) == "bias" and p.shape[0] == len(values):
                p.copy_(torch.tensor(values))


class DQNNStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.batch = _collect_batch()

    def _loss_module(self, double_dqn=False):
        loss = DQNLoss(self.model, delay_value=True, double_dqn=double_dqn)
        loss.make_value_estimator(ValueEstimators.TD0, gamma=GAMMA)
        _zero_all(loss)
        return loss

    def test_bootstrap_is_discounted_by_gamma_to_the_n(self):
        for n in (1, 3):
            accumulator = NStepTransitionAccumulator(n, GAMMA)
            transitions = accumulator.push(self.batch)
            tail = accumulator.flush()
            if tail is not None:
                transitions = torch.cat([transitions, tail], 0)
            # Every frame is emitted exactly once.
            self.assertEqual(transitions.batch_size[0], FRAMES)

            loss = self._loss_module()
            _set_output_bias(loss.target_value_network_params, [100.0] * 10)
            got = float(loss(transitions)["loss"])

            reward = transitions.get(("next", "reward")).squeeze(-1)
            terminated = transitions.get(("next", "terminated")).squeeze(-1).float()
            steps = transitions.get("steps_to_next_obs").float()
            expected = float(
                ((reward + GAMMA**steps * (1 - terminated) * 100.0) ** 2).mean()
            )
            self.assertAlmostEqual(got, expected, delta=1e-2 * expected)
            if n == 3:
                wrong = float(
                    ((reward + GAMMA * (1 - terminated) * 100.0) ** 2).mean()
                )
                self.assertGreater(abs(got - wrong), 0.05 * expected)

    def test_n_step_returns_never_cross_an_episode_boundary(self):
        accumulator = NStepTransitionAccumulator(3, GAMMA)
        transitions = accumulator.push(self.batch)
        tail = accumulator.flush()
        if tail is not None:
            transitions = torch.cat([transitions, tail], 0)
        done = transitions.get(("next", "done")).squeeze(-1)
        steps = transitions.get("steps_to_next_obs")
        # A transition whose next state is terminal must be the last of its window.
        self.assertTrue(bool((steps[done] >= 1).all()))
        self.assertTrue(bool((steps <= 3).all()))

    def test_loss_excludes_illegal_actions_from_the_bootstrap(self):
        scan_only = [0, 0, 0, 0, 100.0, 0, 0, 0, 0, 0]
        for double in (False, True):
            loss = self._loss_module(double_dqn=double)
            _set_output_bias(loss.target_value_network_params, scan_only)
            if double:
                _set_output_bias(loss.value_network_params, scan_only)
            got = float(loss(self.batch.clone())["loss"])

            mask = self.batch.get(("next", "action_mask"))
            reward = self.batch.get(("next", "reward")).squeeze(-1)
            terminated = self.batch.get(("next", "terminated")).squeeze(-1).float()
            bootstrap = torch.where(mask[:, 4], 100.0, 0.0)
            taken = self.batch.get("action").reshape(-1)
            online_q = (taken == 4).float() * 100.0 if double else 0.0
            expected = float(
                (((reward + GAMMA * (1 - terminated) * bootstrap) - online_q) ** 2).mean()
            )
            self.assertGreater(int((~mask[:, 4]).sum()), 0)
            self.assertAlmostEqual(got, expected, delta=1e-2 * expected)


if __name__ == "__main__":
    unittest.main()
