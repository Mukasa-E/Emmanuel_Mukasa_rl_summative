import unittest

import numpy as np
import torch
from tensordict import TensorDict

from environment.custom_env import (
    ACTION_FERTILISE,
    ACTION_MOVE_EAST,
    ACTION_MOVE_NORTH,
    ACTION_HARVEST,
    ACTION_SCAN,
    ACTION_WATER,
    CROP_EMPTY,
    CROP_MATURE,
    CROP_NEEDS_WATER,
    AgriFieldEnv,
)
from training.dqn_training import (
    NStepTransitionAccumulator,
    build_dqn_model,
    legal_action_mask,
    make_env as make_dqn_env,
)
from training.pg_training import (
    build_actor_critic,
    make_env as make_pg_env,
)


class AgriFieldEnvTests(unittest.TestCase):
    def test_dqn_n_step_targets_stop_at_episode_boundary(self):
        done = torch.tensor([[False], [False], [True], [False]])
        batch = TensorDict(
            {
                "observation": torch.arange(4, dtype=torch.float32).unsqueeze(-1),
                "action": torch.zeros(4, 1, dtype=torch.int64),
                "next": TensorDict(
                    {
                        "observation": torch.arange(
                            1, 5, dtype=torch.float32
                        ).unsqueeze(-1),
                        "action_mask": torch.ones(4, 2, dtype=torch.bool),
                        "reward": torch.ones(4, 1),
                        "done": done,
                        "terminated": done.clone(),
                        "truncated": torch.zeros_like(done),
                    },
                    batch_size=[4],
                ),
            },
            batch_size=[4],
        )
        accumulator = NStepTransitionAccumulator(n_steps=3, gamma=0.9)

        targets = accumulator.push(batch)
        tail = accumulator.flush()

        self.assertIsNotNone(targets)
        self.assertIsNotNone(tail)
        assert targets is not None
        assert tail is not None
        torch.testing.assert_close(
            targets.get(("next", "reward")).squeeze(-1),
            torch.tensor([2.71, 1.9, 1.0]),
        )
        torch.testing.assert_close(
            targets.get("steps_to_next_obs"),
            torch.tensor([3, 2, 1]),
        )
        torch.testing.assert_close(
            targets.get(("next", "observation")).squeeze(-1),
            torch.tensor([3.0, 3.0, 3.0]),
        )
        self.assertTrue(targets.get(("next", "done")).all().item())
        self.assertEqual(tail.batch_size, torch.Size([1]))
        self.assertEqual(tail.get("steps_to_next_obs").item(), 1)

    def test_observation_shape_visibility_and_bounds(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        obs, _ = env.reset()
        cell_count = env.grid_size**2
        visibility = obs[4 + cell_count:4 + 2 * cell_count]

        self.assertEqual(obs.shape, (4 + 3 * cell_count,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all((0.0 <= obs) & (obs <= 1.0)))
        self.assertEqual(visibility.sum(), 1)

    def test_initial_mission_has_bounded_actionable_workload(self):
        env = AgriFieldEnv(grid_size=6, max_steps=80, seed=7)
        self.addCleanup(env.close)
        env.reset()

        actionable = np.isin(env.field, [2, 3, 4, 5])
        self.assertGreaterEqual(int(actionable.sum()), 6)
        self.assertLessEqual(int(actionable.sum()), 8)

    def test_torchrl_action_encoding_matches_discrete_policy_outputs(self):
        for make_env in (make_dqn_env, make_pg_env):
            with self.subTest(make_env=make_env.__module__):
                env = make_env(grid_size=6, max_steps=80, seed=3)
                self.addCleanup(env.close)
                tensordict = env.reset()

                self.assertEqual(tuple(env.action_spec.shape), ())
                tensordict.set("action", torch.tensor(9, dtype=torch.int64))
                next_tensordict = env.step(tensordict)["next"]
                self.assertAlmostEqual(
                    next_tensordict["reward"].item(), -0.5, places=5
                )

    def test_dqn_legal_action_mask_uses_only_observed_local_state(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        env.reset()
        env.field[0, 0] = CROP_NEEDS_WATER
        env.obstacles.fill(False)
        observation = torch.as_tensor(env._get_obs())

        mask = legal_action_mask(observation, grid_size=4)
        self.assertFalse(mask[ACTION_MOVE_NORTH].item())
        self.assertTrue(mask[ACTION_MOVE_EAST].item())
        self.assertTrue(mask[ACTION_SCAN].item())
        self.assertTrue(mask[ACTION_WATER].item())
        self.assertFalse(mask[ACTION_FERTILISE].item())
        self.assertFalse(mask[ACTION_HARVEST].item())
        self.assertTrue(mask[9].item())

        scanned_observation, *_ = env.step(ACTION_SCAN)
        scanned_mask = legal_action_mask(
            torch.as_tensor(scanned_observation), grid_size=4
        )
        self.assertFalse(scanned_mask[ACTION_SCAN].item())
        self.assertTrue(scanned_mask[ACTION_WATER].item())

    def test_dqn_legal_action_mask_supports_batched_observations(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        observation, _ = env.reset()
        batch = torch.as_tensor(np.stack([observation, observation]))

        mask = legal_action_mask(batch, grid_size=4)

        self.assertEqual(tuple(mask.shape), (2, 10))
        self.assertTrue(torch.all(mask.any(dim=-1)))

    def test_play_select_action_applies_the_dqn_legal_mask(self):
        from play import select_action

        env = make_dqn_env(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        model = build_dqn_model(env)
        raw_env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(raw_env.close)
        observation, _ = raw_env.reset()
        observation_tensor = torch.as_tensor(observation).unsqueeze(0)

        action = select_action(
            "dqn", model, observation_tensor, grid_size=4
        )
        mask = legal_action_mask(observation_tensor.squeeze(0), grid_size=4)

        self.assertTrue(mask[action].item())

    def test_policy_gradient_actor_samples_only_legal_actions(self):
        env = make_pg_env(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        actor, _ = build_actor_critic(env)
        raw_env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(raw_env.close)
        observation, _ = raw_env.reset()
        raw_env.field[0, 0] = CROP_NEEDS_WATER
        observation = raw_env._get_obs()
        observation_tensor = torch.as_tensor(observation)
        mask = legal_action_mask(observation_tensor, grid_size=4)
        tensordict = TensorDict(
            {
                "observation": observation_tensor,
                "action_mask": mask,
            },
            batch_size=[],
        )

        with torch.no_grad():
            action = int(actor(tensordict)["action"].item())

        self.assertTrue(mask[action].item())
        from play import select_action

        greedy_action = select_action(
            "pg", actor, observation_tensor.unsqueeze(0), grid_size=4
        )
        self.assertTrue(mask[greedy_action].item())

    def test_global_observation_retains_scanned_crop_memory(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        env.reset()
        env.field.fill(CROP_NEEDS_WATER)
        env.obstacles.fill(False)
        env.crop_known.fill(False)
        env._reveal_crop(0, 0)

        scanned_obs, _, _, _, _ = env.step(ACTION_SCAN)
        cell_count = env.grid_size**2
        visibility_slice = slice(4 + cell_count, 4 + 2 * cell_count)
        self.assertEqual(scanned_obs[visibility_slice].sum(), 4)

        moved_obs, _, _, _, _ = env.step(ACTION_MOVE_EAST)
        self.assertEqual(moved_obs[visibility_slice].sum(), 4)

    def test_scan_reveals_local_crops_once(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        env.reset()
        env.field.fill(CROP_NEEDS_WATER)
        env.crop_known.fill(False)
        env._reveal_crop(0, 0)

        obs, reward, _, _, info = env.step(ACTION_SCAN)
        cell_count = env.grid_size**2
        visibility = obs[4 + cell_count:4 + 2 * cell_count]

        self.assertAlmostEqual(reward, 0.65)
        self.assertEqual(info["newly_revealed"], 3)
        self.assertEqual(visibility.sum(), 4)

        obs, reward, _, _, info = env.step(ACTION_SCAN)

        self.assertAlmostEqual(reward, -0.1)
        self.assertEqual(info["newly_revealed"], 0)
        self.assertEqual(obs[4 + cell_count:4 + 2 * cell_count].sum(), 4)

    def test_moving_into_cell_reveals_that_crop(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        env.reset()
        env.field.fill(CROP_NEEDS_WATER)
        env.obstacles.fill(False)
        env.crop_known.fill(False)
        env._reveal_crop(0, 0)

        obs, reward, _, _, info = env.step(ACTION_MOVE_EAST)
        cell_count = env.grid_size**2

        self.assertEqual(info["agent_pos"], (0, 1))
        self.assertEqual(info["newly_revealed"], 1)
        self.assertAlmostEqual(reward, 0.15)
        self.assertEqual(obs[4 + cell_count:4 + 2 * cell_count].sum(), 2)

    def test_completion_ignores_empty_cells_and_counts_only_crops(self):
        env = AgriFieldEnv(grid_size=3, max_steps=10, seed=4)
        self.addCleanup(env.close)
        env.reset()
        env.field.fill(CROP_EMPTY)
        env.field[0, 0] = CROP_NEEDS_WATER
        env.field[0, 1] = CROP_MATURE
        env.obstacles.fill(False)
        env.agent_pos[:] = (0, 0)

        env.step(ACTION_WATER)
        env.step(ACTION_MOVE_EAST)
        _, reward, terminated, truncated, info = env.step(ACTION_HARVEST)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 60.0)
        self.assertEqual(info["crops_resolved"], 2)
        self.assertEqual(info["total_crops"], 2)

    def test_reset_without_seed_advances_rng_but_explicit_seed_replays(self):
        env = AgriFieldEnv(grid_size=5, max_steps=10, seed=8)
        self.addCleanup(env.close)
        first, _ = env.reset()
        first_field = env.field.copy()
        next_obs, _ = env.reset()
        next_field = env.field.copy()

        self.assertFalse(np.array_equal(first_field, next_field))
        self.assertEqual(first.shape, next_obs.shape)

        env.reset(seed=8)
        replay_field = env.field.copy()
        self.assertTrue(np.array_equal(first_field, replay_field))

    def test_battery_depletion_terminates_and_step_limit_truncates(self):
        battery_env = AgriFieldEnv(
            grid_size=3, max_steps=10, max_battery=1, seed=5
        )
        self.addCleanup(battery_env.close)
        battery_env.reset()
        _, _, terminated, truncated, _ = battery_env.step(ACTION_SCAN)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

        horizon_env = AgriFieldEnv(
            grid_size=3, max_steps=1, max_battery=10, seed=5
        )
        self.addCleanup(horizon_env.close)
        horizon_env.reset()
        _, _, terminated, truncated, _ = horizon_env.step(ACTION_SCAN)
        self.assertFalse(terminated)
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
