import unittest

import numpy as np

from baseline import evaluate_policy, task_aware_action
from environment.custom_env import (
    ACTION_FERTILISE,
    ACTION_SCAN,
    ACTION_WATER,
    AgriFieldEnv,
    CROP_NEEDS_FERT,
    CROP_NEEDS_WATER,
)


class BaselineTests(unittest.TestCase):
    def test_task_aware_policy_uses_visible_crop_state_for_treatment(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        observation, _ = env.reset()
        env.field[0, 0] = CROP_NEEDS_WATER
        observation = env._get_obs()
        self.assertEqual(task_aware_action(observation, 4), ACTION_WATER)

        env.field[0, 0] = CROP_NEEDS_FERT
        observation = env._get_obs()
        self.assertEqual(task_aware_action(observation, 4), ACTION_FERTILISE)

    def test_task_aware_policy_scans_when_local_information_is_available(self):
        env = AgriFieldEnv(grid_size=4, max_steps=10, seed=3)
        self.addCleanup(env.close)
        observation, _ = env.reset()
        env.field.fill(1)
        env.crop_known.fill(False)
        env._reveal_crop(0, 0)
        env.obstacles.fill(False)
        observation = env._get_obs()
        self.assertEqual(task_aware_action(observation, 4), ACTION_SCAN)

    def test_evaluator_reports_task_metrics_deterministically(self):
        seeds = (100, 101)
        first = evaluate_policy(
            lambda _observation: 9,
            policy_name="wait",
            grid_size=4,
            max_steps=5,
            seeds=seeds,
        )
        second = evaluate_policy(
            lambda _observation: 9,
            policy_name="wait",
            grid_size=4,
            max_steps=5,
            seeds=seeds,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], 2)
        self.assertIn("mission_completion_rate", first)
        self.assertIn("mean_initial_task_resolution_rate", first)
        self.assertTrue(
            all(result["steps"] <= 5 for result in first["episode_results"])
        )

    def test_evaluator_rejects_empty_seed_list(self):
        with self.assertRaises(ValueError):
            evaluate_policy(
                lambda _observation: 9, policy_name="wait", seeds=()
            )

    def test_policy_rejects_observation_shape_mismatch(self):
        with self.assertRaises(ValueError):
            task_aware_action(np.zeros(10, dtype=np.float32), 4)


if __name__ == "__main__":
    unittest.main()
