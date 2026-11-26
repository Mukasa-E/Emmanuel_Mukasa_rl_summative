import unittest

from environment.custom_env import AgriFieldEnv


class ShapingTests(unittest.TestCase):
    def test_default_environment_is_unshaped(self):
        env = AgriFieldEnv(grid_size=6, max_steps=80, seed=3)
        env.reset(seed=3)
        _, reward, _, _, info = env.step(4)
        self.assertEqual(reward, info["raw_reward"])

    def test_shaping_is_potential_difference_and_leaves_raw_reward_intact(self):
        plain = AgriFieldEnv(grid_size=6, max_steps=80)
        shaped = AgriFieldEnv(grid_size=6, max_steps=80, shaping_coef=0.5, shaping_gamma=1.0)
        plain.reset(seed=7)
        shaped.reset(seed=7)
        phi_start = shaped._frontier_potential()
        total_shift = 0.0
        for action in (1, 1, 2, 4, 2, 2, 4):
            _, r_plain, term, _, _ = plain.step(action)
            _, r_shaped, term_s, _, info = shaped.step(action)
            self.assertAlmostEqual(r_plain, info["raw_reward"], places=6)
            total_shift += r_shaped - r_plain
            if term or term_s:
                break
        phi_end = shaped._frontier_potential()
        # With gamma=1 the shaping terms telescope to coef * (Phi_end - Phi_start).
        self.assertAlmostEqual(total_shift, 0.5 * (phi_end - phi_start), places=5)


if __name__ == "__main__":
    unittest.main()
