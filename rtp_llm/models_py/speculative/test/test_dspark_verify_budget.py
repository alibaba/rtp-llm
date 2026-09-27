"""Native config contract tests: require the rebuilt extension, no model/GPU calls."""

import pickle
import unittest

from rtp_llm.ops import SpeculativeExecutionConfig


class DSparkVerifyBudgetTest(unittest.TestCase):
    def config(self, kind="dspark", gamma=7, budget=0):
        config = SpeculativeExecutionConfig()
        config.type = kind
        config.gen_num_per_cycle = gamma
        config.sp_dspark_verify_tokens = budget
        return config

    def test_default_and_explicit_budget(self):
        for budget, expected in (
            (0, 7),
            (1, 1),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
        ):
            config = self.config(budget=budget)
            self.assertEqual(config.verifySteps(), expected)
            self.assertEqual(config.gen_num_per_cycle, 7)
            self.assertIn(f"sp_dspark_verify_tokens: {budget}", config.to_string())

    def test_invalid_budget_and_legacy_types(self):
        for gamma, budget in ((7, -1), (7, 8), (0, 0), (0, 1), (-1, 0)):
            with self.assertRaises(ValueError):
                self.config(gamma=gamma, budget=budget).verifySteps()
        for kind in ("none", "vanilla", "mtp", "eagle", "eagle3", "deterministic"):
            for gamma in (0, 1, 7):
                self.assertEqual(self.config(kind, gamma).verifySteps(), gamma)
            for budget in (-1, 1, 7):
                with self.assertRaises(ValueError):
                    self.config(kind, budget=budget).verifySteps()

    def restore(self, state):
        config = SpeculativeExecutionConfig.__new__(SpeculativeExecutionConfig)
        config.__setstate__(state)
        return config

    def test_pickle_roundtrip_new_and_old_lengths(self):
        config = self.config(budget=3)
        config.sp_dspark_sample_from_anchor = False
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.verifySteps(), 3)
        self.assertEqual(restored.gen_num_per_cycle, 7)
        self.assertFalse(restored.sp_dspark_sample_from_anchor)
        state = config.__getstate__()
        self.assertEqual(len(state), 15)
        for length in (11, 12, 13, 14):
            restored = self.restore(state[:length])
            self.assertEqual(restored.sp_dspark_verify_tokens, 0)
            self.assertEqual(restored.verifySteps(), 7)
            self.assertEqual(restored.sp_dspark_sample_from_anchor, length < 14)
        # Old10 has no fp8_kv_cache field at index8.
        restored = self.restore(state[:8] + state[9:11])
        self.assertEqual(restored.sp_dspark_verify_tokens, 0)
        self.assertEqual(restored.verifySteps(), 7)
        self.assertEqual(restored.fp8_kv_cache, -1)

    def test_invalid_pickle(self):
        state = self.config().__getstate__()
        for invalid in (state[:9], state + (0,), state[:-1] + (-1,), state[:-1] + (8,)):
            with self.assertRaises(RuntimeError):
                self.restore(invalid)
        state = self.config("mtp").__getstate__()
        with self.assertRaises(RuntimeError):
            self.restore(state[:-1] + (1,))


if __name__ == "__main__":
    unittest.main()
