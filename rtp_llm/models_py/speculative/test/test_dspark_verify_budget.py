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

    def test_adaptive_uses_full_width_and_keeps_budget(self):
        config = self.config(budget=3)
        config.sp_dspark_adaptive_verify = True
        self.assertEqual(config.verifySteps(), 7)
        self.assertEqual(config.verifyBudgetPerRequest(), 3)
        self.assertIn("sp_dspark_adaptive_verify: 1", config.to_string())

    def test_explicit_modes_and_conflicts(self):
        for budget in (0, 7):
            config = self.config(budget=budget)
            config.sp_dspark_verify_mode = "static"
            self.assertFalse(config.isAdaptiveVerify())
            self.assertEqual(config.verifySteps(), 7)
            self.assertEqual(config.verifyBudgetPerRequest(), 7)
        for budget in (0, 1, 3, 7):
            config = self.config(budget=budget)
            config.sp_dspark_verify_mode = "adaptive"
            self.assertTrue(config.isAdaptiveVerify())
            self.assertEqual(config.verifySteps(), 7)
            self.assertEqual(config.verifyBudgetPerRequest(), budget or 7)
        for mode, budget, legacy in (
            ("static", 3, False),
            ("static", 0, True),
            ("bad", 0, False),
        ):
            config = self.config(budget=budget)
            config.sp_dspark_verify_mode = mode
            config.sp_dspark_adaptive_verify = legacy
            for validate in (
                config.isAdaptiveVerify,
                config.verifySteps,
                config.verifyBudgetPerRequest,
            ):
                with self.assertRaises(ValueError):
                    validate()
        for kind in ("none", "vanilla", "mtp", "eagle", "eagle3", "deterministic"):
            for mode in ("static", "adaptive"):
                config = self.config(kind)
                config.sp_dspark_verify_mode = mode
                with self.assertRaises(ValueError):
                    config.verifySteps()

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

    def test_adaptive_verify_batch_capacity(self):
        for gamma, limit in ((7, 146), (4, 256)):
            config = self.config(gamma=gamma)
            config.sp_dspark_verify_mode = "adaptive"
            for batch in (0, 1, limit):
                config.validateVerifyBatchSize(batch)
            for batch in (-1, limit + 1, 2**63 - 1):
                with self.assertRaises(ValueError):
                    config.validateVerifyBatchSize(batch)
            config.sp_dspark_verify_mode = "static"
            config.validateVerifyBatchSize(2**63 - 1)
        for kind in ("none", "vanilla", "mtp", "eagle", "eagle3", "deterministic"):
            self.config(kind).validateVerifyBatchSize(2**63 - 1)

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
        self.assertEqual(len(state), 17)
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
        for length in (15, 16):
            restored = self.restore(state[:length])
            self.assertEqual(restored.sp_dspark_verify_mode, "")
            self.assertEqual(restored.verifySteps(), 3)
        for mode in ("static", "adaptive"):
            config = self.config()
            config.sp_dspark_verify_mode = mode
            restored = pickle.loads(pickle.dumps(config))
            self.assertEqual(restored.sp_dspark_verify_mode, mode)
            self.assertEqual(restored.isAdaptiveVerify(), mode == "adaptive")
            self.assertEqual(restored.verifySteps(), 7)
        config = self.config(budget=3)
        config.sp_dspark_adaptive_verify = True
        restored = self.restore(config.__getstate__()[:16])
        self.assertEqual(restored.sp_dspark_verify_mode, "")
        self.assertTrue(restored.isAdaptiveVerify())
        self.assertEqual(restored.verifySteps(), 7)
        self.assertEqual(restored.verifyBudgetPerRequest(), 3)

    def test_invalid_pickle(self):
        state = self.config().__getstate__()
        for invalid in (
            state[:9],
            state + (0,),
            state[:14] + (-1,) + state[15:],
            state[:14] + (8,) + state[15:],
        ):
            with self.assertRaises(RuntimeError):
                self.restore(invalid)
        state = self.config("mtp").__getstate__()
        with self.assertRaises(RuntimeError):
            self.restore(state[:14] + (1,) + state[15:])
        with self.assertRaises(RuntimeError):
            self.restore(state[:15] + (True,) + state[16:])
        state = self.config().__getstate__()
        for invalid in (
            state[:-1] + ("bad",),
            state[:14] + (3, False, "static"),
            state[:15] + (True, "static"),
        ):
            with self.assertRaises(RuntimeError):
                self.restore(invalid)


if __name__ == "__main__":
    unittest.main()
