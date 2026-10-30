"""Pickle regressions for the GLM base schema (no main-only config fields)."""

import pickle
import unittest

from rtp_llm.ops import HiddenStateCaptureDtype, MoeConfig


def _new_moe_config():
    return MoeConfig.__new__(MoeConfig)


class _LegacyMoeConfig:
    def __reduce__(self):
        # GLM base has no use_mori_ep: the old state contains eleven fields.
        state = (False, True, False, True, True, False, 7, 8192, True, 128, "legacy")
        return _new_moe_config, (), state


class MoeConfigPickleTest(unittest.TestCase):
    def test_fp4_round_trip(self):
        for value in ("auto", "cutlass", "trtllm"):
            with self.subTest(value=value):
                config = MoeConfig()
                config.fp4_moe_op = value
                config.deep_ep_num_sm = 7
                restored = pickle.loads(pickle.dumps(config))
                self.assertEqual(restored.fp4_moe_op, value)
                self.assertEqual(restored.deep_ep_num_sm, 7)
                self.assertEqual(len(config.__getstate__()), 12)

    def test_legacy_eleven_fields(self):
        restored = pickle.loads(pickle.dumps(_LegacyMoeConfig()))
        self.assertEqual(restored.moe_strategy, "legacy")
        self.assertEqual(restored.fp4_moe_op, "auto")
        self.assertTrue(restored.fake_balance_expert)
        self.assertFalse(restored.hack_moe_expert)
        self.assertEqual(restored.deep_ep_num_sm, 7)
        self.assertEqual(restored.masked_max_token_num, 8192)
        self.assertEqual(restored.ll_num_max_token, 128)

    def test_invalid_state_length(self):
        for size in (0, 10, 13):
            with self.subTest(size=size), self.assertRaisesRegex(
                RuntimeError, "Invalid state"
            ):
                config = _new_moe_config()
                config.__setstate__(tuple(range(size)))

    def test_invalid_fp4_type_has_context(self):
        state = list(MoeConfig().__getstate__())
        state[-1] = object()
        with self.assertRaisesRegex(RuntimeError, "MoeConfig unpickle error"):
            config = _new_moe_config()
            config.__setstate__(tuple(state))

    def test_capture_dtype_round_trip(self):
        for dtype in (HiddenStateCaptureDtype.BF16, HiddenStateCaptureDtype.FP8_E4M3):
            with self.subTest(dtype=dtype):
                self.assertEqual(pickle.loads(pickle.dumps(dtype)), dtype)


if __name__ == "__main__":
    unittest.main()
