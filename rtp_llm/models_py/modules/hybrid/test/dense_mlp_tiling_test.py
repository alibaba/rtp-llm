import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.modelopt_mixed import ModelOptMixedPrecisionConfig
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.ops import ActivationType
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DenseMLPTilingTest(unittest.TestCase):
    def test_long_mtp_ffn_keeps_one_collective_and_skip_contract(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 required")
        torch.manual_seed(417)
        weights = {
            W.ffn_w13: torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
            * 0.02,
            W.ffn_w2: torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.02,
        }
        model = DenseMLP(
            ActivationType.Swiglu,
            SimpleNamespace(get_ffn_tp_size=lambda: 2),
            weights,
            ModelOptMixedPrecisionConfig({}),
            max_prefill_tokens=65536,
        )
        for rows in (19, 65539):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
                expected = model._forward_block(x)
                with patch(
                    "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce",
                    side_effect=lambda value, group: value + 1,
                ) as collective:
                    actual = model(x)
                    collective.assert_called_once()
                    self.assertEqual(collective.call_args.args[0].shape, expected.shape)
                    torch.testing.assert_close(
                        actual, expected + 1, rtol=0.01, atol=0.02
                    )
                    collective.reset_mock()
                    actual = model(x, skip_allreduce=True)
                    collective.assert_not_called()
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)

    def test_tiled_prefill_tail_does_not_select_independent_w4a16_weights(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 required")
        torch.manual_seed(417)
        weights = {
            W.ffn_w13: torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
            * 0.02,
            W.ffn_w2: torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.02,
        }
        model = DenseMLP(
            ActivationType.Swiglu,
            SimpleNamespace(get_ffn_tp_size=lambda: 1),
            weights,
            ModelOptMixedPrecisionConfig({}),
            max_prefill_tokens=64,
        )
        with patch.object(
            model,
            "w4a16",
            side_effect=lambda x, activation: x.new_full((x.shape[0], 128), 7),
        ) as alternate_weights:
            for rows in (65, 67, 128):
                with self.subTest(rows=rows):
                    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
                    expected = model._forward_block(x)
                    actual = model(x)
                    alternate_weights.assert_not_called()
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
            x = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
            actual = model(x)
            alternate_weights.assert_called_once()
            torch.testing.assert_close(actual, x.new_full((19, 128), 7))


class GdnPrefillTilingPolicyTest(unittest.TestCase):
    def test_many_short_sequences_keep_batched_gdn(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetPrefill,
        )

        policy = SimpleNamespace(max_prefill_tokens=65536)
        for lengths, expected in [
            ([2048] * 71, False),
            ([65536, 1], False),
            ([65537, 1], True),
            ([0, 262128, 0], True),
            ([], False),
        ]:
            with self.subTest(lengths=lengths):
                metadata = SimpleNamespace(
                    input_lengths=torch.tensor(lengths, dtype=torch.int32)
                )
                actual = Qwen3NextGatedDeltaNetPrefill._should_tile_prefill(
                    policy, metadata, sum(lengths)
                )
                self.assertEqual(actual, expected)
        policy.max_prefill_tokens = 0
        self.assertFalse(
            Qwen3NextGatedDeltaNetPrefill._should_tile_prefill(
                policy, SimpleNamespace(input_lengths=torch.tensor([262128])), 262128
            )
        )


if __name__ == "__main__":
    unittest.main()
