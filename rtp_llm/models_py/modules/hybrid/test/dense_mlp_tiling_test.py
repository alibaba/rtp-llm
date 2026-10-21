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


if __name__ == "__main__":
    unittest.main()
