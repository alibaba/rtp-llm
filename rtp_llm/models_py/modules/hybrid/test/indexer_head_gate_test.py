import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
    CudaF16Linear,
)
from rtp_llm.models_py.modules.hybrid.indexer import Indexer


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class IndexerHeadGateTest(unittest.TestCase):
    def test_bf16_projection_returns_fp32_without_expanding_input(self):
        hidden_size, head_count, token_count = 32, 4, 3
        physical_weight = torch.randn(
            hidden_size,
            head_count,
            dtype=torch.bfloat16,
            device="cuda",
        )
        indexer = Indexer.__new__(Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.weights_proj = CudaF16Linear(physical_weight)
        indexer.softmax_scale = 0.125
        indexer.weights_scale = 0.5

        hidden_states = torch.randn(
            token_count,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_scale = torch.rand(
            token_count,
            head_count,
            1,
            dtype=torch.float32,
            device="cuda",
        )

        with mock.patch.object(torch, "mm", wraps=torch.mm) as mm:
            actual = indexer._get_logits_head_gate(hidden_states, q_scale)

        projected = torch.mm(hidden_states, physical_weight, out_dtype=torch.float32)
        expected = (
            projected.unsqueeze(-1)
            * q_scale
            * indexer.softmax_scale
            * indexer.weights_scale
        )
        mm.assert_called_once()
        self.assertIs(mm.call_args.args[0], hidden_states)
        self.assertEqual(mm.call_args.args[0].dtype, torch.bfloat16)
        self.assertEqual(mm.call_args.args[1].dtype, torch.bfloat16)
        self.assertEqual(mm.call_args.kwargs["out_dtype"], torch.float32)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
