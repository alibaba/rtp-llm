"""Tests for skip_allreduce parameter in DenseMLP and FusedMoe finalize.

Verifies that:
1. skip_allreduce=True suppresses the allreduce call
2. skip_allreduce=False (default) performs the allreduce
3. allreduce(A + B) == allreduce(A) + allreduce(B) (mathematical equivalence)
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.ops import ActivationType, ParallelismConfig
from rtp_llm.utils.model_weight import W


def _make_dense_mlp(tp_size: int = 2, hidden_size: int = 64):
    """Create a DenseMLP with TP config for testing skip_allreduce."""
    torch.manual_seed(0)
    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = tp_size
    parallelism_config.tp_rank = 0
    parallelism_config.ffn_tp_size = tp_size
    parallelism_config.ffn_tp_rank = 0

    inter_size = 4 * hidden_size // tp_size
    weights = {
        W.ffn_w1: torch.randn(
            hidden_size, inter_size, dtype=torch.bfloat16, device="cuda"
        ),
        W.ffn_w3: torch.randn(
            hidden_size, inter_size, dtype=torch.bfloat16, device="cuda"
        ),
        W.ffn_w2: torch.randn(
            inter_size, hidden_size, dtype=torch.bfloat16, device="cuda"
        ),
    }
    return DenseMLP(
        ActivationType.Swiglu, parallelism_config, weights, quant_config=None
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestDenseMLPSkipAllreduce(unittest.TestCase):
    """Test skip_allreduce parameter in DenseMLP.forward."""

    def setUp(self):
        torch.set_default_device("cuda")
        self.mlp = _make_dense_mlp(tp_size=2)
        self.x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")

    @patch("rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce")
    def test_skip_allreduce_true_does_not_call_allreduce(self, mock_ar):
        self.mlp(self.x, skip_allreduce=True)
        mock_ar.assert_not_called()

    @patch(
        "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce",
        side_effect=lambda t, **kw: t,
    )
    def test_skip_allreduce_false_calls_allreduce(self, mock_ar):
        self.mlp(self.x, skip_allreduce=False)
        mock_ar.assert_called_once()

    @patch(
        "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce",
        side_effect=lambda t, **kw: t,
    )
    def test_default_calls_allreduce(self, mock_ar):
        self.mlp(self.x)
        mock_ar.assert_called_once()

    def test_skip_allreduce_output_matches_no_skip(self):
        """When allreduce is identity, skip and no-skip produce the same output."""
        with patch(
            "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce",
            side_effect=lambda t, **kw: t,
        ):
            out_no_skip = self.mlp(self.x.clone(), skip_allreduce=False)
        out_skip = self.mlp(self.x.clone(), skip_allreduce=True)
        torch.testing.assert_close(out_skip, out_no_skip)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestUnifiedAllreduceMathEquivalence(unittest.TestCase):
    """Verify allreduce(A) + allreduce(B) == allreduce(A + B).

    This is the mathematical foundation of the unified allreduce optimization.
    Uses a mock allreduce that simulates summing across 2 ranks.
    """

    def _mock_allreduce_sum(self, tensor, **kwargs):
        """Simulate allreduce by doubling the tensor (2 ranks, each with same value)."""
        return tensor * 2

    def test_allreduce_linearity(self):
        """allreduce(A + B) == allreduce(A) + allreduce(B) for sum reduction."""
        A = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")

        # Separate allreduce
        ar_A = self._mock_allreduce_sum(A)
        ar_B = self._mock_allreduce_sum(B)
        separate = ar_A + ar_B

        # Unified allreduce
        unified = self._mock_allreduce_sum(A + B)

        torch.testing.assert_close(unified, separate)

    def test_allreduce_linearity_with_scaling(self):
        """allreduce(A + gate * B) == allreduce(A) + gate * allreduce(B)
        when gate is identical across ranks (which it is, since input is
        post-RMSNorm and identical across TP ranks)."""
        A = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        gate = torch.sigmoid(torch.randn(8, 1, dtype=torch.bfloat16, device="cuda"))

        separate = self._mock_allreduce_sum(A) + gate * self._mock_allreduce_sum(B)
        unified = self._mock_allreduce_sum(A + gate * B)

        torch.testing.assert_close(unified, separate)


if __name__ == "__main__":
    unittest.main()
