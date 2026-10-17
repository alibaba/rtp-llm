import unittest
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.tp_norm import tp_gather_hidden, tp_rms_norm


class _Norm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rsqrt = torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return (x.float() * rsqrt * self.weight.float()).to(x.dtype)


class TensorParallelRMSNormTest(unittest.TestCase):
    def test_hidden_gather_restores_last_axis_rank_order(self) -> None:
        rank1 = torch.tensor([[5, 6], [7, 8]])
        rank0_shard_major = torch.tensor([[1, 3], [2, 4]])

        def prepend_rank0(rank1_shard_major, _group):
            return torch.cat((rank0_shard_major, rank1_shard_major), dim=0)

        target = "rtp_llm.models_py.distributed.collective_torch.all_gather"
        with mock.patch(target, side_effect=prepend_rank0):
            actual = tp_gather_hidden(rank1, tp_size=2)

        torch.testing.assert_close(actual, torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]]))

    def test_hidden_shard_matches_global_rmsnorm(self) -> None:
        torch.manual_seed(20260823)
        x = torch.randn(5, 8, dtype=torch.bfloat16)
        weight = torch.randn(8, dtype=torch.bfloat16)
        rank, local_dim = 1, 4
        local = x[:, rank * local_dim : (rank + 1) * local_dim]
        rank0_square_sum = x[:, :local_dim].float().square().sum(-1, keepdim=True)

        def add_rank0(square_sum, _group):
            return square_sum + rank0_square_sum

        target = "rtp_llm.models_py.distributed.collective_torch.all_reduce"
        with mock.patch(target, side_effect=add_rank0):
            actual = tp_rms_norm(_Norm(weight), local, tp_size=2, tp_rank=rank)

        expected = _Norm(weight)(x)[:, local_dim:]
        torch.testing.assert_close(actual, expected)

    def test_replicated_hidden_skips_collective(self) -> None:
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        weight = torch.arange(1, 5, dtype=torch.bfloat16)
        target = "rtp_llm.models_py.distributed.collective_torch.all_reduce"
        with mock.patch(target) as all_reduce:
            actual = tp_rms_norm(_Norm(weight), x, tp_size=2, tp_rank=1)

        torch.testing.assert_close(actual, weight.expand_as(x), atol=2e-5, rtol=2e-5)
        all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
