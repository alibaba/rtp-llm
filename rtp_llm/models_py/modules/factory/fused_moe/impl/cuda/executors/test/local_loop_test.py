import unittest

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
    _validate_topk_indices,
)


class _WeightedIdentity(nn.Module):
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return x * weights


class LocalLoopExecutorTest(unittest.TestCase):
    def test_eager_sums_duplicate_slots_for_the_same_expert(self):
        executor = LocalLoopExecutor.__new__(LocalLoopExecutor)
        nn.Module.__init__(executor)
        executor.experts = nn.ModuleList([_WeightedIdentity(), _WeightedIdentity()])
        x = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        weights = torch.tensor([[0.25, 0.75], [0.5, 0.2]])
        indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
        y = torch.zeros_like(x)

        result = executor._forward_eager(
            x,
            weights,
            indices,
            y,
            local_start=0,
            local_end=2,
        )

        expected = torch.tensor([[2.0, 3.0], [3.5, 4.9]])
        torch.testing.assert_close(result, expected)

    def test_topk_index_validation_rejects_negative_and_upper_bound(self):
        for invalid in (-1, 2):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "outside"):
                    _validate_topk_indices(
                        torch.tensor([[0, invalid]], dtype=torch.long),
                        num_experts=2,
                    )


if __name__ == "__main__":
    unittest.main()
