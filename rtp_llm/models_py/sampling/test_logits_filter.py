"""Independent threshold reference for RTP logits filtering."""

import math
import unittest
from collections import Counter

import torch

from rtp_llm.models_py.sampling.logits_filter import filter_logits


def reference_filter(logits: torch.Tensor, k: int, p: float) -> torch.Tensor:
    """Scalar FP64 reference: accumulate whole groups of equal logit values."""
    values = logits.tolist()
    ordered = sorted(values, reverse=True)
    k_cutoff = ordered[min(k, len(values)) - 1] if k else -math.inf
    candidates = [value for value in values if value >= k_cutoff]
    maximum = max(candidates)
    total = math.fsum(math.exp(value - maximum) for value in candidates)
    cutoff = k_cutoff
    if p < 1:
        mass = 0.0
        counts = Counter(candidates)
        for value in sorted(counts, reverse=True):
            mass += counts[value] * math.exp(value - maximum) / total
            cutoff = value
            if mass >= p:
                break
    return logits.masked_fill(logits < cutoff, -math.inf)


class LogitsFilterTest(unittest.TestCase):
    def check_case(self, values: torch.Tensor, k: list[int], p: list[float]) -> None:
        top_k = torch.tensor(k, dtype=torch.int32)
        top_p = torch.tensor(p, dtype=torch.float32)
        source = values.to(device="cuda", dtype=torch.float32)
        before = source.clone()
        actual = filter_logits(source, top_k, top_p).cpu()
        expected = torch.stack(
            [
                reference_filter(row, int(ki), float(pi))
                for row, ki, pi in zip(values, top_k, top_p)
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(source, before, rtol=0, atol=0)

    def test_mixed_rows_and_ties(self) -> None:
        row = torch.tensor([4.0, 3.0, 3.0, 2.0, 1.0, -math.inf])
        self.check_case(
            row.repeat(8, 1),
            [0, 2, 2, 0, 1, 99, 0, 3],
            [1.0, 1.0, 0.8, 0.8, 0.0, 0.8, 0.0, 0.5],
        )
        self.check_case(torch.zeros(4, 9), [0, 2, 9, 1], [0.5, 1.0, 0.0, 0.7])

    def test_topk_is_normalized_before_topp(self) -> None:
        values = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
        self.check_case(values, [2], [0.55])

    def test_extreme_logits_and_strided_input(self) -> None:
        values = torch.tensor([[10000.0, 9999.0, 9999.0, 9998.0, -10000.0, -math.inf]])
        self.check_case(values.repeat(4, 1), [0, 2, 0, 0], [0.8, 0.8, 0.0, 1.0])
        self.check_case(values.repeat(2, 2)[:, ::2], [2, 0], [0.8, 0.8])

    def test_random_and_large_topk(self) -> None:
        generator = torch.Generator().manual_seed(917)
        values = torch.randn(6, 8193, generator=generator)
        self.check_case(
            values, [0, 1, 40, 4097, 8193, 9000], [0.9, 0.9, 0.9, 0.8, 1.0, 0.95]
        )

    def test_empty_batch(self) -> None:
        values = torch.empty(0, 7, device="cuda", dtype=torch.float32)
        result = filter_logits(
            values, torch.empty(0, dtype=torch.int32), torch.empty(0)
        )
        self.assertIs(result, values)


if __name__ == "__main__":
    unittest.main()
