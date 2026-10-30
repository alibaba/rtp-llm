"""CPU contract checks for GC's routed weight ordering and graph shapes."""

import unittest

import torch

from .gc_moe import gc_warmup_token_counts, transform_gc_routed_fp4


class GCMoeContractTest(unittest.TestCase):
    def test_up_gate_checkpoint_uses_gc_transform(self):
        import gc_moe

        w1 = (
            torch.arange(2 * 256 * 128, dtype=torch.int32)
            .to(torch.int8)
            .reshape(2, 256, 128)
        )
        s1 = torch.empty_strided((2, 256, 2), (512, 1, 256), dtype=torch.int32)
        s1.copy_(torch.arange(2 * 256 * 2, dtype=torch.int32).reshape(2, 256, 2))
        w2 = (
            torch.arange(2 * 256 * 64, dtype=torch.int32)
            .to(torch.int8)
            .reshape(2, 256, 64)
        )
        s2 = torch.empty_strided((2, 256, 1), (256, 1, 256), dtype=torch.int32)
        s2.copy_(torch.arange(2 * 256, dtype=torch.int32).reshape(2, 256, 1))
        actual = transform_gc_routed_fp4((w1, s1), (w2, s2), "up_gate")
        expected_s1 = torch.empty_strided(s1.shape, s1.stride(), dtype=s1.dtype)
        expected_s1[:, :128].copy_(s1[:, 128:])
        expected_s1[:, 128:].copy_(s1[:, :128])
        expected = gc_moe.transform_weights_for_gc_moe(
            (
                torch.cat((w1[:, 128:], w1[:, :128]), dim=1),
                expected_s1,
            ),
            (w2, s2),
        )
        self.assertEqual(actual[0][1].stride(), s1.stride())
        self.assertEqual(actual[1][1].stride(), s2.stride())
        for pair_a, pair_b in zip(actual, expected):
            for a, b in zip(pair_a, pair_b):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_eager_warmup_covers_decode_and_eagle_shapes(self):
        self.assertEqual(
            gc_warmup_token_counts(64),
            [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 64],
        )
        self.assertEqual(gc_warmup_token_counts(5), [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
