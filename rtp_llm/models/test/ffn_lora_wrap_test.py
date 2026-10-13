import unittest

import torch

from rtp_llm.model_loader.ffn_weight import (
    w13_lora_a_func_wrap,
    w13_lora_b_func_wrap,
)
from rtp_llm.utils.model_weight import transpose


class _FakeOrigin:
    def __init__(self, n_weights: int):
        self.weights = [None] * n_weights
        self.lora_a_process_func = staticmethod(transpose)
        self.lora_b_process_func = staticmethod(transpose)


class W13PeftLoraWrapTest(unittest.TestCase):
    def test_peft_list_split_matches_per_projection_reference(self):
        # gate rank 4, up rank 3: the wrap must keep each projection's
        # tensors with its own origin (A: rank-concat; B: block-diagonal).
        w1, w3 = _FakeOrigin(1), _FakeOrigin(1)
        h, inter = 8, 12
        gA = torch.randn(4, h)
        uA = torch.randn(3, h)
        gB = torch.randn(inter, 4)
        uB = torch.randn(inter, 3)

        a = w13_lora_a_func_wrap([gA, uA], origin_w1=w1, origin_w3=w3)
        # per-projection: transpose then rank-concat on the column axis
        expect_a = torch.cat([gA.t(), uA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)

        b = w13_lora_b_func_wrap([gB, uB], origin_w1=w1, origin_w3=w3)
        # block-diagonal keeps each rank block on its own projection columns
        expect_b = torch.block_diag(gB.t(), uB.t())
        torch.testing.assert_close(b, expect_b)

        # the merged delta equals the two independent per-projection deltas
        scale = 2.0
        merged = a @ b * scale
        expect_delta = torch.cat(
            [(gB @ gA * scale).t(), (uB @ uA * scale).t()], dim=1
        )
        torch.testing.assert_close(merged, expect_delta)

    def test_peft_list_length_mismatch_raises(self):
        w1, w3 = _FakeOrigin(1), _FakeOrigin(1)
        with self.assertRaises(AssertionError):
            w13_lora_a_func_wrap([torch.randn(2, 4)], origin_w1=w1, origin_w3=w3)


if __name__ == "__main__":
    unittest.main()
