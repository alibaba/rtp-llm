"""Unit tests for swizzle alignment judgement (can_swizzle_kn).

can_swizzle_kn is the single source of truth shared by the data side
(device_impl skips swizzle for unaligned BA) and the dispatch side
(qwen3_next in_proj_ba falls back to NoSwizzle). Its correctness is what
prevents the Qwen3.5 MI308X TP=4 crash (BA out-dim 24, 24 % 16 != 0) while
keeping the swizzle speedup on aligned shapes.

Pure logic + shape math — no CUDA/ROCm needed.
"""

from unittest import TestCase, main

import torch

from rtp_llm.utils.swizzle_utils import can_swizzle_kn, swizzle_tensor


class CanSwizzleKnTest(TestCase):
    def test_ba_alignment_table_bf16(self):
        # BA local weight is (hidden=5120, out=(b+a)=96/TP). Only the out-dim
        # (n) alignment changes across TP; hidden (k=5120) is always %32.
        #   TP=1 -> 96, TP=2 -> 48 : aligned  -> swizzle
        #   TP=4 -> 24, TP=8 -> 12 : unaligned -> fall back
        for tp, out in {1: 96, 2: 48, 4: 24, 8: 12}.items():
            w = torch.empty(5120, out, dtype=torch.bfloat16)
            expected = (out % 16 == 0)
            self.assertEqual(
                can_swizzle_kn(w), expected, f"TP={tp} out={out}"
            )

    def test_k_divisor_bf16_vs_fp8(self):
        # bf16 requires k % 32 == 0; fp8 requires the stricter k % 64 == 0.
        # k=96: passes bf16 (96%32==0) but not fp8 (96%64!=0).
        w_bf16 = torch.empty(96, 32, dtype=torch.bfloat16)
        w_fp8 = torch.empty(96, 32, dtype=torch.float8_e4m3fn)
        self.assertTrue(can_swizzle_kn(w_bf16))
        self.assertFalse(can_swizzle_kn(w_fp8))
        # k=128 passes both.
        self.assertTrue(can_swizzle_kn(torch.empty(128, 32, dtype=torch.bfloat16)))
        self.assertTrue(
            can_swizzle_kn(torch.empty(128, 32, dtype=torch.float8_e4m3fn))
        )

    def test_dtype_override(self):
        # dtype arg overrides the tensor's own dtype (used to reason about a
        # weight as if it were quantized to fp8).
        w = torch.empty(96, 32, dtype=torch.bfloat16)
        self.assertTrue(can_swizzle_kn(w))  # bf16: 96 % 32 == 0
        self.assertFalse(
            can_swizzle_kn(w, dtype=torch.float8_e4m3fn)  # fp8: 96 % 64 != 0
        )

    def test_non_2d_returns_false(self):
        self.assertFalse(can_swizzle_kn(torch.empty(24, dtype=torch.bfloat16)))
        self.assertFalse(
            can_swizzle_kn(torch.empty(2, 5120, 24, dtype=torch.bfloat16))
        )

    def test_judgement_matches_actual_swizzle_constraint(self):
        # The guard is only correct if can_swizzle_kn==False EXACTLY when
        # swizzle_tensor would assert. device_impl calls
        # swizzle_tensor(weight.t(), col_maj=False) on BA (hidden, out).
        # TP=4 BA (5120, 24): must be rejected AND must actually raise.
        ba_bad = torch.empty(5120, 24, dtype=torch.bfloat16)
        self.assertFalse(can_swizzle_kn(ba_bad))
        with self.assertRaises(AssertionError):
            swizzle_tensor(ba_bad.t(), False)

        # Aligned counterpart (pad-to-32): must be accepted AND must not raise.
        ba_ok = torch.empty(5120, 32, dtype=torch.bfloat16)
        self.assertTrue(can_swizzle_kn(ba_ok))
        swizzle_tensor(ba_ok.t(), False)  # no exception


if __name__ == "__main__":
    main()
