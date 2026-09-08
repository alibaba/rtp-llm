"""Unit tests for linear-weight swizzle policy.

The data side and Qwen3Next dispatch must agree on shape support. BA remains
BF16 when qkvz is quantized, so quantization must not independently disable its
swizzled layout.

Pure logic + shape math — no CUDA/ROCm needed.
"""

from unittest import TestCase, main

import torch

from rtp_llm.utils.swizzle_utils import (
    can_fuse_swizzled_kn,
    can_swizzle_kn,
    should_swizzle_linear_attn_ba,
    swizzle_tensor,
)


class CanSwizzleKnTest(TestCase):
    def test_linear_attention_ba_swizzle_depends_on_shape(self):
        aligned_ba = torch.empty(5120, 48, dtype=torch.bfloat16)
        unaligned_ba = torch.empty(5120, 24, dtype=torch.bfloat16)

        self.assertTrue(should_swizzle_linear_attn_ba(aligned_ba))
        self.assertFalse(should_swizzle_linear_attn_ba(unaligned_ba))

    def test_ba_alignment_table_bf16(self):
        # BA local weight is (hidden=5120, out=(b+a)=96/TP). Only the out-dim
        # (n) alignment changes across TP; hidden (k=5120) is always %32.
        #   TP=1 -> 96, TP=2 -> 48 : geometrically swizzle-compatible
        #   TP=4 -> 24, TP=8 -> 12 : unaligned -> fall back
        for tp, out in {1: 96, 2: 48, 4: 24, 8: 12}.items():
            w = torch.empty(5120, out, dtype=torch.bfloat16)
            expected = out % 16 == 0
            self.assertEqual(can_swizzle_kn(w), expected, f"TP={tp} out={out}")

    def test_qwen35_27b_fused_swizzle_alignment_by_tp(self):
        # Qwen3.5-27B local qkvz/BA output dimensions after TP sharding.
        # TP=1/2 keep both source boundaries 16-aligned, while TP=4/8 leave
        # BA raw and therefore cannot be concatenated with the swizzled qkvz.
        qkvz_out_by_tp = {1: 16384, 2: 8192, 4: 4096, 8: 2048}
        ba_out_by_tp = {1: 96, 2: 48, 4: 24, 8: 12}
        for tp in (1, 2, 4, 8):
            qkvz = torch.empty(
                5120, qkvz_out_by_tp[tp], dtype=torch.bfloat16, device="meta"
            )
            ba = torch.empty(
                5120, ba_out_by_tp[tp], dtype=torch.bfloat16, device="meta"
            )
            self.assertEqual(
                can_fuse_swizzled_kn(qkvz, ba),
                tp in (1, 2),
                f"TP={tp}",
            )

    def test_qwen36_35b_a3b_ba_alignment_by_tp(self):
        # Qwen3.6-35B-A3B BA global shape is (2048, 64). TP=4 therefore has
        # local N=16 and must retain swizzle; only TP=8 falls back.
        for tp, out in {1: 64, 2: 32, 4: 16, 8: 8}.items():
            ba = torch.empty(2048, out, dtype=torch.bfloat16, device="meta")
            self.assertEqual(
                should_swizzle_linear_attn_ba(ba),
                tp in (1, 2, 4),
                f"TP={tp} out={out}",
            )

    def test_fused_swizzle_requires_compatible_sources(self):
        aligned = torch.empty(128, 32, dtype=torch.bfloat16, device="meta")
        self.assertFalse(
            can_fuse_swizzled_kn(
                aligned,
                torch.empty(96, 16, dtype=torch.bfloat16, device="meta"),
            )
        )
        self.assertFalse(
            can_fuse_swizzled_kn(
                aligned,
                torch.empty(128, 16, dtype=torch.float16, device="meta"),
            )
        )
        self.assertFalse(
            can_fuse_swizzled_kn(
                aligned,
                torch.empty(128, 8, dtype=torch.bfloat16, device="meta"),
            )
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
        self.assertTrue(can_swizzle_kn(torch.empty(128, 32, dtype=torch.float8_e4m3fn)))

    def test_dtype_override(self):
        # dtype arg overrides the tensor's own dtype (used to reason about a
        # weight as if it were quantized to fp8).
        w = torch.empty(96, 32, dtype=torch.bfloat16)
        self.assertTrue(can_swizzle_kn(w))  # bf16: 96 % 32 == 0
        self.assertFalse(
            can_swizzle_kn(w, dtype=torch.float8_e4m3fn)  # fp8: 96 % 64 != 0
        )

    def test_float32_uses_its_actual_k_divisor(self):
        self.assertTrue(can_swizzle_kn(torch.empty(48, 32, dtype=torch.float32)))

    def test_unsupported_dtype_returns_false(self):
        self.assertFalse(can_swizzle_kn(torch.empty(128, 32, dtype=torch.int8)))

    def test_non_2d_returns_false(self):
        self.assertFalse(can_swizzle_kn(torch.empty(24, dtype=torch.bfloat16)))
        self.assertFalse(can_swizzle_kn(torch.empty(2, 5120, 24, dtype=torch.bfloat16)))

    def test_judgement_matches_actual_swizzle_constraint(self):
        # For bf16, the guard must agree with the actual swizzle constraints.
        # device_impl calls
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
