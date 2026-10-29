import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
    fused_q_rope_quant,
    fused_qk_rope_quant,
)


@unittest.skipUnless(torch.cuda.is_available(), "Indexer kernels require CUDA")
class IndexerFusedRopeQuantTest(unittest.TestCase):
    def _check_reference(self, tokens, *, zero=False):
        heads, dim, rope_dim = 16, 128, 64
        torch.manual_seed(20261021 + tokens)
        q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        if zero:
            q.zero_()
            k.zero_()
        inv_freq = 10000.0 ** (
            -torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32) / rope_dim
        )
        angles = torch.arange(131, device="cuda").float()[:, None] * inv_freq
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64) % 127 + 1
        for scale_fmt in ("none", "ue8m0"):
            for neox in (False, True):
                with self.subTest(
                    tokens=tokens, zero=zero, scale_fmt=scale_fmt, neox=neox
                ):
                    operator = IndexerOp(
                        heads,
                        dim,
                        16,
                        rope_dim,
                        cos_sin_cache=cache,
                        scale_fmt=scale_fmt,
                        is_neox_style=neox,
                        enable_fuse_kernels=False,
                    )
                    query, key = operator.apply_rope_and_rotate_q_k(
                        q.clone(), k.clone(), positions
                    )
                    expected_q, expected_scale = operator.quant_q_only(query)
                    actual_q, actual_scale, actual_key = operator.fused_rope_quant_qk(
                        q, k, positions
                    )
                    single_q, single_scale = operator.fused_rope_quant_q(q, positions)
                    for output in (actual_q, single_q):
                        torch.testing.assert_close(
                            output.float(),
                            expected_q.float(),
                            rtol=0,
                            atol=0,
                        )
                        self.assertTrue(torch.isfinite(output.float()).all())
                    for scale in (actual_scale, single_scale):
                        torch.testing.assert_close(
                            scale, expected_scale, rtol=0, atol=0
                        )
                        self.assertTrue(torch.isfinite(scale).all())
                    torch.testing.assert_close(actual_key, key, rtol=0, atol=0)
                    if zero:
                        self.assertEqual(
                            torch.count_nonzero(actual_q.float()).item(), 0
                        )
                        self.assertEqual(torch.count_nonzero(actual_key).item(), 0)

    def test_decode_matches_unfused_rope_hadamard_quant(self):
        for tokens in (1, 7):
            self._check_reference(tokens)

    def test_prefill_matches_unfused_rope_hadamard_quant(self):
        for tokens in (257, 2048):
            self._check_reference(tokens)

    def test_zero_input_has_finite_matching_scales(self):
        for tokens in (1, 2048):
            self._check_reference(tokens, zero=True)

    def test_explicit_quant_kernel_selection_matches_reference(self):
        for mode in ("legacy", "v2"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}
            ):
                self._check_reference(7)

    def test_direct_helpers_keep_ue8m0_default(self):
        q = torch.ones(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.ones(1, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.zeros(1, device="cuda", dtype=torch.int64)
        cache = torch.cat(
            (torch.ones(1, 32, device="cuda"), torch.zeros(1, 32, device="cuda")),
            dim=-1,
        )
        for helper, args in (
            (fused_q_rope_quant, (q, positions, cache, 2, 128, 64)),
            (fused_qk_rope_quant, (q, k, positions, cache, 2, 128, 64)),
        ):
            implicit = helper(*args)
            explicit = helper(*args, scale_fmt="ue8m0")
            for actual, expected in zip(implicit, explicit):
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
