from __future__ import annotations

import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4.fusions.kv_rope_fp8_quant_pass import (
    apply_kv_rope_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.kv_rope_fp8_quant_runtime import (
    dsv4_kv_rope_fp8_quant_from_provenance,
    dsv4_kv_rope_quant_producer_token,
)
from rtp_llm.models_py.modules.dsv4.fusions.test.graphfx_fusion_test_utils import (
    assert_fp8_quant_close,
    make_fx_pair,
    target_names,
)

torch.fx.wrap("sgl_per_token_group_quant_fp8")


def fused_kv_compress_norm_rope_insert_indexer_attn(x):
    return x


torch.fx.wrap("fused_kv_compress_norm_rope_insert_indexer_attn")


class _KvRopeQuantConsumerPattern(torch.nn.Module):
    def forward(self, x):
        y = fused_kv_compress_norm_rope_insert_indexer_attn(x)
        return sgl_per_token_group_quant_fp8(
            y.contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


class _KvRopeProducerPattern(torch.nn.Module):
    def forward(self, x):
        return fused_kv_compress_norm_rope_insert_indexer_attn(x)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Dsv4KvRopeFp8QuantGraphFXTest(unittest.TestCase):
    def test_graphfx_rewrite_correctness(self):
        pair = make_fx_pair(
            _KvRopeQuantConsumerPattern,
            apply_kv_rope_fp8_quant_fx_pass,
            required_targets=("dsv4_kv_rope_fp8_quant_from_provenance",),
            forbidden_targets=("sgl_per_token_group_quant_fp8",),
        )
        x = (torch.randn(17, 512, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        ref = pair.baseline(x)
        cand = pair.candidate(x)
        torch.cuda.synchronize()
        assert_fp8_quant_close(ref, cand, "kv_rope_quant_graphfx_fallback")

    def test_graphfx_rewrite_inserts_producer_token(self):
        gm = torch.fx.symbolic_trace(_KvRopeProducerPattern())
        gm = apply_kv_rope_fp8_quant_fx_pass(gm)
        gm.recompile()
        names = target_names(gm)
        self.assertIn("dsv4_kv_rope_quant_producer_token", names)

    def test_runtime_reuses_precomputed_payload(self):
        x = (torch.randn(9, 512, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        ref = sgl_per_token_group_quant_fp8(
            x,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        token = dsv4_kv_rope_quant_producer_token(x, ref[0], ref[1])
        cand = dsv4_kv_rope_fp8_quant_from_provenance(
            token,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        torch.cuda.synchronize()
        self.assertIs(cand[0], ref[0])
        self.assertIs(cand[1], ref[1])

    def test_runtime_requires_provenance_when_configured(self):
        old = os.environ.get("DSV4_KV_ROPE_QUANT_REQUIRE_PROVENANCE")
        os.environ["DSV4_KV_ROPE_QUANT_REQUIRE_PROVENANCE"] = "1"
        try:
            x = (torch.randn(5, 512, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            with self.assertRaisesRegex(RuntimeError, "did not find valid producer provenance"):
                dsv4_kv_rope_fp8_quant_from_provenance(
                    x,
                    group_size=128,
                    eps=1e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
        finally:
            if old is None:
                os.environ.pop("DSV4_KV_ROPE_QUANT_REQUIRE_PROVENANCE", None)
            else:
                os.environ["DSV4_KV_ROPE_QUANT_REQUIRE_PROVENANCE"] = old

    def test_graphfx_rewrite_perf(self):
        self.skipTest(
            "KV RoPE FP8 quant pass is still provenance-only; no fused-kernel "
            "performance claim until producer-side dual-output CUDA payload lands"
        )


if __name__ == "__main__":
    unittest.main()
