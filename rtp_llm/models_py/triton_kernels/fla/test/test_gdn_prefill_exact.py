"""Strict acceptance tests for selected prefill normalization/gating kernels."""

import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.common.gated_rmsnorm_prefill import (
    gated_rmsnorm_prefill,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import layer_norm_fwd
from rtp_llm.models_py.triton_kernels.fla.exact_qk_norm import fused_l2norm_qk_exact
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.gdn_gating_prefill import (
    gdn_gating_prefill,
    supports_gdn_gating_prefill,
)
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


class GDNExactPrefill(unittest.TestCase):
    @torch.inference_mode()
    def test_exact_kernels(self):
        torch.manual_seed(107)
        report = {"status": "RUNNING", "cases": []}
        out = Path(
            os.environ.get(
                "GDN_BENCH_OUTPUT",
                os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/gdn-prefill-exact"),
            )
        )
        out.mkdir(parents=True, exist_ok=True)

        def check(name, actual, expected, shape):
            same = torch.equal(actual, expected)
            report["cases"].append(
                {"operator": name, "shape": shape, "bit_exact": same}
            )
            (out / "exact.json").write_text(json.dumps(report, indent=2))
            self.assertTrue(
                same,
                (name, shape, (actual.float() - expected.float()).abs().max().item()),
            )

        for t, h in [
            (2048, 8),
            (2053, 24),
            (10007, 16),
            (16384, 16),
            (24601, 16),
            (32768, 16),
            (40009, 16),
        ]:
            for scale in (0.0, 1e-4, 1.0, 10000.0):
                packed = (
                    torch.randn(1, t, h * 3, 128, device="cuda", dtype=torch.bfloat16)
                    * scale
                )
                q, k, _ = packed.split(h, dim=2)
                qa, ka = fused_l2norm_qk_exact(q, k)
                check("q_norm", qa, l2norm_fwd(q.contiguous()), [t, h, 128, scale])
                check("k_norm", ka, l2norm_fwd(k.contiguous()), [t, h, 128, scale])
                del packed, q, k, _, qa, ka
            h = 64
            x = torch.randn(t, h * 128, device="cuda", dtype=torch.bfloat16)
            z = torch.randn_like(x)
            w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            ref = layer_norm_fwd(
                x, w, None, 1e-6, z=z, group_size=128, is_rms_norm=True
            )[0]
            check("gated_rmsnorm", gated_rmsnorm_prefill(x, z, w), ref, [t, h, 128])
            del x, z, w, ref
            # Preserve a/b strides as produced by fused projection.
            packed = torch.randn(t, h * 2, device="cuda", dtype=torch.bfloat16)
            a, b = packed.split(h, dim=-1)
            al = torch.randn(h, device="cuda")
            bias = torch.randn(h, device="cuda", dtype=torch.bfloat16)
            gr, br = fused_gdn_gating(al, a, b, bias)
            for block in (128, 256, 512):
                g, bt = gdn_gating_prefill(al, a, b, bias, block=block)
                check("gdn_log_gate", g, gr, [t, h, block])
                check("gdn_beta", bt, br, [t, h, block])
            del packed, a, b, al, bias, gr, br, g, bt
        # The loader casts A_log to the model's compute dtype. Both FP32 and
        # BF16 parameters must select and exactly reproduce the fast path.
        for t, h in ((2048, 8), (10007, 16), (24601, 64), (40009, 64)):
            packed = torch.randn(t, h * 2, device="cuda", dtype=torch.bfloat16)
            a, b = packed.split(h, dim=1)
            for ad in (torch.float32, torch.bfloat16):
                for dd in (torch.float32, torch.bfloat16):
                    al = torch.randn(h, device="cuda", dtype=ad)
                    bias = torch.randn(h, device="cuda", dtype=dd)
                    self.assertTrue(supports_gdn_gating_prefill(al, a, b, bias))
                    expected = fused_gdn_gating(al, a, b, bias)
                    actual = gdn_gating_prefill(al, a, b, bias)
                    for name, value, ref in zip(("g", "beta"), actual, expected):
                        check(
                            "gating_dtype_" + name, value, ref, [t, h, str(ad), str(dd)]
                        )
            del packed, a, b, al, bias, expected, actual
        report["status"] = "PASS"
        (out / "exact.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main()
