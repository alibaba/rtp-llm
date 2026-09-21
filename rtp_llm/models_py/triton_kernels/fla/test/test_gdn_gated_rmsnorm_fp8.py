"""Exact prefill fusion comparisons against native RTP tensor boundaries."""

import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.triton_kernels.common.prefill_fusion import prefill_fusion_scope

OUT = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp/gdn-fusion-tests"))
LENGTHS = [10007, 16384, 24601, 32768, 40009]

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.gated_rmsnorm_fp8 import gated_rmsnorm_fp8
from rtp_llm.models_py.triton_kernels.common.gated_rmsnorm_prefill import (
    gated_rmsnorm_prefill,
)


def error(a, b):
    af = a.float()
    bf = b.float()
    d = (af - bf).abs()
    return {
        "max_abs": d.max().item(),
        "rms": d.square().mean().sqrt().item(),
        "mismatches": (
            (a.view(torch.uint8) != b.contiguous().view(torch.uint8)).sum().item()
            if a.dtype == torch.float8_e4m3fn
            else (a != b).sum().item()
        ),
    }


class GatedRmsNormFp8Test(unittest.TestCase):
    @torch.inference_mode()
    def test_correctness(self):
        if os.getenv("IO_MODE") in ("perf", "ncu"):
            self.skipTest("Separate performance run")
        torch.manual_seed(918)
        OUT.mkdir(parents=True, exist_ok=True)
        report = {"status": "RUNNING", "conv": [], "norm_quant": []}

        def save():
            (OUT / "correctness.json").write_text(json.dumps(report, indent=2))

        for t, h, shared, bias, act in [
            (37, 4, True, False, "silu"),
            (2053, 8, False, True, "sigmoid"),
        ] + [(n, 64, True, False, "silu") for n in LENGTHS]:
            x = torch.randn(t, h * 128 + 32, device="cuda", dtype=torch.bfloat16)[
                :, : h * 128
            ]
            z = torch.randn_like(x)
            w = torch.randn(
                128 if shared else h * 128, device="cuda", dtype=torch.bfloat16
            )
            b = torch.randn_like(w) if bias else None
            # Boundary/adversarial groups: zero and subnormal-scale inputs.
            x[0] = 0
            x[1] *= 1e-6
            y = gated_rmsnorm_prefill(x, z, w, b, group_size=128, activation=act)
            for ue in (False, True):
                ref = sgl_per_token_group_quant_fp8(
                    y,
                    128,
                    eps=1e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=ue,
                )
                for tile in (4, 8, 16, 32):
                    got = gated_rmsnorm_fp8(
                        x, z, w, b, activation=act, scale_ue8m0=ue, tile_rows=tile
                    )
                    record = {
                        "t": t,
                        "h": h,
                        "ue8m0": ue,
                        "tile": tile,
                        "fp8": error(got[0], ref[0]),
                        "scale": error(got[1], ref[1]),
                    }
                    report["norm_quant"].append(record)
                    save()
                    print("NORM_QUANT", record, flush=True)
                    self.assertTrue(
                        torch.equal(got[0].view(torch.uint8), ref[0].view(torch.uint8)),
                        record,
                    )
                    self.assertTrue(torch.equal(got[1], ref[1]), record)
            del x, z, w, b, y, ref, got
        report["status"] = "PASS"
        save()

    @torch.inference_mode()
    def test_dispatch(self):
        if os.getenv("IO_MODE") in ("perf", "ncu"):
            self.skipTest("Separate performance run")
        from types import SimpleNamespace
        from unittest.mock import patch

        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

        OUT.mkdir(parents=True, exist_ok=True)
        results = []
        torch.manual_seed(919)
        # Real consumer with caller-provided scales, using its production dispatch.
        n, k = 256, 8192
        shape = (n, k) if is_deep_gemm_e8m0_used() else (k, n)
        w = (torch.randn(shape, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        ws = torch.ones(tuple((d + 127) // 128 for d in shape), device="cuda") * 0.02
        if is_deep_gemm_e8m0_used():
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

            w, ws = requant_weight_ue8m0(w, ws)
        from rtp_llm.models_py.modules.factory.linear import LinearFactory
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )

        factory_linear = LinearFactory.create_linear_from_weights(
            {"w": w, "s": ws},
            "w",
            "s",
            quant_config=SimpleNamespace(get_method=lambda: "FP8_PER_BLOCK"),
        )
        self.assertIsInstance(factory_linear, CudaFp8GEMMLinear)
        norm = RmsNormGated(
            torch.randn(128, device="cuda", dtype=torch.bfloat16), group_size=128
        )
        for linear in (CudaFp8DeepGEMMLinear(w, ws), factory_linear):
            obj = SimpleNamespace(norm=norm, out_proj=linear)
            inner = (
                linear._deepgemm_linear
                if isinstance(linear, CudaFp8GEMMLinear)
                else linear
            )
            for t, enabled in ((37, True), (2053, True), (10007, True), (2053, False)):
                x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
                z = torch.randn(t, k + 128, device="cuda", dtype=torch.bfloat16)[
                    :, 128:
                ]
                with prefill_fusion_scope(True), patch.dict(
                    os.environ,
                    {
                        "RTP_QWEN35_FUSED_GATED_RMSNORM_FP8": "0",
                    },
                ):
                    ref = Qwen3NextGatedDeltaNet._norm_output_project(
                        obj, x, z, enable_fusion=enabled
                    )
                with prefill_fusion_scope(True), patch.dict(
                    os.environ,
                    {
                        "RTP_QWEN35_FUSED_GATED_RMSNORM_FP8": "1",
                    },
                ):
                    with patch.object(
                        inner, "quantize_input", wraps=inner.quantize_input
                    ) as seen:
                        got = Qwen3NextGatedDeltaNet._norm_output_project(
                            obj, x, z, enable_fusion=enabled
                        )
                        self.assertEqual(seen.call_count, int(t < 2048 or not enabled))
                record = {
                    "consumer": type(linear).__name__,
                    "consumer_t": t,
                    "enabled": enabled,
                    "output": error(got, ref),
                }
                results.append(record)
                (OUT / "dispatch.json").write_text(json.dumps(results, indent=2))
                self.assertTrue(torch.equal(ref, got), record)


if __name__ == "__main__":
    unittest.main()
