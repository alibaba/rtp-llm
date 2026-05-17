from __future__ import annotations

import operator
import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_bf16_fp8_quant_pass import (
    apply_rmsnorm_bf16_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_fp8_quant_pass import (
    dsv4_rmsnorm_fx_ref,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.models_py.modules.dsv4.fusions.test.graphfx_fusion_test_utils import (
    DSV4_GRAPHFX_CORRECTNESS_M,
    DSV4_RMSNORM_QUANT_SHAPES,
    assert_bf16_close,
    assert_fp8_quant_close,
    graphfx_m_sweep,
    graphfx_perf_enabled,
    make_fx_pair,
    measured_graph_pair_row,
    trace_dir_for_report,
    write_graphfx_perf_report,
)

torch.fx.wrap("dsv4_rmsnorm_fx_ref")
torch.fx.wrap("sgl_per_token_group_quant_fp8")


def fake_mutating_rmsnorm(output, x, weight, eps, stream=0):
    return None


torch.fx.wrap("fake_mutating_rmsnorm")


FLASH_PRO_MAIN_SHAPES = [(1, 1024), (32, 4096), (257, 4096)]
PRO_MAIN_SHAPES = [(8, 1536), (61, 7168), (251, 7168)]
GUARD_SHAPES = [(17, 2048), (37, 3072)]


def _non_fused_baseline(x: torch.Tensor, weight: torch.Tensor):
    y = torch.empty_like(x)
    rtp_llm_ops.rmsnorm(y, x, weight, 1e-6, torch.cuda.current_stream().cuda_stream)
    q, s = sgl_per_token_group_quant_fp8(
        y,
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return y, q, s


def _fx_old_pattern(x: torch.Tensor, weight: torch.Tensor):
    y = dsv4_rmsnorm_fx_ref(x, weight, 1e-6)
    q, s = sgl_per_token_group_quant_fp8(
        y.contiguous(),
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return y, q, s


class _DualOutputToy(torch.nn.Module):
    def forward(self, x, weight):
        y, q, s = _fx_old_pattern(x, weight)
        return y, q, s, y


class _MutatingRmsnormToy(torch.nn.Module):
    def forward(self, x, weight):
        y = torch.empty_like(x)
        fake_mutating_rmsnorm(y, x, weight, 1e-6, 0)
        q, s = sgl_per_token_group_quant_fp8(
            y.contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        return y, q, s, y


def _fused_graph(module: torch.nn.Module):
    gm = torch.fx.symbolic_trace(module)
    gm = apply_rmsnorm_bf16_fp8_quant_fx_pass(gm)
    gm.recompile()
    return gm


def _target_names(gm: torch.fx.GraphModule) -> list[str]:
    return [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]


def _assert_close_fp8_scale(ref, cand, label: str) -> None:
    y_ref, q_ref, s_ref = ref[:3]
    y_cand, q_cand, s_cand = cand[:3]
    assert_bf16_close(y_ref, y_cand, f"{label}_y", atol=2e-2)
    assert_fp8_quant_close((q_ref, s_ref), (q_cand, s_cand), label)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedRmsnormBf16Fp8QuantPassTest(unittest.TestCase):
    def test_graphfx_rewrite_correctness(self):
        pair = make_fx_pair(
            _DualOutputToy,
            apply_rmsnorm_bf16_fp8_quant_fx_pass,
            required_targets=("fused_rmsnorm_bf16_fp8_quant",),
            forbidden_targets=("sgl_per_token_group_quant_fp8", "dsv4_rmsnorm_fx_ref"),
        )
        for shape_case in DSV4_RMSNORM_QUANT_SHAPES:
            k = shape_case["K"]
            for m in DSV4_GRAPHFX_CORRECTNESS_M:
                torch.manual_seed(2300 + m + k)
                x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
                weight = (torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
                ref = pair.baseline(x, weight)
                cand = pair.candidate(x, weight)
                torch.cuda.synchronize()
                assert_bf16_close(ref[0], cand[0], f"rmsnorm_bf16_fp8_y_M{m}_K{k}", atol=2e-2)
                assert_fp8_quant_close(ref[1:3], cand[1:3], f"rmsnorm_bf16_fp8_qs_M{m}_K{k}")
                assert_bf16_close(ref[3], cand[3], f"rmsnorm_bf16_fp8_alias_M{m}_K{k}", atol=2e-2)

    def test_graphfx_rewrite_perf(self):
        if not graphfx_perf_enabled():
            self.skipTest("set DSV4_GRAPHFX_RUN_PERF_IN_UT=1 or PERF_JSON to run GraphFX perf")
        m_list = graphfx_m_sweep("DSV4_GRAPHFX_RMSNORM_BF16_FP8_M_LIST")
        trace_dir = trace_dir_for_report(
            "DSV4_GRAPHFX_RMSNORM_BF16_FP8_JSON",
            "dsv4_graphfx_rmsnorm_bf16_fp8_quant_perf.json",
            "DSV4_GRAPHFX_RMSNORM_BF16_FP8_TRACE_DIR",
        )
        rows = []
        pair = make_fx_pair(
            _DualOutputToy,
            apply_rmsnorm_bf16_fp8_quant_fx_pass,
            required_targets=("fused_rmsnorm_bf16_fp8_quant",),
            forbidden_targets=("sgl_per_token_group_quant_fp8", "dsv4_rmsnorm_fx_ref"),
        )
        for shape_case in DSV4_RMSNORM_QUANT_SHAPES:
            k = shape_case["K"]
            weight = (torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            for m in m_list:
                torch.manual_seed(2700 + m + k)
                x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
                rows.append(
                    measured_graph_pair_row(
                        op="rmsnorm_bf16_fp8_quant",
                        label=f"rmsnorm_bf16_fp8_M{m}_K{k}",
                        shape_meta={**shape_case, "M": m},
                        baseline_fn=lambda x=x, weight=weight: pair.baseline(x, weight),
                        candidate_fn=lambda x=x, weight=weight: pair.candidate(x, weight),
                        trace_dir=trace_dir,
                        kernel_regex=os.environ.get("DSV4_GRAPHFX_RMSNORM_BF16_FP8_KERNEL_REGEX") or None,
                        warmup=20 if m <= 4096 else 8,
                    )
                )
        path = write_graphfx_perf_report(
            json_env="DSV4_GRAPHFX_RMSNORM_BF16_FP8_JSON",
            default_json="dsv4_graphfx_rmsnorm_bf16_fp8_quant_perf.json",
            rows=rows,
            metadata={
                "title": "DSV4 GraphFX RMSNorm BF16 FP8 Quant Perf",
                "baseline_path": "original FX graph: rmsnorm with BF16 consumer -> quant",
                "candidate_path": "GraphFX rewritten FX graph: fused_rmsnorm_bf16_fp8_quant",
                "m_list": m_list,
                "shape_cases": DSV4_RMSNORM_QUANT_SHAPES,
            },
        )
        print(f"Wrote GraphFX RMSNorm BF16 FP8 quant perf report: {path}")

    def test_fx_pass_rewrites_dual_output_pattern(self):
        gm = _fused_graph(_DualOutputToy())
        targets = _target_names(gm)
        self.assertIn("fused_rmsnorm_bf16_fp8_quant", targets)
        self.assertNotIn("sgl_per_token_group_quant_fp8", targets)
        self.assertNotIn("dsv4_rmsnorm_fx_ref", targets)

    def test_fx_pass_rewrites_mutating_rmsnorm_shape(self):
        gm = _fused_graph(_MutatingRmsnormToy())
        targets = _target_names(gm)
        self.assertIn("fused_rmsnorm_bf16_fp8_quant", targets)
        self.assertNotIn("fake_mutating_rmsnorm", targets)
        self.assertNotIn("sgl_per_token_group_quant_fp8", targets)

    def test_pass_result_matches_original_pattern(self):
        torch.manual_seed(123)
        gm = _fused_graph(_DualOutputToy())
        for m, k in FLASH_PRO_MAIN_SHAPES + PRO_MAIN_SHAPES + GUARD_SHAPES:
            x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            weight = (torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            y_ref, q_ref, s_ref = _non_fused_baseline(x, weight)
            ref = (y_ref, q_ref, s_ref, y_ref)
            cand = gm(x, weight)
            torch.cuda.synchronize()
            _assert_close_fp8_scale(ref, cand, f"M{m}_K{k}")
            assert_bf16_close(ref[3], cand[3], f"M{m}_K{k}_alias", atol=2e-2)

    def test_pass_generated_kernel_rejects_unsupported_k(self):
        gm = _fused_graph(_DualOutputToy())
        x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16).contiguous()
        weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "unsupported fused_rmsnorm_bf16_fp8_quant input"):
            gm(x, weight)

    def test_quant_tuple_users_are_getitem_only(self):
        gm = _fused_graph(_DualOutputToy())
        for node in gm.graph.nodes:
            if getattr(node.target, "__name__", "") == "fused_rmsnorm_bf16_fp8_quant":
                getitem_indices = sorted(
                    user.args[1]
                    for user in node.users
                    if user.op == "call_function" and user.target == operator.getitem
                )
                self.assertEqual(getitem_indices, [0, 1, 2])
                return
        self.fail("fused_rmsnorm_bf16_fp8_quant was not inserted")


if __name__ == "__main__":
    unittest.main()
