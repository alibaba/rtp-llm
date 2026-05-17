from __future__ import annotations

import gc
import os
import unittest

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.base.cuda.norm import rmsnorm as base_cuda_rmsnorm
from rtp_llm.models_py.modules.dsv4.fusions.fusion_registry import apply_registered_dsv4_fusions
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_fp8_quant_pass import (
    apply_rmsnorm_fp8_quant_fx_pass,
    dsv4_rmsnorm_fx_ref,
)
from rtp_llm.models_py.modules.dsv4.fusions.rmsnorm_quant_runtime import (
    dsv4_fused_rmsnorm_fp8_quant_from_provenance,
    dsv4_rmsnorm_quant_producer_token,
    dsv4_rmsnorm_quant_provenance_token,
)
from rtp_llm.models_py.modules.dsv4.fusions.test.graphfx_fusion_test_utils import (
    DSV4_GRAPHFX_CORRECTNESS_M,
    DSV4_RMSNORM_QUANT_SHAPES,
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


def _baseline(x: torch.Tensor, weight: torch.Tensor):
    y = dsv4_rmsnorm_fx_ref(x, weight, 1e-6)
    return sgl_per_token_group_quant_fp8(
        y.contiguous(),
        group_size=128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


def _assert_close_fp8_scale(ref, cand, label: str) -> None:
    q_ref, s_ref = ref
    q_cand, s_cand = cand
    q_diff = (
        q_ref.contiguous().view(torch.uint8).to(torch.int16)
        - q_cand.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()
    s_diff = (
        s_ref.contiguous().view(torch.uint8).to(torch.int16)
        - s_cand.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()
    q_exact = (q_diff == 0).float().mean().item()
    s_exact = (s_diff == 0).float().mean().item()
    print(
        f"[{label}] q_exact={q_exact * 100:.2f}% q_max_ulp={q_diff.max().item()} "
        f"s_exact={s_exact * 100:.2f}% s_max_byte={s_diff.max().item()}"
    )
    assert q_exact >= 0.94
    assert q_diff.max().item() <= 2
    assert s_exact >= 0.98
    assert s_diff.max().item() <= 1


class _RmsnormQuantToy(torch.nn.Module):
    def forward(self, x, weight):
        return _baseline(x, weight)


class _MutatingRmsnormQuantToy(torch.nn.Module):
    def forward(self, x, weight):
        y = torch.empty_like(x)
        fake_mutating_rmsnorm(y, x, weight, 1e-6, 0)
        return sgl_per_token_group_quant_fp8(
            y.contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


class _RmsnormProducerToy(torch.nn.Module):
    def forward(self, x, weight):
        return dsv4_rmsnorm_fx_ref(x, weight, 1e-6)


class _RmsnormLayoutProducerToy(torch.nn.Module):
    def forward(self, x, weight):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        return dsv4_rmsnorm_fx_ref(x_2d, weight, 1e-6).view(orig_shape)


class _MutatingRmsnormProducerToy(torch.nn.Module):
    def forward(self, x, weight):
        y = torch.empty_like(x)
        fake_mutating_rmsnorm(y, x, weight, 1e-6, 0)
        return y


class _QuantConsumerToy(torch.nn.Module):
    def forward(self, y):
        return sgl_per_token_group_quant_fp8(
            y.contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


class _IndexerWqBQuantConsumerToy(torch.nn.Module):
    def forward(self, y):
        return sgl_per_token_group_quant_fp8(
            y.reshape(-1, y.shape[-1]).contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


class _DynamoBaseCudaRmsnormQuantToy(torch.nn.Module):
    def forward(self, x, weight):
        y = base_cuda_rmsnorm(x, weight, 1e-6)
        return sgl_per_token_group_quant_fp8(
            y.contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


class _DynamoReshapedRmsnormQuantToy(torch.nn.Module):
    def forward(self, x, weight):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        y = base_cuda_rmsnorm(x_2d, weight, 1e-6).view(orig_shape)
        return sgl_per_token_group_quant_fp8(
            y.reshape(-1, orig_shape[-1]).contiguous(),
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


class _DynamoLayoutRmsnormProducerToy(torch.nn.Module):
    def forward(self, x, weight):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        return base_cuda_rmsnorm(x_2d, weight, 1e-6).view(orig_shape)


def _fused_graph():
    gm = torch.fx.symbolic_trace(_RmsnormQuantToy())
    gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
    gm.recompile()
    return gm


def _compile_with_local_rmsnorm_quant_pass(module: torch.nn.Module):
    compiled_targets: list[list[str]] = []

    def backend(gm, example_inputs):
        del example_inputs
        gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
        gm.recompile()
        compiled_targets.append(
            [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        )
        return gm.forward

    try:
        import torch._dynamo as dynamo

        dynamo.reset()
    except Exception:
        pass
    return torch.compile(module, backend=backend, dynamic=True, fullgraph=False), compiled_targets


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedRmsnormFp8QuantPassTest(unittest.TestCase):
    def test_graphfx_rewrite_correctness(self):
        pair = make_fx_pair(
            _RmsnormQuantToy,
            apply_rmsnorm_fp8_quant_fx_pass,
            required_targets=("fused_rmsnorm_fp8_quant",),
            forbidden_targets=("sgl_per_token_group_quant_fp8",),
        )
        for shape_case in DSV4_RMSNORM_QUANT_SHAPES:
            k = shape_case["K"]
            for m in DSV4_GRAPHFX_CORRECTNESS_M:
                torch.manual_seed(1300 + m + k)
                x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
                weight = (torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
                ref = pair.baseline(x, weight)
                cand = pair.candidate(x, weight)
                torch.cuda.synchronize()
                assert_fp8_quant_close(ref, cand, f"rmsnorm_fp8_graphfx_M{m}_K{k}")

    def test_graphfx_rewrite_perf(self):
        if not graphfx_perf_enabled():
            self.skipTest("set DSV4_GRAPHFX_RUN_PERF_IN_UT=1 or PERF_JSON to run GraphFX perf")
        m_list = graphfx_m_sweep("DSV4_GRAPHFX_RMSNORM_FP8_M_LIST")
        trace_dir = trace_dir_for_report(
            "DSV4_GRAPHFX_RMSNORM_FP8_JSON",
            "dsv4_graphfx_rmsnorm_fp8_quant_perf.json",
            "DSV4_GRAPHFX_RMSNORM_FP8_TRACE_DIR",
        )
        rows = []
        pair = make_fx_pair(
            _RmsnormQuantToy,
            apply_rmsnorm_fp8_quant_fx_pass,
            required_targets=("fused_rmsnorm_fp8_quant",),
            forbidden_targets=("sgl_per_token_group_quant_fp8",),
        )
        for shape_case in DSV4_RMSNORM_QUANT_SHAPES:
            k = shape_case["K"]
            weight = (torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            for m in m_list:
                torch.manual_seed(1700 + m + k)
                x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
                rows.append(
                    measured_graph_pair_row(
                        op="rmsnorm_fp8_quant",
                        label=f"rmsnorm_fp8_M{m}_K{k}",
                        shape_meta={**shape_case, "M": m},
                        baseline_fn=lambda x=x, weight=weight: pair.baseline(x, weight),
                        candidate_fn=lambda x=x, weight=weight: pair.candidate(x, weight),
                        trace_dir=trace_dir,
                        kernel_regex=os.environ.get("DSV4_GRAPHFX_RMSNORM_FP8_KERNEL_REGEX") or None,
                        warmup=20 if m <= 4096 else 8,
                    )
                )
        path = write_graphfx_perf_report(
            json_env="DSV4_GRAPHFX_RMSNORM_FP8_JSON",
            default_json="dsv4_graphfx_rmsnorm_fp8_quant_perf.json",
            rows=rows,
            metadata={
                "title": "DSV4 GraphFX RMSNorm FP8 Quant Perf",
                "baseline_path": "original FX graph: rmsnorm -> sgl_per_token_group_quant_fp8",
                "candidate_path": "GraphFX rewritten FX graph: fused_rmsnorm_fp8_quant",
                "m_list": m_list,
                "shape_cases": DSV4_RMSNORM_QUANT_SHAPES,
            },
        )
        print(f"Wrote GraphFX RMSNorm FP8 quant perf report: {path}")

    def test_fx_pass_rewrites_pattern(self):
        gm = _fused_graph()
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("fused_rmsnorm_fp8_quant", targets)
        self.assertNotIn("sgl_per_token_group_quant_fp8", targets)

    def test_fx_pass_rewrites_mutating_rmsnorm_pattern(self):
        gm = torch.fx.symbolic_trace(_MutatingRmsnormQuantToy())
        gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
        gm.recompile()
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("fused_rmsnorm_fp8_quant", targets)
        self.assertNotIn("fake_mutating_rmsnorm", targets)
        self.assertNotIn("sgl_per_token_group_quant_fp8", targets)

    def test_fx_pass_rewrites_real_pybind_rmsnorm_pattern(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        y = graph.call_function(torch.empty_like, args=(x,))
        graph.call_function(rtp_llm_ops.rmsnorm, args=(y, x, weight, 1e-6, 0))
        q = graph.call_function(
            sgl_per_token_group_quant_fp8,
            args=(y,),
            kwargs={
                "group_size": 128,
                "eps": 1e-4,
                "column_major_scales": True,
                "scale_tma_aligned": True,
                "scale_ue8m0": True,
            },
        )
        graph.output(q)
        gm = torch.fx.GraphModule({}, graph)
        gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
        gm.recompile()
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("fused_rmsnorm_fp8_quant", targets)
        self.assertNotIn("rmsnorm", targets)
        self.assertNotIn("sgl_per_token_group_quant_fp8", targets)

    def test_fx_pass_inserts_cross_graph_functional_producer_token(self):
        gm = torch.fx.symbolic_trace(_RmsnormProducerToy())
        gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
        gm.recompile()
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("dsv4_rmsnorm_quant_producer_token", targets)
        self.assertNotIn("dsv4_rmsnorm_fx_ref", targets)

    def test_fx_pass_inserts_cross_graph_layout_producer_token(self):
        gm = torch.fx.symbolic_trace(_RmsnormLayoutProducerToy())
        gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
        gm.recompile()
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("dsv4_rmsnorm_quant_producer_token", targets)
        self.assertNotIn("dsv4_rmsnorm_fx_ref", targets)
        self.assertIn("view", targets)

    def test_fx_pass_inserts_cross_graph_mutating_producer_token(self):
        gm = torch.fx.symbolic_trace(_MutatingRmsnormProducerToy())
        gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
        gm.recompile()
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("dsv4_rmsnorm_quant_producer_token", targets)
        self.assertNotIn("fake_mutating_rmsnorm", targets)

    def test_fx_pass_rewrites_cross_graph_quant_consumer(self):
        gm = torch.fx.symbolic_trace(_QuantConsumerToy())
        gm = apply_rmsnorm_fp8_quant_fx_pass(gm)
        gm.recompile()
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("dsv4_fused_rmsnorm_fp8_quant_from_provenance", targets)
        self.assertNotIn("sgl_per_token_group_quant_fp8", targets)

    def test_cross_graph_runtime_safe_fallback_without_provenance(self):
        y = torch.randn(3, 1024, device="cuda", dtype=torch.bfloat16).contiguous()
        ref = sgl_per_token_group_quant_fp8(
            y,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        cand = dsv4_fused_rmsnorm_fp8_quant_from_provenance(
            y,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        torch.cuda.synchronize()
        _assert_close_fp8_scale(ref, cand, "cross_graph_fallback")

    def test_cross_graph_runtime_uses_provenance(self):
        torch.manual_seed(123)
        x = (torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
        y = dsv4_rmsnorm_fx_ref(x, weight, 1e-6)
        token = dsv4_rmsnorm_quant_provenance_token(y, x, weight, 1e-6)
        ref = _baseline(x, weight)
        cand = dsv4_fused_rmsnorm_fp8_quant_from_provenance(
            token,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        torch.cuda.synchronize()
        _assert_close_fp8_scale(ref, cand, "cross_graph_provenance")

    def test_cross_graph_runtime_precomputes_bf16_fp8(self):
        old_env = os.environ.get("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8")
        os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = "1"
        try:
            torch.manual_seed(123)
            x = (torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            ref = _baseline(x, weight)
            token = dsv4_rmsnorm_quant_producer_token(x, weight, 1e-6)
            cand = dsv4_fused_rmsnorm_fp8_quant_from_provenance(
                token,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            torch.cuda.synchronize()
            _assert_close_fp8_scale(ref, cand, "cross_graph_precompute")
        finally:
            if old_env is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = old_env

    def test_cross_graph_runtime_precompute_survives_view_alias(self):
        old_env = os.environ.get("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8")
        os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = "1"
        try:
            torch.manual_seed(123)
            x = (torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            ref = _baseline(x, weight)
            token = dsv4_rmsnorm_quant_producer_token(x, weight, 1e-6)
            alias = token.view(7, 1, 1024).reshape(-1, 1024).contiguous()
            cand = dsv4_fused_rmsnorm_fp8_quant_from_provenance(
                alias,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            torch.cuda.synchronize()
            _assert_close_fp8_scale(ref, cand, "cross_graph_precompute_view_alias")
        finally:
            if old_env is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = old_env

    def test_cross_graph_runtime_precompute_survives_dead_producer_view(self):
        old_env = os.environ.get("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8")
        os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = "1"
        try:
            torch.manual_seed(123)
            x = (torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()

            def make_returned_view():
                token = dsv4_rmsnorm_quant_producer_token(x, weight, 1e-6)
                return token.view(1, 7, 1024).reshape(7, 1024)

            y_view = make_returned_view()
            gc.collect()
            ref = sgl_per_token_group_quant_fp8(
                y_view.contiguous(),
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            cand = dsv4_fused_rmsnorm_fp8_quant_from_provenance(
                y_view,
                fallback_y=y_view.contiguous(),
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            torch.cuda.synchronize()
            self.assertEqual(tuple(ref[0].shape), tuple(cand[0].shape))
            self.assertEqual(tuple(ref[1].shape), tuple(cand[1].shape))
            _assert_close_fp8_scale(ref, cand, "cross_graph_precompute_dead_view")
        finally:
            if old_env is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = old_env

    def test_cross_graph_runtime_precompute_uses_fallback_quant_shape(self):
        old_env = os.environ.get("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8")
        os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = "1"
        try:
            torch.manual_seed(123)
            x = (torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            ref = _baseline(x, weight)
            token = dsv4_rmsnorm_quant_producer_token(x, weight, 1e-6)
            fallback_y = token.clone()
            cand = dsv4_fused_rmsnorm_fp8_quant_from_provenance(
                token,
                fallback_y=fallback_y,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            torch.cuda.synchronize()
            _assert_close_fp8_scale(ref, cand, "cross_graph_precompute_fallback_shape")
        finally:
            if old_env is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = old_env

    def test_torch_compile_rewrites_base_cuda_rmsnorm_quant_pattern(self):
        torch.manual_seed(123)
        module, compiled_targets = _compile_with_local_rmsnorm_quant_pass(
            _DynamoBaseCudaRmsnormQuantToy()
        )
        x = (torch.randn(5, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
        ref = _baseline(x, weight)
        cand = module(x, weight)
        torch.cuda.synchronize()
        self.assertTrue(
            any("fused_rmsnorm_fp8_quant" in targets for targets in compiled_targets),
            compiled_targets,
        )
        _assert_close_fp8_scale(ref, cand, "compile_base_cuda_rmsnorm")

    def test_torch_compile_rewrites_reshaped_rmsnorm_quant_pattern(self):
        torch.manual_seed(123)
        module, compiled_targets = _compile_with_local_rmsnorm_quant_pass(
            _DynamoReshapedRmsnormQuantToy()
        )
        x = (torch.randn(5, 1, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
        ref = _baseline(x.reshape(-1, x.shape[-1]), weight)
        cand = module(x, weight)
        torch.cuda.synchronize()
        self.assertTrue(
            any("fused_rmsnorm_fp8_quant" in targets for targets in compiled_targets),
            compiled_targets,
        )
        _assert_close_fp8_scale(ref, cand, "compile_reshaped_rmsnorm")

    def test_torch_compile_split_producer_consumer_uses_precomputed_quant(self):
        old_precompute = os.environ.get("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8")
        old_require = os.environ.get("DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE")
        os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = "1"
        os.environ["DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE"] = "1"
        try:
            torch.manual_seed(123)
            producer, producer_targets = _compile_with_local_rmsnorm_quant_pass(
                _DynamoLayoutRmsnormProducerToy()
            )
            consumer, consumer_targets = _compile_with_local_rmsnorm_quant_pass(
                _QuantConsumerToy()
            )
            x = (torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            weight = (torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            y = producer(x, weight)
            cand = consumer(y)
            ref = _baseline(x, weight)
            torch.cuda.synchronize()
            self.assertTrue(
                any("dsv4_rmsnorm_quant_producer_token" in targets for targets in producer_targets),
                producer_targets,
            )
            self.assertTrue(
                any(
                    "dsv4_fused_rmsnorm_fp8_quant_from_provenance" in targets
                    for targets in consumer_targets
                ),
                consumer_targets,
            )
            _assert_close_fp8_scale(ref, cand, "compile_split_precomputed")
        finally:
            if old_precompute is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = old_precompute
            if old_require is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE"] = old_require

    def test_torch_compile_indexer_wq_b_consumer_uses_precomputed_quant(self):
        old_precompute = os.environ.get("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8")
        old_require = os.environ.get("DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE")
        os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = "1"
        os.environ["DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE"] = "1"
        try:
            torch.manual_seed(123)
            producer, producer_targets = _compile_with_local_rmsnorm_quant_pass(
                _DynamoLayoutRmsnormProducerToy()
            )
            consumer, consumer_targets = _compile_with_local_rmsnorm_quant_pass(
                _IndexerWqBQuantConsumerToy()
            )
            for shape in ((2, 3, 1024), (1, 17, 1024)):
                x = (
                    torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.2
                ).contiguous()
                weight = (
                    torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1
                    + 1.0
                ).contiguous()
                y = producer(x, weight)
                cand = consumer(y)
                ref = _baseline(x.reshape(-1, x.shape[-1]), weight)
                torch.cuda.synchronize()
                _assert_close_fp8_scale(ref, cand, f"indexer_wq_b_shape_{shape}")
            self.assertTrue(
                any(
                    "dsv4_rmsnorm_quant_producer_token" in targets
                    for targets in producer_targets
                ),
                producer_targets,
            )
            self.assertTrue(
                any(
                    "dsv4_fused_rmsnorm_fp8_quant_from_provenance" in targets
                    for targets in consumer_targets
                ),
                consumer_targets,
            )
        finally:
            if old_precompute is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"] = old_precompute
            if old_require is None:
                os.environ.pop("DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE", None)
            else:
                os.environ["DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE"] = old_require

    def test_registered_pass_rewrites_pattern(self):
        old_env = os.environ.get("DSV4_FUSED_RMSNORM_FP8_QUANT")
        os.environ["DSV4_FUSED_RMSNORM_FP8_QUANT"] = "1"
        try:
            gm = torch.fx.symbolic_trace(_RmsnormQuantToy())
            gm = apply_registered_dsv4_fusions(gm)
            gm.recompile()
            targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
            self.assertIn("fused_rmsnorm_fp8_quant", targets)
        finally:
            if old_env is None:
                os.environ.pop("DSV4_FUSED_RMSNORM_FP8_QUANT", None)
            else:
                os.environ["DSV4_FUSED_RMSNORM_FP8_QUANT"] = old_env

    def test_pass_result_matches_original_pattern(self):
        torch.manual_seed(123)
        gm = _fused_graph()
        for m, k in [(1, 1024), (8, 1024), (32, 4096), (128, 4096), (17, 7168)]:
            x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
            weight = (torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0).contiguous()
            ref = _baseline(x, weight)
            cand = gm(x, weight)
            torch.cuda.synchronize()
            _assert_close_fp8_scale(ref, cand, f"M{m}_K{k}")

    def test_pass_generated_kernel_rejects_bad_layout(self):
        gm = _fused_graph()
        base = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16).contiguous()
        x = base[:, ::2]
        weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "unsupported fused_rmsnorm_fp8_quant input"):
            gm(x, weight)


if __name__ == "__main__":
    unittest.main()
