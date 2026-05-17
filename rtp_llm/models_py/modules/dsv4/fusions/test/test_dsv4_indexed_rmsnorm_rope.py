from __future__ import annotations

import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.kernels.cuda.dsv4_indexed_rope import indexed_rmsnorm_rope_path
from rtp_llm.models_py.modules.dsv4.fusions.indexed_rope_pass import (
    apply_indexed_rope_fx_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.indexed_rope_runtime import (
    dsv4_indexed_rmsnorm_rope_from_freqs,
    dsv4_indexed_rope_freqs_token,
)
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.fusions.test.graphfx_fusion_test_utils import (
    DSV4_GRAPHFX_CORRECTNESS_M,
    DSV4_ROPE_SHAPES,
    assert_bf16_close,
    graphfx_m_sweep,
    graphfx_perf_enabled,
    make_fx_pair,
    measured_graph_pair_row,
    trace_dir_for_report,
    write_graphfx_perf_report,
)

torch.fx.wrap("fused_rmsnorm_rope")
os.environ.setdefault("DSV4_INDEXED_ROPE_CUDA", "1")


def _freqs(max_pos: int, rd: int) -> torch.Tensor:
    return precompute_freqs_cis(
        dim=rd,
        seqlen=max_pos,
        original_seq_len=4096,
        base=10000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
    ).to("cuda")


class _QPattern(torch.nn.Module):
    def forward(self, q, freqs, position_ids):
        selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
        return fused_rmsnorm_rope(q, None, selected, 64, eps=1e-6)


class _KVPattern(torch.nn.Module):
    def forward(self, kv, weight, freqs, position_ids):
        selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
        return fused_rmsnorm_rope(kv, weight, selected, 64, eps=1e-6)


def _fused_graph(module: torch.nn.Module) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(module)
    gm = apply_indexed_rope_fx_pass(gm)
    gm.recompile()
    return gm


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSV4IndexedRmsnormRopePass(unittest.TestCase):
    def test_graphfx_rewrite_correctness(self) -> None:
        q_pair = make_fx_pair(
            _QPattern,
            apply_indexed_rope_fx_pass,
            required_targets=("dsv4_indexed_rmsnorm_rope",),
            forbidden_targets=("fused_rmsnorm_rope",),
        )
        kv_pair = make_fx_pair(
            _KVPattern,
            apply_indexed_rope_fx_pass,
            required_targets=("dsv4_indexed_rmsnorm_rope",),
            forbidden_targets=("fused_rmsnorm_rope",),
        )
        h, rd = 64, 64
        freqs = _freqs(8192, rd)
        for m in DSV4_GRAPHFX_CORRECTNESS_M:
            for d in (128, 512):
                torch.manual_seed(3000 + m + d)
                q = (torch.randn(m, 1, h, d, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
                pos = torch.randint(0, 8192, (m,), dtype=torch.int32, device="cuda")
                ref = q_pair.baseline(q, freqs, pos)
                cand = q_pair.candidate(q, freqs, pos)
                assert_bf16_close(ref, cand, f"indexed_q_rmsnorm_rope_M{m}_D{d}", atol=5e-2 if d == 512 else 2e-2)
            torch.manual_seed(3100 + m)
            kv = (torch.randn(m, 1, 512, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
            weight = (torch.randn(512, dtype=torch.bfloat16, device="cuda").abs() + 0.25).contiguous()
            pos = torch.randint(0, 8192, (m,), dtype=torch.int32, device="cuda")
            ref = kv_pair.baseline(kv, weight, freqs, pos)
            cand = kv_pair.candidate(kv, weight, freqs, pos)
            assert_bf16_close(ref, cand, f"indexed_kv_rmsnorm_rope_M{m}", atol=5e-2)

    def test_graphfx_rewrite_perf(self) -> None:
        if not graphfx_perf_enabled():
            self.skipTest("set DSV4_GRAPHFX_RUN_PERF_IN_UT=1 or PERF_JSON to run GraphFX perf")
        m_list = graphfx_m_sweep("DSV4_GRAPHFX_INDEXED_RMSNORM_ROPE_M_LIST")
        trace_dir = trace_dir_for_report(
            "DSV4_GRAPHFX_INDEXED_RMSNORM_ROPE_JSON",
            "dsv4_graphfx_indexed_rmsnorm_rope_perf.json",
            "DSV4_GRAPHFX_INDEXED_RMSNORM_ROPE_TRACE_DIR",
        )
        rows = []
        q_pair = make_fx_pair(
            _QPattern,
            apply_indexed_rope_fx_pass,
            required_targets=("dsv4_indexed_rmsnorm_rope",),
            forbidden_targets=("fused_rmsnorm_rope",),
        )
        kv_pair = make_fx_pair(
            _KVPattern,
            apply_indexed_rope_fx_pass,
            required_targets=("dsv4_indexed_rmsnorm_rope",),
            forbidden_targets=("fused_rmsnorm_rope",),
        )
        max_m = max(m_list)
        freqs = _freqs(max_m + 4096, 64)
        for m in m_list:
            pos = torch.randint(0, int(freqs.shape[0]), (m,), dtype=torch.int32, device="cuda")
            for shape_case in DSV4_ROPE_SHAPES:
                if shape_case["role"].startswith("q_"):
                    h = shape_case["H"]
                    d = shape_case["D"]
                    q = (torch.randn(m, 1, h, d, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
                    path = indexed_rmsnorm_rope_path(q, None, 64)
                    rows.append(
                        measured_graph_pair_row(
                            op="indexed_rmsnorm_rope",
                            label=f"indexed_rmsnorm_rope_{shape_case['role']}_M{m}",
                            shape_meta={**shape_case, "M": m, "candidate_runtime_path": path},
                            baseline_fn=lambda q=q, pos=pos: q_pair.baseline(q, freqs, pos),
                            candidate_fn=lambda q=q, pos=pos: q_pair.candidate(q, freqs, pos),
                            trace_dir=trace_dir,
                            kernel_regex=os.environ.get("DSV4_GRAPHFX_INDEXED_RMSNORM_ROPE_KERNEL_REGEX") or None,
                        )
                    )
                else:
                    kv = (torch.randn(m, 1, 512, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
                    weight = (torch.randn(512, dtype=torch.bfloat16, device="cuda").abs() + 0.25).contiguous()
                    path = indexed_rmsnorm_rope_path(kv, weight, 64)
                    rows.append(
                        measured_graph_pair_row(
                            op="indexed_rmsnorm_rope",
                            label=f"indexed_rmsnorm_rope_{shape_case['role']}_M{m}",
                            shape_meta={**shape_case, "M": m, "candidate_runtime_path": path},
                            baseline_fn=lambda kv=kv, weight=weight, pos=pos: kv_pair.baseline(kv, weight, freqs, pos),
                            candidate_fn=lambda kv=kv, weight=weight, pos=pos: kv_pair.candidate(kv, weight, freqs, pos),
                            trace_dir=trace_dir,
                            kernel_regex=os.environ.get("DSV4_GRAPHFX_INDEXED_RMSNORM_ROPE_KERNEL_REGEX") or None,
                        )
                    )
        path = write_graphfx_perf_report(
            json_env="DSV4_GRAPHFX_INDEXED_RMSNORM_ROPE_JSON",
            default_json="dsv4_graphfx_indexed_rmsnorm_rope_perf.json",
            rows=rows,
            metadata={
                "title": "DSV4 GraphFX Indexed RMSNorm RoPE Perf",
                "baseline_path": "original FX graph: freqs gather -> fused_rmsnorm_rope",
                "candidate_path": "GraphFX rewritten FX graph: dsv4_indexed_rmsnorm_rope",
                "m_list": m_list,
                "shape_cases": DSV4_ROPE_SHAPES,
            },
        )
        print(f"Wrote GraphFX indexed RMSNorm RoPE perf report: {path}")

    def test_pass_rewrites_q_pattern(self) -> None:
        gm = _fused_graph(_QPattern())
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("dsv4_indexed_rmsnorm_rope", targets)
        self.assertNotIn("fused_rmsnorm_rope", targets)

    def test_indexed_q_rmsnorm_rope_matches_materialized(self) -> None:
        gm = _fused_graph(_QPattern())
        h, rd = 64, 64
        for d in [128, 512]:
            for m in [1, 3, 8, 17, 32, 64, 257]:
                for dtype in [torch.int32, torch.int64]:
                    torch.manual_seed(m + d)
                    q = torch.randn(m, 1, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
                    freqs = _freqs(2048, rd)
                    position_ids = torch.randint(0, 2048, (m,), dtype=dtype, device="cuda")

                    ref = _QPattern()(q, freqs, position_ids)
                    cand = gm(q, freqs, position_ids)
                    diff = (cand.float() - ref.float()).abs()
                    self.assertLessEqual(float(diff.max()), 5e-2 if d == 512 else 2e-2)

    def test_prefill_shapes_select_indexed_path(self) -> None:
        h, rd = 64, 64
        for d in [128, 512]:
            q = torch.empty(1, 17, h, d, dtype=torch.bfloat16, device="cuda")
            self.assertTrue(indexed_rmsnorm_rope_path(q, None, rd).startswith("indexed"))

        kv = torch.empty(1, 17, 512, dtype=torch.bfloat16, device="cuda")
        weight = torch.empty(512, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(indexed_rmsnorm_rope_path(kv, weight, rd).startswith("indexed"))

    def test_indexed_kv_rmsnorm_rope_matches_materialized(self) -> None:
        gm = _fused_graph(_KVPattern())
        d, rd = 512, 64
        for m in [1, 5, 32, 127]:
            torch.manual_seed(100 + m)
            kv = torch.randn(m, 1, d, dtype=torch.bfloat16, device="cuda") * 0.5
            weight = torch.randn(d, dtype=torch.bfloat16, device="cuda").abs() + 0.25
            freqs = _freqs(4096, rd)
            position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")

            ref = _KVPattern()(kv, weight, freqs, position_ids)
            cand = gm(kv, weight, freqs, position_ids)
            diff = (cand.float() - ref.float()).abs()
            self.assertLessEqual(float(diff.max()), 5e-2)

    def test_cross_graph_q_token_path_matches_materialized(self) -> None:
        h, rd = 64, 64
        for d in [128, 512]:
            for m in [1, 8, 33, 257]:
                torch.manual_seed(700 + m + d)
                q = torch.randn(1, m, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
                freqs = _freqs(4096, rd)
                position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")
                selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
                token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

                ref = fused_rmsnorm_rope(q, None, selected, rd, eps=1e-6)
                cand = dsv4_indexed_rmsnorm_rope_from_freqs(q, None, token, rd, eps=1e-6)
                diff = (cand.float() - ref.float()).abs()
                self.assertLessEqual(float(diff.max()), 5e-2 if d == 512 else 2e-2)

    def test_unfused_rmsnorm_rope_consumer_rejects_token(self) -> None:
        h, rd, d, m = 64, 64, 128, 8
        q = torch.randn(1, m, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
        freqs = _freqs(4096, rd)
        position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")
        token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

        self.assertEqual(token.dtype, torch.int8)
        with self.assertRaisesRegex(RuntimeError, "view_as_real|complex"):
            fused_rmsnorm_rope(q, None, token, rd, eps=1e-6)

    def test_from_freqs_rejects_poison_token_row_mismatch(self) -> None:
        h, rd, d = 64, 64, 128
        q = torch.randn(1, 5, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
        freqs = _freqs(4096, rd)
        position_ids = torch.randint(0, 4096, (4,), dtype=torch.int32, device="cuda")
        token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

        with self.assertRaisesRegex(ValueError, "unsupported or unfused consumer"):
            dsv4_indexed_rmsnorm_rope_from_freqs(q, None, token, rd, eps=1e-6)

    def test_cross_graph_flat_q_token_path_matches_materialized(self) -> None:
        h, rd = 64, 64
        for d in [128, 512]:
            for m in [1, 8, 33]:
                torch.manual_seed(710 + m + d)
                q = torch.randn(m * h, d, dtype=torch.bfloat16, device="cuda") * 0.5
                freqs = _freqs(4096, rd)
                position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")
                selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
                token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

                ref = fused_rmsnorm_rope(q, None, selected, rd, eps=1e-6)
                cand = dsv4_indexed_rmsnorm_rope_from_freqs(q, None, token, rd, eps=1e-6)
                diff = (cand.float() - ref.float()).abs()
                self.assertEqual(cand.shape, q.shape)
                self.assertLessEqual(float(diff.max()), 5e-2 if d == 512 else 2e-2)

    def test_cross_graph_3d_q_token_path_matches_materialized(self) -> None:
        h, rd = 64, 64
        for d in [128, 512]:
            for m in [1, 8, 33]:
                torch.manual_seed(720 + m + d)
                q = torch.randn(m, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
                freqs = _freqs(4096, rd)
                position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")
                selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
                token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

                ref = fused_rmsnorm_rope(q, None, selected, rd, eps=1e-6)
                cand = dsv4_indexed_rmsnorm_rope_from_freqs(q, None, token, rd, eps=1e-6)
                diff = (cand.float() - ref.float()).abs()
                self.assertEqual(cand.shape, q.shape)
                self.assertLessEqual(float(diff.max()), 5e-2 if d == 512 else 2e-2)

    def test_from_freqs_q_materialized_path_matches_without_token_provenance(self) -> None:
        h, rd = 64, 64
        for d in [128, 512]:
            torch.manual_seed(701 + d)
            q = torch.randn(1, 17, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
            freqs = _freqs(4096, rd)
            position_ids = torch.randint(0, 4096, (17,), dtype=torch.int32, device="cuda")
            selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()

            ref = fused_rmsnorm_rope(q, None, selected, rd, eps=1e-6)
            cand = dsv4_indexed_rmsnorm_rope_from_freqs(q, None, selected, rd, eps=1e-6)
            diff = (cand.float() - ref.float()).abs()
            self.assertEqual(cand.shape, ref.shape)
            self.assertLessEqual(float(diff.max()), 5e-2 if d == 512 else 2e-2)

    def test_cross_graph_kv_token_path_matches_materialized(self) -> None:
        d, rd = 512, 64
        for m in [1, 5, 32, 127]:
            torch.manual_seed(800 + m)
            kv = torch.randn(m, 1, d, dtype=torch.bfloat16, device="cuda") * 0.5
            weight = torch.randn(d, dtype=torch.bfloat16, device="cuda").abs() + 0.25
            freqs = _freqs(4096, rd)
            position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")
            selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
            token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

            ref = fused_rmsnorm_rope(kv, weight, selected, rd, eps=1e-6)
            cand = dsv4_indexed_rmsnorm_rope_from_freqs(kv, weight, token, rd, eps=1e-6)
            diff = (cand.float() - ref.float()).abs()
            self.assertLessEqual(float(diff.max()), 5e-2)

    def test_from_freqs_kv_materialized_path_matches_without_token_provenance(self) -> None:
        d, rd = 512, 64
        torch.manual_seed(801)
        kv = torch.randn(17, 1, d, dtype=torch.bfloat16, device="cuda") * 0.5
        weight = torch.randn(d, dtype=torch.bfloat16, device="cuda").abs() + 0.25
        freqs = _freqs(4096, rd)
        position_ids = torch.randint(0, 4096, (17,), dtype=torch.int32, device="cuda")
        selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()

        ref = fused_rmsnorm_rope(kv, weight, selected, rd, eps=1e-6)
        cand = dsv4_indexed_rmsnorm_rope_from_freqs(kv, weight, selected, rd, eps=1e-6)
        diff = (cand.float() - ref.float()).abs()
        self.assertEqual(cand.shape, ref.shape)
        self.assertLessEqual(float(diff.max()), 5e-2)

    def test_pass_generated_kernel_rejects_bad_shape(self) -> None:
        gm = _fused_graph(_QPattern())
        x = torch.randn(2, 3, 127, dtype=torch.bfloat16, device="cuda")
        freqs = _freqs(16, 64)
        pos = torch.arange(2, dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "unsupported dsv4_indexed_rmsnorm_rope"):
            gm(x, freqs, pos)

    def test_pass_generated_kernel_strict_position_oob(self) -> None:
        gm = _fused_graph(_QPattern())
        old = os.environ.get("DSV4_INDEXED_ROPE_STRICT")
        os.environ["DSV4_INDEXED_ROPE_STRICT"] = "1"
        try:
            x = torch.randn(2, 1, 128, dtype=torch.bfloat16, device="cuda")
            freqs = _freqs(4, 64)
            pos = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
            with self.assertRaisesRegex(ValueError, "position_ids out of range"):
                gm(x, freqs, pos)
        finally:
            if old is None:
                os.environ.pop("DSV4_INDEXED_ROPE_STRICT", None)
            else:
                os.environ["DSV4_INDEXED_ROPE_STRICT"] = old


if __name__ == "__main__":
    unittest.main()
