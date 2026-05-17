from __future__ import annotations

import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.fusions.indexed_rope_pass import (
    apply_indexed_rope_fx_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.indexed_rope_runtime import (
    dsv4_indexed_inv_rope_fp8_quant_from_freqs,
    dsv4_indexed_rope_freqs_token,
)
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.fusions.test.graphfx_fusion_test_utils import (
    DSV4_GRAPHFX_CORRECTNESS_M,
    graphfx_m_sweep,
    graphfx_perf_enabled,
    make_fx_pair,
    measured_graph_pair_row,
    trace_dir_for_report,
    write_graphfx_perf_report,
)

torch.fx.wrap("fused_inv_rope_fp8_quant")
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


def _dequant(fp8: torch.Tensor, scale: torch.Tensor, head_dim: int) -> torch.Tensor:
    m, g, d_per_group = fp8.shape
    heads_per_group = scale.shape[-1]
    chunks = head_dim // 128
    out = fp8.float().view(m, g, heads_per_group, head_dim)
    bytes_view = scale.to(torch.int32)
    scales = []
    for chunk in range(chunks):
        exp = ((bytes_view >> (chunk * 8)) & 0xFF).float()
        scales.append(torch.exp2(exp - 127.0))
    scale_full = torch.stack(scales, dim=-1).repeat_interleave(128, dim=-1)
    return (out * scale_full).reshape(m, g, d_per_group)


class _InvRopeQuantPattern(torch.nn.Module):
    def __init__(self, n_groups: int, heads_per_group: int, nope_dim: int, rope_head_dim: int):
        super().__init__()
        self.n_groups = n_groups
        self.heads_per_group = heads_per_group
        self.nope_dim = nope_dim
        self.rope_head_dim = rope_head_dim

    def forward(self, o, freqs, position_ids):
        selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
        return fused_inv_rope_fp8_quant(
            o,
            selected,
            n_groups=self.n_groups,
            heads_per_group=self.heads_per_group,
            nope_dim=self.nope_dim,
            rope_head_dim=self.rope_head_dim,
            impl="optimized",
        )


def _fused_graph(module: torch.nn.Module) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(module)
    gm = apply_indexed_rope_fx_pass(gm)
    gm.recompile()
    return gm


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSV4IndexedInvRopeFp8QuantPass(unittest.TestCase):
    def test_graphfx_rewrite_correctness(self) -> None:
        h, g, d, rd = 64, 8, 512, 64
        pair = make_fx_pair(
            lambda: _InvRopeQuantPattern(g, h // g, d - rd, rd),
            apply_indexed_rope_fx_pass,
            required_targets=("dsv4_indexed_inv_rope_fp8_quant",),
            forbidden_targets=("fused_inv_rope_fp8_quant",),
        )
        freqs = _freqs(8192, rd)
        for m in DSV4_GRAPHFX_CORRECTNESS_M:
            for dtype in (torch.int32, torch.int64):
                torch.manual_seed(4000 + m + (0 if dtype is torch.int32 else 1))
                o = (torch.randn(m, h, d, dtype=torch.bfloat16, device="cuda") * 0.25).contiguous()
                pos = torch.randint(0, 8192, (m,), dtype=dtype, device="cuda")
                ref_q, ref_s = pair.baseline(o, freqs, pos)
                cand_q, cand_s = pair.candidate(o, freqs, pos)
                diff = (_dequant(cand_q, cand_s, d) - _dequant(ref_q, ref_s, d)).abs()
                self.assertLessEqual(float(diff.max()), 1.0)
                self.assertLessEqual(float(diff.mean()), 2e-2)

    def test_graphfx_rewrite_perf(self) -> None:
        if not graphfx_perf_enabled():
            self.skipTest("set DSV4_GRAPHFX_RUN_PERF_IN_UT=1 or PERF_JSON to run GraphFX perf")
        h, g, d, rd = 64, 8, 512, 64
        m_list = graphfx_m_sweep("DSV4_GRAPHFX_INDEXED_INV_ROPE_FP8_M_LIST")
        trace_dir = trace_dir_for_report(
            "DSV4_GRAPHFX_INDEXED_INV_ROPE_FP8_JSON",
            "dsv4_graphfx_indexed_inv_rope_fp8_quant_perf.json",
            "DSV4_GRAPHFX_INDEXED_INV_ROPE_FP8_TRACE_DIR",
        )
        pair = make_fx_pair(
            lambda: _InvRopeQuantPattern(g, h // g, d - rd, rd),
            apply_indexed_rope_fx_pass,
            required_targets=("dsv4_indexed_inv_rope_fp8_quant",),
            forbidden_targets=("fused_inv_rope_fp8_quant",),
        )
        freqs = _freqs(max(m_list) + 4096, rd)
        rows = []
        for m in m_list:
            torch.manual_seed(4700 + m)
            o = (torch.randn(m, h, d, dtype=torch.bfloat16, device="cuda") * 0.25).contiguous()
            pos = torch.randint(0, int(freqs.shape[0]), (m,), dtype=torch.int32, device="cuda")
            rows.append(
                measured_graph_pair_row(
                    op="indexed_inv_rope_fp8_quant",
                    label=f"indexed_inv_rope_fp8_quant_M{m}",
                    shape_meta={
                        "model_profile": "flash",
                        "role": "o_projection_inv_rope_quant",
                        "M": m,
                        "H": h,
                        "G": g,
                        "D": d,
                        "head_dim": d,
                        "rope_dim": rd,
                        "nope_dim": d - rd,
                        "downstream_N": [4096, 7168],
                    },
                    baseline_fn=lambda o=o, pos=pos: pair.baseline(o, freqs, pos),
                    candidate_fn=lambda o=o, pos=pos: pair.candidate(o, freqs, pos),
                    trace_dir=trace_dir,
                    kernel_regex=os.environ.get("DSV4_GRAPHFX_INDEXED_INV_ROPE_FP8_KERNEL_REGEX") or None,
                )
            )
        path = write_graphfx_perf_report(
            json_env="DSV4_GRAPHFX_INDEXED_INV_ROPE_FP8_JSON",
            default_json="dsv4_graphfx_indexed_inv_rope_fp8_quant_perf.json",
            rows=rows,
            metadata={
                "title": "DSV4 GraphFX Indexed InvRoPE FP8 Quant Perf",
                "baseline_path": "original FX graph: freqs gather -> fused_inv_rope_fp8_quant",
                "candidate_path": "GraphFX rewritten FX graph: dsv4_indexed_inv_rope_fp8_quant",
                "m_list": m_list,
            },
        )
        print(f"Wrote GraphFX indexed InvRoPE FP8 quant perf report: {path}")

    def test_pass_rewrites_pattern(self) -> None:
        gm = _fused_graph(_InvRopeQuantPattern(8, 8, 448, 64))
        targets = [getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes]
        self.assertIn("dsv4_indexed_inv_rope_fp8_quant", targets)
        self.assertNotIn("fused_inv_rope_fp8_quant", targets)

    def test_indexed_inv_rope_fp8_quant_matches_materialized(self) -> None:
        h, g, d, rd = 64, 8, 512, 64
        hpg = h // g
        module = _InvRopeQuantPattern(g, hpg, d - rd, rd)
        gm = _fused_graph(module)
        for m in [1, 3, 8, 31, 64]:
            for dtype in [torch.int32, torch.int64]:
                torch.manual_seed(200 + m)
                o = torch.randn(m, h, d, dtype=torch.bfloat16, device="cuda") * 0.25
                freqs = _freqs(4096, rd)
                position_ids = torch.randint(0, 4096, (m,), dtype=dtype, device="cuda")

                ref_q, ref_s = module(o, freqs, position_ids)
                cand_q, cand_s = gm(o, freqs, position_ids)
                diff = (_dequant(cand_q, cand_s, d) - _dequant(ref_q, ref_s, d)).abs()
                self.assertLessEqual(float(diff.max()), 1.0)
                self.assertLessEqual(float(diff.mean()), 2e-2)

    def test_indexed_inv_rope_fp8_quant_compact_shape_matches_materialized(self) -> None:
        h, g, d, rd = 16, 4, 128, 64
        hpg = h // g
        module = _InvRopeQuantPattern(g, hpg, d - rd, rd)
        gm = _fused_graph(module)
        for m in [1, 17, 257]:
            torch.manual_seed(500 + m)
            o = torch.randn(m, h, d, dtype=torch.bfloat16, device="cuda") * 0.25
            freqs = _freqs(4096, rd)
            position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")

            ref_q, ref_s = module(o, freqs, position_ids)
            cand_q, cand_s = gm(o, freqs, position_ids)
            diff = (_dequant(cand_q, cand_s, d) - _dequant(ref_q, ref_s, d)).abs()
            self.assertLessEqual(float(diff.max()), 1.0)
            self.assertLessEqual(float(diff.mean()), 2e-2)

    def test_cross_graph_token_path_matches_materialized(self) -> None:
        h, g, d, rd = 64, 8, 512, 64
        hpg = h // g
        for m in [1, 3, 8, 31, 64, 257]:
            torch.manual_seed(900 + m)
            o = torch.randn(m, h, d, dtype=torch.bfloat16, device="cuda") * 0.25
            freqs = _freqs(4096, rd)
            position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")
            selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
            token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

            ref_q, ref_s = fused_inv_rope_fp8_quant(
                o,
                selected,
                n_groups=g,
                heads_per_group=hpg,
                nope_dim=d - rd,
                rope_head_dim=rd,
                impl="optimized",
            )
            cand_q, cand_s = dsv4_indexed_inv_rope_fp8_quant_from_freqs(
                o,
                token,
                n_groups=g,
                heads_per_group=hpg,
                nope_dim=d - rd,
                rope_head_dim=rd,
            )
            diff = (_dequant(cand_q, cand_s, d) - _dequant(ref_q, ref_s, d)).abs()
            self.assertLessEqual(float(diff.max()), 1.0)
            self.assertLessEqual(float(diff.mean()), 2e-2)

    def test_unfused_inv_rope_quant_consumer_rejects_token(self) -> None:
        h, g, d, rd, m = 64, 8, 512, 64, 8
        hpg = h // g
        o = torch.randn(m, h, d, dtype=torch.bfloat16, device="cuda") * 0.25
        freqs = _freqs(4096, rd)
        position_ids = torch.randint(0, 4096, (m,), dtype=torch.int32, device="cuda")
        token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

        self.assertEqual(token.dtype, torch.int8)
        with self.assertRaises(AssertionError):
            fused_inv_rope_fp8_quant(
                o,
                token,
                n_groups=g,
                heads_per_group=hpg,
                nope_dim=d - rd,
                rope_head_dim=rd,
                impl="optimized",
            )

    def test_from_freqs_rejects_poison_token_row_mismatch(self) -> None:
        h, g, d, rd = 64, 8, 512, 64
        hpg = h // g
        o = torch.randn(5, h, d, dtype=torch.bfloat16, device="cuda") * 0.25
        freqs = _freqs(4096, rd)
        position_ids = torch.randint(0, 4096, (4,), dtype=torch.int32, device="cuda")
        token = dsv4_indexed_rope_freqs_token(freqs, position_ids)

        with self.assertRaisesRegex(ValueError, "unsupported or unfused consumer"):
            dsv4_indexed_inv_rope_fp8_quant_from_freqs(
                o,
                token,
                n_groups=g,
                heads_per_group=hpg,
                nope_dim=d - rd,
                rope_head_dim=rd,
            )

    def test_from_freqs_falls_back_without_token_provenance(self) -> None:
        h, g, d, rd = 64, 8, 512, 64
        hpg = h // g
        torch.manual_seed(901)
        o = torch.randn(17, h, d, dtype=torch.bfloat16, device="cuda") * 0.25
        freqs = _freqs(4096, rd)
        position_ids = torch.randint(0, 4096, (17,), dtype=torch.int32, device="cuda")
        selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()

        ref_q, ref_s = fused_inv_rope_fp8_quant(
            o,
            selected,
            n_groups=g,
            heads_per_group=hpg,
            nope_dim=d - rd,
            rope_head_dim=rd,
            impl="optimized",
        )
        cand_q, cand_s = dsv4_indexed_inv_rope_fp8_quant_from_freqs(
            o,
            selected,
            n_groups=g,
            heads_per_group=hpg,
            nope_dim=d - rd,
            rope_head_dim=rd,
        )
        diff = (_dequant(cand_q, cand_s, d) - _dequant(ref_q, ref_s, d)).abs()
        self.assertEqual(cand_q.shape, ref_q.shape)
        self.assertEqual(cand_s.shape, ref_s.shape)
        self.assertLessEqual(float(diff.max()), 0.0)

    def test_pass_generated_kernel_rejects_bad_layout(self) -> None:
        gm = _fused_graph(_InvRopeQuantPattern(4, 4, 64, 64))
        o = torch.randn(2, 16, 128, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
        freqs = _freqs(16, 64)
        pos = torch.arange(16, dtype=torch.int32, device="cuda")
        with self.assertRaisesRegex(ValueError, "unsupported dsv4_indexed_inv_rope_fp8_quant"):
            gm(o, freqs, pos)


if __name__ == "__main__":
    unittest.main()
