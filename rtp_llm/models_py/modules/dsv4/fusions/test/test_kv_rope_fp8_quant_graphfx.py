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
    DSV4_GRAPHFX_CORRECTNESS_M,
    assert_fp8_quant_close,
    graphfx_m_sweep,
    graphfx_perf_enabled,
    make_fx_pair,
    measured_graph_pair_row,
    target_names,
    trace_dir_for_report,
    write_graphfx_perf_report,
)

torch.fx.wrap("sgl_per_token_group_quant_fp8")


DSV4_KV_ROPE_QUANT_SHAPES = [
    {
        "model_profile": "flash",
        "shape_group": "main",
        "role": "kv_rope_indexer",
        "K": 128,
        "head_dim": 128,
    },
    {
        "model_profile": "flash",
        "shape_group": "main",
        "role": "kv_rope_attention",
        "K": 512,
        "head_dim": 512,
    },
]


class _ScopedEnv:
    def __init__(self, **values: str):
        self._values = values
        self._old: dict[str, str | None] = {}

    def __enter__(self):
        self._old = {key: os.environ.get(key) for key in self._values}
        for key, value in self._values.items():
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= int(size)
    return tuple(reversed(strides))


def _materialize_for_fp8_assert(tensor: torch.Tensor) -> torch.Tensor:
    shape = tuple(int(dim) for dim in tensor.shape)
    out = torch.empty_strided(
        shape,
        _contiguous_strides(shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    out.copy_(tensor)
    return out


def _active_ue8m0_scale_bytes(scale: torch.Tensor, k: int) -> torch.Tensor:
    group_count = k // 128
    materialized = _materialize_for_fp8_assert(scale)
    byte_view = materialized.view(torch.uint8)
    byte_view = byte_view.reshape(
        *materialized.shape[:-1],
        materialized.shape[-1] * materialized.element_size(),
    )
    return byte_view[..., :group_count].contiguous()


def _assert_kv_rope_fp8_quant_close(
    ref: tuple[torch.Tensor, torch.Tensor],
    cand: tuple[torch.Tensor, torch.Tensor],
    label: str,
    k: int,
) -> None:
    q_ref, s_ref = ref
    q_cand, s_cand = cand
    assert_fp8_quant_close(
        (
            _materialize_for_fp8_assert(q_ref),
            _active_ue8m0_scale_bytes(s_ref, k),
        ),
        (
            _materialize_for_fp8_assert(q_cand),
            _active_ue8m0_scale_bytes(s_cand, k),
        ),
        label,
        s_exact_min=1.0,
        s_max_byte=0,
    )


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
            required_targets=(
                "dsv4_kv_rope_quant_producer_token",
                "dsv4_kv_rope_fp8_quant_from_provenance",
            ),
            forbidden_targets=("sgl_per_token_group_quant_fp8",),
        )
        for shape_case in DSV4_KV_ROPE_QUANT_SHAPES:
            k = shape_case["K"]
            for m in DSV4_GRAPHFX_CORRECTNESS_M:
                torch.manual_seed(8100 + m + k)
                x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
                ref = pair.baseline(x)
                cand = pair.candidate(x)
                torch.cuda.synchronize()
                _assert_kv_rope_fp8_quant_close(ref, cand, f"kv_rope_quant_graphfx_M{m}_K{k}", k)

    def test_graphfx_precompute_reuses_provenance_when_required(self):
        pair = make_fx_pair(
            _KvRopeQuantConsumerPattern,
            apply_kv_rope_fp8_quant_fx_pass,
            required_targets=(
                "dsv4_kv_rope_quant_producer_token",
                "dsv4_kv_rope_fp8_quant_from_provenance",
            ),
            forbidden_targets=("sgl_per_token_group_quant_fp8",),
        )
        with _ScopedEnv(
            DSV4_KV_ROPE_QUANT_PRECOMPUTE_FP8="1",
            DSV4_KV_ROPE_QUANT_REQUIRE_PROVENANCE="1",
        ):
            for shape_case in DSV4_KV_ROPE_QUANT_SHAPES:
                k = shape_case["K"]
                for m in (1, 17, 257):
                    torch.manual_seed(8200 + m + k)
                    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
                    ref = pair.baseline(x)
                    cand = pair.candidate(x)
                    torch.cuda.synchronize()
                    _assert_kv_rope_fp8_quant_close(ref, cand, f"kv_rope_quant_precompute_M{m}_K{k}", k)

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
        if not graphfx_perf_enabled():
            self.skipTest("set DSV4_GRAPHFX_RUN_PERF_IN_UT=1 or PERF_JSON to run GraphFX perf")
        m_list = graphfx_m_sweep("DSV4_GRAPHFX_KV_ROPE_FP8_M_LIST")
        trace_dir = trace_dir_for_report(
            "DSV4_GRAPHFX_KV_ROPE_FP8_JSON",
            "dsv4_graphfx_kv_rope_fp8_quant_perf.json",
            "DSV4_GRAPHFX_KV_ROPE_FP8_TRACE_DIR",
        )
        pair = make_fx_pair(
            _KvRopeQuantConsumerPattern,
            apply_kv_rope_fp8_quant_fx_pass,
            required_targets=(
                "dsv4_kv_rope_quant_producer_token",
                "dsv4_kv_rope_fp8_quant_from_provenance",
            ),
            forbidden_targets=("sgl_per_token_group_quant_fp8",),
        )
        rows = []
        with _ScopedEnv(DSV4_KV_ROPE_QUANT_PRECOMPUTE_FP8="1"):
            for shape_case in DSV4_KV_ROPE_QUANT_SHAPES:
                k = shape_case["K"]
                for m in m_list:
                    torch.manual_seed(8700 + m + k)
                    x = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
                    rows.append(
                        measured_graph_pair_row(
                            op="kv_rope_fp8_quant",
                            label=f"kv_rope_fp8_M{m}_K{k}",
                            shape_meta={**shape_case, "M": m},
                            baseline_fn=lambda x=x: pair.baseline(x),
                            candidate_fn=lambda x=x: pair.candidate(x),
                            trace_dir=trace_dir,
                            kernel_regex=os.environ.get("DSV4_GRAPHFX_KV_ROPE_FP8_KERNEL_REGEX") or None,
                            warmup=20 if m <= 4096 else 8,
                        )
                    )
        path = write_graphfx_perf_report(
            json_env="DSV4_GRAPHFX_KV_ROPE_FP8_JSON",
            default_json="dsv4_graphfx_kv_rope_fp8_quant_perf.json",
            rows=rows,
            metadata={
                "title": "DSV4 GraphFX KV RoPE FP8 Quant Perf",
                "baseline_path": "original FX graph: kv_rope producer -> sgl_per_token_group_quant_fp8",
                "candidate_path": "GraphFX rewritten FX graph: kv_rope producer token -> provenance FP8 quant",
                "m_list": m_list,
                "shape_cases": DSV4_KV_ROPE_QUANT_SHAPES,
            },
        )
        print(f"Wrote GraphFX KV RoPE FP8 quant perf report: {path}")


if __name__ == "__main__":
    unittest.main()
