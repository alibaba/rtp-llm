"""FX-pass-level tests for ``apply_rmsnorm_gated_fp8_quant_fx_pass``."""

from __future__ import annotations

import operator
import unittest

import torch
from torch import fx

from rtp_llm.models_py.modules.fuse_kernel_fx.rmsnorm_gated_fp8_quant_pass import (
    apply_rmsnorm_gated_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.fuse_kernel_fx.test.graphfx_fusion_test_utils import (
    make_dummy_tensor_meta,
    target_names,
)


def layer_norm_fwd(x, weight, bias, eps, **kwargs):
    return x, None, None


def sgl_per_token_group_quant_fp8(
    x,
    *,
    group_size=128,
    eps=1e-4,
    column_major_scales=True,
    scale_tma_aligned=True,
    scale_ue8m0=True,
    fuse_silu_and_mul=False,
    masked_m=None,
):
    return x, x


def _build_graph(
    *,
    last_dim: int = 128,
    is_rms_norm: bool = True,
    norm_before_gate: bool = True,
    num_heads: int | None = 8,
    has_z: bool = True,
):
    graph = fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    bias = graph.placeholder("bias")
    z = graph.placeholder("z")
    x.meta["tensor_meta"] = make_dummy_tensor_meta((32, last_dim))
    weight.meta["tensor_meta"] = make_dummy_tensor_meta((last_dim,))
    z.meta["tensor_meta"] = make_dummy_tensor_meta((32, last_dim))

    layer_kwargs = {
        "is_rms_norm": is_rms_norm,
        "norm_before_gate": norm_before_gate,
    }
    if has_z:
        layer_kwargs["z"] = z
    if num_heads is not None:
        layer_kwargs["num_heads"] = num_heads
    fused = graph.call_function(
        layer_norm_fwd, args=(x, weight, bias, 1e-6), kwargs=layer_kwargs
    )
    out = graph.call_function(operator.getitem, args=(fused, 0))
    out.meta["tensor_meta"] = make_dummy_tensor_meta((32, last_dim))
    quant = graph.call_function(
        sgl_per_token_group_quant_fp8,
        args=(out,),
        kwargs={
            "group_size": 128,
            "eps": 1e-4,
            "column_major_scales": True,
            "scale_tma_aligned": True,
            "scale_ue8m0": True,
            "fuse_silu_and_mul": False,
            "masked_m": None,
        },
    )
    fp8 = graph.call_function(operator.getitem, args=(quant, 0))
    scale = graph.call_function(operator.getitem, args=(quant, 1))
    graph.output((fp8, scale))
    return fx.GraphModule(torch.nn.Module(), graph)


class RmsNormGatedFp8QuantPassTest(unittest.TestCase):
    def test_replaces_rmsnorm_gated_then_quant(self):
        gm = _build_graph()
        apply_rmsnorm_gated_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("fused_rmsnorm_gated_fp8_quant", names)
        self.assertNotIn("layer_norm_fwd", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)

    def test_skips_when_not_rms_norm(self):
        gm = _build_graph(is_rms_norm=False)
        apply_rmsnorm_gated_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("layer_norm_fwd", names)
        self.assertIn("sgl_per_token_group_quant_fp8", names)

    def test_skips_when_no_gate(self):
        gm = _build_graph(has_z=False)
        apply_rmsnorm_gated_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("layer_norm_fwd", names)

    def test_skips_when_num_heads_missing(self):
        gm = _build_graph(num_heads=None)
        apply_rmsnorm_gated_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("layer_norm_fwd", names)
        self.assertNotIn("fused_rmsnorm_gated_fp8_quant", names)


if __name__ == "__main__":
    unittest.main()
