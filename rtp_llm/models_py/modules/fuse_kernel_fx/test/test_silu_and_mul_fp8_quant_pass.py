"""FX-pass-level tests for ``apply_silu_and_mul_fp8_quant_fx_pass``."""

from __future__ import annotations

import operator
import unittest

import torch
from torch import fx

from rtp_llm.models_py.modules.fuse_kernel_fx.silu_and_mul_fp8_quant_pass import (
    apply_silu_and_mul_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.fuse_kernel_fx.test.graphfx_fusion_test_utils import (
    make_dummy_tensor_meta,
    target_names,
)


def silu_and_mul(up):
    return up


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


def _build_graph(*, last_dim: int, quant_kwargs: dict | None = None):
    graph = fx.Graph()
    up = graph.placeholder("up")
    up.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))
    activated = graph.call_function(silu_and_mul, args=(up,))
    activated.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim // 2))
    quant_kw = {
        "group_size": 128,
        "eps": 1e-4,
        "column_major_scales": True,
        "scale_tma_aligned": True,
        "scale_ue8m0": True,
        "fuse_silu_and_mul": False,
        "masked_m": None,
    }
    if quant_kwargs:
        quant_kw.update(quant_kwargs)
    quant = graph.call_function(
        sgl_per_token_group_quant_fp8, args=(activated,), kwargs=quant_kw
    )
    fp8 = graph.call_function(operator.getitem, args=(quant, 0))
    scale = graph.call_function(operator.getitem, args=(quant, 1))
    graph.output((fp8, scale))
    return fx.GraphModule(torch.nn.Module(), graph)


class SiluAndMulFp8QuantPassTest(unittest.TestCase):
    def test_replaces_silu_then_quant(self):
        gm = _build_graph(last_dim=4096)
        apply_silu_and_mul_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd", names)
        self.assertNotIn("silu_and_mul", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)

    def test_skips_when_quant_contract_mismatch(self):
        gm = _build_graph(last_dim=4096, quant_kwargs={"column_major_scales": False})
        apply_silu_and_mul_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("silu_and_mul", names)
        self.assertIn("sgl_per_token_group_quant_fp8", names)

    def test_skips_when_input_last_dim_misaligned(self):
        # silu_and_mul halves the last dim; 384/2=192, not divisible by 128.
        gm = _build_graph(last_dim=384)
        apply_silu_and_mul_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("silu_and_mul", names)


if __name__ == "__main__":
    unittest.main()
