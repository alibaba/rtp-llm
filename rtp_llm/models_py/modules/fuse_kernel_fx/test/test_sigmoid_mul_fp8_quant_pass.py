"""FX-pass-level tests for ``apply_sigmoid_mul_fp8_quant_fx_pass``."""

from __future__ import annotations

import operator
import unittest

import torch
from torch import fx

from rtp_llm.models_py.modules.fuse_kernel_fx.sigmoid_mul_fp8_quant_pass import (
    apply_sigmoid_mul_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.fuse_kernel_fx.test.graphfx_fusion_test_utils import (
    make_dummy_tensor_meta,
    target_names,
)


def sigmoid_mul_inplace_triton(attn, gate):
    return attn


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
    attn = graph.placeholder("attn")
    gate = graph.placeholder("gate")
    attn.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))
    gate.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))
    fused = graph.call_function(sigmoid_mul_inplace_triton, args=(attn, gate))
    fused.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))
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
        sgl_per_token_group_quant_fp8, args=(fused,), kwargs=quant_kw
    )
    fp8 = graph.call_function(operator.getitem, args=(quant, 0))
    scale = graph.call_function(operator.getitem, args=(quant, 1))
    graph.output((fp8, scale))
    return fx.GraphModule(torch.nn.Module(), graph)


class SigmoidMulFp8QuantPassTest(unittest.TestCase):
    def test_replaces_sigmoid_mul_then_quant(self):
        gm = _build_graph(last_dim=2048)
        apply_sigmoid_mul_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("sigmoid_mul_fp8_quant_fwd", names)
        self.assertNotIn("sigmoid_mul_inplace_triton", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)

    def test_skips_misaligned_last_dim(self):
        gm = _build_graph(last_dim=120)
        apply_sigmoid_mul_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("sigmoid_mul_inplace_triton", names)
        self.assertIn("sgl_per_token_group_quant_fp8", names)

    def test_skips_when_quant_contract_mismatch(self):
        gm = _build_graph(last_dim=2048, quant_kwargs={"scale_tma_aligned": False})
        apply_sigmoid_mul_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("sigmoid_mul_inplace_triton", names)
        self.assertIn("sgl_per_token_group_quant_fp8", names)

    def test_replaces_pure_pytorch_baseline(self):
        """After the model_desc refactor the eager chain becomes
        ``attn * torch.sigmoid(gate)`` followed by an FP8 quant inside the
        linear; GraphFX must still pick it up.
        """
        graph = fx.Graph()
        attn = graph.placeholder("attn")
        gate = graph.placeholder("gate")
        attn.meta["tensor_meta"] = make_dummy_tensor_meta((4, 2048))
        gate.meta["tensor_meta"] = make_dummy_tensor_meta((4, 2048))
        sig = graph.call_function(torch.sigmoid, args=(gate,))
        sig.meta["tensor_meta"] = make_dummy_tensor_meta((4, 2048))
        mul = graph.call_function(operator.mul, args=(attn, sig))
        mul.meta["tensor_meta"] = make_dummy_tensor_meta((4, 2048))
        quant = graph.call_function(
            sgl_per_token_group_quant_fp8,
            args=(mul,),
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
        gm = fx.GraphModule(torch.nn.Module(), graph)
        apply_sigmoid_mul_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("sigmoid_mul_fp8_quant_fwd", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)
        self.assertNotIn("mul", names)


if __name__ == "__main__":
    unittest.main()
