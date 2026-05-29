"""Cross-graph FX rewrite tests for add+RMSNorm+FP8 quant.

These tests build two separate ``GraphModule``s — a producer graph that
contains the mutating ``fused_add_rmsnorm`` and a consumer graph that
contains ``sgl_per_token_group_quant_fp8`` on a placeholder — and verify
that the pass:

* Rewrites the producer-side mutating call to the
  ``fused_add_rmsnorm_fp8_quant_producer`` custom_op (which preserves
  ``mutates_args`` for Dynamo mutation tracking).
* Rewrites the consumer-side quant call to
  ``graphfx_fused_add_rmsnorm_fp8_quant_from_provenance`` when its input is
  a graph placeholder.
"""

from __future__ import annotations

import operator
import unittest

import torch
from torch import fx

from rtp_llm.models_py.modules.fuse_kernel_fx.add_rmsnorm_fp8_quant_pass import (
    apply_add_rmsnorm_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.fuse_kernel_fx.test.graphfx_fusion_test_utils import (
    make_dummy_tensor_meta,
    target_names,
)


def fused_add_rmsnorm(hidden, residual, weight, eps, stream):
    return None


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


def some_bf16_consumer(x):
    return x


def _build_producer_only_graph(*, last_dim: int = 4096):
    """Producer graph: mutating add_rmsnorm with hidden returned through output."""
    graph = fx.Graph()
    hidden = graph.placeholder("hidden")
    residual = graph.placeholder("residual")
    weight = graph.placeholder("weight")
    hidden.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))
    residual.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))
    weight.meta["tensor_meta"] = make_dummy_tensor_meta((last_dim,))
    graph.call_function(fused_add_rmsnorm, args=(hidden, residual, weight, 1e-6, 0))
    graph.output(hidden)
    return fx.GraphModule(torch.nn.Module(), graph)


def _build_consumer_only_graph(*, last_dim: int = 4096):
    """Consumer graph: quant on a placeholder."""
    graph = fx.Graph()
    y = graph.placeholder("y")
    y.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))
    quant = graph.call_function(
        sgl_per_token_group_quant_fp8,
        args=(y,),
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


class CrossGraphProducerCustomOpTest(unittest.TestCase):
    def test_producer_only_graph_emits_custom_op(self):
        gm = _build_producer_only_graph()
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertTrue(
            any("fused_add_rmsnorm_fp8_quant_producer" in n for n in names),
            f"producer custom_op not found in {names}",
        )
        self.assertNotIn("fused_add_rmsnorm", names)

    def test_producer_with_extra_call_consumer_skips(self):
        graph = fx.Graph()
        hidden = graph.placeholder("hidden")
        residual = graph.placeholder("residual")
        weight = graph.placeholder("weight")
        hidden.meta["tensor_meta"] = make_dummy_tensor_meta((4, 4096))
        residual.meta["tensor_meta"] = make_dummy_tensor_meta((4, 4096))
        weight.meta["tensor_meta"] = make_dummy_tensor_meta((4096,))
        graph.call_function(fused_add_rmsnorm, args=(hidden, residual, weight, 1e-6, 0))
        graph.call_function(some_bf16_consumer, args=(hidden,))
        graph.output(hidden)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertFalse(
            any("fused_add_rmsnorm_fp8_quant_producer" in n for n in names),
        )
        self.assertIn("fused_add_rmsnorm", names)


class CrossGraphConsumerTokenTest(unittest.TestCase):
    def test_consumer_only_graph_swaps_to_provenance(self):
        gm = _build_consumer_only_graph()
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("graphfx_fused_add_rmsnorm_fp8_quant_from_provenance", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)

    def test_consumer_with_inline_producer_keeps_same_graph_path(self):
        graph = fx.Graph()
        hidden = graph.placeholder("hidden")
        residual = graph.placeholder("residual")
        weight = graph.placeholder("weight")
        hidden.meta["tensor_meta"] = make_dummy_tensor_meta((4, 4096))
        residual.meta["tensor_meta"] = make_dummy_tensor_meta((4, 4096))
        weight.meta["tensor_meta"] = make_dummy_tensor_meta((4096,))
        graph.call_function(fused_add_rmsnorm, args=(hidden, residual, weight, 1e-6, 0))
        quant = graph.call_function(
            sgl_per_token_group_quant_fp8,
            args=(hidden,),
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
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("fused_add_rmsnorm_fp8_quant_with_bf16_output", names)
        self.assertNotIn("graphfx_fused_add_rmsnorm_fp8_quant_from_provenance", names)


if __name__ == "__main__":
    unittest.main()
