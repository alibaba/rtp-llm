"""Cross-graph FX rewrite tests for add+RMSNorm+FP8 quant.

These tests build two separate ``GraphModule``s — a producer graph that
contains the mutating ``fused_add_rmsnorm`` and a consumer graph that
contains ``sgl_per_token_group_quant_fp8`` on a placeholder — and verify
that the pass:

* Rewrites the producer-side mutating call to ``qwen35_fused_add_rmsnorm_producer_token``
  when the BF16 result only leaves the graph.
* Rewrites the consumer-side quant call to
  ``qwen35_fused_add_rmsnorm_fp8_quant_from_provenance`` when its input is
  a graph placeholder.
* The runtime registry honours precompute-mode lookups.

These tests run on CPU; the runtime helpers do not require triton/CUDA when
the registry is exercised through ``_remember`` directly.
"""

from __future__ import annotations

import operator
import os
import unittest

import torch
from torch import fx

from rtp_llm.models_py.modules.fuse_kernel_fx.add_rmsnorm_fp8_quant_pass import (
    apply_add_rmsnorm_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.fuse_kernel_fx.add_rmsnorm_runtime import (
    _clear_registries_for_tests,
    _lookup,
    _remember,
    qwen35_fused_add_rmsnorm_fp8_quant_from_provenance,
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
    """Consumer graph: quant on a placeholder + a non-quant BF16 reader."""
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


class CrossGraphProducerTokenTest(unittest.TestCase):
    def test_producer_only_graph_emits_token(self):
        gm = _build_producer_only_graph()
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("qwen35_fused_add_rmsnorm_producer_token", names)
        self.assertNotIn("fused_add_rmsnorm", names)

    def test_producer_with_extra_call_consumer_skips_producer_token(self):
        # When hidden has a non-layout user we cannot safely emit the
        # producer-token; this ensures the rewrite stays conservative.
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
        # Cross-graph producer rewrite must NOT fire (non-layout user blocks).
        self.assertNotIn("qwen35_fused_add_rmsnorm_producer_token", names)
        self.assertIn("fused_add_rmsnorm", names)


class CrossGraphConsumerTokenTest(unittest.TestCase):
    def test_consumer_only_graph_swaps_to_provenance(self):
        gm = _build_consumer_only_graph()
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("qwen35_fused_add_rmsnorm_fp8_quant_from_provenance", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)

    def test_consumer_with_inline_producer_keeps_same_graph_path(self):
        # When the producer mutating call IS in the same graph, we want the
        # same-graph rewrite, not the cross-graph consumer rewrite.
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
        self.assertIn("fused_add_rmsnorm_fp8_quant", names)
        self.assertNotIn("qwen35_fused_add_rmsnorm_fp8_quant_from_provenance", names)


class ProvenanceRegistryTest(unittest.TestCase):
    def setUp(self):
        _clear_registries_for_tests()
        self._old_env = os.environ.pop(
            "QWEN35_FUSED_ADD_RMSNORM_REQUIRE_PROVENANCE", None
        )

    def tearDown(self):
        if self._old_env is not None:
            os.environ["QWEN35_FUSED_ADD_RMSNORM_REQUIRE_PROVENANCE"] = self._old_env

    def test_lookup_returns_precomputed_fp8_when_remembered(self):
        token = torch.zeros(4, 128)
        weight = torch.ones(128)
        q = torch.zeros(4, 128, dtype=torch.uint8)
        scale = torch.zeros(4, 1)
        _remember(token, None, None, weight, 1e-6, True, q, scale)
        entry = _lookup(token)
        self.assertIsNotNone(entry)
        # entry layout: (token_ref, x_orig, residual_ref, weight, eps, ue8m0, q, scale)
        self.assertIs(entry[6], q)
        self.assertIs(entry[7], scale)

    def test_provenance_consumer_returns_precomputed(self):
        token = torch.zeros(4, 128)
        weight = torch.ones(128)
        q = torch.zeros(4, 128, dtype=torch.uint8)
        scale = torch.zeros(4, 1)
        _remember(token, None, None, weight, 1e-6, True, q, scale)
        out_q, out_s = qwen35_fused_add_rmsnorm_fp8_quant_from_provenance(token)
        self.assertIs(out_q, q)
        self.assertIs(out_s, scale)

    def test_require_provenance_raises_on_miss(self):
        os.environ["QWEN35_FUSED_ADD_RMSNORM_REQUIRE_PROVENANCE"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                qwen35_fused_add_rmsnorm_fp8_quant_from_provenance(torch.zeros(4, 128))
        finally:
            os.environ.pop("QWEN35_FUSED_ADD_RMSNORM_REQUIRE_PROVENANCE", None)


if __name__ == "__main__":
    unittest.main()
