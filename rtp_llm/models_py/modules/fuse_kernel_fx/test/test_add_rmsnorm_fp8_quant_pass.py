"""FX-pass-level tests for ``apply_add_rmsnorm_fp8_quant_fx_pass``.

These tests build small ``torch.fx.GraphModule`` graphs by hand so they run
on CPU without requiring CUDA / Triton.  They verify the pass:

* Replaces ``fused_add_rmsnorm`` (mutating) + ``sgl_per_token_group_quant_fp8``
  with ``fused_add_rmsnorm_fp8_quant`` when the BF16 normed value has no
  other consumer.
* Uses ``fused_add_rmsnorm_fp8_quant_with_bf16_output`` and redirects BF16
  consumers when the normed value also feeds a non-quant user.
* Records misses and skips when ``hidden_dim`` is unsupported, when the
  quant contract doesn't match, or when no quant consumer is found.
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

# ---- stand-in callables (FX matches by ``__name__``) ----


def fused_add_rmsnorm(hidden, residual, weight, eps, stream):
    """Stand-in for the wrapped ``rtp_llm_ops.fused_add_rmsnorm`` mutating op."""
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


# ---- helpers ----


def _attach_bf16_meta(node: fx.Node, last_dim: int) -> None:
    node.meta["tensor_meta"] = make_dummy_tensor_meta((4, last_dim))


def _build_single_graph(
    *,
    last_dim: int,
    quant_kwargs: dict | None = None,
    add_extra_bf16_consumer: bool = False,
) -> fx.GraphModule:
    graph = fx.Graph()
    hidden = graph.placeholder("hidden")
    residual = graph.placeholder("residual")
    weight = graph.placeholder("weight")
    _attach_bf16_meta(hidden, last_dim)
    _attach_bf16_meta(residual, last_dim)
    weight.meta["tensor_meta"] = make_dummy_tensor_meta((last_dim,))

    add_node = graph.call_function(
        fused_add_rmsnorm, args=(hidden, residual, weight, 1e-6, 0)
    )
    add_node.meta["is_mutating_add_rmsnorm"] = True

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

    quant_node = graph.call_function(
        sgl_per_token_group_quant_fp8, args=(hidden,), kwargs=quant_kw
    )
    fp8 = graph.call_function(operator.getitem, args=(quant_node, 0))
    scale = graph.call_function(operator.getitem, args=(quant_node, 1))

    if add_extra_bf16_consumer:
        bf16_user = graph.call_function(some_bf16_consumer, args=(hidden,))
        graph.output((fp8, scale, bf16_user))
    else:
        graph.output((fp8, scale))

    return fx.GraphModule(torch.nn.Module(), graph)


class AddRmsnormFp8QuantPassTest(unittest.TestCase):
    def test_same_graph_single_output_fusion(self):
        gm = _build_single_graph(last_dim=4096)
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("fused_add_rmsnorm_fp8_quant", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)
        self.assertNotIn("fused_add_rmsnorm", names)

    def test_dual_output_fusion_when_bf16_consumer_present(self):
        gm = _build_single_graph(last_dim=4096, add_extra_bf16_consumer=True)
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("fused_add_rmsnorm_fp8_quant_with_bf16_output", names)
        self.assertNotIn("sgl_per_token_group_quant_fp8", names)
        # The mutating add_rmsnorm node is erased; the BF16 consumer now
        # reads the dual kernel's BF16 output via getitem(0).
        bf16_consumer_call = next(
            node
            for node in gm.graph.nodes
            if getattr(node.target, "__name__", "") == "some_bf16_consumer"
        )
        bf16_input = bf16_consumer_call.args[0]
        self.assertEqual(getattr(bf16_input.target, "__name__", ""), "getitem")
        self.assertEqual(bf16_input.args[1], 0)

    def test_unsupported_hidden_dim_skips(self):
        gm = _build_single_graph(last_dim=999)
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("sgl_per_token_group_quant_fp8", names)
        self.assertIn("fused_add_rmsnorm", names)
        self.assertNotIn("fused_add_rmsnorm_fp8_quant", names)

    def test_quant_contract_mismatch_skips(self):
        gm = _build_single_graph(
            last_dim=4096, quant_kwargs={"column_major_scales": False}
        )
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("sgl_per_token_group_quant_fp8", names)
        self.assertNotIn("fused_add_rmsnorm_fp8_quant", names)

    def test_no_quant_consumer_skips(self):
        graph = fx.Graph()
        hidden = graph.placeholder("hidden")
        residual = graph.placeholder("residual")
        weight = graph.placeholder("weight")
        _attach_bf16_meta(hidden, 4096)
        _attach_bf16_meta(residual, 4096)
        weight.meta["tensor_meta"] = make_dummy_tensor_meta((4096,))
        graph.call_function(fused_add_rmsnorm, args=(hidden, residual, weight, 1e-6, 0))
        graph.call_function(some_bf16_consumer, args=(hidden,))
        graph.output(hidden)
        gm = fx.GraphModule(torch.nn.Module(), graph)
        apply_add_rmsnorm_fp8_quant_fx_pass(gm)
        names = target_names(gm)
        self.assertIn("fused_add_rmsnorm", names)
        self.assertNotIn("fused_add_rmsnorm_fp8_quant", names)


if __name__ == "__main__":
    unittest.main()
