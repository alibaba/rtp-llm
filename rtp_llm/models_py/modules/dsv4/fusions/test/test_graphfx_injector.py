from __future__ import annotations

import operator
import os
import unittest
from unittest import mock

import torch
from torch import fx

from rtp_llm.models_py.kernels.cuda.dsv4_indexed_rope import (
    dsv4_indexed_inv_rope_fp8_quant,
    dsv4_indexed_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.fusions.kv_rope_fp8_quant_pass import (
    apply_kv_rope_fp8_quant_fx_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.kv_rope_fp8_quant_runtime import (
    dsv4_kv_rope_fp8_quant_from_provenance,
    dsv4_kv_rope_quant_producer_token,
)
from rtp_llm.models_py.modules.dsv4.fusions.indexed_rope_pass import (
    apply_indexed_rope_fx_pass,
)
from rtp_llm.models_py.modules.dsv4.fusions.indexed_rope_runtime import (
    dsv4_indexed_inv_rope_fp8_quant_from_freqs,
    dsv4_indexed_rmsnorm_rope_from_freqs,
    dsv4_indexed_rope_freqs_token,
)
from rtp_llm.models_py.modules.dsv4.fusions.graphfx_injector import (
    maybe_install_dsv4_graphfx_fusions,
)


class _FakePyModel:
    def __init__(self):
        self.initialized = False

    def forward(self, x):
        return x

    def initialize(self, init_resource=None):
        self.initialized = True
        return True


def fused_rmsnorm_rope(x, weight, freqs_cis, rope_head_dim, *, eps=1e-6):
    return x


def fused_inv_rope_fp8_quant(
    o,
    freqs_cis,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_head_dim,
    *,
    quant_group_size=128,
    eps=1e-10,
    impl=None,
    heads_per_cta=None,
):
    return o, freqs_cis


def triton_kernel_wrapper_mutation(**kwargs):
    return None


def fused_kv_compress_norm_rope_insert_indexer_attn(x):
    return x


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


class GraphFXInjectorTest(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "DSV4_GRAPHFX_FUSION",
                "DSV4_FUSION_REGISTRY_DEBUG",
                "DSV4_GRAPHFX_COMPILE_SCOPE",
            )
        }
        os.environ["DSV4_GRAPHFX_FUSION"] = "1"
        os.environ["DSV4_GRAPHFX_COMPILE_SCOPE"] = "forward"

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_direct_install_compiles_model_forward(self):
        calls = []

        def fake_compile(fn, **kwargs):
            calls.append(kwargs)

            def compiled(*args, **inner_kwargs):
                return fn(*args, **inner_kwargs)

            return compiled

        py_model = _FakePyModel()
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fusions.graphfx_injector.compile_with_dsv4_fusions",
            side_effect=fake_compile,
        ):
            self.assertTrue(maybe_install_dsv4_graphfx_fusions(py_model))

        self.assertEqual(len(calls), 1)
        self.assertTrue(all(item.get("dynamic") is True for item in calls))
        self.assertTrue(all(item.get("fullgraph") is False for item in calls))

    def test_direct_install_is_idempotent(self):
        calls = []

        def fake_compile(fn, **kwargs):
            calls.append(kwargs)
            return fn

        py_model = _FakePyModel()
        py_model.initialize(None)
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fusions.graphfx_injector.compile_with_dsv4_fusions",
            side_effect=fake_compile,
        ):
            self.assertTrue(maybe_install_dsv4_graphfx_fusions(py_model))
            self.assertFalse(maybe_install_dsv4_graphfx_fusions(py_model))

        self.assertEqual(len(calls), 1)

    def test_unsupported_scope_falls_back_to_model_forward(self):
        calls = []

        def fake_compile(fn, **kwargs):
            calls.append((getattr(fn, "__name__", ""), kwargs))

            def compiled(*args, **inner_kwargs):
                return fn(*args, **inner_kwargs)

            return compiled

        os.environ["DSV4_GRAPHFX_COMPILE_SCOPE"] = "decode_layers"
        py_model = _FakePyModel()
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fusions.graphfx_injector.compile_with_dsv4_fusions",
            side_effect=fake_compile,
        ):
            self.assertTrue(maybe_install_dsv4_graphfx_fusions(py_model))

        self.assertEqual([name for name, _ in calls], ["forward"])
        self.assertTrue(getattr(py_model.forward, "_dsv4_graphfx_compiled", False))
        self.assertTrue(all(item.get("dynamic") is True for _, item in calls))

    def test_indexed_rope_pass_rewrites_same_graph_pattern(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        freqs = graph.placeholder("freqs_cis")
        pos = graph.placeholder("position_ids")
        to_long = graph.call_method("to", args=(pos,), kwargs={"dtype": torch.long})
        gathered = graph.call_method("index_select", args=(freqs, 0, to_long))
        contiguous = graph.call_method("contiguous", args=(gathered,))
        fused = graph.call_function(
            fused_rmsnorm_rope,
            args=(x, weight, contiguous, 64),
            kwargs={"eps": 1e-6},
        )
        graph.output(fused)
        gm = fx.GraphModule({}, graph)

        out = apply_indexed_rope_fx_pass(gm)
        targets = [node.target for node in out.graph.nodes if node.op == "call_function"]

        self.assertIn(dsv4_indexed_rmsnorm_rope, targets)

    def test_indexed_rope_pass_dces_rewritten_consumer_before_freq_token(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        freqs = graph.placeholder("freqs_cis")
        pos = graph.placeholder("position_ids")
        to_long = graph.call_method("to", args=(pos,), kwargs={"dtype": torch.long})
        gathered = graph.call_method("index_select", args=(freqs, 0, to_long))
        contiguous = graph.call_method("contiguous", args=(gathered,))
        fused = graph.call_function(
            fused_rmsnorm_rope,
            args=(x, weight, contiguous, 64),
            kwargs={"eps": 1e-6},
        )
        graph.output((fused, contiguous))
        gm = fx.GraphModule({}, graph)

        out = apply_indexed_rope_fx_pass(gm)
        targets = [node.target for node in out.graph.nodes if node.op == "call_function"]
        target_names = [getattr(target, "__name__", str(target)) for target in targets]

        self.assertIn(dsv4_indexed_rmsnorm_rope, targets)
        self.assertIn(dsv4_indexed_rope_freqs_token, targets)
        self.assertNotIn("index_select", target_names)
        self.assertFalse(
            any(
                node.op == "call_method" and getattr(node.target, "__name__", node.target) == "index_select"
                for node in out.graph.nodes
            )
        )

    def test_indexed_rope_pass_rewrites_producer_only_freqs_gather(self):
        graph = fx.Graph()
        freqs = graph.placeholder("freqs_cis")
        pos = graph.placeholder("position_ids")
        to_long = graph.call_method("to", args=(pos,), kwargs={"dtype": torch.long})
        gathered = graph.call_method("index_select", args=(freqs, 0, to_long))
        contiguous = graph.call_method("contiguous", args=(gathered,))
        graph.output(contiguous)
        gm = fx.GraphModule({}, graph)

        out = apply_indexed_rope_fx_pass(gm)
        targets = [node.target for node in out.graph.nodes if node.op == "call_function"]

        self.assertIn(dsv4_indexed_rope_freqs_token, targets)

    def test_indexed_rope_pass_rewrites_tensor_getitem_freqs(self):
        graph = fx.Graph()
        o = graph.placeholder("o")
        freqs = graph.placeholder("freqs_cis")
        pos = graph.placeholder("position_ids")
        gathered = graph.call_function(operator.getitem, args=(freqs, pos))
        contiguous = graph.call_method("contiguous", args=(gathered,))
        fused = graph.call_function(
            fused_inv_rope_fp8_quant,
            args=(o, contiguous, 8, 8, 448, 64),
            kwargs={"eps": 1e-10, "impl": "optimized"},
        )
        graph.output(fused)
        gm = fx.GraphModule({}, graph)

        out = apply_indexed_rope_fx_pass(gm)
        targets = [node.target for node in out.graph.nodes if node.op == "call_function"]

        self.assertIn(dsv4_indexed_inv_rope_fp8_quant, targets)

    def test_indexed_rope_pass_rewrites_cross_graph_consumer(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        weight = graph.placeholder("weight")
        freqs = graph.placeholder("freqs_cis")
        rd = graph.placeholder("rd")
        fused = graph.call_function(
            fused_rmsnorm_rope,
            args=(x, weight, freqs, rd),
            kwargs={"eps": 1e-6},
        )
        graph.output(fused)
        gm = fx.GraphModule({}, graph)

        out = apply_indexed_rope_fx_pass(gm)
        targets = [node.target for node in out.graph.nodes if node.op == "call_function"]

        self.assertIn(dsv4_indexed_rmsnorm_rope_from_freqs, targets)

    def test_indexed_rope_pass_rewrites_lowered_triton_rmsnorm_rope(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        freqs = graph.placeholder("freqs_cis")
        s95 = graph.placeholder("s95")
        x_size = graph.call_method("size", args=(x,))
        head_dim = graph.call_function(operator.getitem, args=(x_size, 3))
        x_flat = graph.call_method("view", args=(x, -1, head_dim))
        out_flat = graph.call_function(torch.empty_like, args=(x_flat,))
        w = graph.call_function(
            torch.empty,
            args=(1,),
            kwargs={"dtype": torch.float32, "device": torch.device("cuda")},
        )
        freqs_flat = graph.call_method("view", args=(freqs, -1, s95))
        freqs_ri = graph.call_function(torch.view_as_real, args=(freqs_flat,))
        rope_dim = graph.call_function(operator.mul, args=(2, s95))
        graph.call_function(
            triton_kernel_wrapper_mutation,
            kwargs={
                "kernel_idx": 3,
                "constant_args_idx": 3,
                "grid": [(1, 1, 1)],
                "tma_descriptor_metadata": {},
                "kwargs": {
                    "x_ptr": x_flat,
                    "w_ptr": w,
                    "freqs_ri_ptr": freqs_ri,
                    "out_ptr": out_flat,
                    "x_stride_n": 128,
                    "out_stride_n": 128,
                    "freqs_stride_b": rope_dim,
                },
            },
        )
        out = graph.call_method("view", args=(out_flat, 1, -1, 128))
        graph.output(out)
        gm = fx.GraphModule({}, graph)

        out_gm = apply_indexed_rope_fx_pass(gm)
        targets = [node.target for node in out_gm.graph.nodes if node.op == "call_function"]

        self.assertIn(dsv4_indexed_rmsnorm_rope_from_freqs, targets)

    def test_indexed_inv_rope_fallback_preserves_reference_kwargs(self):
        graph = fx.Graph()
        o = graph.placeholder("o")
        freqs = graph.placeholder("freqs_cis")
        fused = graph.call_function(
            fused_inv_rope_fp8_quant,
            args=(o, freqs, 8, 8, 448, 64),
            kwargs={"eps": 1e-7, "impl": "legacy", "heads_per_cta": 2},
        )
        graph.output(fused)
        gm = fx.GraphModule({}, graph)

        out = apply_indexed_rope_fx_pass(gm)
        rewritten = [
            node
            for node in out.graph.nodes
            if node.op == "call_function"
            and node.target is dsv4_indexed_inv_rope_fp8_quant_from_freqs
        ]

        self.assertEqual(len(rewritten), 1)
        self.assertEqual(rewritten[0].kwargs["impl"], "legacy")
        self.assertEqual(rewritten[0].kwargs["heads_per_cta"], 2)

    def test_kv_rope_quant_pass_rewrites_consumer(self):
        graph = fx.Graph()
        x = graph.placeholder("kv_compress")
        producer = graph.call_function(
            fused_kv_compress_norm_rope_insert_indexer_attn,
            args=(x,),
        )
        quant = graph.call_function(
            sgl_per_token_group_quant_fp8,
            args=(producer,),
            kwargs={
                "group_size": 128,
                "eps": 1e-4,
                "column_major_scales": True,
                "scale_tma_aligned": True,
                "scale_ue8m0": True,
            },
        )
        graph.output(quant)
        gm = fx.GraphModule({}, graph)

        out = apply_kv_rope_fp8_quant_fx_pass(gm)
        targets = [node.target for node in out.graph.nodes if node.op == "call_function"]

        self.assertIn(dsv4_kv_rope_fp8_quant_from_provenance, targets)
        self.assertNotIn(sgl_per_token_group_quant_fp8, targets)

    def test_kv_rope_quant_pass_inserts_producer_token(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        producer = graph.call_function(
            fused_kv_compress_norm_rope_insert_indexer_attn,
            args=(x,),
        )
        graph.output(producer)
        gm = fx.GraphModule({}, graph)

        out = apply_kv_rope_fp8_quant_fx_pass(gm)
        targets = [node.target for node in out.graph.nodes if node.op == "call_function"]

        self.assertIn(dsv4_kv_rope_quant_producer_token, targets)


if __name__ == "__main__":
    unittest.main()
