"""Projection pointer, CUDA Graph and model-weight ownership regressions."""

import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.qwen4_exp.gdn_projection import Sm90GdnLinear, _library


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "CUDA required"
)
class GdnProjectionTest(unittest.TestCase):
    def assert_exact(self, actual, expected):
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def require_fast_path(self):
        if (
            torch.cuda.get_device_capability() != (9, 0)
            or torch.cuda.get_device_properties().multi_processor_count != 78
            or _library() is None
        ):
            self.skipTest("requires the qualified H20 cuBLASLt library")

    def test_misaligned_inputs_and_weights(self):
        for columns, inner in ((8240, 2560), (2560, 3072)):
            native = CudaF16Linear(
                torch.full((inner, columns), 0.125, device="cuda", dtype=torch.bfloat16)
            )
            candidate = Sm90GdnLinear(native)
            for rows in (16, 32, 64, 32768):
                storage = torch.full(
                    (rows * inner + 1,), 0.125, device="cuda", dtype=torch.bfloat16
                )
                value = storage[1:].view(rows, inner)
                self.assertEqual(value.data_ptr() % 16, 2)
                self.assert_exact(candidate(value), native(value))
            for row_major in (False, True):
                storage = torch.full(
                    (inner * columns + 1,), 0.125, device="cuda", dtype=torch.bfloat16
                )
                weight = (
                    storage[1:].view(columns, inner)
                    if row_major
                    else storage[1:].view(inner, columns).T
                )
                native.weight = candidate.weight = weight
                for rows in (16, 32768):
                    value = torch.full(
                        (rows, inner), 0.125, device="cuda", dtype=torch.bfloat16
                    )
                    self.assert_exact(candidate(value), native(value))

    def test_live_graph_after_eager_and_other_shape(self):
        self.require_fast_path()
        native = CudaF16Linear(
            torch.full((2560, 8240), 0.125, device="cuda", dtype=torch.bfloat16)
        )
        candidate = Sm90GdnLinear(native)
        values = [
            torch.full((rows, 2560), 0.125, device="cuda", dtype=torch.bfloat16)
            for rows in (16, 32)
        ]
        for value in values:
            candidate(value)
        # Recreate the real pack event after the plan warmup so both captures
        # retain EventWait nodes even when preparation finishes quickly.
        candidate._packed = None
        candidate._refresh_weight(True)
        graphs, outputs = [], []
        for value in values:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = candidate(value)
            graphs.append(graph)
            outputs.append(output)
        for _ in range(3):
            for graph, output, value in zip(graphs, outputs, values):
                graph.replay()
                self.assert_exact(output, native(value))
                candidate(value)
                torch.cuda.synchronize()
            for graph, output, value in zip(graphs, outputs, values):
                graph.replay()
                self.assert_exact(output, native(value))
        for graph in graphs:
            graph.reset()

    def test_library_fallback(self):
        native = CudaF16Linear(
            torch.ones(2560, 8240, device="cuda", dtype=torch.bfloat16)
        )
        candidate = Sm90GdnLinear(native)
        value = torch.full((16, 2560), 0.125, device="cuda", dtype=torch.bfloat16)
        with mock.patch(
            "rtp_llm.models_py.modules.qwen4_exp.gdn_projection._library",
            return_value=None,
        ):
            self.assert_exact(candidate(value), native(value))
        self.assertEqual(candidate._plans, {})

    def test_model_weight_update_guards(self):
        self.require_fast_path()
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.model_loader.weight_manager import WeightManager
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
            prepare_compute_weights,
        )
        from rtp_llm.ops import DataType, ParallelismConfig
        from rtp_llm.utils.model_weight import W

        config = ModelConfig()
        linear = config.linear_attention_config
        linear.linear_num_key_heads, linear.linear_num_value_heads = 16, 48
        linear.linear_key_head_dim = linear.linear_value_head_dim = 128
        linear.linear_conv_kernel_dim = 4
        linear.ssm_state_dtype, linear.conv_state_dtype = (
            DataType.TYPE_FP32,
            DataType.TYPE_BF16,
        )
        parallel = ParallelismConfig()
        parallel.tp_size = parallel.world_size = 2
        parallel.dp_size = parallel.ep_size = 1
        shapes = {
            W.linear_attn_qkvz_w: (2560, 8192),
            W.linear_attn_ba_w: (2560, 48),
            W.linear_attn_conv1d_w: (5120, 1, 4),
            W.linear_attn_dt_b: (24,),
            W.linear_attn_alog: (24,),
            W.linear_attn_norm_w: (128,),
            W.linear_attn_out_w: (3072, 2560),
        }
        for prepared, dtype in (
            (False, torch.bfloat16),
            (True, torch.bfloat16),
            (False, torch.float16),
        ):
            owner = ModelWeights(1, "cuda", dtype)
            owner.weights[0] = {
                name: torch.full(shape, 0.125, device="cuda", dtype=dtype)
                for name, shape in shapes.items()
            }
            compute = prepare_compute_weights(owner) if prepared else owner
            layer = Qwen3NextGatedDeltaNet(
                linear, parallel, compute.weights[0], 1e-6, norm_activation="sigmoid"
            )
            if dtype == torch.float16:
                self.assertIs(type(layer.in_proj_fused), CudaF16Linear)
                for name in (
                    W.linear_attn_qkvz_w,
                    W.linear_attn_ba_w,
                    W.linear_attn_out_w,
                ):
                    value = owner.weights[0][name]
                    self.assertFalse(
                        getattr(value, "_rtp_requires_restart_for_update", False)
                    )
                    owner.update_layer_weight(0, name, value.clone())
                continue
            self.assertIsInstance(layer.in_proj_fused, Sm90GdnLinear)
            for weights in (owner, compute):
                for name in (
                    W.linear_attn_qkvz_w,
                    W.linear_attn_ba_w,
                    W.linear_attn_out_w,
                ):
                    value = weights.weights[0][name]
                    saved = value.clone()
                    with self.assertRaises(PermissionError):
                        weights.update_layer_weight(0, name, torch.zeros_like(value))
                    self.assert_exact(value, saved)
                    manager = object.__new__(WeightManager)
                    manager._weights = weights
                    with self.assertRaises(PermissionError):
                        manager.update(
                            {
                                "name": f"model.layers.0.{name}",
                                "method": "shm",
                                "desc": "unused-before-guard",
                            }
                        )
                    self.assert_exact(value, saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
