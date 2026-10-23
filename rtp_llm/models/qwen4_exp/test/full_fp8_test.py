import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.model_loader.fp8_weight import FP8_STORAGE_SCALE_SUFFIX, Fp8StorageWeight
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models_py.modules.base.common.fp8_embedding import Fp8VocabEmbedding
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
    prepare_compute_weights as compute_weight_view,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W


class _Source(TensorSource):
    def load_tensor(self, name, data_type=torch.float16):
        return [torch.arange(32).reshape(4, 8).to(data_type) - 16]

    def has_tensor(self, name):
        return True

    def get_database(self):
        return None


def quantized(value):
    scale = value.float().abs().amax().clamp_min(1e-12).reshape(1) / 448
    return (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


class FullFp8StorageTest(unittest.TestCase):
    def test_inventory_summary_counts_backing_storage_and_shared_compute_once(self):
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
            ComputeWeightInventory,
            WeightView,
            summarize_compute_inventory,
        )

        # Offset views share padded backing storage, even when data_ptr differs.
        storage = torch.zeros(32, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1)
        compute = torch.zeros(64, dtype=torch.bfloat16)
        entries = (
            WeightView(W.moe_w1, storage[:8], scale, compute[:8], True),
            WeightView(W.moe_w2, storage[8:16], scale, compute[8:16], True),
            WeightView("scale", scale, None, scale, False),
        )
        summary = summarize_compute_inventory(
            ComputeWeightInventory("FP8", None, (entries,), (entries[0],))
        )
        self.assertEqual(
            summary["source_storage"],
            [
                {
                    "device": "cpu",
                    "dtype": "torch.float32",
                    "bytes": 4,
                    "allocations": 1,
                },
                {
                    "device": "cpu",
                    "dtype": "torch.float8_e4m3fn",
                    "bytes": 32,
                    "allocations": 1,
                },
            ],
        )
        self.assertEqual(
            summary["extra_compute_storage"],
            [
                {
                    "device": "cpu",
                    "dtype": "torch.bfloat16",
                    "bytes": 128,
                    "allocations": 1,
                }
            ],
        )
        self.assertEqual(summary["layouts"][0]["role"], "routed_down")
        self.assertEqual(summary["layouts"][0]["shape"], [8])
        self.assertEqual(summary["layouts"][1]["tensors"], 2)

    def test_original_descriptor_split_runs_before_quantization(self):
        source = AtomicWeight(W.embedding, [CkptWeightInfo("weight")])
        source.split_func = lambda t, **kwargs: t[:, :4]
        config = SimpleNamespace(
            tp_size=2,
            tp_rank=0,
            ep_rank=0,
            dp_rank=0,
            ffn_tp_rank=0,
            ffn_tp_size=2,
            hidden_size=8,
            head_num=1,
            head_num_kv=1,
            size_per_head=8,
            moe_pure_tp_mode=False,
            bit=16,
            dp_size=1,
            ep_size=1,
            merge_lora=False,
            compute_dtype=torch.bfloat16,
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda key, value, **kw: value
            ),
            use_swizzleA=False,
        )
        descriptor = Fp8StorageWeight(source)
        loaded = descriptor.load(_Source(), None, "cpu", config)
        self.assertEqual(loaded[W.embedding].dtype, torch.float8_e4m3fn)
        self.assertEqual(loaded[W.embedding].element_size(), 1)
        scale = loaded[W.embedding + FP8_STORAGE_SCALE_SUFFIX]
        self.assertEqual(scale.numel(), 1)
        self.assertIn(
            W.embedding + FP8_STORAGE_SCALE_SUFFIX, descriptor.additional_output_names
        )
        torch.testing.assert_close(
            loaded[W.embedding].float() * scale,
            _Source().load_tensor("weight", torch.float32)[0][:, :4],
            rtol=0.08,
            atol=0.5,
        )

    def test_custom_residual_consumers_record_actual_compute_copies(self):
        from torch import nn

        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import (
            annotate_compute_inventory,
        )

        weights = ModelWeights(1, "cpu", torch.bfloat16)
        projections = (
            (W.qwen4_hc_attn_mix_down, (16, 128)),
            (W.qwen4_hc_attn_mix_up, (128, 16)),
            (W.qwen4_hc_attn_inject, (4, 128)),
            (W.qwen4_hc_mlp_mix_down, (16, 128)),
            (W.qwen4_hc_mlp_mix_up, (128, 16)),
            (W.qwen4_hc_mlp_inject, (4, 128)),
        )
        for name, shape in projections:
            value, scale = quantized(torch.randn(shape))
            weights.weights[0][name] = value
            weights.weights[0][name + FP8_STORAGE_SCALE_SUFFIX] = scale
        for name in (W.qwen4_hc_attn_norm, W.qwen4_hc_mlp_norm):
            weights.weights[0][name] = torch.zeros(128)
        prepared = compute_weight_view(weights)
        layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
        nn.Module.__init__(layer)
        layer._build_residual_modules(
            SimpleNamespace(hc_mult=4, layernorm_eps=1e-6), prepared.weights[0]
        )
        inventory = annotate_compute_inventory(prepared)
        entries = {entry.name: entry for entry in inventory.layers[0]}
        for unit, names in (
            (layer.attn_hyper_connection, [x[0] for x in projections[:3]]),
            (layer.mlp_hyper_connection, [x[0] for x in projections[3:]]),
        ):
            for name, module in zip(
                names,
                (unit.down_projection, unit.up_projection, unit.inject_projection),
            ):
                entry = entries[name]
                self.assertEqual(
                    entry.storage.data_ptr(), weights.weights[0][name].data_ptr()
                )
                self.assertEqual(
                    entry.compute.data_ptr(), module.compute_weight.data_ptr()
                )
                self.assertEqual(entry.compute.dtype, torch.bfloat16)
                self.assertTrue(entry.update_requires_restart)
                self.assertTrue(entry.compute._rtp_requires_restart_for_update)

    def test_compute_view_preserves_fp8_inventory_and_matrices(self):
        weights = ModelWeights(1, "cpu", torch.bfloat16)
        matrix, ms = quantized(torch.randn(32, 32))
        norm, ns = quantized(torch.randn(32))
        weights.weights[0] = {
            "matrix": matrix,
            "matrix" + FP8_STORAGE_SCALE_SUFFIX: ms,
            "norm": norm,
            "norm" + FP8_STORAGE_SCALE_SUFFIX: ns,
        }
        view = compute_weight_view(weights)
        self.assertIs(view.weights[0]["matrix"], matrix)
        self.assertEqual(view.weights[0]["norm"].dtype, torch.bfloat16)
        self.assertEqual(weights.weights[0]["norm"].dtype, torch.float8_e4m3fn)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FullFp8CudaTest(unittest.TestCase):
    def test_dense_gemm_eager_side_stream_and_graph_live_inputs(self):
        torch.manual_seed(873)
        weight, scale = quantized(torch.randn(128, 256, device="cuda"))
        module = Fp8Linear(weight, scale)
        x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = module(x)
        torch.cuda.current_stream().wait_stream(stream)
        expected = F.linear(x, (weight.float() * scale).to(x.dtype))
        torch.testing.assert_close(actual, expected, rtol=0.1, atol=1.5)
        for _ in range(3):
            module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(x)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.testing.assert_close(output, module(x), rtol=0, atol=0)
        self.assertIsNone(module.compute_weight)
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)

    def test_ple_fp8_projection_scales_in_eager_and_graph(self):
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple import (
            Qwen4ExpNGramEmbedding,
            Qwen4ExpPLELayer,
        )

        torch.manual_seed(917)
        device = "cuda"
        key = torch.randint(-2, 3, (128, 64), device=device).to(torch.float8_e4m3fn)
        value = torch.randint(-2, 3, (32, 64), device=device).to(torch.float8_e4m3fn)
        key_scale = torch.tensor([0.25], device=device)
        value_scale = torch.tensor([0.5], device=device)
        conv = torch.zeros(128, 1, 2, device=device, dtype=torch.bfloat16)
        norm = torch.zeros(128, device=device, dtype=torch.bfloat16)

        def layer(k, v, scales):
            return Qwen4ExpPLELayer(
                Qwen4ExpNGramEmbedding(
                    [torch.zeros(8, 32, device=device, dtype=torch.bfloat16)],
                    torch.tensor([3, 4], device=device, dtype=torch.int64),
                    torch.tensor([0, 4], device=device, dtype=torch.int64),
                    torch.tensor([1, 3], device=device, dtype=torch.int64),
                    ngram_size=2,
                    eos_token_id=0,
                ),
                k,
                v,
                conv,
                norm,
                norm,
                norm,
                hc_mult=4,
                hidden_size=32,
                conv_kernel_size=2,
                norm_eps=1e-6,
                projection_scales=scales,
            )

        actual = layer(key, value, (key_scale, value_scale))
        reference = layer(
            (key.float() * key_scale).to(torch.bfloat16),
            (value.float() * value_scale).to(torch.bfloat16),
            None,
        )
        self.assertIsInstance(actual.key_projection, Fp8Linear)
        self.assertIsInstance(actual.value_projection, Fp8Linear)
        self.assertEqual(actual.key_projection.weight.data_ptr(), key.data_ptr())
        torch.testing.assert_close(
            actual.key_projection.scale, key_scale, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual.value_projection.scale, value_scale, rtol=0, atol=0
        )
        hidden = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
        # Max magnitude seven makes the dynamic FP8 input scale exactly 1/64.
        embeddings = torch.randint(-7, 8, (2, 4, 64), device=device).to(torch.bfloat16)
        embeddings.flatten()[0] = 7
        history = torch.zeros(2, 5, device=device, dtype=torch.long)
        state = torch.zeros(2, 2, 128, device=device, dtype=torch.bfloat16)

        def forward(module):
            return module.decode_chunk(hidden, history, state, embeddings=embeddings)

        for _ in range(3):
            forward(actual)
        expected = forward(reference)
        eager = forward(actual)
        for a, b in zip(eager, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward(actual)
        embeddings.neg_()
        hidden.neg_()
        graph.replay()
        for a, b in zip(output, forward(reference)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_dense_quantization_handles_zero_activations(self):
        weight, scale = quantized(torch.randn(32, 256, device="cuda"))
        module = Fp8Linear(weight, scale)
        result = module(torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.count_nonzero().item(), 0)

    def test_ple_pinned_owner_replaces_pageable_storage_inventory(self):

        from torch import nn

        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel

        storage = ModelWeights(1, "cpu", torch.bfloat16)
        table, scale = quantized(torch.randn(8, 32))
        name = W.qwen4_ple_ngram_shards + ".0"
        storage.weights[0] = {
            name: table,
            W.qwen4_ple_ngram_scale: scale,
            W.qwen4_ple_ngram_vocab_sizes: torch.tensor([3, 4], device="cuda"),
            W.qwen4_ple_ngram_offsets: torch.tensor([0, 4], device="cuda"),
            W.qwen4_ple_multipliers: torch.tensor([2, 3], device="cuda"),
            W.qwen4_ple_key_proj_w: torch.zeros(
                32, 64, device="cuda", dtype=torch.bfloat16
            ),
            W.qwen4_ple_value_proj_w: torch.zeros(
                8, 64, device="cuda", dtype=torch.bfloat16
            ),
            W.qwen4_ple_conv_w: torch.zeros(
                32, 1, 2, device="cuda", dtype=torch.bfloat16
            ),
            W.qwen4_ple_norm_key_gamma: torch.zeros(
                32, device="cuda", dtype=torch.bfloat16
            ),
            W.qwen4_ple_norm_query_gamma: torch.zeros(
                32, device="cuda", dtype=torch.bfloat16
            ),
            W.qwen4_ple_norm_conv_gamma: torch.zeros(
                32, device="cuda", dtype=torch.bfloat16
            ),
        }
        model = Qwen4ExpModel.__new__(Qwen4ExpModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            _qwen4_ple_layer_ids=[1],
            _qwen4_split_ngram_parts=1,
            _qwen4_ngram_size=2,
            hc_mult=4,
            hidden_size=8,
            layernorm_eps=1e-6,
            special_tokens=SimpleNamespace(eos_token_id=0),
        )
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 1, get_attn_tp_rank=lambda: 0
        )
        model.layers = nn.ModuleList([nn.Identity()])
        model.ple_layers = nn.ModuleDict()
        model.weight = compute_weight_view(storage)
        model._build_ple_layers()
        owner = model.ple_layers["0"].ple_embedding.shards[0]
        self.assertTrue(owner.is_pinned())
        self.assertIs(storage.weights[0][name], owner)
        self.assertIs(model.weight.weights[0][name], owner)
        self.assertNotEqual(owner.data_ptr(), table.data_ptr())

    def test_fp8_device_vocab_graph(self):
        weight, scale = quantized(torch.randn(257, 320))
        ids = torch.tensor([0, 17, 256, -1], dtype=torch.int32, device="cuda")
        stored = weight.cuda()
        module = Fp8VocabEmbedding(
            stored, 1, torch.device("cuda", 0), weight_scale=scale
        )
        for _ in range(3):
            module(ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(ids)
        ids.copy_(torch.tensor([256, 3, 19, -1], dtype=torch.int32, device="cuda"))
        graph.replay()
        expected = (weight[[256, 3, 19]].float() * scale).to(torch.bfloat16).cuda()
        torch.testing.assert_close(output[:3], expected, rtol=0.008, atol=0.02)
        self.assertEqual(output[-1].count_nonzero().item(), 0)
        self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
