"""CPU contracts for always-on optimizations; no GPU accuracy claim."""

import ast
import inspect
import os
import textwrap
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal import mm_process_engine as process
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import packed_vision
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    MLP,
    RMSNorm,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.packed_graph import (
    PackedVisionGraph,
)

REMOVED_FLAGS = (
    "V41_GPU_PREPROCESS",
    "V41_PACKED_FORWARD",
    "V41_PACKED_CUDA_GRAPH",
    "V41_GPU_PREPROCESS_TABLE_CACHE",
    "V41_PACKED_ROPE_BHSD",
    "V41_VISION_SILU_MUL_FUSED",
    "V41_VISION_SILU_MUL_DIRECT",
    "V41_VISION_RMSNORM_BACKEND",
    "V41_VISION_RMSNORM_FUSED",
    "V41_VISION_RMSNORM_EPILOGUE_FUSED",
    "V41_VISION_RESIDUAL_RMSNORM_FUSED",
    "V41_VISION_RMSNORM_REDUCE_FUSED",
    "V41_PACKED_MAX_IMAGES",
    "V41_PACKED_MAX_PATCHES",
    "V41_PACKED_STATS",
    "V41_GPU_PREFETCH_MAX_BYTES",
    "RTP_MM_FAST_INPUT_METADATA",
    "RTP_MM_PREFETCH_FILL_BATCH",
)


def tiny_config():
    return V41Config.from_dict(
        {
            "text_config": {
                "hidden_size": 8,
                "vocab_size": 129280,
                "max_position_embeddings": 128,
            },
            "vision_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "patch_size": 14,
                "downsample_ratio": 3,
                "rope_theta": 10000.0,
                "max_image_tokens": 1024,
                "min_pixels": 295936,
            },
            "quantization_config": {},
            "dtype": "bfloat16",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "image_token_id": 129264,
        }
    )


class DefaultOptimizationsTest(unittest.TestCase):
    def setUp(self):
        environment = {name: "0" for name in REMOVED_FLAGS}
        environment["V41_VISION_ATTN_BACKEND"] = "sdpa"
        self.env = patch.dict(os.environ, environment)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.model = DeepSeekV41VisionEmbedding(tiny_config(), device="cpu").eval()

    def test_packed_and_graph_enabled_without_environment_switches(self):
        self.assertTrue(self.model.packed_forward_stats()["enabled"])
        self.assertTrue(self.model.packed_forward_stats()["stats_enabled"])
        self.assertIsInstance(self.model._packed_graph, PackedVisionGraph)
        self.assertEqual(self.model._packed_graph.full_batch_images, 126)
        self.assertEqual(self.model._packed_max_images, 126)
        self.assertEqual(self.model.get_batch_work_budget(126).input_patches, 1058400)
        self.assertNotIn("_packed_forward_enabled", vars(self.model))

    def test_gpu_preprocess_follows_device_without_touching_cuda(self):
        self.assertFalse(self.model.get_preprocess_params()["gpu_preprocess"])
        self.assertIsNone(self.model.prepare_embedding_batch([], []))
        with patch.object(
            DeepSeekV41VisionEmbedding, "_device", new_callable=PropertyMock
        ) as device:
            device.return_value = torch.device("cuda:0")
            self.assertTrue(self.model.get_preprocess_params()["gpu_preprocess"])
        default = inspect.signature(self.model.preprocess_input).parameters[
            "gpu_preprocess"
        ]
        self.assertIs(default.default, True)

    def test_cpu_graph_fallback_still_owns_results(self):
        source = torch.arange(4.0).reshape(4, 1)
        with torch.inference_mode():
            result = self.model._packed_graph.run(
                [source],
                [(2, 2)],
                lambda patches, grids: patches,
                lambda outputs: [value.clone() for value in outputs],
            )
        self.assertTrue(torch.equal(result[0], source))
        self.assertNotEqual(result[0].data_ptr(), source.data_ptr())
        self.assertEqual(self.model._packed_graph.stats()["captures"], 0)
        self.assertEqual(self.model._packed_graph.stats()["eager"], 1)

    def test_silu_uses_exact_lookup_helper_by_default(self):
        from rtp_llm.models_py.modules.dsv41 import vision_silu_mul_fused as fused

        mlp = MLP({"vision_dim": 8, "vision_inter_dim": 16}).eval()
        value = torch.ones(2, 8)
        intermediate = torch.full((2, 16), 0.25)
        with torch.inference_mode(), patch.object(
            fused, "apply_vision_silu_mul", return_value=intermediate
        ) as helper:
            actual = mlp(value)
            expected = mlp.w2(intermediate)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(helper.call_count, 1)
        self.assertEqual(helper.call_args.kwargs, {})  # Never selects direct-exp.

    def test_unsupported_silu_retains_original_math(self):
        from rtp_llm.models_py.modules.dsv41 import vision_silu_mul_fused as fused

        mlp = MLP({"vision_dim": 8, "vision_inter_dim": 16}).eval()
        value = torch.ones(2, 8)
        with torch.inference_mode():
            gate, up = mlp.w1(value).chunk(2, dim=-1)
            expected = mlp.w2(F.silu(gate) * up)
            self.assertIsNone(fused.apply_vision_silu_mul(gate, up))
            actual = mlp(value)
        self.assertTrue(torch.equal(actual, expected))

    def test_bhsd_layout_is_default_and_reference_override_remains(self):
        default = inspect.signature(packed_vision.forward_packed).parameters[
            "rope_bhsd"
        ]
        self.assertIs(default.default, True)

    def test_device_tables_always_use_bounded_cache(self):
        from rtp_llm.models_py.modules.dsv41 import vision_preprocess_triton as prep

        factory = Mock()
        tables = (torch.ones(2),)
        with patch.object(prep._DEVICE_TABLES, "get", return_value=tables) as get:
            actual = prep._tables_on_device("cuda:0", ("shape", 14), factory)
        self.assertIs(actual, tables)
        get.assert_called_once_with("cuda:0", ("shape", 14), factory)
        factory.assert_not_called()

    def test_native_metadata_does_not_materialize_url(self):
        class Input:
            def has_url(self):
                return True

            @property
            def url(self):
                raise AssertionError("native metadata copied the URL")

        self.assertTrue(process._input_has_url(Input()))
        self.assertTrue(
            process._input_has_url(type("Legacy", (), {"url": "image.jpg"})())
        )
        self.assertFalse(process._input_has_url(type("Legacy", (), {"url": ""})()))

    def test_lazy_cache_key_default_preserves_required_identity(self):
        media, cache = Mock(), Mock()
        cache.try_acquire_lazy.return_value = ("", "miss", "entry")
        self.assertEqual(
            process._acquire_input_cache(cache, media), ("", "miss", "entry")
        )
        cache.try_acquire_lazy.assert_called_once_with(media.cache_key)
        media.cache_key.assert_not_called()
        media.cache_key.return_value = "key"
        cache.try_acquire.return_value = ("hit", "cached")
        self.assertEqual(
            process._acquire_input_cache(cache, media, require_key=True),
            ("key", "hit", "cached"),
        )
        media.cache_key.assert_called_once_with()

    def test_prefetch_fill_follows_existing_resource_budget(self):
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(process.MMProcessEngine.__init__))
        )
        scheduler = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MMScheduler"
        )
        expression = next(
            k.value for k in scheduler.keywords if k.arg == "prefetch_fill_batch"
        )
        code = compile(ast.Expression(expression), "<prefetch-wiring>", "eval")
        for settings, expected in (
            ({}, False),
            ({"prefetch_batches": 0}, False),
            ({"prefetch_batches": 1}, True),
        ):
            self.assertIs(eval(code, {"scheduler_args": settings}), expected)

    def test_only_attention_selector_remains_configurable(self):
        self.assertEqual(self.model.vision.blocks[0].attn._attention_backend, "sdpa")
        self.assertEqual(self.model.vision.norm._rmsnorm_backend, "flashinfer")
        self.assertFalse(hasattr(self.model.vision.norm, "_fused_rmsnorm"))
        with patch.dict(
            os.environ,
            V41_VISION_RMSNORM_BACKEND="native",
            V41_VISION_ATTN_BACKEND="fa4",
        ):
            model = DeepSeekV41VisionEmbedding(tiny_config(), device="cpu").eval()
        self.assertEqual(model.vision.blocks[0].attn._attention_backend, "fa4")
        self.assertEqual(model.vision.norm._rmsnorm_backend, "flashinfer")
        self.assertTrue(
            all(
                m.weight.dtype == torch.bfloat16
                for m in model.vision.modules()
                if isinstance(m, RMSNorm)
            )
        )

    def test_common_media_limit_bounds_model_chunks_and_graph(self):
        for limit, expected in ((1, 1), (8, 8), (126, 126), (192, 126)):
            with self.subTest(limit=limit):
                model = DeepSeekV41VisionEmbedding(
                    tiny_config(), device="cpu", max_batch_images=limit
                )
                self.assertEqual(model._packed_max_images, expected)
                self.assertEqual(model._packed_graph.full_batch_images, expected)
        with self.assertRaisesRegex(ValueError, "budget must be positive"):
            DeepSeekV41VisionEmbedding(tiny_config(), device="cpu", max_batch_images=0)

    def test_norm_cpu_and_autograd_keep_regular_expression(self):
        norm = RMSNorm(8).eval()
        value = torch.linspace(-1, 1, 24).reshape(3, 8).requires_grad_(True)
        with patch.object(
            norm,
            "_flashinfer_rmsnorm",
            side_effect=AssertionError("CPU/autograd must not launch CUDA norm"),
        ):
            actual = norm(value)
        expected = norm.weight * (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm.eps)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(norm.weight.grad)

    def test_norm_cuda_inference_dispatch_and_failure_propagation(self):
        norm = RMSNorm(8).eval()
        fake_cuda = Mock(is_cuda=True)
        with torch.inference_mode():
            self.assertTrue(norm._uses_flashinfer(fake_cuda))
        with torch.enable_grad():
            self.assertFalse(norm._uses_flashinfer(fake_cuda))
        norm.train()
        with torch.inference_mode():
            self.assertFalse(norm._uses_flashinfer(fake_cuda))
        norm.eval()
        with patch.object(norm, "_uses_flashinfer", return_value=True), patch.object(
            norm, "_flashinfer_rmsnorm", side_effect=RuntimeError("norm launch failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "norm launch failed"):
                norm(torch.zeros(1, 8))

    def test_only_attention_reads_model_specific_environment(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
            deepseek_v41_vision,
            deepseek_vision,
        )

        reads = []
        for module in (deepseek_v41_vision, deepseek_vision, packed_vision):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "getenv")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.startswith("V41_")
                ):
                    reads.append(node.args[0].value)
        self.assertEqual(reads, ["V41_VISION_ATTN_BACKEND"])

    def test_attention_defaults_to_fa4_without_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            model = DeepSeekV41VisionEmbedding(tiny_config(), device="cpu").eval()
        self.assertEqual(model.vision.blocks[0].attn._attention_backend, "fa4")

    def test_invalid_attention_selector_is_rejected(self):
        with patch.dict(os.environ, V41_VISION_ATTN_BACKEND="unknown"):
            with self.assertRaisesRegex(ValueError, "must be sdpa or fa4"):
                DeepSeekV41VisionEmbedding(tiny_config(), device="cpu")


if __name__ == "__main__":
    unittest.main()
