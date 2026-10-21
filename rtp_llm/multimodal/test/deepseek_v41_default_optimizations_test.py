"""CPU contracts for always-on optimizations; no GPU accuracy claim."""

import ast
import inspect
import os
import sys
import textwrap
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal import mm_process_engine as process
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import packed_vision
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import deepseek_vision
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
        self.assertEqual(self.model._packed_graph.max_batch_images, sys.maxsize)
        self.assertEqual(self.model._packed_max_images, sys.maxsize)
        budget = self.model.get_batch_work_budget(126)
        self.assertEqual(budget.input_patches, 0)
        self.assertGreater(budget.estimated_workspace_bytes, 0)
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
        for limit, expected in ((1, 1), (8, 8), (126, 126), (192, 192), (256, 256)):
            with self.subTest(limit=limit):
                model = DeepSeekV41VisionEmbedding(
                    tiny_config(), device="cpu", max_batch_images=limit
                )
                self.assertEqual(model._packed_max_images, expected)
                self.assertEqual(model._packed_graph.max_batch_images, expected)
        with self.assertRaisesRegex(ValueError, "budget must be positive"):
            DeepSeekV41VisionEmbedding(tiny_config(), device="cpu", max_batch_images=0)

    def test_workspace_is_actual_geometry_not_reference_image_count(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            _RuntimeImage,
        )

        small = _RuntimeImage(torch.zeros(6, 588), 2, 3)
        large = _RuntimeImage(torch.zeros(60, 588), 6, 10)
        a, b = (self.model.estimate_work(x) for x in (small, large))
        self.assertLess(a.estimated_workspace_bytes, b.estimated_workspace_bytes)
        self.assertEqual(self.model.estimate_work([small, large]), a + b)
        self.assertTrue(self.model.strict_work_budget)

    def test_192_images_reach_one_forward_without_hidden_126_ceiling(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            _RuntimeImage,
        )

        image = _RuntimeImage(torch.zeros(9, 588), 3, 3)
        with patch.object(self.model, "_forward_packed") as forward:
            forward.return_value = [torch.zeros(1, 8)] * 192
            values = self.model.encode_images_packed([image] * 192)
        self.assertEqual(len(values), 192)
        self.assertEqual(len(forward.call_args.args[0]), 192)
        self.assertEqual(forward.call_count, 1)

    def test_byte_budget_splits_actual_work_and_rejects_oversized_image(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
            _RuntimeImage,
        )

        image = _RuntimeImage(torch.zeros(9, 588), 3, 3)
        cost = self.model.estimate_work(image).estimated_workspace_bytes
        self.model._batch_max_bytes = cost * 2
        batches = []

        def forward(parts, grids, **kwargs):
            batches.append(len(parts))
            return [torch.zeros(1, 8)] * len(parts)

        with patch.object(self.model, "_forward_packed", side_effect=forward):
            self.model.encode_images_packed([image] * 3)
        self.assertEqual(batches, [2, 1])
        self.model._batch_max_bytes = cost - 1
        self.model._resource_budget = None
        with self.assertRaisesRegex(ValueError, "workspace budget"):
            self.model.encode_images_packed([image])

    def test_mixin_role_controls_graph_reservation_not_image_cap(self):
        from types import SimpleNamespace

        from rtp_llm.config.py_config_modules import VitConfig
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_mixin import (
            DeepSeekV41Mixin,
        )
        from rtp_llm.ops import VitSeparation

        cfg = VitConfig()
        cfg.gpu_max_batch_size = 256
        related = SimpleNamespace(config={"v41_config": tiny_config()})
        local = DeepSeekV41Mixin._get_mm_module(related, cfg)
        self.assertEqual(local._packed_max_images, 256)
        self.assertEqual(local._packed_graph.max_entries, 0)
        cfg.vit_separation = VitSeparation.VIT_SEPARATION_ROLE
        role = DeepSeekV41Mixin._get_mm_module(related, cfg)
        self.assertEqual(role._packed_max_images, 256)
        self.assertEqual(
            role._packed_graph.max_entries, cfg.gpu_graph_cache_max_entries
        )

    def test_weight_reload_invalidates_pending_shape_generation(self):
        graph = self.model._packed_graph
        generation = graph._generation
        graph._pending["stale"] = 1
        self.model.load_state_dict(self.model.state_dict())
        self.assertFalse(graph._pending)
        self.assertGreater(graph._generation, generation)

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

    def test_residual_norm_fusion_dispatch_fallback_and_failure(self):
        norm = RMSNorm(1024).eval()
        value = torch.ones(2, 1024, dtype=torch.bfloat16)
        residual = torch.full_like(value, 2)
        fused = (torch.full_like(value, 3), torch.full_like(value, 4))
        with torch.inference_mode(), patch.object(
            norm, "_uses_flashinfer", return_value=True
        ), patch.object(
            deepseek_vision, "apply_vision_residual_rmsnorm", return_value=fused
        ) as helper:
            self.assertIs(norm.forward_with_residual(value, residual), fused)
        helper.assert_called_once_with(value, residual, norm.weight, norm.eps)

        normalized = torch.full_like(value, 5)
        with torch.inference_mode(), patch.object(
            norm, "_uses_flashinfer", return_value=True
        ), patch.object(
            deepseek_vision, "apply_vision_residual_rmsnorm", return_value=None
        ), patch.object(
            norm, "_flashinfer_rmsnorm", return_value=normalized
        ) as fallback:
            summed, actual = norm.forward_with_residual(value, residual)
        torch.testing.assert_close(summed, value + residual, rtol=0, atol=0)
        self.assertIs(actual, normalized)
        fallback.assert_called_once_with(summed)

        with torch.inference_mode(), patch.object(
            norm, "_uses_flashinfer", return_value=True
        ), patch.object(
            deepseek_vision,
            "apply_vision_residual_rmsnorm",
            side_effect=RuntimeError("fused residual norm failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "fused residual norm failed"):
                norm.forward_with_residual(value, residual)

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
