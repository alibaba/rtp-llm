"""Qwen4 checkpoint/assembly tests. Processor fixture is not raw-media coverage."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import requires_native_fst
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewLoaderLoadMethod
from rtp_llm.models_py.new_models.qwen4_exp.media_text_embedding import (
    Qwen4MediaTextEmbedding,
    load_qwen4_media_text_embedding,
)
from rtp_llm.models_py.new_models.qwen_service_vision import load_service_vision
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)


def make_visual(cfg):
    config = Qwen3_5MoeVisionConfig(**cfg)
    config._attn_implementation = "sdpa"
    return Qwen3_5MoeVisionModel(config).eval()


class Qwen4MediaEmbeddingTest(unittest.TestCase):
    def test_pure_main_assembly_after_real_checkpoint_load(self):
        # Exercise the actual pure processor file without the unrelated eager
        # mixin package initializer; the full mixin test below remains separate.
        path = (
            Path(__file__).parents[4]
            / "multimodal/multimodal_mixins/qwen4_exp_processor.py"
        )
        spec = importlib.util.spec_from_file_location(
            "qwen4_processor_direct_test", path
        )
        processor_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = processor_module
        spec.loader.exec_module(processor_module)
        cfg = dict(
            depth=2,
            hidden_size=16,
            intermediate_size=24,
            num_heads=2,
            hidden_act="gelu_pytorch_tanh",
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=8,
            num_position_embeddings=16,
        )
        source = make_visual(cfg)
        table = torch.randn(20, 8)
        weights = {
            "model.visual." + name: value.clone()
            for name, value in source.state_dict().items()
        }
        weights["model.language_model.embed_tokens.weight"] = table
        processor = types.SimpleNamespace(
            image_processor=types.SimpleNamespace(merge_size=2),
            image_token_id=18,
            video_token_id=19,
            vision_start_token_id=2,
            vision_end_token_id=3,
            tokenizer=types.SimpleNamespace(encode=lambda text, **kwargs: [4, 5]),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, os.path.join(directory, "model.safetensors"))
            visual = load_service_vision(
                make_visual(cfg),
                directory,
                torch.float32,
                "cpu",
                "scratch",
                prefix=None,
            )
            loaded_table = load_qwen4_media_text_embedding(
                directory, torch.float32
            ).weight
        spatial = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
        for frames, timestamps in ((1, None), (2, [0.0, 1.0])):
            grid = torch.tensor([[frames, 4, 4]])
            pixels = torch.randn(frames * 16, 24)
            with torch.inference_mode():
                features = visual(pixels, grid).pooler_output
            data = processor_module.Qwen4VisualInput(pixels, grid, timestamps)
            actual, positions, semantic_ids = (
                processor_module.assemble_visual_embedding(
                    data, features, spatial, processor, loaded_table
                )
            )
            tokens, expected_parts = [], []
            for frame in range(frames):
                prefix = [2] if timestamps is None else [4, 5, 2]
                tokens += prefix + ([18] if timestamps is None else [19]) * 4 + [3]
                expected_parts.extend(
                    [table[prefix], features[frame * 4 : (frame + 1) * 4], table[[3]]]
                )
            torch.testing.assert_close(
                actual, torch.cat(expected_parts), rtol=0, atol=0
            )
            self.assertEqual(semantic_ids.tolist(), tokens)
            self.assertEqual(tuple(positions.shape), (len(tokens), 3))

    def test_cpu_full_vocab_priority_and_dtype_not_visual_fp16_staging(self):
        value = torch.arange(120, dtype=torch.float32).reshape(15, 8) * 0.01
        value[0, 0] = 1e-9
        for indexed in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                filename = "model.safetensors"
                save_file(
                    {
                        "model.language_model.embed_tokens.weight": value,
                        "model.embed_tokens.weight": torch.full_like(value, -1),
                    },
                    os.path.join(directory, filename),
                )
                if indexed:
                    with open(
                        os.path.join(directory, "model.safetensors.index.json"), "w"
                    ) as stream:
                        json.dump(
                            {
                                "weight_map": {
                                    key: filename
                                    for key in (
                                        "model.language_model.embed_tokens.weight",
                                        "model.embed_tokens.weight",
                                    )
                                }
                            },
                            stream,
                        )
                loaded = load_qwen4_media_text_embedding(
                    directory, torch.bfloat16
                ).weight
            self.assertEqual(loaded.device.type, "cpu")
            self.assertEqual(tuple(loaded.shape), (15, 8))
            torch.testing.assert_close(loaded, value.bfloat16(), rtol=0, atol=0)
            self.assertNotEqual(loaded[0, 0].item(), 0)

    def test_fallback_name_and_missing_weight(self):
        value = torch.ones(9, 8)
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"model.embed_tokens.weight": value},
                os.path.join(directory, "model.safetensors"),
            )
            torch.testing.assert_close(
                load_qwen4_media_text_embedding(directory, torch.float32).weight, value
            )
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"model.norm.weight": torch.ones(8)},
                os.path.join(directory, "model.safetensors"),
            )
            with self.assertRaisesRegex(ValueError, "missing its text embedding"):
                load_qwen4_media_text_embedding(directory, torch.float32)

    def test_cpu_only_replicated_contract(self):
        cfg = dict(checkpoint_name="model.embed_tokens.weight", embedding_shape=[9, 8])
        with self.assertRaisesRegex(ValueError, "CPU"):
            Qwen4MediaTextEmbedding(cfg, NewLoaderConfig(device="cuda"))
        with self.assertRaisesRegex(ValueError, "replicated"):
            Qwen4MediaTextEmbedding(
                cfg, NewLoaderConfig(device="cpu", tp_size=2, tp_rank=1)
            )

    def test_real_pytorch_checkpoint(self):
        value = torch.randn(9, 8)
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {"model.embed_tokens.weight": value},
                os.path.join(directory, "pytorch_model.bin"),
            )
            torch.testing.assert_close(
                load_qwen4_media_text_embedding(directory, torch.float32).weight, value
            )

    @requires_native_fst
    def test_explicit_native_fst_cpu_without_scratch(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"), native.__file__)
        value = torch.randn(15, 8)
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"model.embed_tokens.weight": value},
                os.path.join(directory, "model.safetensors"),
            )
            with patch(
                "rtp_llm.models_py.weight_mapper.get_all_weights",
                side_effect=AssertionError("scratch reader must not run"),
            ):
                owner = load_qwen4_media_text_embedding(
                    directory, torch.float32, NewLoaderLoadMethod.FASTSAFETENSORS
                )
            torch.testing.assert_close(owner.weight, value, rtol=0, atol=0)
            self.assertEqual(owner.weight.device.type, "cpu")

    def test_real_mixin_load_image_video_assembly_without_legacy_database(self):
        self._check_real_mixin(NewLoaderLoadMethod.SCRATCH)

    @requires_native_fst
    def test_real_mixin_native_fst_image_video_without_scratch(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"), native.__file__)
        with patch(
            "rtp_llm.models_py.weight_mapper.get_all_weights",
            side_effect=AssertionError("scratch reader must not run"),
        ):
            self._check_real_mixin(NewLoaderLoadMethod.FASTSAFETENSORS)

    def _check_real_mixin(self, method):
        from rtp_llm.multimodal.multimodal_mixins.qwen4_exp_mixin import Qwen4ExpMixin
        from rtp_llm.multimodal.multimodal_mixins.qwen4_exp_processor import (
            Qwen4VisualInput,
        )

        cfg = dict(
            depth=2,
            hidden_size=16,
            intermediate_size=24,
            num_heads=2,
            hidden_act="gelu_pytorch_tanh",
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=8,
            num_position_embeddings=16,
        )
        torch.manual_seed(97)
        visual = make_visual(cfg)
        table = torch.randn(20, 8)
        weights = {
            "model.visual." + name: value.clone()
            for name, value in visual.state_dict().items()
        }
        weights["model.language_model.embed_tokens.weight"] = table
        processor = types.SimpleNamespace(
            image_processor=types.SimpleNamespace(merge_size=2),
            image_token_id=18,
            video_token_id=19,
            vision_start_token_id=2,
            vision_end_token_id=3,
            tokenizer=types.SimpleNamespace(encode=lambda text, **kwargs: [4, 5]),
        )
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, os.path.join(directory, "model.safetensors"))
            visual.config.save_pretrained(directory)
            with patch(
                "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin.AutoProcessor.from_pretrained",
                return_value=processor,
            ):
                owner = Qwen4ExpMixin.from_new_loader(
                    torch.float32,
                    "cpu",
                    types.SimpleNamespace(config={"ckpt_path": directory}),
                    method,
                    None,
                    directory,
                )
            self.assertFalse(hasattr(owner, "mm_mixin_loader"))
            self.assertFalse(hasattr(owner.mm_related_params, "vit_weights"))
            self.assertEqual(owner.mm_part.text_embedding.device.type, "cpu")
            self.assertIs(
                owner.mm_part.text_embedding,
                owner.mm_part.text_embedding_owner.weight,
            )
            with torch.no_grad():
                for param in visual.parameters():
                    param.copy_(param.half().float())
            for actual, expected in zip(
                owner.mm_part.visual.parameters(), visual.parameters()
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for frames, timestamps in ((1, None), (2, [0.0, 1.0])):
                grid = torch.tensor([[frames, 4, 4]])
                pixels = torch.randn(frames * 16, 24)
                data = Qwen4VisualInput(pixels, grid, timestamps)
                with torch.inference_mode():
                    features = visual(pixels, grid).pooler_output
                    actual, positions, semantic_ids = owner.mm_part.embedding(data)
                tokens = []
                expected_parts = []
                for frame in range(frames):
                    prefix = [2] if timestamps is None else [4, 5, 2]
                    tokens += prefix + ([18] if timestamps is None else [19]) * 4 + [3]
                    expected_parts.extend(
                        [
                            table[prefix],
                            features[frame * 4 : (frame + 1) * 4],
                            table[[3]],
                        ]
                    )
                torch.testing.assert_close(
                    actual, torch.cat(expected_parts), rtol=1e-5, atol=1e-6
                )
                self.assertEqual(semantic_ids.tolist(), tokens)
                self.assertEqual(tuple(positions.shape), (len(tokens), 3))
                if timestamps is None:
                    self.assertEqual(
                        positions.tolist(),
                        [
                            [0, 0, 0],
                            [1, 1, 1],
                            [1, 1, 2],
                            [1, 2, 1],
                            [1, 2, 2],
                            [3, 3, 3],
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
