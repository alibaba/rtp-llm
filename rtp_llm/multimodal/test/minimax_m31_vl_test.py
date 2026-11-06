import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.minimax_m31_vl.image_processor import (
    MiniMaxM31VLImageProcessor,
    get_hw_multiple_of,
    smart_resize,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m31_vl.minimax_m31_vl_mixin import (
    MiniMaxM31VLImageEmbedding,
    MiniMaxM31VLMixin,
    _MiniMaxM31VLVisionGraphCache,
)
from rtp_llm.multimodal.multimodal_mixins.minimax_m31_vl.minimax_m31_vl_vit import (
    MiniMaxM31VLVisionModel,
    MiniMaxM31VLVisionTower,
    get_fused_qkv_checkpoint_names,
)


class MiniMaxM31VLTest(unittest.TestCase):
    def test_mixin_rejects_non_bf16_before_visual_initialization(self):
        with self.assertRaisesRegex(ValueError, "ACT_TYPE=bf16"):
            MiniMaxM31VLMixin(torch.float16)

    def test_transformers_fast_image_processor_import_contract(self):
        from transformers.image_processing_utils_fast import (
            BaseImageProcessorFast,
            group_images_by_shape,
            reorder_images,
        )

        self.assertTrue(callable(group_images_by_shape))
        self.assertTrue(callable(reorder_images))
        self.assertTrue(callable(BaseImageProcessorFast.from_pretrained))

    def test_image_processor_round_trip_pretrained_and_preprocess(self):
        with tempfile.TemporaryDirectory() as root:
            MiniMaxM31VLImageProcessor().save_pretrained(root)
            processor = MiniMaxM31VLImageProcessor.from_pretrained(root)
            output = processor.preprocess(
                Image.new("RGB", (56, 56)), return_tensors="pt"
            )
        self.assertGreater(output["pixel_values"].numel(), 0)
        self.assertEqual(tuple(output["image_grid_thw"].shape), (1, 3))

    def test_unindexed_word_embedding_checkpoint(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as root:
            expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            save_file(
                {"language_model.model.embed_tokens.weight": expected},
                str(Path(root) / "model.safetensors"),
            )
            actual = MiniMaxM31VLImageEmbedding._load_word_embedding(root)
            torch.testing.assert_close(actual, expected)

    def test_checkpoint_with_unimplemented_vision_post_norm_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "vision_tower.vision_model.post_layernorm.weight": "model.safetensors"
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "post_layernorm is unsupported"):
                MiniMaxM31VLImageEmbedding._load_word_embedding(root)

    def test_long_side_resize_respects_area_and_dimensions(self):
        height, width = smart_resize(
            4000, 1000, max_long_side_pixel=4096, max_pixels=451584
        )
        self.assertLessEqual(height * width, 451584)
        self.assertLessEqual(max(height, width), 4096)
        with self.assertRaises(FtRuntimeException):
            smart_resize(0, 100, max_long_side_pixel=4096)
        with self.assertRaises(FtRuntimeException):
            smart_resize(1, 10000, max_long_side_pixel=4096)
        self.assertEqual(get_hw_multiple_of((10000, 1), 28, 1000), (980, 28))

    def test_m31_only_mixin_registration(self):
        self.assertIs(get_multimodal_mixin_cls("minimax_m31_vl"), MiniMaxM31VLMixin)
        self.assertEqual(
            get_fused_qkv_checkpoint_names(
                "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj.weight"
            ),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight",
                "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight",
            ),
        )

    def test_batch_work_budget_scales_additive_costs_only(self):
        embedding = SimpleNamespace(
            mm_processor=SimpleNamespace(patch_size=14, max_pixels=19600),
            merge_size=2,
            visual=SimpleNamespace(
                vision_config=SimpleNamespace(vision_segment_max_frames=4)
            ),
            _workspace_bytes_per_patch=lambda: 10,
        )
        estimate = MiniMaxM31VLImageEmbedding.get_batch_work_budget(embedding, 2)
        self.assertEqual(estimate.input_patches, 200)
        self.assertEqual(estimate.output_tokens, 54)
        self.assertEqual(estimate.estimated_workspace_bytes, 2000)
        self.assertEqual(estimate.max_attention_segment, 400)
        self.assertEqual(estimate.attention_work, 20000)

    def test_nonpositive_frame_limit_disables_segmentation(self):
        for limit in (0, -1):
            model = SimpleNamespace(vision_segment_max_frames=limit)
            self.assertEqual(
                MiniMaxM31VLVisionModel._apply_max_frames_limit(model, [[5, 2, 2]]),
                [[5, 2, 2]],
            )

    def test_vision_graph_cache_cpu_fallback(self):
        class Visual(torch.nn.Module):
            def forward(self, pixels, grid):
                return pixels.sum(dim=0)

        cache = _MiniMaxM31VLVisionGraphCache(Visual())
        output = cache.run(torch.ones((4, 3)), torch.tensor([[1, 2, 2]]))
        self.assertEqual(output.tolist(), [4.0, 4.0, 4.0])
        self.assertEqual(cache.stats()["fallback"], 1)

    def test_vision_graph_capture_type_error_disables_shape(self):
        cache = _MiniMaxM31VLVisionGraphCache(torch.nn.Identity(), capture_after=1)
        pixels = SimpleNamespace(is_cuda=True, shape=(4, 3))
        grid = torch.tensor([[1, 2, 2]])
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch(
                "rtp_llm.multimodal.multimodal_mixins.minimax_m31_vl.minimax_m31_vl_mixin._select_attention_backend",
                return_value="flashinfer",
            ),
            patch.object(cache, "_signature", return_value=("shape",)),
            patch.object(
                cache, "_capture", side_effect=TypeError("unsupported plan")
            ) as capture,
            patch.object(cache, "_eager", return_value="eager"),
        ):
            self.assertEqual(cache.run(pixels, grid), "eager")
            self.assertEqual(cache.run(pixels, grid), "eager")
        self.assertEqual(capture.call_count, 1)

    def test_small_vision_forward_cpu(self):
        config = {
            "vision_config": {
                "hidden_size": 24,
                "num_attention_heads": 3,
                "num_hidden_layers": 1,
                "intermediate_size": 48,
                "patch_size": 2,
                "image_size": 4,
            },
            "text_config": {"hidden_size": 32},
            "projector_hidden_size": 32,
        }
        model = MiniMaxM31VLVisionTower(config).to_bfloat16().eval()
        rope = model.vision_tower.vision_model
        self.assertEqual(rope.inv_freq_t.dtype, torch.float32)
        self.assertEqual(rope.inv_freq_h.dtype, torch.float32)
        self.assertEqual(rope.inv_freq_w.dtype, torch.float32)
        phases = rope._get_3d_rope_embed(1, 2, 258, 2)
        width_phase = rope.t_dim // 2 + rope.h_dim // 2
        self.assertNotEqual(
            float(phases[512, width_phase]), float(phases[513, width_phase])
        )
        self.assertEqual(
            MiniMaxM31VLImageEmbedding._data_type.fget(SimpleNamespace(visual=model)),
            torch.bfloat16,
        )
        pixels = torch.randn(4, 24, dtype=torch.bfloat16)
        grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
        with torch.no_grad():
            output = model(pixels, grid)
        self.assertEqual(output.shape, (1, 32))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
