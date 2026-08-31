import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np
import torch

from rtp_llm.multimodal.multimodal_mixins.glm5_3_flash import (
    glm5_3_flash_mixin as glm53_mixin,
)
from rtp_llm.multimodal.multimodal_mixins.glm5_3_flash.glm5_3_flash_mixin import (
    GLM53_MM_LAYOUT_MAGIC,
    Glm53FlashImageEmbedding,
    Glm53FlashVisionAttention,
    Glm53FlashVisionModel,
    glm5_sample_frame_indices,
    glm5_smart_resize,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Glm53FlashVitTest(unittest.TestCase):
    def test_video_sampler_keeps_temporal_pairs(self):
        indices = glm5_sample_frame_indices(
            300,
            30.0,
            10.0,
            target_fps=2.0,
            max_frame_count=64,
            temporal_patch_size=2,
        )
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(indices) % 2, 0)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 299)

    def test_short_video_duplicates_tail_for_temporal_patch(self):
        self.assertEqual(
            glm5_sample_frame_indices(
                1,
                30.0,
                1 / 30,
                target_fps=2.0,
                max_frame_count=64,
                temporal_patch_size=2,
            ),
            [0, 0],
        )

    def test_odd_max_frames_is_a_strict_budget(self):
        indices = glm5_sample_frame_indices(
            300,
            30.0,
            10.0,
            target_fps=2.0,
            max_frame_count=3,
            temporal_patch_size=2,
        )
        self.assertEqual(len(indices), 2)
        self.assertLessEqual(len(indices), 3)

    def test_video_resize_accounts_for_all_frames(self):
        height, width = glm5_smart_resize(
            1080,
            1920,
            temporal_patch_size=2,
            factor=28,
            min_pixels=2 * 28 * 28,
            max_pixels=64 * 28 * 28,
            frames=16,
        )
        self.assertLessEqual(16 * height * width, 64 * 28 * 28)

    def test_interleaved_layout_tensor(self):
        layout = Glm53FlashImageEmbedding._layout_tensor(
            group_start=True,
            prefix_ids=[11],
            suffix_ids=[12, 21, 22],
        )
        self.assertEqual(
            layout.tolist(), [GLM53_MM_LAYOUT_MAGIC, 1, 1, 3, 11, 12, 21, 22]
        )

    def test_video_embedding_splits_temporal_frames_and_builds_timestamps(self):
        class _FakeVisual:
            patch_embed = SimpleNamespace(
                proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
            )

            def __call__(self, pixel_values, grid_thw):
                return torch.arange(16, dtype=torch.float32).view(4, 4)

        class _FakeTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                assert text in ("0.0 seconds", "1.0 seconds")
                assert not add_special_tokens
                return [91, 92]

        embedding = Glm53FlashImageEmbedding.__new__(Glm53FlashImageEmbedding)
        embedding.visual = _FakeVisual()
        embedding.special_token_ids = {"image_start": 11, "image_end": 12}
        embedding._timestamp_tokenizer = _FakeTokenizer()

        features, positions, layouts = embedding.embedding(
            (
                torch.zeros(4, 6),
                torch.tensor([[2, 2, 2]], dtype=torch.int64),
                [0, 1],
            )
        )

        self.assertIsNone(positions)
        self.assertEqual([feature.shape for feature in features], [(2, 4), (2, 4)])
        self.assertEqual(
            layouts[0].tolist(),
            [GLM53_MM_LAYOUT_MAGIC, 1, 1, 3, 11, 12, 91, 92],
        )
        self.assertEqual(
            layouts[1].tolist(),
            [GLM53_MM_LAYOUT_MAGIC, 0, 1, 3, 11, 12, 91, 92],
        )

    def test_video_preprocess_samples_once_and_honors_request_budget(self):
        class _FakeBatch:
            def __init__(self, frames):
                self.frames = frames

            def asnumpy(self):
                return self.frames

        class _FakeVideoReader:
            requested_indices = None

            def __init__(self, data, ctx, num_threads):
                self.frames = np.zeros((4, 56, 84, 3), dtype=np.uint8)

            def __len__(self):
                return len(self.frames)

            def get_avg_fps(self):
                return 2.0

            def get_batch(self, indices):
                _FakeVideoReader.requested_indices = list(indices)
                return _FakeBatch(self.frames[indices])

        class _FakeVideoProcessor:
            frames = None

            def preprocess(self, *, videos, return_tensors, do_resize):
                self.frames = videos
                self.assertions = (return_tensors, do_resize)
                return {
                    "pixel_values_videos": torch.zeros(4, 6),
                    "video_grid_thw": torch.tensor([[2, 2, 2]]),
                }

        preprocess_config = SimpleNamespace(
            fps=2.0,
            min_pixels=2 * 28 * 28,
            max_pixels=4 * 28 * 28,
            max_frames=4,
        )
        mm_input = SimpleNamespace(
            url="memory://video",
            mm_type=MMUrlType.VIDEO,
            mm_preprocess_config=preprocess_config,
        )
        media_config = {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "min_image_tokens": 16,
            "max_image_tokens": 240000,
            "fps": 2,
        }
        processor_config = {
            "image_processor": media_config,
            "video_processor": media_config,
        }
        video_processor = _FakeVideoProcessor()

        with (
            patch.object(glm53_mixin, "VideoReader", _FakeVideoReader),
            patch.object(glm53_mixin, "cpu", lambda _: object()),
            patch.object(
                glm53_mixin,
                "get_bytes_io_from_url",
                return_value=object(),
            ),
        ):
            pixel_values, grid_thw, timestamps = (
                Glm53FlashImageEmbedding.preprocess_input(
                    [mm_input],
                    SimpleNamespace(download_headers="", mm_video_max_frames=8),
                    processor=None,
                    video_processor=video_processor,
                    processor_config=processor_config,
                )
            )

        self.assertEqual(_FakeVideoReader.requested_indices, [0, 1, 2, 3])
        self.assertEqual(video_processor.assertions, ("pt", False))
        self.assertEqual(
            [frame.size for frame in video_processor.frames], [(28, 28)] * 4
        )
        self.assertEqual(pixel_values.shape, (4, 6))
        self.assertEqual(grid_thw.tolist(), [[2, 2, 2]])
        self.assertEqual(timestamps, [0, 1])

    def test_real_mp4_decord_preprocess(self):
        video_path = Path(os.environ["TEST_TMPDIR"]) / "glm53_decord_test.mp4"
        with av.open(str(video_path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=4)
            stream.width = 84
            stream.height = 56
            stream.pix_fmt = "yuv420p"
            for frame_index in range(8):
                pixels = np.zeros((56, 84, 3), dtype=np.uint8)
                pixels[:, :, frame_index % 3] = 32 * frame_index
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        self.assertGreater(video_path.stat().st_size, 0)

        media_config = {
            "do_rescale": True,
            "patch_expand_factor": 1,
            "merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "temporal_patch_size": 2,
            "patch_size": 14,
            "min_image_tokens": 16,
            "max_image_tokens": 240000,
            "fps": 2,
        }
        video_processor = glm53_mixin.Qwen2VLImageProcessor(
            do_resize=False,
            do_rescale=True,
            image_mean=media_config["image_mean"],
            image_std=media_config["image_std"],
            patch_size=14,
            temporal_patch_size=2,
            merge_size=2,
        )
        preprocess_config = MMPreprocessConfig(
            -1,
            -1,
            -1,
            128 * 2 * 28 * 28,
            2.0,
            -1,
            8,
            [],
            -1,
            -1,
        )
        mm_input = MultimodalInput(
            str(video_path),
            int(MMUrlType.VIDEO),
            torch.empty(0),
            preprocess_config,
        )

        pixel_values, grid_thw, timestamps = (
            Glm53FlashImageEmbedding.preprocess_input(
                [mm_input],
                SimpleNamespace(download_headers="", mm_video_max_frames=8),
                processor=None,
                video_processor=video_processor,
                processor_config={
                    "image_processor": media_config,
                    "video_processor": media_config,
                },
            )
        )

        grid_t, grid_h, grid_w = grid_thw[0].tolist()
        self.assertGreater(grid_t, 0)
        self.assertLessEqual(grid_t, 4)
        self.assertEqual(len(timestamps), grid_t)
        self.assertEqual(pixel_values.ndim, 2)
        self.assertEqual(pixel_values.shape[0], grid_t * grid_h * grid_w)
        self.assertEqual(pixel_values.shape[1], 3 * 2 * 14 * 14)
        self.assertLessEqual(grid_t * grid_h * grid_w // 4, 128)

    def test_resize_is_upward_aligned(self):
        height, width = glm5_smart_resize(
            101,
            203,
            temporal_patch_size=2,
            factor=28,
            min_pixels=2 * 28 * 28,
            max_pixels=2 * 280 * 280,
        )
        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)
        self.assertLessEqual(2 * height * width, 2 * 280 * 280)

    def test_tiny_vision_forward(self):
        config = SimpleNamespace(
            attention_bias=True,
            depth=1,
            hidden_size=32,
            in_channels=3,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=32,
            patch_size=2,
            projection_intermediate_size=64,
            rms_norm_eps=1e-5,
            spatial_merge_size=2,
            swiglu_limit=10.0,
            temporal_patch_size=2,
        )
        model = Glm53FlashVisionModel(config)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
        pixel_values = torch.randn(16, 3 * 2 * 2 * 2)
        self.assertEqual(model._rotary_freqs(grid_thw).shape, (16, 4))
        output = model(pixel_values, grid_thw)
        self.assertEqual(output.shape, (4, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_rope_preserves_bfloat16(self):
        config = SimpleNamespace(
            attention_bias=True,
            hidden_size=32,
            num_heads=4,
        )
        attention = Glm53FlashVisionAttention(config)
        q = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        freqs = torch.randn(3, 4)
        q, k = attention._apply_rope(q, k, freqs)
        self.assertEqual(q.dtype, torch.bfloat16)
        self.assertEqual(k.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
