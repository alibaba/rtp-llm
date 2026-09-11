import io
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen3VLVideoProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import Qwen3_VLImageEmbedding
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Qwen35VideoBatchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=144,
            num_heads=2,
            intermediate_size=256,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=16,
        )
        config._attn_implementation = "sdpa"
        config.vit_attention_backend = "sdpa"
        self.part = object.__new__(Qwen3_5MoeImageEmbedding)
        self.part.visual = Qwen3_5MoeVisionModel(config).eval()

    def media(self, grid, offset=0.0):
        return torch.randn((int(torch.tensor(grid).prod()), 24)) + offset, torch.tensor(
            [grid]
        )

    def test_mixed_videos_and_image_share_one_forward_without_contamination(self):
        data = [
            self.media([2, 4, 4]),
            self.media([3, 2, 4], 2.0),
            self.media([1, 4, 2], -2.0),
        ]
        kinds = [MMUrlType.VIDEO, MMUrlType.VIDEO, MMUrlType.DEFAULT]
        reference = [self.part.embedding(d, mm_type=t) for d, t in zip(data, kinds)]
        with mock.patch.object(
            self.part.visual, "forward", wraps=self.part.visual.forward
        ) as forward:
            result = self.part.batched_embedding(data, kinds)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual([len(v[0]) for v in result], [8, 6, 2])
        for actual, expected in zip(result, reference):
            torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        reordered = self.part.batched_embedding(
            list(reversed(data)), list(reversed(kinds))
        )
        for actual, expected in zip(reordered, reversed(reference)):
            torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-4)

    def test_cost_counts_per_temporal_segment(self):
        estimate = self.part.estimate_work(self.media([3, 4, 6]), MMUrlType.VIDEO)
        self.assertEqual(estimate.input_patches, 72)
        self.assertEqual(estimate.output_tokens, 18)
        self.assertEqual(estimate.max_attention_segment, 24)
        self.assertEqual(estimate.attention_work, 3 * 24 * 24)
        self.assertGreater(estimate.estimated_workspace_bytes, 0)

    def test_malformed_grid_and_result_mapping_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            self.part.batched_embedding([self.media([1, 2, 2])], [])
        data = self.media([1, 2, 2])
        with self.assertRaisesRegex(ValueError, "length does not match"):
            self.part.estimate_work((data[0][:-1], data[1]))
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.part.estimate_work(self.media([1, 3, 2]))

    def test_video_processor_keeps_already_sampled_frames(self):
        video = (
            torch.arange(14, dtype=torch.float32)
            .view(14, 1, 1, 1)
            .expand(14, 3, 64, 64)
        )
        video_processor = Qwen3VLVideoProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 4096, "longest_edge": 25165824},
        )
        processor = SimpleNamespace(video_processor=video_processor)
        item = SimpleNamespace(
            mm_type=MMUrlType.VIDEO,
            url="unused",
            mm_preprocess_config=SimpleNamespace(),
        )
        module = "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin"
        with mock.patch(
            module + ".get_bytes_io_from_url", return_value=io.BytesIO(b"video")
        ):
            with mock.patch.object(
                Qwen3_VLImageEmbedding, "load_video", return_value=video
            ) as load:
                pixels, grid = Qwen3_VLImageEmbedding.preprocess_input(
                    [item], VitConfig(), processor, factor=32
                )
        self.assertEqual(grid.tolist(), [[7, 4, 4]])
        self.assertEqual(pixels.shape[0], 112)
        self.assertEqual(load.call_args.kwargs["factor"], 32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_auto_backend_keeps_fp32_on_sdpa(self):
        self.part.visual.to(device="cuda", dtype=torch.float32)
        self.part.visual.config.vit_attention_backend = "auto"
        result = self.part.embedding(self.media([1, 2, 2]))
        self.assertEqual(self.part.visual.last_backend, "sdpa")
        self.assertTrue(torch.isfinite(result[0]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_sm103_fa4_matches_segmented_sdpa(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("SM103-specific backend check")
        self.part.visual.to(device="cuda", dtype=torch.bfloat16)
        data = [self.media([2, 4, 4]), self.media([3, 2, 4], 1)]
        kinds = [MMUrlType.VIDEO] * 2
        reference = self.part.batched_embedding(data, kinds)
        self.part.visual.config.vit_attention_backend = "fa4"
        result = self.part.batched_embedding(data, kinds)
        self.assertEqual(self.part.visual.last_backend, "fa4")
        for actual, expected in zip(result, reference):
            self.assertTrue(torch.isfinite(actual[0]).all())
            torch.testing.assert_close(actual[0], expected[0], atol=0.03, rtol=0.03)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
