import copy
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen3VLVideoProcessor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionModel as HFVisionModel,
)

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
        self.part.visual.load_weights(
            HFVisionModel(copy.deepcopy(config)).state_dict().items()
        )
        self.part.word_embedding_weight = torch.randn(128, 64)
        self.part.vision_start_token_id = 11
        self.part.vision_end_token_id = 12

    def media(self, grid, offset=0.0):
        pixels = torch.randn((int(torch.tensor(grid).prod()), 24)) + offset
        timestamps = [[10 + frame] for frame in range(grid[0])]
        return pixels, torch.tensor([grid]), timestamps

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
        self.assertEqual([len(v[0]) for v in result], [14, 15, 2])
        for actual, expected in zip(result, reference):
            torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        for actual in result:
            self.assertEqual(len(actual), 2)
            self.assertEqual(actual[1].shape, (actual[0].shape[0], 3))
        reordered = self.part.batched_embedding(
            list(reversed(data)), list(reversed(kinds))
        )
        for actual, expected in zip(reordered, reversed(reference)):
            torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-4)

    def test_cost_counts_per_temporal_segment(self):
        estimate = self.part.estimate_work(self.media([3, 4, 6]), MMUrlType.VIDEO)
        self.assertEqual(estimate.input_patches, 72)
        self.assertEqual(estimate.output_tokens, 27)
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

    def test_video_assembly_matches_text_lookup_and_frame_mrope(self):
        features = torch.arange(8 * 64, dtype=torch.float32).reshape(8, 64)
        text = self.part.word_embedding_weight[
            torch.tensor([10, 11, 12, 20, 21, 11, 12])
        ]
        emb, pos, consumed = self.part._assemble_video(
            features, torch.tensor([[2, 4, 4]]), [[10], [20, 21]], text, 0
        )
        expected = torch.cat(
            (
                text[:2],
                features[:4],
                text[2:3],
                text[3:6],
                features[4:],
                text[6:],
            )
        )
        torch.testing.assert_close(emb, expected, atol=0, rtol=0)
        # Text advances in all axes; every frame's visual temporal axis resets.
        expected_pos = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
                [8, 8, 9],
                [8, 9, 8],
                [8, 9, 9],
                [10, 10, 10],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(pos, expected_pos, atol=0, rtol=0)
        self.assertEqual(consumed, 7)
        self.assertTrue(emb.is_contiguous())
        self.assertTrue(pos.is_contiguous())

    def test_complete_video_positions_match_hf_with_mixed_media(self):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

        start, end = self.part.vision_start_token_id, self.part.vision_end_token_id
        image_id, video_id = 100, 101
        data = [self.media([2, 4, 6]), self.media([1, 6, 2]), self.media([3, 2, 4])]
        data[0] = (*data[0][:2], [[20], [21, 22]])
        data[2] = (*data[2][:2], [[30, 31], [32], [33, 34, 35]])
        kinds = [MMUrlType.VIDEO, MMUrlType.IMAGE, MMUrlType.VIDEO]
        results = self.part.batched_embedding(data, kinds)
        prompt, positions, base = [7], [torch.zeros((1, 3), dtype=torch.int32)], 1
        for item, kind, (_, relative) in zip(data, kinds, results):
            prompt.append(start)
            positions.append(torch.full((1, 3), base, dtype=torch.int32))
            base += 1
            t, h, w = item[1][0].tolist()
            visual_count = h * w // 4
            if kind == MMUrlType.VIDEO:
                for ids in item[2]:
                    prompt.extend([*ids, start, *([video_id] * visual_count), end])
            else:
                prompt.extend([image_id] * visual_count)
            positions.append(relative + base)
            base += int(relative[-1].max()) + 1
            prompt.extend([end, 8])
            positions.append(
                torch.arange(base, base + 2, dtype=torch.int32)[:, None].expand(-1, 3)
            )
            base += 2
        config = SimpleNamespace(
            vision_config=self.part.visual.config,
            image_token_id=image_id,
            video_token_id=video_id,
            vision_start_token_id=start,
        )
        reference, _ = Qwen3_5MoeModel.get_rope_index(
            SimpleNamespace(config=config),
            input_ids=torch.tensor([prompt]),
            image_grid_thw=data[1][1],
            video_grid_thw=torch.cat([data[0][1], data[2][1]]),
        )
        actual = torch.cat(positions)
        torch.testing.assert_close(actual.long(), reference[:, 0].T, atol=0, rtol=0)

    def test_word_embedding_loading_supports_checkpoint_prefixes(self):
        for key in (
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ):
            with self.subTest(key=key):
                database = mock.Mock()
                database.get_pretrain_tensor_names.return_value = [key]
                weight = torch.randn(128, 64, dtype=torch.bfloat16)
                database.load_tensor.return_value = [weight]
                self.part.load_word_embedding(database)
                database.load_tensor.assert_called_once_with(key, data_type=None)
                self.assertEqual(self.part.word_embedding_weight.device.type, "cpu")
                torch.testing.assert_close(self.part.word_embedding_weight, weight)
        database.get_pretrain_tensor_names.return_value = []
        with self.assertRaisesRegex(ValueError, "no language word embedding"):
            self.part.load_word_embedding(database)
        database.get_pretrain_tensor_names.return_value = [key]
        database.load_tensor.return_value = [torch.empty(128, 32)]
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.part.load_word_embedding(database)

    def test_video_requires_timestamp_metadata_and_word_embedding(self):
        data = self.media([2, 4, 4])
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self.part.batched_embedding([data[:2]], [MMUrlType.VIDEO])
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self.part.estimate_work((*data[:2], [[10]]), MMUrlType.VIDEO)
        self.part.word_embedding_weight = None
        with self.assertRaisesRegex(RuntimeError, "word embedding"):
            self.part.embedding(data, mm_type=MMUrlType.VIDEO)

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
            with mock.patch(
                module + ".decode_video", return_value=(video, None)
            ) as load:
                pixels, grid = Qwen3_VLImageEmbedding.preprocess_input(
                    [item], VitConfig(), processor, factor=32
                )
        self.assertEqual(grid.tolist(), [[7, 4, 4]])
        self.assertEqual(pixels.shape[0], 112)
        self.assertIs(load.call_args.args[1], item.mm_preprocess_config)

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
