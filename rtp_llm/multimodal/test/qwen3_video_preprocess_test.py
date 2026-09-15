"""CPU regressions against the HF processor used by vLLM for Qwen3-VL."""

import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from transformers import Qwen3VLVideoProcessor

from rtp_llm.multimodal.qwen3_vl_video import (
    decode_video,
    resize_video,
    resolve_video_size,
    sample_frame_indices,
    video_resize_shape,
    video_timestamps,
    video_token_layout,
)


class Qwen3VideoPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def processor(self, min_pixels=128, max_pixels=4096):
        return Qwen3VLVideoProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
        )

    def test_timestamps_pad_last_frame_and_preserve_rounding(self):
        indices = [0, 5, 10, 15, 21]
        self.assertEqual(
            video_timestamps(indices, 30, 2),
            [(0 / 30 + 5 / 30) / 2, (10 / 30 + 15 / 30) / 2, 21 / 30],
        )
        self.assertEqual(indices, [0, 5, 10, 15, 21])
        self.assertEqual(video_timestamps([0], 30, 2), [0.0])
        for fps in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                video_timestamps([0], fps, 2)

    def test_video_layout_encodes_each_timestamp_independently(self):
        tokenizer = mock.Mock()
        tokenizer.convert_tokens_to_ids.side_effect = [100, 101]
        tokenizer.encode.side_effect = [[20, 21], [22, 23]]
        processor = SimpleNamespace(
            video_processor=self.processor(), tokenizer=tokenizer
        )
        layout = video_token_layout(
            torch.tensor([[2, 4, 8]]), [0, 5, 10], 30, processor
        )
        self.assertEqual(layout.tolist(), [20, 21, 100, -8, 101, 22, 23, 100, -8, 101])
        self.assertEqual(
            tokenizer.encode.call_args_list,
            [
                mock.call("<0.1 seconds>", add_special_tokens=False),
                mock.call("<0.3 seconds>", add_special_tokens=False),
            ],
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            video_token_layout(torch.tensor([[3, 4, 8]]), [0, 5, 10], 30, processor)

    def test_sampling_keeps_odd_frame_count(self):
        self.assertEqual(len(sample_frame_indices(233, 30, SimpleNamespace())), 15)
        self.assertEqual(
            sample_frame_indices(10, 10, SimpleNamespace(fps=3, min_frames=1)),
            [0, 4, 9],
        )
        self.assertEqual(
            sample_frame_indices(30, 30, SimpleNamespace(fps=5)), [0, 7, 14, 22, 29]
        )

    def test_sampling_limits_short_and_long_videos(self):
        self.assertEqual(sample_frame_indices(2, 30, SimpleNamespace()), [0, 1])
        self.assertEqual(sample_frame_indices(1, 30, SimpleNamespace()), [0])
        self.assertEqual(
            sample_frame_indices(30, 30, SimpleNamespace(fps=30, max_frames=4)),
            [0, 10, 19, 29],
        )
        self.assertEqual(len(sample_frame_indices(100000, 30, SimpleNamespace())), 768)
        self.assertEqual(
            len(sample_frame_indices(233, 30, SimpleNamespace(fps=6, max_frames=180))),
            46,
        )

    def test_invalid_sampling_metadata_is_rejected(self):
        for total, fps in ((0, 30), (10, 0), (10, float("nan")), (10, float("inf"))):
            with self.subTest(total=total, fps=fps), self.assertRaises(ValueError):
                sample_frame_indices(total, fps, SimpleNamespace())
        for cfg in (
            SimpleNamespace(fps=0),
            SimpleNamespace(min_frames=8, max_frames=4),
        ):
            with self.assertRaises(ValueError):
                sample_frame_indices(30, 30, cfg)

    def test_resize_and_pixels_match_hf_for_video_shapes_and_budgets(self):
        generator = torch.Generator().manual_seed(17)
        # Odd/even time counts, portrait/landscape, tiny frames, down/up scaling.
        for frames in (2, 3, 5, 15):
            for height, width in ((4, 8), (19, 33), (33, 19), (64, 96)):
                for minimum, maximum in ((128, 4096), (4096, 16384)):
                    with self.subTest(
                        frames=frames, height=height, width=width, budget=maximum
                    ):
                        vp = self.processor(minimum, maximum)
                        video = torch.randint(
                            256,
                            (frames, 3, height, width),
                            dtype=torch.uint8,
                            generator=generator,
                        )
                        reference = vp(
                            video, return_tensors="pt", do_sample_frames=False
                        )
                        resized = resize_video(
                            video, SimpleNamespace(), vp, resolve_video_size(vp)
                        )
                        actual = vp(
                            resized,
                            return_tensors="pt",
                            do_resize=False,
                            do_sample_frames=False,
                        )
                        torch.testing.assert_close(
                            actual["video_grid_thw"],
                            reference["video_grid_thw"],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            actual["pixel_values_videos"],
                            reference["pixel_values_videos"],
                            atol=0,
                            rtol=0,
                        )

    def test_optional_hf_per_frame_cap_uses_the_same_grid_and_pixels(self):
        vp = self.processor(128, 16384)
        if not hasattr(vp, "cap_pixels_per_frame"):
            self.skipTest("installed Transformers has no optional per-frame cap")
        vp.cap_pixels_per_frame = True
        vp.max_video_tokens = 8
        video = torch.randint(256, (5, 3, 32, 64), dtype=torch.uint8)
        reference = vp(video, return_tensors="pt", do_sample_frames=False)
        resized = resize_video(video, SimpleNamespace(), vp, resolve_video_size(vp))
        actual = vp(
            resized, return_tensors="pt", do_resize=False, do_sample_frames=False
        )
        torch.testing.assert_close(
            actual["video_grid_thw"], reference["video_grid_thw"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["pixel_values_videos"],
            reference["pixel_values_videos"],
            atol=0,
            rtol=0,
        )

    def test_single_frame_has_same_resize_validation_as_hf(self):
        vp = self.processor()
        video = torch.zeros((1, 3, 32, 32), dtype=torch.uint8)
        try:
            reference = vp(video, return_tensors="pt", do_sample_frames=False)
        except ValueError:
            with self.assertRaises(ValueError):
                resize_video(video, SimpleNamespace(), vp, resolve_video_size(vp))
        else:
            resized = resize_video(video, SimpleNamespace(), vp, resolve_video_size(vp))
            actual = vp(
                resized, return_tensors="pt", do_resize=False, do_sample_frames=False
            )
            for key in ("pixel_values_videos", "video_grid_thw"):
                torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)

    def test_total_budget_overrides_do_not_mutate_model_defaults(self):
        vp = self.processor()
        original = dict(vp.size)
        size = resolve_video_size(vp, 2500000, 73728000)
        self.assertEqual(size, {"shortest_edge": 2500000, "longest_edge": 73728000})
        self.assertEqual(dict(vp.size), original)
        for minimum, maximum in ((-1, 0), (0, -1), (8192, 1024)):
            with self.assertRaises(ValueError):
                resolve_video_size(vp, minimum, maximum)

    def test_benchmark_grid_and_odd_default_grid(self):
        vp = Qwen3VLVideoProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 4096, "longest_edge": 25165824},
        )
        self.assertEqual(
            video_resize_shape(
                SimpleNamespace(),
                46,
                720,
                1280,
                vp,
                resolve_video_size(vp, 2500000, 73728000),
            ),
            (704, 1280),
        )
        # The vLLM default keeps 15 frames and pads to 8 temporal groups.
        self.assertEqual(
            video_resize_shape(
                SimpleNamespace(), 15, 720, 1280, vp, resolve_video_size(vp)
            ),
            (704, 1280),
        )

    def test_explicit_request_dimensions_and_pixel_limits(self):
        vp = self.processor()
        size = resolve_video_size(vp)
        self.assertEqual(
            video_resize_shape(
                SimpleNamespace(height=19, width=33), 5, 100, 100, vp, size
            ),
            (20, 32),
        )
        for cfg in (
            SimpleNamespace(height=19),
            SimpleNamespace(height=0, width=32),
            SimpleNamespace(min_pixels=8192, max_pixels=1024),
        ):
            with self.assertRaises(ValueError):
                video_resize_shape(cfg, 5, 64, 64, vp, size)
        per_frame = SimpleNamespace(min_pixels=1024, max_pixels=4096)
        expected = resolve_video_size(vp, 5 * 1024, 5 * 4096)
        self.assertEqual(
            video_resize_shape(per_frame, 5, 32, 64, vp, size),
            video_resize_shape(SimpleNamespace(), 5, 32, 64, vp, expected),
        )

    def test_decode_preserves_native_rgb_and_sample_metadata(self):
        reader = mock.MagicMock()
        reader.__len__.return_value = 30
        reader.get_avg_fps.return_value = 30.0
        raw = np.arange(30, dtype=np.uint8)[:, None, None, None] * np.ones(
            (30, 8, 12, 3), dtype=np.uint8
        )
        reader.get_batch.side_effect = lambda indices: SimpleNamespace(
            asnumpy=lambda: raw[indices]
        )
        with mock.patch("decord.VideoReader", return_value=reader):
            video, metadata = decode_video(io.BytesIO(b"video"), SimpleNamespace(fps=5))
        self.assertEqual(tuple(video.shape), (5, 3, 8, 12))
        self.assertEqual(video[:, 0, 0, 0].tolist(), [0, 7, 14, 22, 29])
        self.assertEqual(metadata.frames_indices, [0, 7, 14, 22, 29])
        self.assertEqual(metadata.total_num_frames, 30)
        self.assertEqual(metadata.fps, 30.0)
        self.assertEqual(metadata.duration, 1.0)


if __name__ == "__main__":
    unittest.main()
