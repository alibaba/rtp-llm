import io
import os
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLImageProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import Qwen3_VLImageEmbedding
from rtp_llm.multimodal.qwen35_token_length import token_length, video_output_length
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


class Qwen35TokenLengthTest(unittest.TestCase):
    def setUp(self):
        self.vit = VitConfig()
        self.processor = SimpleNamespace(
            image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2),
            video_processor=Qwen3VLVideoProcessor(),
            tokenizer=SimpleNamespace(
                encode=lambda text, **kwargs: list(text.encode())
            ),
        )
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)

    def test_image_metadata_matches_pixel_grid_without_forward(self):
        for height, width, config in (
            (64, 96, MMPreprocessConfig()),
            (401, 717, MMPreprocessConfig(128, 96)),
            (64, 128, MMPreprocessConfig(-1, -1, 4096, 16384)),
        ):
            with self.subTest(shape=(height, width)):
                data = io.BytesIO()
                Image.new("RGB", (width, height), color=(11, 29, 51)).save(data, "PNG")
                item = MultimodalInput(
                    "fixture.png", MMUrlType.IMAGE, torch.empty(0), config
                )
                with patch(
                    "rtp_llm.multimodal.qwen35_token_length.get_bytes_io_from_url",
                    side_effect=lambda *a, **k: io.BytesIO(data.getvalue()),
                ):
                    length = token_length(item, self.vit, self.processor, 32)
                with patch(
                    "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.get_bytes_io_from_url",
                    side_effect=lambda *a, **k: io.BytesIO(data.getvalue()),
                ):
                    _, grid = Qwen3_VLImageEmbedding.preprocess_input(
                        [item], self.vit, processor=self.processor, factor=32
                    )
                self.assertEqual(length, int(grid.prod().item()) // 4)

    def test_video_complete_length_includes_timestamps_and_temporal_padding(self):
        from transformers.video_utils import VideoMetadata

        from rtp_llm.multimodal.qwen3_vl_video import video_timestamp_tokens

        for frames in (4, 5, 18, 37):
            with self.subTest(frames=frames):
                indices = list(range(frames))
                vp = self.processor.video_processor
                video = torch.zeros((frames, 3, 64, 96), dtype=torch.uint8)
                actual = vp(
                    video,
                    return_tensors="pt",
                    do_resize=False,
                    do_sample_frames=False,
                    video_metadata=VideoMetadata(
                        total_num_frames=frames, fps=30, frames_indices=indices
                    ),
                )
                grid = actual["video_grid_thw"]
                timestamps = video_timestamp_tokens(grid, indices, 30, self.processor)
                length = video_output_length(grid, indices, 30, self.processor)
                self.assertEqual(
                    length, int(grid.prod()) // 4 + sum(len(t) + 2 for t in timestamps)
                )
                self.assertGreater(length, int(grid.prod()) // 4)
                self.assertEqual(
                    len(timestamps),
                    (frames + vp.temporal_patch_size - 1) // vp.temporal_patch_size,
                )

    def test_service_sampling_matches_cpu_pixels_and_nvdec_metadata(self):
        import av
        import numpy as np

        # Decode a real, tiny indexed video on CPU; NVDEC is only metadata-planned.
        encoded = io.BytesIO()
        with av.open(encoded, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height = 96, 64
            stream.pix_fmt = "yuv420p"
            for index in range(24):
                frame = av.VideoFrame.from_ndarray(
                    np.full((64, 96, 3), index * 7, dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        payload = encoded.getvalue()
        self.vit.mm_video_fps = 6.0
        self.vit.mm_video_max_frames = 8
        # The same serialized VitConfig is passed to preprocessing workers.
        service = pickle.loads(pickle.dumps(self.vit))
        modules = (
            "rtp_llm.multimodal.qwen35_token_length",
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin",
            "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin",
        )
        from contextlib import ExitStack

        with ExitStack() as stack:
            for module in modules:
                stack.enter_context(
                    patch(
                        module + ".get_bytes_io_from_url",
                        side_effect=lambda *a, **k: io.BytesIO(payload),
                    )
                )
            for fps, maximum, expected_frames in (
                (-1, -1, 8),
                (3, -1, 7),
                (-1, 12, 12),
                (5, 6, 6),
            ):
                with self.subTest(fps=fps, max_frames=maximum):
                    config = MMPreprocessConfig(96, 64, -1, -1, fps, -1, maximum)
                    item = MultimodalInput(
                        "fixture.mp4", MMUrlType.VIDEO, torch.empty(0), config
                    )
                    length = token_length(
                        item, service, self.processor, 32, video_backend="cpu"
                    )
                    nvdec_length = token_length(
                        item, service, self.processor, 32, video_backend="nvdec"
                    )
                    pixels, grid, timestamps = (
                        Qwen3_5MoeImageEmbedding.preprocess_input(
                            [item], service, self.processor, 32, video_backend="cpu"
                        )
                    )
                    plan, planned_grid, planned_timestamps = (
                        Qwen3_5MoeImageEmbedding.preprocess_input(
                            [item], service, self.processor, 32, video_backend="nvdec"
                        )
                    )
                    self.assertEqual(len(plan.frame_indices), expected_frames)
                    self.assertEqual(int(grid[0, 0]), (expected_frames + 1) // 2)
                    self.assertTrue(torch.equal(grid, planned_grid))
                    self.assertEqual(timestamps, planned_timestamps)
                    self.assertEqual(length, nvdec_length)
                    self.assertEqual(
                        length,
                        int(grid.prod()) // 4 + sum(len(t) + 2 for t in timestamps),
                    )
                    self.assertEqual(pixels.shape[0], int(grid.prod()))
                    # Explicit old-style requests produce the exact same pixels.
                    explicit = MultimodalInput(
                        "fixture.mp4",
                        MMUrlType.VIDEO,
                        torch.empty(0),
                        MMPreprocessConfig(
                            96,
                            64,
                            -1,
                            -1,
                            int(service.mm_video_fps) if fps == -1 else fps,
                            -1,
                            service.mm_video_max_frames if maximum == -1 else maximum,
                        ),
                    )
                    reference, reference_grid, reference_timestamps = (
                        Qwen3_5MoeImageEmbedding.preprocess_input(
                            [explicit],
                            VitConfig(),
                            self.processor,
                            32,
                            video_backend="cpu",
                        )
                    )
                    self.assertTrue(torch.equal(pixels, reference))
                    self.assertTrue(torch.equal(grid, reference_grid))
                    self.assertEqual(timestamps, reference_timestamps)
                    self.assertEqual(config.fps, fps)
                    self.assertEqual(config.max_frames, maximum)

    def test_engine_length_query_does_not_submit_or_read_embedding(self):
        engine = object.__new__(MMProcessEngine)
        engine.vit_config = self.vit
        calls = []
        engine.mm_part = SimpleNamespace(
            validate_inputs=lambda inputs: calls.append("validate"),
            token_length=lambda item, config: len(item),
        )
        self.assertEqual(engine.token_lengths(["one", "four"]), [3, 4])
        self.assertEqual(calls, ["validate"])
        engine.mm_part.token_length = lambda *args: 0
        with self.assertRaises(ValueError):
            engine.token_lengths(["invalid"])

    @unittest.skipUnless(
        os.environ.get("RTP_MM_TEST_VIDEO_DIR"),
        "external four-video acceptance inputs not configured",
    )
    def test_real_four_video_lengths_match_production_preprocessing(self):
        processor = AutoProcessor.from_pretrained(
            os.environ["RTP_MM_TEST_CHECKPOINT"], local_files_only=True
        )
        processor.image_processor = Qwen2VLImageProcessor.from_pretrained(
            os.environ["RTP_MM_TEST_CHECKPOINT"], local_files_only=True
        )
        # The exact four-request preprocessing parameters requested by the user.
        config = MMPreprocessConfig(1280, 704, 2500000, 73728000, 6, -1, 180)
        for name in ("video.mp4", "video_02.mp4", "video_03.mp4", "video_04.mp4"):
            with self.subTest(video=name):
                path = os.path.join(os.environ["RTP_MM_TEST_VIDEO_DIR"], name)
                item = MultimodalInput(path, MMUrlType.VIDEO, torch.empty(0), config)
                length = token_length(
                    item, self.vit, processor, 32, video_backend="cpu"
                )
                nvdec_length = token_length(
                    item, self.vit, processor, 32, video_backend="nvdec"
                )
                pixels, grid, timestamps = Qwen3_5MoeImageEmbedding.preprocess_input(
                    [item],
                    self.vit,
                    processor=processor,
                    factor=32,
                    video_backend="cpu",
                )
                self.assertEqual(
                    length,
                    int(grid.prod().item()) // 4 + sum(len(t) + 2 for t in timestamps),
                )
                self.assertEqual(length, nvdec_length)
                self.assertEqual(pixels.shape[0], int(grid.prod().item()))
                print(
                    f"METADATA_ACCEPT video={name} grid={grid.tolist()} length={length}",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
