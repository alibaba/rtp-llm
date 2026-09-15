import io
import pickle
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen3VLVideoProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import gpu_video
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import qwen3_5_moe_mixin as qwen35
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.gpu_video import (
    GpuVideoInput,
    prepare_gpu_video,
    preprocess_video_cuda,
)
from rtp_llm.multimodal.qwen3_vl_video import (
    resolve_video_size,
    sample_frame_indices,
    video_resize_shape,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class GpuVideoTest(unittest.TestCase):
    def processor(self):
        return Qwen3VLVideoProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 4096, "longest_edge": 25165824},
        )

    def configs(self):
        return SimpleNamespace(
            width=-1,
            height=-1,
            min_pixels=-1,
            max_pixels=-1,
            fps=-1,
            min_frames=-1,
            max_frames=-1,
        )

    def test_decoder_import_restores_flags_on_success_and_failure(self):
        before = sys.getdlopenflags()
        module = object()
        with mock.patch.object(
            gpu_video.importlib, "import_module", return_value=module
        ):
            self.assertIs(gpu_video._load_nvcodec(), module)
        self.assertEqual(sys.getdlopenflags(), before)
        with mock.patch.object(
            gpu_video.importlib, "import_module", side_effect=ImportError("missing")
        ):
            with self.assertRaisesRegex(RuntimeError, "libnvcuvid"):
                gpu_video._load_nvcodec()
        self.assertEqual(sys.getdlopenflags(), before)

    def test_metadata_matches_real_video_sampling_without_cuda(self):
        container = mock.MagicMock()
        container.__enter__.return_value.streams.video = [
            SimpleNamespace(
                frames=233,
                average_rate=30,
                height=720,
                width=1280,
                codec_context=SimpleNamespace(
                    colorspace=1, color_range=1, format=SimpleNamespace(name="yuv420p")
                ),
            )
        ]
        with (
            mock.patch("av.open", return_value=container),
            mock.patch(
                "torch.cuda.init",
                side_effect=AssertionError("CPU worker initialized CUDA"),
            ),
        ):
            data, grid = prepare_gpu_video(
                b"compressed", self.configs(), self.processor(), 32
            )
        self.assertEqual(
            data.frame_indices,
            (0, 17, 33, 50, 66, 83, 99, 116, 133, 149, 166, 182, 199, 215, 232),
        )
        self.assertEqual(grid.tolist(), [[8, 44, 80]])
        self.assertEqual(data.shape, (28160, 1536))
        self.assertEqual(pickle.loads(pickle.dumps(data)), data)
        self.assertEqual(grid.device.type, "cpu")
        self.assertGreater(data.workspace_bytes, 94 * 1024 * 1024)

    def test_vllm_video_total_pixel_budgets(self):
        processor = self.processor()
        configs = self.configs()
        configs.min_pixels = 2500000
        configs.max_pixels = 73728000
        # H/W results from the Qwen3-VL whole-video resize formula.
        for frames, expected in (
            (6, (1088, 1920)),
            (60, (800, 1472)),
            (180, (480, 832)),
        ):
            with self.subTest(frames=frames):
                self.assertEqual(
                    video_resize_shape(
                        self.configs(),
                        frames,
                        1080,
                        1920,
                        processor,
                        resolve_video_size(
                            processor, configs.min_pixels, configs.max_pixels
                        ),
                    ),
                    expected,
                )
        # The lower budget is also across frames; it is not 2.5M per frame.
        self.assertEqual(
            video_resize_shape(
                self.configs(),
                180,
                64,
                64,
                processor,
                resolve_video_size(processor, configs.min_pixels, configs.max_pixels),
            ),
            (128, 128),
        )

    def test_request_sampling_controls(self):
        configs = self.configs()
        configs.fps = 6
        configs.max_frames = 180
        self.assertEqual(len(sample_frame_indices(900, 30, configs)), 180)
        configs.max_frames = 5
        self.assertEqual(sample_frame_indices(80, 30, configs), [0, 20, 40, 59, 79])

    def test_request_pixel_budgets_do_not_mutate_model_defaults(self):
        processor = self.processor()
        original = dict(processor.size)
        configs = self.configs()
        configs.min_pixels = 2500000
        configs.max_pixels = 73728000
        self.assertEqual(
            resolve_video_size(
                processor, max(0, configs.min_pixels), max(0, configs.max_pixels)
            ),
            {"shortest_edge": 2500000, "longest_edge": 73728000},
        )
        self.assertEqual(processor.size, original)
        self.assertEqual(resolve_video_size(processor), original)
        configs.min_pixels = -1
        configs.max_pixels = 4194304
        self.assertEqual(
            resolve_video_size(
                processor, max(0, configs.min_pixels), max(0, configs.max_pixels)
            ),
            {"shortest_edge": original["shortest_edge"], "longest_edge": 4194304},
        )

    def test_video_preprocess_defers_nvdec_with_request_config(self):
        processor = self.processor()
        container = mock.MagicMock()
        container.__enter__.return_value.streams.video = [
            SimpleNamespace(
                frames=80,
                average_rate=30,
                height=64,
                width=64,
                codec_context=SimpleNamespace(
                    colorspace=1, color_range=1, format=SimpleNamespace(name="yuv420p")
                ),
            )
        ]
        preprocess_config = self.configs()
        preprocess_config.fps = 2
        preprocess_config.max_frames = 5
        preprocess_config.min_pixels = 4096
        preprocess_config.max_pixels = 10240
        item = SimpleNamespace(
            mm_type=MMUrlType.VIDEO,
            url="unused",
            mm_preprocess_config=preprocess_config,
        )
        tokenizer = mock.Mock()
        tokenizer.encode.return_value = [10, 20]
        with mock.patch("av.open", return_value=container), mock.patch.object(
            qwen35, "get_bytes_io_from_url", return_value=io.BytesIO(b"video")
        ), mock.patch.object(
            qwen35.Qwen3_VLImageEmbedding,
            "load_video",
            side_effect=AssertionError("video used the inherited CPU decoder"),
        ), mock.patch(
            "torch.cuda.init", side_effect=AssertionError("CPU worker initialized CUDA")
        ):
            gpu_input, grid, timestamps = (
                qwen35.Qwen3_5MoeImageEmbedding.preprocess_input(
                    [item],
                    VitConfig(),
                    SimpleNamespace(video_processor=processor, tokenizer=tokenizer),
                )
            )
        self.assertIsInstance(gpu_input, GpuVideoInput)
        self.assertEqual(gpu_input.encoded, b"video")
        self.assertEqual(gpu_input.frame_indices, (0, 20, 40, 59, 79))
        # Per-frame request budget retains 64x64; the fifth frame is padded.
        self.assertEqual(grid.tolist(), [[3, 4, 4]])
        self.assertEqual(gpu_input.shape, (48, 1536))
        self.assertEqual(timestamps, [[10, 20]] * 3)
        self.assertEqual(grid.device.type, "cpu")
        self.assertEqual(pickle.loads(pickle.dumps(gpu_input)), gpu_input)

    def test_image_preprocessing_is_inherited(self):
        item = SimpleNamespace(mm_type=MMUrlType.IMAGE)
        config, processor = VitConfig(), object()
        expected = object()
        with mock.patch.object(
            qwen35.Qwen3_VLImageEmbedding, "preprocess_input", return_value=expected
        ) as parent:
            actual = qwen35.Qwen3_5MoeImageEmbedding.preprocess_input(
                [item], config, processor
            )
        self.assertIs(actual, expected)
        parent.assert_called_once_with(
            [item], config, processor, 32, return_video_metadata=True
        )

    def test_unknown_frame_count_is_rejected_before_gpu(self):
        container = mock.MagicMock()
        container.__enter__.return_value.streams.video = [
            SimpleNamespace(
                frames=0,
                average_rate=30,
                height=720,
                width=1280,
                codec_context=SimpleNamespace(
                    colorspace=1, color_range=1, format=SimpleNamespace(name="yuv420p")
                ),
            )
        ]
        with mock.patch("av.open", return_value=container):
            with self.assertRaisesRegex(ValueError, "frame count"):
                prepare_gpu_video(b"bad", self.configs(), self.processor(), 32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_processor_preserves_frame_patch_order_and_grid(self):
        processor = self.processor()
        for count in (4, 5):
            data = GpuVideoInput(b"", count, tuple(range(count)), 64, 64, 64, 64, 16, 2)
            frames = (
                torch.arange(count, dtype=torch.uint8)
                .view(count, 1, 1, 1)
                .expand(count, 3, 64, 64)
            )
            expected = processor(
                frames.float(),
                return_tensors="pt",
                do_resize=False,
                do_sample_frames=False,
            )["pixel_values_videos"]
            with mock.patch.object(
                gpu_video, "decode_video_cuda", return_value=frames.cuda()
            ):
                actual = preprocess_video_cuda(data, processor, "cuda:0")
            self.assertTrue(actual.is_cuda)
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_resized_nvdec_input_matches_cpu_uint8_rounding(self):
        processor = self.processor()
        generator = torch.Generator().manual_seed(37)
        frames = torch.randint(
            256, (5, 3, 64, 96), dtype=torch.uint8, generator=generator
        )
        data = GpuVideoInput(b"", 5, tuple(range(5)), 64, 96, 32, 64, 16, 2)
        resized = gpu_video.resize_video_to_shape(frames, processor, 32, 64)
        expected = processor(
            resized, return_tensors="pt", do_resize=False, do_sample_frames=False
        )["pixel_values_videos"]
        with mock.patch.object(
            gpu_video, "decode_video_cuda", return_value=frames.cuda()
        ):
            actual = preprocess_video_cuda(data, processor, "cuda:0")
        torch.testing.assert_close(
            actual.cpu().to(torch.bfloat16), expected.to(torch.bfloat16), atol=0, rtol=0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_session_drains_reuses_and_discards_failure_without_frame_cache(self):
        device = torch.device("cuda", torch.cuda.current_device())
        codec = mock.MagicMock()
        state = gpu_video._decode_state
        state.key = None
        state.decoder = None
        payloads = []

        class Demux:
            def GetNvCodecId(self):
                return 4

            def __iter__(self):
                return iter([0, 1, 2])

        def demux(feed):
            buf = bytearray(64)
            n = feed(buf)
            payloads.append(bytes(buf[:n]))
            return Demux()

        decoder = codec.CreateDecoder.return_value
        codec.CreateDemuxer.side_effect = demux
        generation = [30]
        calls = []

        def decode(packet):
            calls.append(packet)
            frame = torch.full((48, 32), 128, device=device, dtype=torch.uint8)
            frame[:32] = generation[0] + packet
            return [frame]

        decoder.Decode.side_effect = decode
        data = GpuVideoInput(b"first", 3, (0, 2, 2), 32, 32, 32, 32, 16, 2)
        with mock.patch.dict("sys.modules", {"PyNvVideoCodec": codec}):
            first = gpu_video.decode_video_cuda(data, device)
            generation[0] = 60
            second = gpu_video.decode_video_cuda(
                GpuVideoInput(b"second", 3, (0, 2, 2), 32, 32, 32, 32, 16, 2), device
            )
            self.assertEqual(codec.CreateDecoder.call_count, 1)
            self.assertEqual(calls, [0, 1, 2, 0, 1, 2])
            self.assertEqual(payloads, [b"first", b"second"])
            self.assertEqual(first[:, 0, 0, 0].tolist(), [16, 18, 18])
            self.assertEqual(second[:, 0, 0, 0].tolist(), [51, 53, 53])
            consumer = torch.cuda.Stream(device=device)
            with torch.cuda.stream(consumer):
                third = gpu_video.decode_video_cuda(data, device)
                self.assertEqual(torch.cuda.current_stream(device), consumer)
                torch.testing.assert_close(third, second, atol=0, rtol=0)
            consumer.synchronize()
            self.assertEqual(codec.CreateDecoder.call_count, 1)
            self.assertNotEqual(
                codec.CreateDecoder.call_args.kwargs["cudastream"], consumer.cuda_stream
            )
            decoder.Decode.side_effect = RuntimeError("broken bitstream")
            with self.assertRaisesRegex(RuntimeError, "broken bitstream"):
                gpu_video.decode_video_cuda(data, device)
            self.assertIsNone(state.decoder)
            self.assertIsNone(state.key)


if __name__ == "__main__":
    unittest.main()
