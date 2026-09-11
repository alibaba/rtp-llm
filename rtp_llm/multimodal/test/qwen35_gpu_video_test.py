import io
import pickle
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen3VLVideoProcessor

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import gpu_video
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.gpu_video import (
    GpuVideoInput,
    prepare_gpu_video,
    preprocess_video_cuda,
)


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
        with mock.patch("av.open", return_value=container), mock.patch(
            "torch.cuda.init", side_effect=AssertionError("CPU worker initialized CUDA")
        ):
            data, grid = prepare_gpu_video(
                b"compressed", self.configs(), self.processor(), 32
            )
        self.assertEqual(
            data.frame_indices,
            tuple(torch.linspace(0, 232, 14).round().long().tolist()),
        )
        self.assertEqual(grid.tolist(), [[7, 36, 64]])
        self.assertEqual(data.shape, (16128, 1536))
        self.assertEqual(pickle.loads(pickle.dumps(data)), data)
        self.assertEqual(grid.device.type, "cpu")
        self.assertGreater(data.workspace_bytes, 94 * 1024 * 1024)

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
            decoder.Decode.side_effect = RuntimeError("broken bitstream")
            with self.assertRaisesRegex(RuntimeError, "broken bitstream"):
                gpu_video.decode_video_cuda(data, device)
            self.assertIsNone(state.decoder)
            self.assertIsNone(state.key)


if __name__ == "__main__":
    unittest.main()
