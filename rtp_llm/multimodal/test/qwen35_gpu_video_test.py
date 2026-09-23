import io
import pickle
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen3VLVideoProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
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
from rtp_llm.multimodal.vit_metrics import collect_vit_preprocess_metrics
from rtp_llm.utils.base_model_datatypes import MMUrlType


class GpuVideoMetricTest(unittest.TestCase):
    def test_cuda_timer_defers_event_waits_until_metric_reporting(self):
        start, end, stream = mock.Mock(), mock.Mock(), mock.Mock()
        start.elapsed_time.return_value = 1.25
        with mock.patch("torch.cuda.Event", side_effect=[start, end]), mock.patch(
            "torch.cuda.current_stream", return_value=stream
        ), mock.patch.object(gpu_video, "_submit_video_metric") as submit, mock.patch(
            "torch.cuda.synchronize", side_effect=AssertionError("global CUDA sync")
        ):
            with gpu_video._cuda_video_timer(
                GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC,
                "decode",
                "cuda:0",
                base_us=2500,
                sampled_frames=5,
            ):
                pass
        start.record.assert_called_once_with(stream)
        end.record.assert_called_once_with(stream)
        start.elapsed_time.assert_not_called()
        end.synchronize.assert_not_called()
        sample = submit.call_args.args[0]
        with mock.patch.object(gpu_video.kmonitor, "report") as report:
            sample.report()
        end.synchronize.assert_called_once_with()
        self.assertEqual(report.call_args_list[0].args[1], 3750)
        self.assertEqual(report.call_args_list[0].args[2]["timing"], "wall_plus_cuda")
        self.assertEqual(
            report.call_args_list[1].args[:2],
            (GaugeMetrics.VIT_VIDEO_FRAME_COUNT_METRIC, 5),
        )

    def test_failed_stage_does_not_submit_successful_sample(self):
        with mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.current_stream"
        ), mock.patch.object(gpu_video, "_submit_video_metric") as submit:
            with self.assertRaisesRegex(ValueError, "decode failed"):
                with gpu_video._cuda_video_timer(
                    GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC,
                    "decode",
                    "cuda:0",
                    sampled_frames=5,
                ):
                    raise ValueError("decode failed")
        submit.assert_not_called()

    def test_idle_final_request_is_reported_without_another_request(self):
        reported = threading.Event()
        sample = mock.Mock(stage="processor")
        sample.report.side_effect = reported.set
        reporter = gpu_video._CudaVideoMetricReporter(capacity=1)
        reporter.submit(sample)
        self.assertTrue(reported.wait(5))
        self.assertTrue(reporter._thread.daemon)
        sample.report.assert_called_once_with()

    def test_report_failure_is_isolated_and_next_sample_still_reports(self):
        reported = threading.Event()
        failed = mock.Mock(stage="decode")
        failed.report.side_effect = RuntimeError("CUDA event failed")
        final = mock.Mock(stage="processor")
        final.report.side_effect = reported.set
        with mock.patch.object(
            gpu_video.kmonitor, "report"
        ) as report, mock.patch.object(gpu_video.logging, "warning"):
            reporter = gpu_video._CudaVideoMetricReporter(capacity=2)
            reporter.submit(failed)
            reporter.submit(final)
            self.assertTrue(reported.wait(5))
        self.assertEqual(report.call_args.args[2]["reason"], "report_error")
        self.assertEqual(report.call_args.args[1], 1)

    def test_full_queue_only_accumulates_until_background_flush(self):
        with mock.patch.object(threading.Thread, "start"):
            reporter = gpu_video._CudaVideoMetricReporter(capacity=1)
        first, second = mock.Mock(stage="resize"), mock.Mock(stage="processor")
        with mock.patch.object(
            gpu_video.kmonitor, "report"
        ) as report, mock.patch.object(gpu_video.logging, "warning") as warning:
            reporter.submit(first)
            reporter.submit(second)
            report.assert_not_called()
            warning.assert_not_called()
            self.assertEqual(reporter._drops, {("processor", "queue_full"): 1})
            reporter._flush_drops()
        self.assertIs(reporter._pending.get_nowait(), first)
        self.assertEqual(report.call_args.args[1], 1)

    def test_drop_logs_are_limited_and_counts_are_aggregated(self):
        with mock.patch.object(threading.Thread, "start"):
            reporter = gpu_video._CudaVideoMetricReporter(capacity=1)
        with mock.patch.object(
            gpu_video.time, "monotonic", side_effect=[10, 11, 70]
        ), mock.patch.object(
            gpu_video.logging, "warning"
        ) as warning, mock.patch.object(
            gpu_video.kmonitor, "report"
        ) as report:
            for count in (3, 1, 1):
                for _ in range(count):
                    reporter.record_drop("resize", "queue_full")
                reporter._flush_drops()
        self.assertEqual(warning.call_count, 2)
        self.assertEqual([c.args[1] for c in report.call_args_list], [3, 1, 1])
        self.assertEqual(
            report.call_args.args[0],
            AccMetrics.VIT_PREPROCESS_METRIC_DROPPED_QPS_METRIC,
        )

    def test_blocked_backend_does_not_block_saturated_submit(self):
        backend_entered = threading.Event()
        release_backend = threading.Event()
        submitted = threading.Event()
        dropped_reported = threading.Event()

        def backend(metric, value, tags):
            if metric == AccMetrics.VIT_PREPROCESS_METRIC_DROPPED_QPS_METRIC:
                dropped_reported.set()
                return
            backend_entered.set()
            release_backend.wait(5)

        first = gpu_video._CudaVideoMetricSample(
            GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC,
            "resize",
            mock.Mock(),
            mock.Mock(),
        )
        first.start.elapsed_time.return_value = 1.0
        queued, dropped = mock.Mock(stage="processor"), mock.Mock(stage="decode")
        with mock.patch.object(
            gpu_video.kmonitor, "report", side_effect=backend
        ), mock.patch.object(gpu_video.logging, "warning"):
            reporter = gpu_video._CudaVideoMetricReporter(capacity=1)
            reporter.submit(first)
            try:
                self.assertTrue(backend_entered.wait(5))
                reporter.submit(queued)

                def submit():
                    reporter.submit(dropped)
                    submitted.set()

                producer = threading.Thread(target=submit, daemon=True)
                producer.start()
                self.assertTrue(submitted.wait(1), "saturated submit waited on backend")
                self.assertFalse(release_backend.is_set())
            finally:
                release_backend.set()
            self.assertTrue(dropped_reported.wait(5))

    def test_final_event_init_failure_flushes_without_another_request(self):
        reported = threading.Event()
        with mock.patch.object(
            gpu_video, "_cuda_video_metric_reporter", None
        ), mock.patch("torch.cuda.current_stream"), mock.patch(
            "torch.cuda.Event", side_effect=RuntimeError("event init failed")
        ), mock.patch.object(
            gpu_video.kmonitor, "report", side_effect=lambda *args: reported.set()
        ) as report, mock.patch.object(
            gpu_video.logging, "warning"
        ):
            with gpu_video._cuda_video_timer(
                GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC, "resize", "cuda:0"
            ):
                pass
            self.assertTrue(reported.wait(5))
        self.assertEqual(report.call_args.args[2]["reason"], "report_error")

    def test_drop_backend_failure_does_not_recursively_generate_drops(self):
        with mock.patch.object(threading.Thread, "start"):
            reporter = gpu_video._CudaVideoMetricReporter(capacity=1)
        reporter.record_drop("decode", "report_error")
        with mock.patch.object(
            gpu_video.kmonitor, "report", side_effect=RuntimeError("backend failed")
        ), mock.patch.object(gpu_video.logging, "warning"):
            reporter._flush_drops()
        self.assertEqual(reporter._drops, {})
        self.assertEqual(reporter._drop_total, 1)

    def test_resize_pixel_count_uses_sampled_frames_without_temporal_padding(self):
        start, end = mock.Mock(), mock.Mock()
        start.elapsed_time.return_value = 0.5
        with mock.patch("torch.cuda.Event", side_effect=[start, end]), mock.patch(
            "torch.cuda.current_stream"
        ), mock.patch.object(gpu_video, "_submit_video_metric") as submit:
            with gpu_video._cuda_video_timer(
                GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC,
                "resize",
                "cuda:0",
                resized_pixels=5 * 32 * 64,
            ):
                pass
        with mock.patch.object(gpu_video.kmonitor, "report") as report:
            submit.call_args.args[0].report()
        self.assertEqual(
            report.call_args.args[:2],
            (GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC, 10240),
        )

    def test_cpu_video_counts_sampled_frames_without_temporal_padding(self):
        grid = torch.tensor([[3, 4, 4]])
        metadata = SimpleNamespace(frames_indices=[0, 20, 40, 59, 79], fps=30)
        pixels = torch.empty(48, 1536)
        item = SimpleNamespace(mm_type=MMUrlType.VIDEO)
        with mock.patch.object(
            qwen35.Qwen3_VLImageEmbedding,
            "preprocess_input",
            return_value=(pixels, grid, metadata),
        ), mock.patch.object(qwen35, "video_timestamp_tokens", return_value=[[1]] * 3):
            with collect_vit_preprocess_metrics() as metrics:
                qwen35.Qwen3_5MoeImageEmbedding.preprocess_input(
                    [item], VitConfig(), object(), video_backend="cpu"
                )
        self.assertEqual(len(metrics.samples), 1)
        self.assertEqual(
            metrics.samples[0].metric, GaugeMetrics.VIT_VIDEO_FRAME_COUNT_METRIC
        )
        self.assertEqual(metrics.samples[0].value, 5)
        self.assertEqual(metrics.samples[0].tags["backend"], "cpu")


class GpuVideoTest(unittest.TestCase):

    def test_selected_surfaces_copy_to_ordered_buffer_before_decoder_reuse(self):
        from contextlib import nullcontext

        device = torch.device("cpu")
        surface = torch.full((48, 32), 128, dtype=torch.uint8)
        decoder = mock.Mock()

        def decode(packet):
            surface[:32].fill_(30 + packet)
            return [surface]

        decoder.Decode.side_effect = decode
        demuxer = mock.MagicMock()
        demuxer.GetNvCodecId.return_value = 99
        demuxer.__iter__.return_value = iter([0, 1, 2])
        codec = mock.Mock()
        codec.CreateDemuxer.return_value = demuxer
        codec.CreateDecoder.return_value = decoder
        producer, consumer = mock.Mock(), mock.Mock()
        data = GpuVideoInput(b"test", 3, (2, 0, 2), 32, 32, 32, 32, 16, 2)
        gpu_video._decode_state.key = None
        with mock.patch(
            "torch.cuda.stream", side_effect=lambda stream: nullcontext()
        ), mock.patch.object(torch.Tensor, "record_stream"), mock.patch(
            "torch.stack", side_effect=AssertionError("sample stack")
        ), mock.patch.object(
            torch.Tensor, "clone", side_effect=AssertionError("sample clone")
        ), mock.patch.object(
            gpu_video, "nv12_to_rgb", side_effect=lambda x, *args: x
        ):
            result = gpu_video._decode_on_stream(
                data, device, codec, producer, consumer
            )
        self.assertEqual(result.shape, (3, 48, 32))
        self.assertEqual(result[:, 0, 0].tolist(), [32, 30, 32])
        self.assertEqual(producer.synchronize.call_count, 2)
        consumer.wait_stream.assert_called_once_with(producer)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_parallel_gpu_pixels_match_reference_across_consumer_streams(self):
        processor = self.processor()
        data = [
            GpuVideoInput(bytes([value]), 5, tuple(range(5)), 32, 64, 32, 64, 16, 2)
            for value in (27, 219, 63, 173)
        ]
        expected = [
            processor(
                torch.full((5, 3, 32, 64), x.encoded[0], dtype=torch.uint8),
                do_resize=False,
                do_sample_frames=False,
                return_tensors="pt",
            )["pixel_values_videos"].bfloat16()
            for x in data
        ]

        def decode(item, device):
            return torch.full(
                (5, 3, 32, 64), item.encoded[0], dtype=torch.uint8, device=device
            )

        pool = gpu_video.VideoDecodePool(2)
        try:
            with mock.patch.object(gpu_video, "decode_video_cuda", side_effect=decode):
                with pool.preprocess(
                    data, processor, "cuda:0", torch.bfloat16
                ) as values:
                    prepared = list(values)
            consumer = torch.cuda.Stream()
            with torch.cuda.stream(consumer):
                actual = [x.consume(consumer).clone() for x in prepared]
            consumer.synchronize()
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a.cpu(), b, atol=0, rtol=0)
        finally:
            pool.close()

    def test_single_video_pixels_match_processor_without_singleton_copies(self):
        generator = torch.Generator().manual_seed(841)
        for count in (4, 5):
            for normalize in (False, True):
                with self.subTest(frames=count, normalize=normalize):
                    processor = self.processor()
                    processor.do_normalize = normalize
                    frames = torch.randint(
                        256,
                        (count, 3, 32, 64),
                        dtype=torch.uint8,
                        generator=generator,
                    )
                    before = frames.clone()
                    expected = processor(
                        frames,
                        do_resize=False,
                        do_sample_frames=False,
                        return_tensors="pt",
                    )["pixel_values_videos"]
                    # An even frame count needs neither stack nor cat; odd
                    # counts still concatenate the required repeated last frame.
                    with mock.patch("torch.stack", side_effect=AssertionError("stack")):
                        if count % 2 == 0:
                            with mock.patch(
                                "torch.cat", side_effect=AssertionError("cat")
                            ):
                                actual = gpu_video.process_video_pixels(
                                    frames, processor
                                )
                        else:
                            actual = gpu_video.process_video_pixels(frames, processor)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(frames, before, rtol=0, atol=0)

    def test_video_preparation_pool_runs_whole_tasks_concurrently_in_order(self):
        barrier = threading.Barrier(2)

        def prepare(value, processor, device, dtype):
            barrier.wait(timeout=5)
            return value + 10

        with mock.patch.object(
            gpu_video, "_preprocess_video_ready", side_effect=prepare
        ):
            pool = gpu_video.VideoDecodePool(2)
            try:
                with pool.preprocess(
                    range(4), None, "cuda:0", torch.bfloat16
                ) as values:
                    self.assertEqual(list(values), [10, 11, 12, 13])
            finally:
                pool.close()

    def test_prepared_video_waits_and_records_consumer_stream(self):
        source = GpuVideoInput(b"", 2, (0, 1), 32, 32, 32, 32, 16, 2)
        pixels = mock.Mock()
        stream = mock.Mock()
        ready = mock.sentinel.ready
        prepared = gpu_video.PreparedGpuVideo(source, pixels, ready)
        self.assertIs(prepared.consume(stream), pixels)
        stream.wait_event.assert_called_once_with(ready)
        pixels.record_stream.assert_called_once_with(stream)
        self.assertEqual(prepared.shape, source.shape)

    def test_parallel_decode_preserves_order_and_bounds_prefetch(self):
        second_done = threading.Event()
        started = []
        finished = []
        lock = threading.Lock()

        def decode(index, device):
            with lock:
                started.append(index)
            if index == 0:
                self.assertTrue(second_done.wait(5))
            result = SimpleNamespace(index=index, record_stream=mock.Mock())
            with lock:
                finished.append(index)
            if index == 1:
                second_done.set()
            return result

        with mock.patch.object(
            gpu_video, "_decode_video_ready", side_effect=decode
        ), mock.patch("torch.cuda.current_stream", return_value=mock.sentinel.stream):
            pool = gpu_video.VideoDecodePool(2)
            try:
                with pool.decode(range(6), "cuda:0") as results:
                    first = next(results)
                    self.assertEqual(first.index, 0)
                    self.assertEqual(set(started), {0, 1})
                    self.assertEqual(finished[:2], [1, 0])
                    rest = list(results)
                self.assertEqual(
                    [first.index, *[r.index for r in rest]], list(range(6))
                )
                for item in [first, *rest]:
                    item.record_stream.assert_called_once_with(mock.sentinel.stream)
            finally:
                pool.close()

    def test_parallel_decode_drains_errors_and_can_recover(self):
        second_started = threading.Event()
        drained = threading.Event()

        def decode(index, device):
            if index == 0:
                self.assertTrue(second_started.wait(5))
                raise ValueError("broken video")
            second_started.set()
            time.sleep(0.05)
            drained.set()
            return SimpleNamespace(record_stream=mock.Mock())

        with mock.patch.object(
            gpu_video, "_decode_video_ready", side_effect=decode
        ), mock.patch("torch.cuda.current_stream", return_value=mock.sentinel.stream):
            pool = gpu_video.VideoDecodePool(2)
            try:
                with self.assertRaisesRegex(ValueError, "broken video"):
                    with pool.decode(range(8), "cuda:0") as results:
                        next(results)
                self.assertTrue(drained.is_set())
                with pool.decode([1, 2], "cuda:0") as results:
                    self.assertEqual(len(list(results)), 2)
            finally:
                pool.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_prefetched_frames_skip_decode_and_keep_processor_output(self):
        processor = self.processor()
        data = GpuVideoInput(b"", 5, tuple(range(5)), 64, 96, 32, 64, 16, 2)
        frames = torch.randint(256, (5, 3, 64, 96), dtype=torch.uint8, device="cuda")
        with mock.patch.object(gpu_video, "decode_video_cuda", return_value=frames):
            expected = preprocess_video_cuda(data, processor, "cuda:0")
        with mock.patch.object(
            gpu_video, "decode_video_cuda", side_effect=AssertionError("decoded twice")
        ):
            actual = preprocess_video_cuda(data, processor, "cuda:0", video=frames)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

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
    def test_resized_nvdec_input_stays_on_cuda(self):
        processor = self.processor()
        generator = torch.Generator().manual_seed(37)
        frames = torch.randint(
            256, (5, 3, 64, 96), dtype=torch.uint8, generator=generator
        ).cuda()
        for height, width in ((32, 64), (96, 160)):
            with self.subTest(height=height, width=width):
                data = GpuVideoInput(
                    b"", 5, tuple(range(5)), 64, 96, height, width, 16, 2
                )
                resized = gpu_video.resize_video_to_shape(
                    frames, processor, height, width
                )
                self.assertTrue(resized.is_cuda)
                self.assertEqual(resized.dtype, torch.uint8)
                expected = processor(
                    resized,
                    return_tensors="pt",
                    do_resize=False,
                    do_sample_frames=False,
                )["pixel_values_videos"]
                with mock.patch.object(
                    gpu_video, "decode_video_cuda", return_value=frames
                ), mock.patch.object(
                    torch.Tensor,
                    "cpu",
                    side_effect=AssertionError("GPU video left CUDA"),
                ):
                    actual = preprocess_video_cuda(data, processor, "cuda:0")
                self.assertTrue(actual.is_cuda)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

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


class DecoderReconfigurationTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(gpu_video, "_decode_state", threading.local())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.nvc = mock.MagicMock()
        self.nvc.CreateDecoder.side_effect = lambda **kwargs: mock.Mock()

    def get_decoder(self, width, height, codec=4, stream=7):
        return gpu_video._get_video_decoder(
            SimpleNamespace(width=width, height=height),
            torch.device("cuda:0"),
            self.nvc,
            SimpleNamespace(cuda_stream=stream),
            codec,
        )

    def test_landscape_portrait_and_repeat_reuse_one_session(self):
        decoder = self.get_decoder(1280, 720)
        for width, height in ((720, 1280), (1920, 788), (1920, 870), (1280, 720)):
            self.assertIs(self.get_decoder(width, height), decoder)
        self.assertIs(self.get_decoder(1280, 720), decoder)
        self.assertEqual(self.nvc.CreateDecoder.call_count, 1)
        self.assertEqual(
            decoder.setReconfigParams.call_args_list,
            [
                mock.call(720, 1280),
                mock.call(1920, 788),
                mock.call(1920, 870),
                mock.call(1280, 720),
            ],
        )

    def test_larger_input_grows_session_then_reuses_smaller_input(self):
        original = self.get_decoder(1280, 720)
        larger = self.get_decoder(3840, 2160)
        self.assertIsNot(original, larger)
        self.assertEqual(self.nvc.CreateDecoder.call_count, 2)
        self.assertGreaterEqual(
            self.nvc.CreateDecoder.call_args.kwargs["maxwidth"], 3840
        )
        self.assertGreaterEqual(
            self.nvc.CreateDecoder.call_args.kwargs["maxheight"], 2160
        )
        self.assertIs(self.get_decoder(720, 1280), larger)
        self.assertIs(self.get_decoder(3840, 2160), larger)
        self.assertEqual(self.nvc.CreateDecoder.call_count, 2)

    def test_codec_and_stream_changes_do_not_share_session(self):
        first = self.get_decoder(1280, 720)
        second = self.get_decoder(1280, 720, codec=8)
        third = self.get_decoder(1280, 720, codec=8, stream=9)
        self.assertIsNot(first, second)
        self.assertIsNot(second, third)
        self.assertEqual(self.nvc.CreateDecoder.call_count, 3)

    def test_failed_reconfiguration_discards_session_before_retry(self):
        decoder = self.get_decoder(1280, 720)
        decoder.setReconfigParams.side_effect = RuntimeError("bad reconfiguration")
        with self.assertRaisesRegex(RuntimeError, "bad reconfiguration"):
            self.get_decoder(720, 1280)
        self.assertIsNone(gpu_video._decode_state.decoder)
        self.assertIsNone(gpu_video._decode_state.key)
        self.assertIsNot(self.get_decoder(720, 1280), decoder)
        self.assertEqual(self.nvc.CreateDecoder.call_count, 2)

    def test_failed_growth_discards_previous_session(self):
        self.get_decoder(1280, 720)
        self.nvc.CreateDecoder.side_effect = RuntimeError("allocation failed")
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self.get_decoder(3840, 2160)
        self.assertIsNone(gpu_video._decode_state.decoder)
        self.assertIsNone(gpu_video._decode_state.key)


if __name__ == "__main__":
    unittest.main()
