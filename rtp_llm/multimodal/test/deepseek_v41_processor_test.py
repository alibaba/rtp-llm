"""Official-revision parity for backend-owned V4.1 geometry and pixels."""

import base64
import hashlib
import importlib.util
import io
import os
import sys
import threading
import unittest
from concurrent.futures import CancelledError, Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from urllib3.exceptions import ReadTimeoutError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.mm_preprocess import LocalPreprocessExecutor
from rtp_llm.multimodal.mm_work_item import MMWorkItem
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
    deepseek_v41_vision as vision_module,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    V41ImageProcessorConfig,
    num_image_tokens,
    plan_image_grid,
    preprocess_image,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType

PROCESSOR_SHA256 = "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272"


def load_reference(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROCESSOR_SHA256:
        raise ValueError(f"reference source does not match fixed HF revision: {path}")
    spec = importlib.util.spec_from_file_location(
        "dsv41_official_image_processor", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def image_data(width, height, mode="RGB", offset=0):
    values = (
        np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3)
        + offset
    ) % 256
    image = Image.fromarray(values.astype(np.uint8), "RGB").convert(mode)
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


class V41BackendProcessorParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.environ.get("DSV41_MODEL_PATH")
        if not model_path:
            raise unittest.SkipTest("DSV41_MODEL_PATH is required for official parity")
        cls.reference = load_reference(
            Path(model_path) / "inference/image_processor.py"
        )
        cls.config = V41ImageProcessorConfig()
        cls.reference_args = type(
            "Args",
            (),
            {
                **cls.config.__dict__,
                "vision_enabled": True,
                "vision_max_wh_ratio": None,
            },
        )()

    def test_geometry_and_complete_span_length_match_official(self):
        for width, height in (
            (1, 1),
            (1, 8192),
            (8192, 1),
            (17, 31),
            (544, 544),
            (1000, 999),
            (3072, 4096),
            (16, 65535),
            (65535, 16),
        ):
            with self.subTest(size=(width, height)):
                actual = plan_image_grid(width, height, self.config)
                expected = self.reference.plan_image_grid(
                    width, height, self.reference_args
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    num_image_tokens(*actual[:2]),
                    self.reference.image_token_types(*expected[:2]).numel(),
                )
                self.assertLessEqual(num_image_tokens(*actual[:2]), 1024)

    def test_pixels_and_patch_order_match_official_byte_for_byte(self):
        for width, height, mode in (
            (1, 1, "RGB"),
            (17, 31, "RGBA"),
            (545, 546, "L"),
            (1, 8192, "RGB"),
            (8192, 1, "RGB"),
            (1400, 900, "RGB"),
        ):
            with self.subTest(size=(width, height), mode=mode):
                data = image_data(width, height, mode)
                with Image.open(io.BytesIO(data)) as image:
                    actual = preprocess_image(image, self.config)
                expected = self.reference.load_image(
                    {"data": data}, self.reference_args
                )
                self.assertEqual(actual[1:], expected[1:])
                self.assertEqual(actual[0].dtype, torch.bfloat16)
                self.assertTrue(
                    torch.equal(
                        actual[0].contiguous().view(torch.uint8),
                        expected[0].contiguous().view(torch.uint8),
                    )
                )


class V41ImageDownloadTimingTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.multimodal import multimodal_util

        self.cache = multimodal_util.URLDataCache(cache_size=2)
        cache_patch = patch.object(multimodal_util, "url_data_cache_", self.cache)
        cache_patch.start()
        self.addCleanup(cache_patch.stop)

    def test_http_attempts_share_the_download_deadline(self):
        from rtp_llm.multimodal import multimodal_util as util
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
            deepseek_v41_processor as processor,
        )

        for caller_deadline, budget in ((None, 30), (190.0, 30), (102.0, 2)):
            with self.subTest(caller_deadline=caller_deadline):
                clock = [100.0]
                timeouts = []

                def advance(seconds):
                    clock[0] += seconds

                def stalled_get(url, headers, timeout):
                    connect, read = timeout
                    timeouts.append(timeout)
                    if len(timeouts) == 1:
                        advance(connect)
                        raise requests.ConnectTimeout()
                    advance(read)
                    raise requests.ReadTimeout()

                fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=advance)
                with patch.object(processor, "time", fake_time), patch.object(
                    util, "time", fake_time
                ), patch.object(util, "REQUEST_GET", stalled_get):
                    with self.assertRaises(FtRuntimeException) as raised:
                        processor.load_image_source(
                            "https://example.invalid/image.png",
                            deadline=caller_deadline,
                        )
                self.assertEqual(
                    raised.exception.exception_type, ExceptionType.MM_DOWNLOAD_FAILED
                )
                self.assertAlmostEqual(clock[0] - 100.0, budget)
                if budget == 2:
                    self.assertEqual(timeouts, [(2, 2)])
                else:
                    # The first connect retry and the next outer GET both spend
                    # the original budget (including 0.2 s of retry backoff).
                    self.assertEqual(timeouts[:2], [(3, 20), (3, 20)])
                    self.assertEqual(len(timeouts), 3)
                    self.assertEqual(timeouts[2][0], 3)
                    self.assertAlmostEqual(timeouts[2][1], 6.8)

    def run_loader(
        self,
        statuses,
        *,
        repeats=1,
        deadline=None,
        cancellation_event=None,
        chunk_error=None,
    ):
        from rtp_llm.multimodal.multimodal_util import collect_download_timing

        processor = (
            "rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor"
        )
        clock = [100.0]
        status_iter = iter(statuses)

        def get_response(*args, **kwargs):
            clock[0] += 0.125
            status = next(status_iter)
            if isinstance(status, Exception):
                raise status
            response = MagicMock(status_code=status)
            response.__enter__.return_value = response
            return response

        def sleep(seconds):
            clock[0] += seconds

        if cancellation_event is not None:
            cancellation_event.wait.side_effect = sleep

        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
            load_image_source,
        )

        # Replace module bindings, not the shared time module: metric/scheduler
        # background threads must not advance this deterministic retry clock.
        fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
        with patch(processor + ".time", fake_time), patch(
            "rtp_llm.multimodal.multimodal_util.time", fake_time
        ), patch(
            "rtp_llm.multimodal.multimodal_util.request_get", side_effect=get_response
        ) as get, patch(
            processor + "._image_download_chunks",
            return_value=[b"image"],
            side_effect=chunk_error,
        ), collect_download_timing() as timing:
            try:
                result = []
                for _ in range(repeats):
                    with load_image_source(
                        "https://example.invalid/image.png",
                        deadline=deadline,
                        cancellation_event=cancellation_event,
                    ) as image:
                        result.append(image.read())
            except FtRuntimeException as error:
                result = error
        return result, timing.elapsed_ms, get.call_count

    def test_http_cache_hit_skips_download_and_backoff_is_not_timed(self):
        result, elapsed_ms, requests = self.run_loader([503, 200], repeats=3)
        self.assertEqual(result, [b"image"] * 3)
        self.assertEqual(requests, 2)
        self.assertEqual(elapsed_ms, 250.0)

    def test_http_phase_metrics_cover_v41_retry_and_cache_hit(self):
        from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics
        from rtp_llm.multimodal import vit_metrics
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
            deepseek_v41_processor as processor,
        )

        with patch.object(processor, "VIT_PHASE_METRICS", True), patch.object(
            vit_metrics, "VIT_PHASE_METRICS", True
        ), vit_metrics.collect_vit_preprocess_metrics() as metrics:
            result, elapsed_ms, requests = self.run_loader([503, 200], repeats=2)

        self.assertEqual(result, [b"image"] * 2)
        self.assertEqual((elapsed_ms, requests), (250.0, 2))
        phases = [
            sample.tags["phase"]
            for sample in metrics.samples
            if sample.metric == GaugeMetrics.VIT_HTTP_PHASE_RT_US_METRIC
        ]
        self.assertEqual(phases, ["headers", "backoff", "headers", "body"])
        values = [
            (sample.metric, sample.value, sample.tags)
            for sample in metrics.samples
            if sample.metric != GaugeMetrics.VIT_HTTP_PHASE_RT_US_METRIC
        ]
        self.assertEqual(
            values,
            [
                (AccMetrics.VIT_HTTP_SOURCE_RETRY_QPS_METRIC, 1, {}),
                (GaugeMetrics.VIT_HTTP_BODY_BYTES_METRIC, 5, {}),
                (AccMetrics.VIT_URL_SOURCE_QPS_METRIC, 1, {"source": "load"}),
                (AccMetrics.VIT_URL_SOURCE_QPS_METRIC, 1, {"source": "cache"}),
            ],
        )

    def test_http_phase_metrics_off_preserves_download(self):
        from rtp_llm.multimodal import vit_metrics
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
            deepseek_v41_processor as processor,
        )

        with patch.object(processor, "VIT_PHASE_METRICS", False), patch.object(
            vit_metrics, "VIT_PHASE_METRICS", False
        ), vit_metrics.collect_vit_preprocess_metrics() as metrics:
            result, elapsed_ms, requests = self.run_loader([503, 200], repeats=2)

        self.assertEqual(result, [b"image"] * 2)
        self.assertEqual((elapsed_ms, requests), (250.0, 2))
        self.assertFalse(metrics.samples)

    def test_disabled_url_cache_downloads_each_time(self):
        self.cache.resize_cache(0)
        result, elapsed_ms, requests = self.run_loader([200, 200], repeats=2)
        self.assertEqual(result, [b"image", b"image"])
        self.assertEqual(requests, 2)
        self.assertEqual(elapsed_ms, 250.0)

    def test_native_download_timing_includes_each_callers_queue_wait(self):
        from rtp_llm.multimodal.media_network import SourceBatch

        # A legacy wheel returns seven fields. A new wheel appends queue
        # milliseconds and the shared-transfer flag.
        def completed(worker_ms, hit=False, extra=()):
            result = Future()
            result.set_result((200, "", b"data", hit, 0, (0, 0, 0), worker_ms) + extra)
            return result

        network = MagicMock()
        network.config.mm_image_max_file_size_kb = 1
        network.image.side_effect = [
            completed(3),
            completed(5, extra=(100, False)),
            completed(8, extra=(25, True)),
            completed(1, hit=True, extra=(0, False)),
        ]
        item = SimpleNamespace(
            mm_inputs=[
                SimpleNamespace(
                    url=f"https://example.invalid/{i}", mm_type=MMUrlType.IMAGE
                )
                for i in range(4)
            ],
            remaining_timeout_s=lambda: 30,
            region=None,
            preprocess_ready_event=threading.Event(),
        )
        batch = SourceBatch(network, item)
        self.assertEqual(batch.result(), (b"data",) * 4)
        self.assertEqual(batch.fetch_times_us, (3000, 105000, 33000, 1000))
        self.assertEqual(batch.network_ms, 105)

    def test_native_join_reports_wait_once_without_recounting_transfer(self):
        from rtp_llm.metrics import AccMetrics
        from rtp_llm.multimodal.media_network import MediaNetwork

        network = MagicMock(spec=MediaNetwork)
        network.native = SimpleNamespace(Request=lambda: SimpleNamespace(proxy=""))
        network.proxy_for = None
        future = Future()
        network.submit.return_value = future
        with patch("rtp_llm.multimodal.vit_metrics.VIT_PHASE_METRICS", True), patch(
            "rtp_llm.metrics.kmonitor.report"
        ) as report:
            MediaNetwork.image(network, "http://example.invalid/img", "", 0, 30, None)
            future.set_result((200, "", b"data", False, 1, (0, 0, 1), 8, 25, True))
        self.assertEqual(
            [call.args[0] for call in report.call_args_list],
            [
                AccMetrics.VIT_URL_SOURCE_QPS_METRIC,
                GaugeMetrics.VIT_HTTP_PHASE_RT_US_METRIC,
            ],
        )
        report.assert_any_call(
            AccMetrics.VIT_URL_SOURCE_QPS_METRIC, tags={"source": "shared"}
        )
        report.assert_any_call(
            GaugeMetrics.VIT_HTTP_PHASE_RT_US_METRIC,
            25000,
            tags={"phase": "worker_queue", "source": "shared"},
        )

    def test_native_download_delivers_while_metrics_reporter_is_blocked(self):
        from rtp_llm.config.py_config_modules import VitConfig
        from rtp_llm.multimodal.media_network import MediaNetwork

        reporting = threading.Event()
        release = threading.Event()
        config = VitConfig()

        def block_report(*args, **kwargs):
            reporting.set()
            release.wait(5)

        with (
            patch.dict(os.environ, {"MM_VIT_DOWNLOAD_FAST_PATH": "1"}),
            patch("rtp_llm.multimodal.vit_metrics.VIT_PHASE_METRICS", True),
            patch("rtp_llm.metrics.kmonitor.report", side_effect=block_report) as report,
        ):
            network = MediaNetwork(config)
            try:
                first = network.image("http://example.invalid/a", "", 0, 0, None)
                self.assertTrue(reporting.wait(3))
                second = network.image("http://example.invalid/b", "", 0, 0, None)
                ready = threading.Event()
                second.add_done_callback(lambda _: ready.set())
                self.assertTrue(ready.wait(2))
                self.assertEqual(first.result()[1], "timeout")
                self.assertEqual(second.result()[1], "timeout")
            finally:
                release.set()
                network.close()
                network.close()
        source_reports = [
            call for call in report.call_args_list
            if call.kwargs.get("tags") == {"source": "load"}
        ]
        self.assertEqual(len(source_reports), 2)

    def test_configured_image_size_limit_on_python_and_native_downloads(self):
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        from tempfile import TemporaryDirectory

        from rtp_llm.config.py_config_modules import VitConfig
        from rtp_llm.multimodal.media_network import MediaNetwork, SourceBatch
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
            load_image_source,
        )

        size_kb = 11 * 1024
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with (Path(directory.name) / "image.png").open("wb") as output:
            output.write(image_data(3, 5))
            output.truncate(size_kb * 1024)
        handler = partial(SimpleHTTPRequestHandler, directory=directory.name)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}/image.png"
        network = MediaNetwork(VitConfig())
        self.addCleanup(network.close)

        def native_load(url, limit_kb):
            network.config.mm_image_max_file_size_kb = limit_kb
            item = SimpleNamespace(
                mm_inputs=[SimpleNamespace(url=url, mm_type=MMUrlType.IMAGE)],
                remaining_timeout_s=lambda: 30,
                region=None,
                preprocess_ready_event=threading.Event(),
            )
            return SourceBatch(network, item).result()[0]

        def python_load(url, limit_kb):
            return load_image_source(url, max_file_size_kb=limit_kb).getvalue()

        for load in (python_load, native_load):
            with self.subTest(loader=load.__name__):
                self.assertEqual(len(load(url, size_kb)), size_kb * 1024)
                for source in (url, url + "?uncached"):
                    with self.assertRaises(FtRuntimeException) as error:
                        load(source, size_kb - 1)
                    self.assertEqual(
                        error.exception.exception_type, ExceptionType.INVALID_PARAMS
                    )
                self.assertEqual(len(load(url + "?unlimited", 0)), size_kb * 1024)

    def test_cached_bytes_respect_current_size_limit_deadline_and_cancellation(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
            load_image_source,
        )

        url = "https://example.invalid/image.png"
        self.cache.get_or_load(url, "", lambda: io.BytesIO(b"x" * 1025))
        cancelled = threading.Event()
        cancelled.set()
        with patch("rtp_llm.multimodal.multimodal_util.request_get") as get:
            with self.assertRaises(FtRuntimeException) as size_error:
                load_image_source(url, max_file_size_kb=1)
            self.assertEqual(
                size_error.exception.exception_type, ExceptionType.INVALID_PARAMS
            )
            with self.assertRaises(FtRuntimeException) as timeout:
                load_image_source(url, deadline=0)
            self.assertEqual(
                timeout.exception.exception_type, ExceptionType.MM_DOWNLOAD_FAILED
            )
            with self.assertRaises(CancelledError):
                load_image_source(url, cancellation_event=cancelled)
            get.assert_not_called()

    def test_exhausted_attempts_are_timed_and_preserve_download_error(self):
        result, elapsed_ms, requests = self.run_loader([503, 503, 503])
        self.assertIsInstance(result, FtRuntimeException)
        self.assertEqual(result.exception_type, ExceptionType.MM_DOWNLOAD_FAILED)
        self.assertEqual(
            result.message, "Temporary multimodal download failure: HTTP 503"
        )
        self.assertEqual(requests, 3)
        self.assertEqual(elapsed_ms, 375.0)

    def test_nonretryable_attempt_is_timed_and_preserves_invalid_params(self):
        result, elapsed_ms, requests = self.run_loader([404])
        self.assertIsInstance(result, FtRuntimeException)
        self.assertEqual(result.exception_type, ExceptionType.INVALID_PARAMS)
        self.assertEqual(
            result.message, "Failed to download multimodal content: HTTP 404"
        )
        self.assertEqual(requests, 1)
        self.assertEqual(elapsed_ms, 125.0)

    def test_denied_url_rejects_before_source_cache_lookup(self):
        from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
            load_image_source,
        )
        from rtp_llm.multimodal.multimodal_util import url_data_cache_

        with patch.dict(
            os.environ, {"MM_DOWNLOAD_DOMAIN_DENYLIST": '["web.archive.org"]'}
        ), patch.object(
            url_data_cache_, "check_cache", return_value=io.BytesIO(b"cached source")
        ) as lookup:
            with self.assertRaises(FtRuntimeException) as error:
                load_image_source("https://web.archive.org/cached.png")
        self.assertEqual(
            error.exception.exception_type, ExceptionType.MM_WRONG_FORMAT_ERROR
        )
        lookup.assert_not_called()

    def test_ssrf_client_error_is_preserved_without_retry(self):
        error = FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR, "url ssrf check failed"
        )
        result, elapsed_ms, attempts = self.run_loader([error])
        self.assertIs(result, error)
        self.assertEqual(attempts, 1)
        self.assertEqual(elapsed_ms, 125.0)

    def test_network_timeouts_are_download_errors_and_exclude_retry_backoff(self):
        for error in (
            requests.ConnectTimeout("connect timeout"),
            requests.ReadTimeout("read timeout"),
            ReadTimeoutError(None, "/image.png", "read timeout"),
            TimeoutError("download deadline exceeded"),
        ):
            with self.subTest(error=type(error).__name__):
                result, elapsed_ms, attempts = self.run_loader([error] * 3)
                self.assertIsInstance(result, FtRuntimeException)
                self.assertEqual(
                    result.exception_type, ExceptionType.MM_DOWNLOAD_FAILED
                )
                self.assertEqual(
                    result.message,
                    f"Temporary multimodal download failure: {type(error).__name__}",
                )
                self.assertIs(result.__cause__, error)
                self.assertEqual(attempts, 3)
                self.assertEqual(elapsed_ms, 375.0)

    def test_streaming_read_timeout_is_a_download_error(self):
        error = ReadTimeoutError(None, "/image.png", "stream timeout")
        result, elapsed_ms, attempts = self.run_loader([200] * 3, chunk_error=error)
        self.assertEqual(result.exception_type, ExceptionType.MM_DOWNLOAD_FAILED)
        self.assertIs(result.__cause__, error)
        self.assertEqual(attempts, 3)
        self.assertEqual(elapsed_ms, 375.0)

    def test_download_deadline_expiring_during_backoff_is_a_download_error(self):
        for cancellable in (False, True):
            with self.subTest(cancellable=cancellable):
                event = MagicMock() if cancellable else None
                if event is not None:
                    event.is_set.return_value = False
                result, elapsed_ms, attempts = self.run_loader(
                    [503], deadline=100.2, cancellation_event=event
                )
                self.assertEqual(
                    result.exception_type, ExceptionType.MM_DOWNLOAD_FAILED
                )
                self.assertIsInstance(result.__cause__, TimeoutError)
                self.assertEqual(attempts, 1)
                self.assertEqual(elapsed_ms, 125.0)

    def test_cancellation_during_backoff_is_not_a_download_error(self):
        event = MagicMock()
        event.is_set.side_effect = lambda: event.wait.called
        with self.assertRaises(CancelledError):
            self.run_loader([503], cancellation_event=event)


class V41BackendPreprocessContractTest(unittest.TestCase):
    def setUp(self):
        self.config = V41ImageProcessorConfig(vision_min_pixels=0)
        self.vit_config = SimpleNamespace(
            download_headers="", mm_image_max_file_size_kb=10240
        )

    def media(self, data):
        return MultimodalInput(
            "data:image/png;base64," + base64.b64encode(data).decode(),
            MMUrlType.IMAGE,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 0),
        )

    def prepare(self, data, **kwargs):
        # This helper tests the CPU patch contract; production defaults to RGB
        # decoding for GPU preprocessing. Tests below cover that mode explicitly.
        kwargs.setdefault("gpu_preprocess", False)
        return vision_module.DeepSeekV41VisionEmbedding.preprocess_input(
            [self.media(data)], self.vit_config, processor_config=self.config, **kwargs
        )

    def prepare_through_executor_boundary(self, data):
        executor = LocalPreprocessExecutor(
            vision_module.DeepSeekV41VisionEmbedding.preprocess_input,
            self.vit_config,
            {"processor_config": self.config, "gpu_preprocess": False},
        )
        work_item = MMWorkItem([self.media(data)])
        executor.submit(work_item)
        executor.get_result(work_item)
        return work_item.preprocess_result

    def test_oversized_image_is_a_typed_input_error_before_resize(self):
        data = image_data(8, 8)
        with patch.object(Image, "MAX_IMAGE_PIXELS", 16), self.assertRaises(
            FtRuntimeException
        ) as raised:
            self.prepare_through_executor_boundary(data)
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.MM_IMAGE_PIXEL_LIMIT_EXCEEDED
        )
        self.assertIn("64 pixels", raised.exception.message)
        self.assertIn("32 pixels", raised.exception.message)

    def test_cpu_pixels_and_rgb_decode_keep_order_and_owned_storage(self):
        sources = [
            image_data(31, 42),
            image_data(42, 31, offset=17),
            image_data(31, 42),
        ]
        for gpu in (False, True):
            actual = vision_module.DeepSeekV41VisionEmbedding.preprocess_input(
                [self.media(data) for data in sources],
                self.vit_config,
                processor_config=self.config,
                gpu_preprocess=gpu,
            )
            for item, data in zip(actual, sources):
                with Image.open(io.BytesIO(data)) as image:
                    if gpu:
                        expected = torch.from_numpy(
                            np.array(image.convert("RGB"), copy=True)
                        )
                        self.assertTrue(torch.equal(item.rgb, expected))
                        self.assertEqual(item.rgb.device.type, "cpu")
                    else:
                        expected = preprocess_image(image, self.config)
                        self.assertTrue(torch.equal(item.patches, expected[0]))
                        self.assertEqual((item.n_vit_h, item.n_vit_w), expected[1:3])
            first = actual[0].rgb if gpu else actual[0].patches
            last = actual[2].rgb if gpu else actual[2].patches
            self.assertTrue(torch.equal(first, last))
            self.assertNotEqual(first.data_ptr(), last.data_ptr())

    def test_preprocessing_reports_stage_samples(self):
        with patch("rtp_llm.metrics.kmonitor.report") as report:
            self.prepare_through_executor_boundary(image_data(3, 5))
        metrics = [call.args[0] for call in report.call_args_list]
        for metric in (
            GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC,
            GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC,
            GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC,
            GaugeMetrics.VIT_IMAGE_PROCESSOR_RT_US_METRIC,
            GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC,
        ):
            self.assertEqual(metrics.count(metric), 1)
        for call in report.call_args_list:
            self.assertGreaterEqual(call.args[1], 0)

    def test_metric_failure_preserves_pixels_and_public_parse_error(self):
        data = image_data(3, 5)
        expected = self.prepare(data)
        with patch(
            "rtp_llm.metrics.kmonitor.report",
            side_effect=RuntimeError("metrics unavailable"),
        ):
            with self.assertLogs(level="ERROR"):
                actual = self.prepare_through_executor_boundary(data)
            with self.assertRaisesRegex(FtRuntimeException, MMErr.IMG_OPEN):
                self.prepare_through_executor_boundary(b"not an image")
        torch.testing.assert_close(actual.patches, expected.patches, rtol=0, atol=0)

    def test_bad_images_preserve_public_error_and_never_report_resize(self):
        for data in (b"not an image", image_data(3, 5)[:24]):
            with self.subTest(length=len(data)), patch(
                "rtp_llm.metrics.kmonitor.report"
            ) as report:
                with self.assertRaisesRegex(
                    FtRuntimeException, MMErr.IMG_OPEN
                ) as raised:
                    self.prepare(data)
                self.assertEqual(
                    raised.exception.exception_type, ExceptionType.MM_WRONG_FORMAT_ERROR
                )
                self.assertIsInstance(raised.exception.__cause__, OSError)
                self.assertNotIn(
                    GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC,
                    [call.args[0] for call in report.call_args_list],
                )

    def test_lazy_decode_failure_preserves_public_error_in_both_modes(self):
        data = image_data(3, 5)
        for gpu_preprocess in (False, True):
            with self.subTest(gpu_preprocess=gpu_preprocess), patch.object(
                Image.Image, "load", side_effect=OSError("image decode failed")
            ):
                with self.assertRaisesRegex(
                    FtRuntimeException, MMErr.IMG_OPEN
                ) as raised:
                    self.prepare(data, gpu_preprocess=gpu_preprocess)
                self.assertEqual(
                    raised.exception.exception_type, ExceptionType.MM_WRONG_FORMAT_ERROR
                )
                self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_byte_limit_rejects_before_image_decode(self):
        self.vit_config.mm_image_max_file_size_kb = 1
        with patch.object(Image, "open", wraps=Image.open) as decode:
            with self.assertRaisesRegex(Exception, "Multimodal file size is too large"):
                self.prepare(image_data(3, 5).ljust(1025, b"0"))
            decode.assert_not_called()
        self.prepare(image_data(3, 5).ljust(1024, b"0"))

    def test_heic_image_payload(self):
        from pillow_heif import from_pillow

        data = io.BytesIO()
        from_pillow(Image.new("RGB", (64, 64), (255, 0, 0))).save(data, quality=100)
        patches = self.prepare(data.getvalue()).patches.float()
        self.assertTrue(torch.isfinite(patches).all())
        self.assertGreater(patches[:, 0].mean().item(), 0.9)
        self.assertLess(patches[:, 1:].mean().item(), -0.9)

    def test_inspection_retains_original_bytes_and_canonical_patches(self):
        import time

        data = image_data(31, 42, "RGBA")
        adapter = object.__new__(vision_module.DeepSeekV41VisionEmbedding)
        torch.nn.Module.__init__(adapter)
        media = self.media(data)
        inspected = adapter.preprocess_input(
            [media, media],
            self.vit_config,
            processor_config=self.config,
            gpu_preprocess=False,
            capture_inspection_sources=True,
            deadline=time.monotonic() + 30,
        )
        self.assertEqual(inspected.inspection_sources, [data, data])
        original = self.prepare(data)
        for item in inspected.data:
            torch.testing.assert_close(item.patches, original.patches, rtol=0, atol=0)
        self.assertFalse(hasattr(original, "inspection_sources"))

    def test_low_detail_uses_the_model_resize_grid_for_each_image(self):
        self.config = V41ImageProcessorConfig()
        source = image_data(900, 630)
        media = [self.media(source), self.media(source)]
        media[1].mm_preprocess_config.width = media[1].mm_preprocess_config.height = 512
        adapter = object.__new__(vision_module.DeepSeekV41VisionEmbedding)
        adapter.processor_config = self.config
        for gpu in (False, True):
            result = adapter.preprocess_input(
                media,
                self.vit_config,
                processor_config=self.config,
                gpu_preprocess=gpu,
                capture_inspection_sources=True,
            )
            self.assertEqual(result.inspection_sources, [source, source])
            for item, grid_size in zip(result.data, [(900, 630), (512, 512)]):
                _, _, height, width = plan_image_grid(*grid_size, self.config)
                with Image.open(io.BytesIO(source)) as image:
                    if gpu:
                        self.assertTrue(
                            torch.equal(item.rgb, torch.from_numpy(np.array(image)))
                        )
                        self.assertEqual(item.resize_tables.target_hw, (height, width))
                    else:
                        expected = ImageOps.pad(
                            image, (width, height), color=(127, 127, 127)
                        )
                        reference = preprocess_image(expected, self.config)[0]
                        self.assertTrue(torch.equal(item.patches, reference))
                self.assertEqual(adapter._vit_grid(item), (height // 14, width // 14))

    def test_low_detail_keeps_a_one_pixel_strip_and_handles_gpu_size_limit(self):
        for width in (8192, 65537):
            with self.subTest(width=width):
                payload = io.BytesIO()
                Image.new("RGB", (width, 1), "white").save(payload, "PNG")
                media = self.media(payload.getvalue())
                media.mm_preprocess_config.width = media.mm_preprocess_config.height = (
                    512
                )
                options = dict(processor_config=V41ImageProcessorConfig())
                prepare = vision_module.DeepSeekV41VisionEmbedding.preprocess_input
                cpu = prepare([media], self.vit_config, gpu_preprocess=False, **options)
                self.assertEqual((cpu.patches[:, 0] == 1).sum().item(), 546)
                item = prepare([media], self.vit_config, gpu_preprocess=True, **options)
                if width == 8192:
                    self.assertEqual(item.resize_tables.target_hw, (546, 546))
                else:
                    self.assertTrue(torch.equal(item.patches, cpu.patches))

    def test_media_hints_leave_model_preprocessing_unchanged(self):
        data = [image_data(3, 5), image_data(5, 3)]
        media = [self.media(item) for item in data]
        media[1].mm_preprocess_config.width = 123
        media[1].mm_preprocess_config.min_pixels = 0
        media[1].mm_preprocess_config.max_pixels = 28901376
        adapter = object.__new__(vision_module.DeepSeekV41VisionEmbedding)
        adapter.validate_inputs(media)
        for gpu_preprocess in (False, True):
            with self.subTest(gpu_preprocess=gpu_preprocess):
                actual = adapter.preprocess_input(
                    media,
                    self.vit_config,
                    processor_config=self.config,
                    gpu_preprocess=gpu_preprocess,
                )
                expected = [
                    self.prepare(item, gpu_preprocess=gpu_preprocess) for item in data
                ]
                self.assertEqual(len(actual), len(expected))
                for result, reference in zip(actual, expected):
                    field = "rgb" if gpu_preprocess else "patches"
                    torch.testing.assert_close(
                        getattr(result, field),
                        getattr(reference, field),
                        rtol=0,
                        atol=0,
                    )

    def test_invalid_later_media_type_rejects_whole_group_before_io(self):
        media = [self.media(image_data(3, 5)), self.media(image_data(5, 3))]
        media[1].mm_type = MMUrlType.VIDEO
        with patch.object(vision_module, "load_image_source") as loader:
            with self.assertRaisesRegex(ValueError, "non-empty IMAGE inputs"):
                vision_module.DeepSeekV41VisionEmbedding.preprocess_input(
                    media, self.vit_config, processor_config=self.config
                )
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
