"""Official-revision parity for backend-owned V4.1 geometry and pixels."""

import base64
import hashlib
import importlib.util
import io
import os
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests
import torch
from PIL import Image

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.mm_process_engine import (
    _report_preprocess_timing,
    _run_preprocess_task,
    _unpack_preprocess_result,
)
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
    def run_loader(self, statuses, *, repeats=1):
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
            processor + "._image_download_chunks", return_value=[b"image"]
        ), collect_download_timing() as timing:
            try:
                result = [
                    load_image_source("https://example.invalid/image.png").getvalue()
                    for _ in range(repeats)
                ]
            except FtRuntimeException as error:
                result = error
        return result, timing.elapsed_ms, get.call_count

    def test_actual_attempts_exclude_backoff_and_never_cache_mutable_http_urls(self):
        result, elapsed_ms, requests = self.run_loader([503, 200, 200], repeats=2)
        self.assertEqual(result, [b"image", b"image"])
        self.assertEqual(requests, 3)
        self.assertEqual(elapsed_ms, 375.0)

    def test_exhausted_attempts_are_timed_and_preserve_download_error(self):
        result, elapsed_ms, requests = self.run_loader([503, 503, 503])
        self.assertIsInstance(result, FtRuntimeException)
        self.assertEqual(result.exception_type, ExceptionType.MM_DOWNLOAD_FAILED)
        self.assertEqual(result.message, "Temporary multimodal download failure")
        self.assertEqual(requests, 3)
        self.assertEqual(elapsed_ms, 375.0)

    def test_nonretryable_attempt_is_timed_and_preserves_invalid_params(self):
        result, elapsed_ms, requests = self.run_loader([404])
        self.assertIsInstance(result, FtRuntimeException)
        self.assertEqual(result.exception_type, ExceptionType.INVALID_PARAMS)
        self.assertEqual(result.message, "Failed to download multimodal content")
        self.assertEqual(requests, 1)
        self.assertEqual(elapsed_ms, 125.0)

    def test_network_timeouts_keep_timeout_code_and_exclude_retry_backoff(self):
        result, elapsed_ms, attempts = self.run_loader(
            [requests.Timeout("download timeout") for _ in range(3)]
        )
        self.assertIsInstance(result, FtRuntimeException)
        self.assertEqual(result.exception_type, ExceptionType.CONNECT_TIMEOUT)
        self.assertEqual(result.message, "Temporary multimodal download failure")
        self.assertEqual(attempts, 3)
        self.assertEqual(elapsed_ms, 375.0)


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
        return vision_module.DeepSeekV41VisionEmbedding.preprocess_input(
            [self.media(data)], self.vit_config, processor_config=self.config, **kwargs
        )

    def prepare_through_executor_boundary(self, data):
        payload = _run_preprocess_task(
            vision_module.DeepSeekV41VisionEmbedding.preprocess_input,
            [self.media(data)],
            self.vit_config,
            {"processor_config": self.config},
        )
        result, total_ms, download_ms = _unpack_preprocess_result(payload)
        _report_preprocess_timing(total_ms, download_ms)
        return result

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
            with self.assertRaisesRegex(ValueError, MMErr.IMG_OPEN):
                self.prepare_through_executor_boundary(b"not an image")
        torch.testing.assert_close(actual.patches, expected.patches, rtol=0, atol=0)

    def test_bad_images_preserve_public_error_and_never_report_resize(self):
        for data in (b"not an image", image_data(3, 5)[:24]):
            with self.subTest(length=len(data)), patch(
                "rtp_llm.metrics.kmonitor.report"
            ) as report:
                with self.assertRaisesRegex(ValueError, MMErr.IMG_OPEN) as raised:
                    self.prepare(data)
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
                with self.assertRaisesRegex(ValueError, MMErr.IMG_OPEN) as raised:
                    self.prepare(data, gpu_preprocess=gpu_preprocess)
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
        inspected = adapter.prepare_inspection_inputs(
            [media, media],
            self.vit_config,
            deadline=time.monotonic() + 30,
            cancellation_event=threading.Event(),
        )
        self.assertEqual(len(inspected), 2)
        for item in inspected:
            self.assertEqual(base64.b64decode(item.url.split(",", 1)[1]), data)
        original = self.prepare(data)
        result = adapter.preprocess_input(
            inspected, self.vit_config, processor_config=self.config
        )
        for item in result:
            torch.testing.assert_close(item.patches, original.patches, rtol=0, atol=0)

    def test_invalid_later_media_rejects_whole_group_before_io(self):
        media = [self.media(image_data(3, 5)), self.media(image_data(5, 3))]
        media[1].mm_preprocess_config.width = 123
        with patch.object(vision_module, "load_image_source") as loader:
            with self.assertRaisesRegex(ValueError, "fixed by the model config"):
                vision_module.DeepSeekV41VisionEmbedding.preprocess_input(
                    media, self.vit_config, processor_config=self.config
                )
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
