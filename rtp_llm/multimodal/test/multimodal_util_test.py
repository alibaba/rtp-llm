import base64
import io
import logging
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import requests
from PIL import Image

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MMPreprocessConfigPB
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.multimodal_util import (
    collect_download_timing,
    get_bytes_io_from_url,
    request_get,
    trans_config,
    url_data_cache_,
)


class _FakeResponse:
    def __init__(self, *, status_code=200, headers=None, content=b"payload"):
        self.status_code = status_code
        self.headers = headers or {}
        self._content = content
        self.content_accessed = False
        self.closed = False

    @property
    def content(self):
        self.content_accessed = True
        return self._content

    def iter_content(self, chunk_size):
        self.content_accessed = True
        for offset in range(0, len(self._content), chunk_size):
            yield self._content[offset : offset + chunk_size]

    def close(self):
        self.closed = True


class TestMultiModalUtil(unittest.TestCase):
    def assert_mm_error(self, exception_type, message, callable_):
        with self.assertRaises(FtRuntimeException) as context:
            callable_()
        self.assertEqual(context.exception.exception_type, exception_type)
        self.assertEqual(context.exception.message, message)

    def test_get_bytes(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            temp_path = tmp_file.name

            image = Image.new("RGB", (200, 200), "white")
            image.save(temp_path, format="PNG")

            self.assertTrue(
                Image.open(get_bytes_io_from_url(temp_path)).size == image.size
            )

    def test_trans_config_preserves_fractional_fps(self):
        config = trans_config(MMPreprocessConfigPB(fps=0.2, max_long_side_pixel=1008))

        self.assertAlmostEqual(config.fps, 0.2)
        self.assertEqual(config.max_long_side_pixel, 1008)

    def test_base64(self):
        buffer = io.BytesIO()

        image = Image.new("RGB", (200, 200), "white")
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        base64_str = "data:image/png;base64," + base64.b64encode(image_bytes).decode(
            "utf-8"
        )

        self.assertTrue(
            Image.open(get_bytes_io_from_url(base64_str)).size == image.size
        )

    def test_http_checks_content_length_before_body(self):
        response = _FakeResponse(headers={"Content-Length": str(2 * 1024)})
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            self.assert_mm_error(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                MMErr.FILE_TOO_LARGE,
                lambda: get_bytes_io_from_url(
                    "https://example.com/too-large", max_file_size_kb=1
                ),
            )
        self.assertFalse(response.content_accessed)
        self.assertTrue(response.closed)

    def test_http_checks_streamed_body_size(self):
        response = _FakeResponse(
            headers={"Content-Length": "1"},
            content=b"x" * (2 * 1024),
        )
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            self.assert_mm_error(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                MMErr.FILE_TOO_LARGE,
                lambda: get_bytes_io_from_url(
                    "https://example.com/incorrect-content-length",
                    max_file_size_kb=1,
                ),
            )
        self.assertTrue(response.content_accessed)
        self.assertTrue(response.closed)

    def test_http_requires_content_length(self):
        response = _FakeResponse()
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            self.assert_mm_error(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                MMErr.MISS_CONTENT_LEN,
                lambda: get_bytes_io_from_url(
                    "https://example.com/no-content-length", max_file_size_kb=1
                ),
            )
        self.assertFalse(response.content_accessed)
        self.assertTrue(response.closed)

    def test_http_success(self):
        response = _FakeResponse(headers={"Content-Length": "7"}, content=b"payload")
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            result = get_bytes_io_from_url(
                "https://example.com/success", max_file_size_kb=1
            )
        self.assertEqual(result.read(), b"payload")
        self.assertTrue(response.content_accessed)
        self.assertTrue(response.closed)

    def test_download_timing_excludes_cache_hits(self):
        """The preprocess timing context records misses and leaves hits at zero."""

        def slow_download(*args):
            time.sleep(0.01)
            return io.BytesIO(b"payload")

        with patch.object(
            url_data_cache_, "check_cache", return_value=None
        ), patch.object(url_data_cache_, "insert_cache"), patch(
            "rtp_llm.multimodal.multimodal_util._download_http_content",
            side_effect=slow_download,
        ):
            with collect_download_timing() as timing:
                get_bytes_io_from_url("https://example.com/timing")
        self.assertGreaterEqual(timing.elapsed_ms, 5.0)

        with patch.object(
            url_data_cache_, "check_cache", return_value=io.BytesIO(b"payload")
        ):
            with collect_download_timing() as cache_timing:
                get_bytes_io_from_url("https://example.com/cached")
        self.assertEqual(cache_timing.elapsed_ms, 0.0)

    def test_request_get_retries_connect_timeout_twice(self):
        response = _FakeResponse()
        request_impl = Mock(
            side_effect=[
                requests.exceptions.ConnectTimeout(),
                requests.exceptions.ConnectTimeout(),
                response,
            ]
        )
        with patch("rtp_llm.multimodal.multimodal_util.REQUEST_GET", request_impl):
            self.assertIs(request_get("https://example.com/image", {}), response)

        self.assertEqual(request_impl.call_count, 3)

    def test_request_get_stops_after_two_connect_timeout_retries(self):
        request_impl = Mock(side_effect=requests.exceptions.ConnectTimeout())
        with patch("rtp_llm.multimodal.multimodal_util.REQUEST_GET", request_impl):
            with self.assertRaises(requests.exceptions.ConnectTimeout):
                request_get("https://example.com/image", {})

        self.assertEqual(request_impl.call_count, 3)

    def test_request_get_does_not_retry_read_timeout(self):
        request_impl = Mock(side_effect=requests.exceptions.ReadTimeout())
        with patch("rtp_llm.multimodal.multimodal_util.REQUEST_GET", request_impl):
            with self.assertRaises(requests.exceptions.ReadTimeout):
                request_get("https://example.com/image", {})

        self.assertEqual(request_impl.call_count, 1)

    def test_http_timeout(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get",
            side_effect=requests.Timeout,
        ):
            self.assert_mm_error(
                ExceptionType.MM_PROCESS_ERROR,
                MMErr.DL_TIMEOUT,
                lambda: get_bytes_io_from_url("https://example.com/timeout"),
            )

    def test_http_invalid_url(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get",
            side_effect=requests.ConnectionError,
        ):
            self.assert_mm_error(
                ExceptionType.MM_PROCESS_ERROR,
                MMErr.URL_INVALID,
                lambda: get_bytes_io_from_url("https://example.invalid/image"),
            )

    def test_http_failure_status(self):
        response = _FakeResponse(status_code=404)
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            self.assert_mm_error(
                ExceptionType.MM_DOWNLOAD_FAILED,
                MMErr.DL_FAILED,
                lambda: get_bytes_io_from_url("https://example.com/not-found"),
            )
        self.assertFalse(response.content_accessed)
        self.assertTrue(response.closed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
