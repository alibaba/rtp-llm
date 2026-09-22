import base64
import io
import logging
import tempfile
import unittest
from unittest.mock import patch

import requests
from PIL import Image

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.mm_error_messages import MMErr, format_mm_rpc_error
from rtp_llm.multimodal.multimodal_util import (
    _record_download_time,
    collect_download_timing,
    get_bytes_io_from_url,
    get_json_result_from_url,
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


class TestDownloadTiming(unittest.TestCase):
    def test_http_body_is_timed_and_cache_hit_is_not(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.url_data_cache_"
        ) as cache, patch(
            "rtp_llm.multimodal.multimodal_util.request_get",
            return_value=_FakeResponse(),
        ), patch(
            "rtp_llm.multimodal.multimodal_util.time.monotonic",
            side_effect=[1.0, 1.025],
        ):
            cache.check_cache.return_value = None
            with collect_download_timing() as timing:
                loaded = get_bytes_io_from_url("https://example.com/timed")
            self.assertEqual(loaded.read(), b"payload")
            self.assertAlmostEqual(timing.elapsed_ms, 25.0)
            cache.check_cache.return_value = io.BytesIO(b"cached")
            with collect_download_timing() as cached_timing:
                self.assertEqual(
                    get_bytes_io_from_url("https://example.com/timed").read(),
                    b"cached",
                )
            self.assertEqual(cached_timing.elapsed_ms, 0)

    def test_local_data_url_and_json_loading_are_timed(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.url_data_cache_"
        ) as cache, patch(
            "rtp_llm.multimodal.multimodal_util.time.monotonic",
            side_effect=[1.0, 1.002, 2.0, 2.003, 3.0, 3.004],
        ):
            cache.check_cache.return_value = None
            with tempfile.NamedTemporaryFile() as stream:
                stream.write(b"local")
                stream.flush()
                with collect_download_timing() as timing:
                    self.assertEqual(
                        get_bytes_io_from_url(stream.name).read(), b"local"
                    )
                    self.assertEqual(
                        get_bytes_io_from_url(
                            "data:application/octet-stream;base64,eA=="
                        ).read(),
                        b"x",
                    )
                    self.assertEqual(
                        get_json_result_from_url("data:application/json;base64,e30="),
                        "{}",
                    )
            self.assertAlmostEqual(timing.elapsed_ms, 9.0)

    def test_nested_and_concurrent_collectors_are_isolated(self):
        import concurrent.futures

        def collect_one():
            with collect_download_timing() as timing:
                _record_download_time(9.0)
            return timing.elapsed_ms

        with patch(
            "rtp_llm.multimodal.multimodal_util.time.monotonic", return_value=10.0
        ):
            with collect_download_timing() as outer:
                _record_download_time(9.0)
                with collect_download_timing() as inner:
                    _record_download_time(8.0)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    values = list(pool.map(lambda _: collect_one(), range(2)))
                _record_download_time(7.0)
            self.assertEqual(outer.elapsed_ms, 4000.0)
            self.assertEqual(inner.elapsed_ms, 2000.0)
            self.assertEqual(values, [1000.0, 1000.0])

    def test_failed_load_preserves_error_and_records_time(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.url_data_cache_"
        ) as cache, patch(
            "rtp_llm.multimodal.multimodal_util.request_get",
            return_value=_FakeResponse(status_code=503),
        ), patch(
            "rtp_llm.multimodal.multimodal_util.time.monotonic",
            side_effect=[1.0, 1.010],
        ):
            cache.check_cache.return_value = None
            with collect_download_timing() as timing:
                with self.assertRaises(FtRuntimeException):
                    get_bytes_io_from_url("https://example.com/failed")
            self.assertAlmostEqual(timing.elapsed_ms, 10.0)


class TestMultiModalUtil(unittest.TestCase):
    def assert_mm_error(self, exception_type, message, callable_):
        with self.assertRaises(FtRuntimeException) as context:
            callable_()
        self.assertEqual(context.exception.exception_type, exception_type)
        self.assertEqual(context.exception.message, message)

    def test_format_mm_rpc_error(self):
        error = FtRuntimeException(ExceptionType.MM_WRONG_FORMAT_ERROR, "invalid image")
        self.assertEqual(
            format_mm_rpc_error(error), "[MM_WRONG_FORMAT_ERROR] invalid image"
        )

    def test_get_bytes(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            temp_path = tmp_file.name

            image = Image.new("RGB", (200, 200), "white")
            image.save(temp_path, format="PNG")

            self.assertTrue(
                Image.open(get_bytes_io_from_url(temp_path)).size == image.size
            )

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

    def test_http_allows_missing_content_length(self):
        response = _FakeResponse()
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            result = get_bytes_io_from_url(
                "https://example.com/no-content-length", max_file_size_kb=1
            )
        self.assertEqual(result.read(), b"payload")
        self.assertTrue(response.content_accessed)
        self.assertTrue(response.closed)

    def test_http_checks_streamed_body_without_content_length(self):
        response = _FakeResponse(content=b"x" * (2 * 1024))
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            self.assert_mm_error(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                MMErr.FILE_TOO_LARGE,
                lambda: get_bytes_io_from_url(
                    "https://example.com/chunked-too-large", max_file_size_kb=1
                ),
            )
        self.assertTrue(response.content_accessed)
        self.assertTrue(response.closed)

    def test_http_allows_invalid_content_length(self):
        response = _FakeResponse(
            headers={"Content-Length": "invalid"}, content=b"payload"
        )
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            result = get_bytes_io_from_url(
                "https://example.com/invalid-content-length", max_file_size_kb=1
            )
        self.assertEqual(result.read(), b"payload")
        self.assertTrue(response.content_accessed)
        self.assertTrue(response.closed)

    def test_http_non_positive_limit_disables_size_check(self):
        response = _FakeResponse(content=b"x" * (2 * 1024))
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            result = get_bytes_io_from_url(
                "https://example.com/no-size-limit", max_file_size_kb=0
            )
        self.assertEqual(result.read(), b"x" * (2 * 1024))
        self.assertTrue(response.content_accessed)
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

    def test_http_cache_returns_independent_streams(self):
        response = _FakeResponse(content=b"payload")
        url = "https://example.com/cache-isolation"
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get", return_value=response
        ):
            first = get_bytes_io_from_url(url, max_file_size_kb=1)
        first.read(1)
        second = get_bytes_io_from_url(url, max_file_size_kb=1)

        self.assertIsNot(first, second)
        self.assertEqual(second.read(), b"payload")
        self.assertEqual(first.tell(), 1)
        self.assertTrue(response.closed)

    def test_http_timeout(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get",
            side_effect=requests.Timeout,
        ):
            self.assert_mm_error(
                ExceptionType.MM_DOWNLOAD_FAILED,
                MMErr.DL_TIMEOUT,
                lambda: get_bytes_io_from_url("https://example.com/timeout"),
            )

    def test_http_invalid_url(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get",
            side_effect=requests.exceptions.InvalidURL,
        ):
            self.assert_mm_error(
                ExceptionType.MM_WRONG_FORMAT_ERROR,
                MMErr.URL_INVALID,
                lambda: get_bytes_io_from_url("https://example.com/invalid-url"),
            )

    def test_http_connection_error(self):
        with patch(
            "rtp_llm.multimodal.multimodal_util.request_get",
            side_effect=requests.ConnectionError,
        ):
            self.assert_mm_error(
                ExceptionType.MM_DOWNLOAD_FAILED,
                MMErr.DL_FAILED,
                lambda: get_bytes_io_from_url("https://example.com/image"),
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
