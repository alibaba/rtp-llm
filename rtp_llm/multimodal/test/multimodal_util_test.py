import base64
import io
import logging
import tempfile
import unittest
from unittest.mock import patch

import requests
from PIL import Image

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url


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
