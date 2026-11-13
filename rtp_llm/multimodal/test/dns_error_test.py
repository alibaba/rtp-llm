import multiprocessing
import socket
import unittest

from rtp_llm.config.exceptions import ExceptionType
from rtp_llm.multimodal.mm_error_messages import (
    MediaDownloadError,
    MMErr,
    dns_error_code,
    find_dns_error,
)


def _raise_dns_error():
    raise MediaDownloadError(
        ExceptionType.MM_WRONG_FORMAT_ERROR,
        MMErr.DNS_FAILED,
        "dns_error",
        None,
        "media.example",
        1.5,
        0,
    )


class DnsErrorTest(unittest.TestCase):
    def test_dns_failure_survives_spawn_worker(self):
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            result = pool.apply_async(_raise_dns_error)
            with self.assertRaises(MediaDownloadError) as raised:
                result.get(timeout=10)
        error = raised.exception
        self.assertEqual(error.exception_type, ExceptionType.MM_WRONG_FORMAT_ERROR)
        self.assertEqual(error.domain, "media.example")
        self.assertEqual(
            error.metric_tags(), {"failure_type": "dns_error", "http_status": "none"}
        )

    def test_only_permanent_lookup_failures_are_input_errors(self):
        for errno in (
            socket.EAI_NONAME,
            getattr(socket, "EAI_NODATA", socket.EAI_NONAME),
        ):
            with self.subTest(errno=errno):
                self.assertEqual(
                    dns_error_code(socket.gaierror(errno, "lookup failed")),
                    ExceptionType.MM_WRONG_FORMAT_ERROR,
                )
        for errno in (socket.EAI_AGAIN, socket.EAI_FAIL, None):
            with self.subTest(errno=errno):
                self.assertEqual(
                    dns_error_code(socket.gaierror(errno, "lookup failed")),
                    ExceptionType.MM_DOWNLOAD_FAILED,
                )

    def test_wrapped_error_and_cycle(self):
        dns_error = socket.gaierror(socket.EAI_NONAME, "lookup failed")
        wrapper = RuntimeError(RuntimeError(dns_error))
        wrapper.__cause__ = wrapper
        self.assertIs(find_dns_error(wrapper), dns_error)

    def test_suppressed_context_is_not_misclassified(self):
        error = RuntimeError("Name or service not known")
        error.__context__ = socket.gaierror(socket.EAI_NONAME, "lookup failed")
        error.__suppress_context__ = True
        self.assertIsNone(find_dns_error(error))
        error.__suppress_context__ = False
        self.assertIs(find_dns_error(error), error.__context__)


if __name__ == "__main__":
    unittest.main()
