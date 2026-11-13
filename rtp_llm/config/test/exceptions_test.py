import multiprocessing
import pickle
import unittest

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.mm_error_messages import MediaDownloadError, MMErr


def _raise_download_error():
    raise FtRuntimeException(
        ExceptionType.MM_DOWNLOAD_FAILED, "Failed to download multimodal content"
    )


def _raise_dns_error():
    raise MediaDownloadError(
        ExceptionType.MM_WRONG_FORMAT_ERROR,
        MMErr.DNS_FAILED,
        "dns_error",
        None,
        "kb-prod.dsbj.com",
        1.5,
        0,
    )


class FtRuntimeExceptionTest(unittest.TestCase):
    def test_dns_error_survives_multiprocessing_pool(self):
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            result = pool.apply_async(_raise_dns_error)
            with self.assertRaises(MediaDownloadError) as raised:
                result.get(timeout=5)
        error = raised.exception
        self.assertEqual(error.exception_type, ExceptionType.MM_WRONG_FORMAT_ERROR)
        self.assertEqual(error.message, MMErr.DNS_FAILED)
        self.assertEqual(error.domain, "kb-prod.dsbj.com")
        self.assertEqual(
            error.metric_tags(), {"failure_type": "dns_error", "http_status": "none"}
        )

    def test_pickle_round_trip_preserves_error(self):
        error = FtRuntimeException(
            ExceptionType.MM_DOWNLOAD_FAILED,
            "Failed to download multimodal content",
        )

        restored = pickle.loads(pickle.dumps(error))

        self.assertIsInstance(restored, FtRuntimeException)
        self.assertEqual(restored.exception_type, ExceptionType.MM_DOWNLOAD_FAILED)
        self.assertEqual(restored.message, error.message)
        self.assertEqual(str(restored), error.message)

    def test_multiprocessing_pool_propagates_error(self):
        context = multiprocessing.get_context("spawn")
        with context.Pool(1) as pool:
            result = pool.apply_async(_raise_download_error)

            with self.assertRaises(FtRuntimeException) as raised:
                result.get(timeout=5)

        self.assertEqual(
            raised.exception.exception_type, ExceptionType.MM_DOWNLOAD_FAILED
        )
        self.assertEqual(
            raised.exception.message, "Failed to download multimodal content"
        )


if __name__ == "__main__":
    unittest.main()
