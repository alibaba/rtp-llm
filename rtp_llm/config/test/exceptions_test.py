import multiprocessing
import pickle
import unittest

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException


def _raise_download_error():
    raise FtRuntimeException(
        ExceptionType.MM_DOWNLOAD_FAILED, "Failed to download multimodal content"
    )


class FtRuntimeExceptionTest(unittest.TestCase):
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
