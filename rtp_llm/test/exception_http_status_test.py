import unittest

from rtp_llm.config.exceptions import (
    ExceptionType,
    FtRuntimeException,
    http_status_for,
)


class HttpStatusForTest(unittest.TestCase):
    def test_bad_request_category_maps_to_400(self):
        for exception_type in (
            ExceptionType.INVALID_PARAMS,
            ExceptionType.ERROR_GENERATE_CONFIG_FORMAT,
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
        ):
            e = FtRuntimeException(exception_type, "bad params")
            self.assertEqual(http_status_for(e), 400, exception_type)

    def test_non_bad_request_ft_exception_maps_to_500(self):
        for exception_type in (
            ExceptionType.MALLOC_ERROR,
            ExceptionType.DECODE_MALLOC_FAILED,
            ExceptionType.GENERATE_TIMEOUT,
        ):
            e = FtRuntimeException(exception_type, "engine failure")
            self.assertEqual(http_status_for(e), 500, exception_type)

    def test_plain_exception_maps_to_500(self):
        self.assertEqual(http_status_for(RuntimeError("boom")), 500)


if __name__ == "__main__":
    unittest.main()
