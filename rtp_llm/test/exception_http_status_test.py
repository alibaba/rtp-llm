import asyncio
import json
import unittest

from fastapi.exceptions import RequestValidationError

from rtp_llm.config.exceptions import (
    ExceptionType,
    FtRuntimeException,
    http_status_for,
)
from rtp_llm.frontend.frontend_app import request_validation_exception_handler_400


class HttpStatusForTest(unittest.TestCase):
    def test_fastapi_request_validation_maps_to_400(self):
        exc = RequestValidationError(
            [
                {
                    "type": "missing",
                    "loc": ("body", "messages"),
                    "msg": "Field required",
                    "input": {},
                }
            ]
        )

        response = asyncio.run(request_validation_exception_handler_400(None, exc))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            json.loads(response.body)["detail"][0]["loc"], ["body", "messages"]
        )

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
