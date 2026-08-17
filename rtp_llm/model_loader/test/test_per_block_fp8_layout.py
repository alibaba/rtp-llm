import os
import unittest
from unittest import mock

from rtp_llm.utils.sm120_fp8_backend import (
    SM120_FP8_BACKEND_ENV,
    get_sm120_fp8_backend,
    resolve_sm120_fp8_backend,
)


class SM120Fp8BackendConfigTest(unittest.TestCase):
    def test_auto_is_default_and_preserves_deepgemm(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_sm120_fp8_backend(), "auto")
            self.assertEqual(resolve_sm120_fp8_backend(), "deepgemm")

    def test_explicit_backends(self):
        for backend in ("cutlass", "deepgemm"):
            with self.subTest(backend=backend), mock.patch.dict(
                os.environ, {SM120_FP8_BACKEND_ENV: backend}
            ):
                self.assertEqual(resolve_sm120_fp8_backend(), backend)

    def test_value_is_case_and_whitespace_insensitive(self):
        with mock.patch.dict(os.environ, {SM120_FP8_BACKEND_ENV: " CUTLASS "}):
            self.assertEqual(resolve_sm120_fp8_backend(), "cutlass")

    def test_invalid_backend_fails_early(self):
        with mock.patch.dict(os.environ, {SM120_FP8_BACKEND_ENV: "cublas"}):
            with self.assertRaisesRegex(ValueError, SM120_FP8_BACKEND_ENV):
                resolve_sm120_fp8_backend()


if __name__ == "__main__":
    unittest.main()
