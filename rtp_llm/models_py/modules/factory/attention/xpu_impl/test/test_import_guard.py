"""Focused unit tests for the XPU skip-vs-fail import guard.

Regression coverage for the P1 review finding: an `_IMPORT_OK = False`
(guarded package import failure) must FAIL the test on a real XPU test run
and only SKIP elsewhere. This deliberately does not import `rtp_llm.ops` or
any XPU-only module, so it runs (and is meaningful) on every platform,
including when the package under test is completely unbuildable.
"""

import os
from unittest import SkipTest, TestCase, main
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.attention.xpu_impl.test._import_guard import (
    is_xpu_test_env,
    skip_or_fail_on_missing_import,
)


class TestIsXpuTestEnv(TestCase):
    def test_no_env_is_not_xpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_xpu_test_env())

    def test_test_using_device_xpu(self):
        with patch.dict(os.environ, {"TEST_USING_DEVICE": "XPU"}, clear=True):
            self.assertTrue(is_xpu_test_env())

    def test_test_using_device_is_case_insensitive(self):
        with patch.dict(os.environ, {"TEST_USING_DEVICE": "xpu"}, clear=True):
            self.assertTrue(is_xpu_test_env())

    def test_test_using_device_cuda_is_not_xpu(self):
        with patch.dict(os.environ, {"TEST_USING_DEVICE": "CUDA"}, clear=True):
            self.assertFalse(is_xpu_test_env())

    def test_rtp_llm_device_type_xpu(self):
        with patch.dict(os.environ, {"RTP_LLM_DEVICE_TYPE": "xpu"}, clear=True):
            self.assertTrue(is_xpu_test_env())

    def test_rtp_llm_device_type_is_case_insensitive(self):
        with patch.dict(os.environ, {"RTP_LLM_DEVICE_TYPE": "XPU"}, clear=True):
            self.assertTrue(is_xpu_test_env())

    def test_rtp_llm_device_type_cuda_is_not_xpu(self):
        with patch.dict(os.environ, {"RTP_LLM_DEVICE_TYPE": "cuda"}, clear=True):
            self.assertFalse(is_xpu_test_env())


class TestSkipOrFailOnMissingImport(TestCase):
    def test_import_ok_is_a_noop_on_xpu(self):
        with patch.dict(os.environ, {"TEST_USING_DEVICE": "XPU"}, clear=True):
            skip_or_fail_on_missing_import(self, True, None)  # must not raise

    def test_import_ok_is_a_noop_off_xpu(self):
        with patch.dict(os.environ, {}, clear=True):
            skip_or_fail_on_missing_import(self, True, None)  # must not raise

    def test_import_failure_fails_on_xpu_via_test_using_device(self):
        with patch.dict(os.environ, {"TEST_USING_DEVICE": "XPU"}, clear=True):
            with self.assertRaises(AssertionError):
                skip_or_fail_on_missing_import(self, False, RuntimeError("boom"))

    def test_import_failure_fails_on_xpu_via_rtp_llm_device_type(self):
        with patch.dict(os.environ, {"RTP_LLM_DEVICE_TYPE": "xpu"}, clear=True):
            with self.assertRaises(AssertionError):
                skip_or_fail_on_missing_import(self, False, RuntimeError("boom"))

    def test_import_failure_skips_off_xpu(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SkipTest):
                skip_or_fail_on_missing_import(self, False, RuntimeError("boom"))

    def test_import_failure_skips_on_non_xpu_device(self):
        with patch.dict(os.environ, {"TEST_USING_DEVICE": "CUDA"}, clear=True):
            with self.assertRaises(SkipTest):
                skip_or_fail_on_missing_import(self, False, RuntimeError("boom"))


if __name__ == "__main__":
    main()
