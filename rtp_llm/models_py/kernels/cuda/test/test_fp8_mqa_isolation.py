"""CPU tests for backend isolation and the legacy-platform fallback."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "mqa_under_test", Path(__file__).resolve().parents[1] / "fp8_mqa.py"
)
mqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mqa)


def legacy():
    return SimpleNamespace(
        fp8_mqa_logits=Mock(return_value="prefill"),
        fp8_paged_mqa_logits=Mock(return_value="paged"),
        get_paged_mqa_logits_metadata=Mock(return_value="schedule"),
        get_num_sms=Mock(return_value=72),
        get_pdl=Mock(return_value=True),
    )


class IsolationTest(unittest.TestCase):
    def tearDown(self):
        mqa._backend.cache_clear()

    def test_isolated_library_is_available_without_global_fp8_apis(self):
        with patch.object(mqa, "_binary", return_value=Path("private.so")):
            with patch.dict("sys.modules", {"deep_gemm": SimpleNamespace()}):
                self.assertTrue(mqa.available())

    def test_platform_without_isolated_library_keeps_legacy_backend(self):
        old = legacy()
        with patch.object(mqa, "_binary", return_value=None):
            with patch.dict("sys.modules", {"deep_gemm": old}):
                self.assertTrue(mqa.available())
                self.assertIs(mqa._backend(), old)
                self.assertEqual(mqa.fp8_mqa_logits(1, clean_logits=True), "prefill")
                old.fp8_mqa_logits.assert_called_once_with(1, clean_logits=True)

    def test_new_global_version_does_not_hide_missing_isolated_library(self):
        with patch.object(mqa, "_binary", return_value=None):
            with patch.dict("sys.modules", {"deep_gemm": SimpleNamespace()}):
                self.assertFalse(mqa.available())
                with self.assertRaisesRegex(RuntimeError, "missing.*isolated"):
                    mqa._backend()

    def test_private_runtime_receives_global_budget_without_mutating_global(self):
        global_gemm = legacy()
        before = dict(vars(global_gemm))
        native = Mock()
        native.get_num_sms.return_value = 72
        with patch.object(mqa, "_backend", return_value=native):
            with patch.dict("sys.modules", {"deep_gemm": global_gemm}):
                self.assertEqual(mqa.get_num_sms(), 72)
                native.set_num_sms.assert_called_once_with(72)
                native.set_pdl.assert_called_once_with(True)
                self.assertEqual(vars(global_gemm), before)

    def test_first_load_requires_warmup_before_graph_capture(self):
        with patch.object(mqa, "_binary", return_value=Path("private.so")):
            with patch.object(mqa.torch.cuda, "is_initialized", return_value=True):
                with patch.object(
                    mqa.torch.cuda, "is_current_stream_capturing", return_value=True
                ):
                    with self.assertRaisesRegex(RuntimeError, "Warm up"):
                        mqa._backend()


if __name__ == "__main__":
    unittest.main()
