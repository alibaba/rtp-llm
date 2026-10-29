"""Optional wheel import failures are diagnostics; process interrupts propagate."""

import builtins
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class FlashMLAProbeTest(unittest.TestCase):
    def _load(self, failure=None):
        original_import = builtins.__import__

        def import_wheel(name, *args, **kwargs):
            if name == "rtp_llm.models_py.utils.flash_mla_legacy":
                if failure is not None:
                    raise failure
                return SimpleNamespace(
                    flash_mla_with_kvcache=object(), get_mla_metadata=object()
                )
            return original_import(name, *args, **kwargs)

        path = Path(__file__).parent.parent / "_flash_mla_probe.py"
        spec = importlib.util.spec_from_file_location(
            "flash_mla_probe_under_test", path
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch("builtins.__import__", side_effect=import_wheel):
            spec.loader.exec_module(module)
        return module

    def test_success_has_no_unavailable_reason(self):
        probe = self._load()
        self.assertTrue(probe.flash_mla_available())
        self.assertEqual(probe.flash_mla_unavailable_reason(), "")

    def test_ordinary_wheel_failures_are_reported(self):
        for exception in (
            ImportError,
            OSError,
            RuntimeError,
            AssertionError,
            TypeError,
            NameError,
        ):
            with self.subTest(exception=exception), self.assertLogs(level="WARNING"):
                failure = exception("incompatible wheel")
                probe = self._load(failure)
                self.assertFalse(probe.flash_mla_available())
                self.assertIs(probe.FLASH_MLA_IMPORT_ERROR, failure)
                self.assertEqual(
                    probe.flash_mla_unavailable_reason(),
                    f"{exception.__name__}: incompatible wheel",
                )

    def test_process_control_exceptions_propagate(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception), self.assertRaises(exception):
                self._load(exception())


if __name__ == "__main__":
    unittest.main()
