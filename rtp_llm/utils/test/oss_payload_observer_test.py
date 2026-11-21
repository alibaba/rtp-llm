import builtins
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


OBSERVER_MODULE = "internal_source.rtp_llm.utils.media_download_metrics"


class OssPayloadObserverTest(unittest.TestCase):
    def setUp(self):
        self.object = Mock()
        self.object.read.return_value = b"\x00media\xff"
        self.bucket = Mock()
        self.bucket.get_object.return_value = self.object
        self.oss2 = types.ModuleType("oss2")
        self.oss2.Bucket = Mock(return_value=self.bucket)
        self.oss2.ProviderAuth = Mock()
        credentials = types.ModuleType("oss2.credentials")
        credentials.EnvironmentVariableCredentialsProvider = Mock()
        config = types.ModuleType("rtp_llm.config.py_config_modules")
        config.StaticConfig = types.SimpleNamespace(
            model_config=types.SimpleNamespace(oss_endpoint="https://configured.invalid")
        )
        self.observer = types.ModuleType(OBSERVER_MODULE)
        self.observer.observe_oss_payload = Mock()
        self.modules = patch.dict(
            sys.modules,
            {
                "oss2": self.oss2,
                "oss2.credentials": credentials,
                "rtp_llm.config.py_config_modules": config,
                OBSERVER_MODULE: self.observer,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        path = Path(__file__).resolve().parents[1] / "oss_util.py"
        spec = importlib.util.spec_from_file_location("oss_payload_subject", path)
        self.subject = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.subject)

    def test_success_preserves_payload_position_and_single_sdk_read(self):
        result = self.subject.get_bytes_io_from_oss_path("oss://bucket/media")
        self.assertEqual(result.tell(), 0)
        self.assertEqual(result.read(), b"\x00media\xff")
        self.assertEqual(result.getvalue(), b"\x00media\xff")
        self.object.read.assert_called_once_with()
        self.bucket.get_object.assert_called_once_with("media")
        self.oss2.Bucket.assert_called_once_with(
            self.oss2.ProviderAuth.return_value,
            "https://configured.invalid",
            "bucket",
            connect_timeout=100,
        )
        self.observer.observe_oss_payload.assert_called_once_with(7)

    def test_zero_length_success_is_observed_as_zero(self):
        self.object.read.return_value = b""
        self.assertEqual(
            self.subject.get_bytes_io_from_oss_path("oss://bucket/empty").read(), b""
        )
        self.observer.observe_oss_payload.assert_called_once_with(0)

    def test_get_failure_propagates_without_observation(self):
        failure = RuntimeError("get failed")
        self.bucket.get_object.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            self.subject.get_bytes_io_from_oss_path("oss://bucket/media")
        self.assertIs(raised.exception, failure)
        self.object.read.assert_not_called()
        self.observer.observe_oss_payload.assert_not_called()

    def test_read_failure_propagates_without_estimated_bytes(self):
        failure = OSError("partial read failed")
        self.object.read.side_effect = failure
        with self.assertRaises(OSError) as raised:
            self.subject.get_bytes_io_from_oss_path("oss://bucket/media")
        self.assertIs(raised.exception, failure)
        self.observer.observe_oss_payload.assert_not_called()

    def test_missing_optional_observer_preserves_standalone_download(self):
        original_import = builtins.__import__

        def without_observer(name, *args, **kwargs):
            if name == OBSERVER_MODULE:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_observer):
            result = self.subject.get_bytes_io_from_oss_path("oss://bucket/media")
        self.assertEqual(result.read(), b"\x00media\xff")
        self.observer.observe_oss_payload.assert_not_called()

    def test_failed_observer_does_not_change_download(self):
        self.observer.observe_oss_payload.side_effect = RuntimeError("collector unavailable")
        result = self.subject.get_bytes_io_from_oss_path("oss://bucket/media")
        self.assertEqual(result.read(), b"\x00media\xff")
        self.object.read.assert_called_once_with()

    def test_invalid_url_never_reaches_sdk_or_observer(self):
        with self.assertRaises(Exception):
            self.subject.get_bytes_io_from_oss_path("https://example.invalid/media")
        self.oss2.Bucket.assert_not_called()
        self.observer.observe_oss_payload.assert_not_called()

    def test_observer_process_control_exceptions_are_not_swallowed(self):
        for kind in (KeyboardInterrupt, SystemExit):
            with self.subTest(kind=kind):
                self.observer.observe_oss_payload.side_effect = kind()
                with self.assertRaises(kind):
                    self.subject.get_bytes_io_from_oss_path("oss://bucket/media")


if __name__ == "__main__":
    unittest.main()
