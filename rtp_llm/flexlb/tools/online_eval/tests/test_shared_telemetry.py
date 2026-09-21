import unittest
from unittest.mock import Mock, patch
from online_eval import telemetry


class SharedTelemetryTest(unittest.TestCase):
    def test_registered_failure_never_scrapes_exporter_again(self):
        owner = Mock()
        owner.read.side_effect = RuntimeError("target down")
        with patch.dict(telemetry._REGISTRY, {"http://mock/metrics": owner}), patch(
            "urllib.request.urlopen"
        ) as fetch:
            with self.assertRaisesRegex(RuntimeError, "target down"):
                telemetry.http_text("http://mock/metrics")
            fetch.assert_not_called()

    def test_tsdb_cursor_delegated_without_private_history(self):
        owner = Mock()
        owner.samples_since.return_value = [{"sequence": 42}]
        with patch.dict(telemetry._REGISTRY, {"http://mock/metrics": owner}):
            self.assertEqual(
                telemetry.shared_samples_since("http://mock/metrics", 41),
                [{"sequence": 42}],
            )
            owner.samples_since.assert_called_once_with(41)
