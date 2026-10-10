import unittest
from unittest.mock import Mock, patch
from monitoring import telemetry


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
