import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import standalone_master


class StandaloneMasterTest(unittest.TestCase):
    def test_readiness_timeout_terminates_master(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"RTP_LLM_MOCK_BUNDLE": "1", "MODEL_SERVICE_CONFIG": "{}",
                   "MOCK_BUNDLE_RUN_DIR": directory}
            child = MagicMock()
            child.poll.return_value = None
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(standalone_master.signal, "signal"), \
                    patch.object(standalone_master.subprocess, "Popen", return_value=child) as start, \
                    patch.object(standalone_master.urllib.request, "urlopen", side_effect=OSError()), \
                    patch.object(standalone_master.time, "monotonic", side_effect=[0, 121]):
                with self.assertRaises(TimeoutError):
                    standalone_master.run()
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=10)
            self.assertNotIn("mock.jar", " ".join(start.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
