import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.utils.graph_replay_util import (
    graph_replay_count,
    require_graph_replay,
)

_COMPLETED = "[PyWrappedModel] CUDA graph forward completed\n"


class GraphReplayUtilTest(unittest.TestCase):
    def test_old_replay_does_not_validate_next_request(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "process.log"
            log.write_text(_COMPLETED)
            previous = graph_replay_count(str(log))
            with patch("rtp_llm.test.utils.graph_replay_util.time.monotonic", side_effect=[0, 6]):
                with self.assertRaisesRegex(AssertionError, "possible eager fallback"):
                    require_graph_replay(str(log), previous)

    def test_entering_graph_without_completion_does_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "process.log"
            log.write_text("[PyWrappedModel] using CUDA graph forward\n")
            with patch("rtp_llm.test.utils.graph_replay_util.time.monotonic", side_effect=[0, 6]):
                with self.assertRaises(AssertionError):
                    require_graph_replay(str(log), 0)

    def test_waits_for_new_request_completion_to_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "process.log"
            log.write_text(_COMPLETED)
            previous = graph_replay_count(str(log))
            with patch("rtp_llm.test.utils.graph_replay_util.time.sleep") as sleep:
                sleep.side_effect = lambda _: log.write_text(_COMPLETED * 2)
                require_graph_replay(str(log), previous)
            self.assertEqual(graph_replay_count(str(log)), previous + 1)


if __name__ == "__main__":
    unittest.main()
