import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from rtp_llm.frontend.frontend_app import FrontendApp


class FrontendPreBindTest(unittest.TestCase):
    def test_barrier_is_called_with_listener_metadata(self):
        app = FrontendApp.__new__(FrontendApp)
        app.server_config = SimpleNamespace(
            rank_id=2,
            frontend_server_id=9,
            server_port=18080,
        )
        app._app_prebind_barrier = Mock()
        app._app_prebind_barrier.prebind_ready.return_value = True

        app._wait_for_app_prebind()

        app._app_prebind_barrier.prebind_ready.assert_called_once_with(
            metadata={
                "rank_id": 2,
                "server_id": 9,
                "port": 18080,
                "listener": "frontend-http",
            }
        )

    def test_missing_barrier_keeps_legacy_path(self):
        app = FrontendApp.__new__(FrontendApp)
        app.server_config = SimpleNamespace()
        app._wait_for_app_prebind()


if __name__ == "__main__":
    unittest.main()
