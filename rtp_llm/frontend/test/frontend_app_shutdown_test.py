import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rtp_llm.frontend.frontend_app import FrontendApp


class FrontendAppShutdownTest(unittest.TestCase):

    def test_backend_ready_wait_returns_when_uvicorn_requests_shutdown(self):
        app = object.__new__(FrontendApp)
        app.frontend_server = SimpleNamespace(is_embedding=False)
        app.server_config = SimpleNamespace(rank_id=0, frontend_server_id=2)
        app._uvicorn_server = SimpleNamespace(should_exit=False)

        async def request_and_trigger_shutdown(*args, **kwargs):
            app._uvicorn_server.should_exit = True
            return {"status": "error"}

        app.grpc_client = SimpleNamespace(
            post_request=AsyncMock(side_effect=request_and_trigger_shutdown)
        )

        asyncio.run(app._wait_backend_health_ready_impl())

        app.grpc_client.post_request.assert_awaited_once_with("health_check", {})


if __name__ == "__main__":
    unittest.main()
