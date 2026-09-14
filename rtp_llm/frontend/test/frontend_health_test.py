import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.start_server import check_server_health


class FrontendHealthTest(unittest.IsolatedAsyncioTestCase):
    def test_launcher_can_check_liveness_without_claiming_readiness(self):
        with patch("rtp_llm.start_server.requests.get") as get:
            get.return_value = SimpleNamespace(status_code=503, text='"ok"')
            self.assertFalse(check_server_health(18005))
            get.return_value = SimpleNamespace(status_code=200, text='"ok"')
            self.assertTrue(check_server_health(18005, "live"))
            get.assert_called_with("http://localhost:18005/live", timeout=60)
            get.return_value = SimpleNamespace(
                status_code=200, text='{"error":"unavailable"}'
            )
            self.assertFalse(check_server_health(18005))

    async def test_unready_backend_is_not_wrapped_in_http_200(self):
        frontend = object.__new__(FrontendApp)
        frontend.separated_frontend = False
        frontend.frontend_server = Mock()
        frontend.frontend_server._global_controller = SimpleNamespace(max_concurrency=2)
        app = frontend.create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            with patch(
                "rtp_llm.frontend.frontend_app.async_request_server",
                new_callable=AsyncMock,
            ) as backend:
                for result in (
                    {"error": "HTTP Error 503"},
                    {"error": "Connection failed"},
                    None,
                ):
                    backend.return_value = result
                    for path in (
                        "/health",
                        "/health_check",
                        "/GraphService/cm2_status",
                        "/",
                    ):
                        response = await (
                            client.post(path)
                            if path == "/health_check"
                            else client.get(path)
                        )
                        self.assertEqual(503, response.status_code)
                backend.side_effect = asyncio.TimeoutError
                self.assertEqual(503, (await client.get("/health")).status_code)
                backend.side_effect = None
                backend.return_value = "ok"
                self.assertEqual(200, (await client.get("/health")).status_code)
                self.assertEqual(200, (await client.get("/live")).status_code)
                backend.assert_awaited_with("get", unittest.mock.ANY, "live", {})
                backend.return_value = {"status": "home"}
                self.assertEqual(200, (await client.get("/")).status_code)


if __name__ == "__main__":
    unittest.main()
