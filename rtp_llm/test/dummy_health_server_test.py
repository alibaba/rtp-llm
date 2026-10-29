import json
import os
import signal
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from rtp_llm.utils import dummy_health_server

REPO_ROOT = Path(__file__).resolve().parents[2]
START_SERVER_PATH = REPO_ROOT / "rtp_llm" / "start_server.py"


def _start_server_in_thread():
    stats = dummy_health_server._Stats()
    server = dummy_health_server._DummyHTTPServer(
        ("127.0.0.1", 0), dummy_health_server._DummyHandler, stats
    )
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return server, thread


class DummyHealthServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.thread = _start_server_in_thread()
        cls.base_url = "http://127.0.0.1:%d" % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _request(self, method, path, body=None, raw_body=None, headers=None):
        url = self.base_url + path
        if raw_body is not None:
            data = raw_body
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
        else:
            data = None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers=dict(headers or {}),
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            try:
                result = (error.code, dict(error.headers), error.read())
            finally:
                error.close()
            return result

    def test_health_get_and_post(self):
        for method in ("GET", "POST"):
            status, headers, body = self._request(method, "/health")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), {"status": "ok"})

    def test_metrics_prometheus_text(self):
        status, headers, body = self._request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn("text/plain", headers.get("Content-Type", ""))
        text = body.decode("utf-8")
        self.assertIn("rtp_llm_dummy_up 1", text)
        self.assertIn("rtp_llm_dummy_requests_total ", text)

    def test_root_info(self):
        status, headers, body = self._request("GET", "/")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["mode"], "dummy")
        self.assertEqual(payload["service"], "rtp-llm-dummy-health-server")

    def test_models(self):
        status, headers, body = self._request("GET", "/v1/models")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["object"], "list")
        self.assertTrue(payload["data"])

    def test_chat_completion_non_stream(self):
        request = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        status, headers, body = self._request(
            "POST", "/v1/chat/completions", body=request
        )
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["model"], "my-model")
        choice = payload["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIn("hello", choice["message"]["content"])

    def test_chat_completion_stream(self):
        request = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        status, headers, body = self._request(
            "POST", "/v1/chat/completions", body=request
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("Content-Type", ""))
        text = body.decode("utf-8")
        self.assertTrue(text.strip().endswith("data: [DONE]"))
        self.assertIn('"delta"', text)

    def test_invalid_json_returns_400(self):
        status, headers, body = self._request(
            "POST", "/v1/chat/completions", raw_body=b"{not json"
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_unknown_path_returns_404(self):
        status, headers, body = self._request("GET", "/no/such/path")
        self.assertEqual(status, 404)

    def test_chat_get_returns_405(self):
        status, headers, body = self._request("GET", "/v1/chat/completions")
        self.assertEqual(status, 405)


class DummyHealthServerLogicTest(unittest.TestCase):
    def test_is_enabled_values(self):
        cases = {
            "1": True,
            " true ": True,
            "TRUE": True,
            "0": False,
            "": False,
            "yes": False,
        }
        for value, expected in cases.items():
            self.assertEqual(
                dummy_health_server.is_enabled({"RTP_DUMMY_SERVE": value}), expected
            )
        self.assertFalse(dummy_health_server.is_enabled({}))

    def test_resolve_start_port(self):
        cases = {
            "": dummy_health_server.DEFAULT_START_PORT,
            "12345": 12345,
            " 80 ": 80,
        }
        for value, expected in cases.items():
            port, error = dummy_health_server._resolve_start_port({"START_PORT": value})
            self.assertIsNone(error)
            self.assertEqual(port, expected)
        for value in ("notaport", "0", "65536", "-1"):
            port, error = dummy_health_server._resolve_start_port({"START_PORT": value})
            self.assertIsNone(port)
            self.assertIsNotNone(error)

    def test_run_dummy_server_invalid_port_returns_two(self):
        with mock.patch.dict(os.environ, {"START_PORT": "notaport"}):
            self.assertEqual(dummy_health_server.run_dummy_server(), 2)

    def test_shutdown_stops_serve_forever(self):
        server, thread = _start_server_in_thread()
        try:
            self.assertTrue(thread.is_alive())
            dummy_health_server._shutdown_from_signal(server, signal.SIGTERM, None)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        finally:
            server.server_close()


class DummyHealthServerSourceTest(unittest.TestCase):
    def test_module_does_not_import_torch(self):
        source = Path(dummy_health_server.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)

    def test_start_server_hook_present(self):
        source = START_SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("RTP_DUMMY_SERVE", source)
        self.assertIn("run_dummy_server", source)
        # The hook must run before the engine imports.
        self.assertLess(source.index("RTP_DUMMY_SERVE"), source.index("import torch"))
        self.assertLess(
            source.index("RTP_DUMMY_SERVE"), source.index("import requests")
        )


if __name__ == "__main__":
    unittest.main()
