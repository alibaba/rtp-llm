import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rtp_llm.cpp.media_network import _media_network as native


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, location=None):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        if location:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.visits.append(self.path)
        if self.path == "/slow":
            self.server.slow_started.set()
            self.server.release.wait(2)
            try:
                self.reply(200, b"slow")
            except (OSError, BrokenPipeError):
                pass
        elif self.path == "/redirect":
            self.reply(302, b"", "/binary")
        elif self.path == "/short-redirect":
            self.send_response(302)
            self.send_header("Location", "/binary")
            self.send_header("Content-Length", "100")
            self.send_header("Connection", "close")
            self.end_headers()
        elif self.path == "/redirect-v6":
            self.reply(302, b"", "http://[::1]/image")
        elif self.path == "/policy":
            self.reply(302, b"", "http://denied.invalid/image")
        elif self.path == "/retry" and self.server.visits.count("/retry") == 1:
            self.reply(503, b"")
        else:
            self.reply(200, b"\x00\xffbinary")

    def do_PUT(self):
        self.reply(200, self.rfile.read(int(self.headers["Content-Length"])))

    do_POST = do_PUT


class MediaNetworkTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.visits = []
        self.server.slow_started = threading.Event()
        self.server.release = threading.Event()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        policy = native.Policy()
        policy.internal = True
        policy.set_networks(["192.0.2.0/24"])
        policy.denied = {"denied.invalid"}
        self.net = native.Network(2, 1, policy)
        self.addCleanup(self.net.close)

    def fetch(self, path, **options):
        req = native.Request()
        req.url = "http://127.0.0.1:%d%s" % (self.server.server_port, path)
        for name, value in options.items():
            if name == "body":
                req.set_body(value)
            else:
                setattr(req, name, value)
        job = self.net.submit(req)
        self.assertIs(self.net.next(), job)
        return job.result()

    def test_redirect_cache_limit_eviction(self):
        self.assertEqual(
            self.fetch("/redirect", cache_key="a")[:4],
            (200, "", b"\x00\xffbinary", False),
        )
        self.assertTrue(self.fetch("/redirect", cache_key="a")[3])
        self.assertEqual(
            self.fetch("/redirect", cache_key="a", limit=2)[1], "too_large"
        )
        self.assertEqual(self.server.visits, ["/redirect", "/binary"])
        self.assertEqual(self.fetch("/binary", cache_key="b")[1], "")
        self.assertFalse(self.fetch("/redirect", cache_key="a")[3])

    def test_policy_deadline_retry_and_size(self):
        self.assertEqual(self.fetch("/policy")[1], "blocked")
        self.assertEqual(
            self.fetch("/unused", url="http://192.0.2.1/test")[1], "blocked"
        )
        self.assertEqual(
            self.fetch(
                "/binary", url=f"http://localhost:{self.server.server_port}/binary"
            )[0],
            200,
        )
        self.assertEqual(self.fetch("/unused", budget_ms=0)[1], "timeout")
        self.assertEqual(self.fetch("/binary", limit=1)[1], "too_large")
        self.assertEqual(self.fetch("/retry", attempts=2)[4], 1)
        self.assertEqual(self.server.visits.count("/retry"), 2)

    def test_slow_job_allows_other_transfers_and_python_threads_to_progress(self):
        slow = native.Request()
        slow.url = f"http://127.0.0.1:{self.server.server_port}/slow"
        pending = self.net.submit(slow)
        try:
            self.assertTrue(self.server.slow_started.wait(1))
            self.assertEqual(self.fetch("/binary")[2], b"\x00\xffbinary")
            pending.cancel()
        finally:
            self.server.release.set()
        self.assertIs(self.net.next(), pending)
        self.assertEqual(pending.result()[1], "cancelled")

    def test_media_literal_policy_and_proxy(self):
        self.assertEqual(
            self.fetch("/unused", url="http://[::1]/image")[1], "invalid_url"
        )
        self.assertEqual(self.fetch("/redirect-v6")[1], "invalid_url")
        proxy = f"http://localhost:{self.server.server_port}"
        target = "http://127.0.0.1:1/binary"
        self.assertEqual(self.fetch("/unused", url=target, proxy=proxy)[0], 200)
        self.assertIn(target, self.server.visits)

    def test_ipv6_connection_for_origin_redirect_and_proxy(self):
        class IPv6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        try:
            server = IPv6Server(("::1", 0), Handler)
        except OSError as error:
            self.skipTest(f"IPv6 loopback unavailable: {error}")
        server.visits = []
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)

        origin = f"http://localhost:{server.server_port}"
        self.assertEqual(self.fetch("/unused", url=origin + "/redirect")[0], 200)
        target = "http://127.0.0.1:1/binary"
        self.assertEqual(self.fetch("/unused", url=target, proxy=origin)[0], 200)
        self.assertEqual(server.visits, ["/redirect", "/binary", target])

    def test_ipv4_wins_while_ipv6_connect_is_pending(self):
        port = self.server.server_port
        try:
            listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            self.addCleanup(listener.close)
            listener.bind(("::1", port))
        except OSError as error:
            self.skipTest(f"IPv6 loopback unavailable: {error}")
        # Fill the IPv6 accept queue so the next handshake stays pending.
        listener.listen(0)
        with socket.create_connection(("::1", port), timeout=1):
            result = self.fetch("/unused", url=f"http://localhost:{port}/binary")
        self.assertEqual(result[:2], (200, ""))
        self.assertEqual(self.server.visits, ["/binary"])

    def test_redirect_ignores_truncated_redirect_body(self):
        self.assertEqual(self.fetch("/short-redirect")[:2], (200, ""))
        self.assertEqual(self.server.visits, ["/short-redirect", "/binary"])

    def test_service_body(self):
        for method in ("PUT", "POST"):
            self.assertEqual(
                self.fetch("/service", method=method, media=False, body=b"\xff\x00")[
                    :3
                ],
                (200, "", b"\xff\x00"),
            )


if __name__ == "__main__":
    unittest.main()
