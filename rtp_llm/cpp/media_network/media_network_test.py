import gzip
import socket
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from urllib3.exceptions import DecodeError

from rtp_llm.cpp.media_network import _media_network as native


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, location=None, encoding=None):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        if isinstance(encoding, str):
            encoding = (encoding,)
        for value in encoding or ():
            self.send_header("Content-Encoding", value)
        if location:
            self.send_header("Location", location)
        self.end_headers()
        self.wfile.write(body)

    def do_CONNECT(self):
        self.server.tunnels.append(self.path)
        if getattr(self.server, "delay_connect", None) is not None:
            self.server.delay_connect.wait(5)
            self.close_connection = True
            return
        self.send_response(200)
        self.end_headers()
        self.close_connection = False

    def do_GET(self):
        self.server.visits.append(self.path)
        if self.path in getattr(self.server, "encoded", {}):
            encoding, body = self.server.encoded[self.path]
            self.reply(200, body, encoding=encoding)
        elif self.path in self.server.gates:
            started, release = self.server.gates[self.path]
            started.set()
            release.wait(10)
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
        elif self.path == "/redirect-host":
            self.reply(302, b"", "http://localhost/image")
        elif self.path == "/echo-host":
            self.reply(200, self.headers["Host"].encode())
        elif self.path == "/policy":
            self.reply(302, b"", "http://denied.invalid/image")
        elif self.path == "/busy":
            self.reply(503, b"")
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
        self.server.tunnels = []
        self.server.gates = {
            path: (threading.Event(), threading.Event()) for path in ("/slow", "/hold")
        }
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        policy = native.Policy()
        policy.ca = requests.certs.where()
        policy.internal = True
        policy.set_networks(["192.0.2.0/24"])
        policy.denied = {"denied.invalid"}
        self.net = native.Network(2, 1, policy)
        self.addCleanup(self.net.close)

    def fetch(self, path, network=None, **options):
        network = self.net if network is None else network
        req = native.Request()
        req.url = "http://127.0.0.1:%d%s" % (self.server.server_port, path)
        for name, value in options.items():
            if name == "body":
                req.set_body(value)
            else:
                setattr(req, name, value)
        job = network.submit(req)
        self.assertIs(network.next(), job)
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

    def test_hot_url_completes_while_all_download_workers_are_busy(self):
        self.assertFalse(self.fetch("/binary", cache_key="hot")[3])
        slow_jobs = []
        for path, (started, release) in self.server.gates.items():
            self.addCleanup(release.set)
            req = native.Request()
            req.url = f"http://127.0.0.1:{self.server.server_port}{path}"
            slow_jobs.append(self.net.submit(req))
            self.assertTrue(started.wait(3))

        hit = self.fetch("/binary", cache_key="hot", budget_ms=250)
        self.assertEqual(hit[:4], (200, "", b"\x00\xffbinary", True))
        self.assertEqual(self.server.visits.count("/binary"), 1)
        self.assertEqual(
            self.fetch("/binary", cache_key="hot", limit=1)[1], "too_large"
        )
        self.assertEqual(
            self.fetch("/binary", cache_key="hot", budget_ms=0)[1], "timeout"
        )
        for _, release in self.server.gates.values():
            release.set()
        self.assertCountEqual([self.net.next() for _ in slow_jobs], slow_jobs)

    def test_proxy_connect_timeout_uses_direct_with_remaining_attempts(self):
        self.server.delay_connect = threading.Event()
        self.addCleanup(self.server.delay_connect.set)
        proxy = f"http://127.0.0.1:{self.server.server_port}"
        for path, attempts, status, direct_visits in (
            ("/binary", 1, 200, 1),
            ("/busy", 3, 503, 2),
        ):
            with self.subTest(path=path):
                result = self.fetch(
                    path,
                    url=f"http://localhost:{self.server.server_port}{path}",
                    proxy=proxy,
                    attempts=attempts,
                    connect_ms=200,
                    budget_ms=3000,
                )
                self.assertEqual(result[:2], (status, ""))
                self.assertEqual(result[5], (1, 1, 1))
                self.assertEqual(self.server.visits.count(path), direct_visits)
        self.assertEqual(len(self.server.tunnels), 2)

    def test_proxy_timeout_at_total_deadline_does_not_start_direct(self):
        self.server.delay_connect = threading.Event()
        self.addCleanup(self.server.delay_connect.set)
        port = self.server.server_port
        result = self.fetch(
            "/binary",
            url=f"http://localhost:{port}/binary",
            proxy=f"http://127.0.0.1:{port}",
            attempts=3,
            connect_ms=2000,
            budget_ms=200,
        )
        self.assertEqual(result[1], "timeout")
        self.assertEqual(result[5], (1, 0, 0))
        self.assertEqual(self.server.visits, [])

    def test_direct_connect_timeouts_share_the_three_attempt_budget(self):
        # Accept TCP and hold the TLS handshake open to time out just the connect phase.
        class PendingTLS(Handler):
            def handle(self):
                self.server.connections.append(self.client_address)
                self.server.release.wait(5)

        origin = ThreadingHTTPServer(("127.0.0.1", 0), PendingTLS)
        origin.connections = []
        origin.release = threading.Event()
        threading.Thread(target=origin.serve_forever, daemon=True).start()
        self.addCleanup(origin.server_close)
        self.addCleanup(origin.shutdown)
        self.addCleanup(origin.release.set)
        result = self.fetch(
            "/binary",
            url=f"https://127.0.0.1:{origin.server_port}/binary",
            attempts=3,
            connect_ms=200,
            budget_ms=5000,
        )
        self.assertEqual(result[1], "timeout")
        self.assertEqual(result[5], (0, 0, 2))
        self.assertEqual(len(origin.connections), 3)

    def test_slow_image_response_does_not_fall_back_to_direct(self):
        port = self.server.server_port
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        result = self.fetch(
            "/slow",
            url=f"http://localhost:{port}/slow",
            proxy=f"http://127.0.0.1:{port}",
            idle_ms=200,
            budget_ms=3000,
            attempts=1,
        )
        self.assertTrue(started.is_set())
        self.assertEqual(result[1], "timeout")
        self.assertEqual(result[5], (1, 0, 0))
        self.assertEqual(self.server.visits.count("/slow"), 1)

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

    def test_resolved_addresses_checked_for_each_hop_and_proxy(self):
        policy = native.Policy()
        policy.internal = True
        # localhost supplies both families; reject the whole answer even if
        # the other address and the initial redirect origin are permitted.
        policy.set_networks(["::1/128"])
        network = native.Network(1, 0, policy)
        self.addCleanup(network.close)
        proxy = f"http://127.0.0.1:{self.server.server_port}"
        for via in ("", proxy):
            with self.subTest(proxy=bool(via)):
                self.assertEqual(
                    self.fetch("/", network, url="http://localhost/image", proxy=via)[
                        1
                    ],
                    "blocked",
                )
        self.assertEqual(self.fetch("/redirect-host", network)[1], "blocked")
        self.assertEqual(self.server.visits, ["/redirect-host"])
        self.assertEqual(self.server.tunnels, [])

    def test_proxy_uses_checked_address_and_preserves_origin_host(self):
        proxy = f"http://127.0.0.1:{self.server.server_port}"
        result = self.fetch("/", url="http://localhost:8731/echo-host", proxy=proxy)
        self.assertEqual(result[:3], (200, "", b"localhost:8731"))
        self.assertEqual(self.server.tunnels, ["127.0.0.1:8731"])

    def test_oss_exception_is_scoped_to_configured_endpoint_and_port(self):
        policy = native.Policy()
        policy.internal = policy.rewrite_oss = True
        policy.group_hosts = {"media.example.test": "localhost"}
        policy.set_networks(["127.0.0.0/8", "::1/128"])
        network = native.Network(1, 0, policy)
        self.addCleanup(network.close)
        proxy = f"http://127.0.0.1:{self.server.server_port}"
        self.assertEqual(
            self.fetch(
                "/", network, url="http://media.example.test/image", proxy=proxy
            )[0],
            200,
        )
        self.assertEqual(self.server.visits, ["http://localhost/image"])
        self.assertEqual(
            self.fetch(
                "/", network, url="http://media.example.test:81/image", proxy=proxy
            )[1],
            "blocked",
        )

    def test_slow_job_allows_other_transfers_and_python_threads_to_progress(self):
        slow = native.Request()
        slow.url = f"http://127.0.0.1:{self.server.server_port}/slow"
        pending = self.net.submit(slow)
        try:
            self.assertTrue(self.server.gates["/slow"][0].wait(2))
            self.assertEqual(self.fetch("/binary")[2], b"\x00\xffbinary")
            pending.cancel()
        finally:
            self.server.gates["/slow"][1].set()
        self.assertIs(self.net.next(), pending)
        self.assertEqual(pending.result()[1], "cancelled")

    def test_keepalive_reused_across_workers(self):
        peers = []

        class Origin(Handler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                peers.append(self.client_address)
                self.reply(200, self.path.encode())

        origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
        self.addCleanup(origin.server_close)
        threading.Thread(target=origin.serve_forever, daemon=True).start()
        self.addCleanup(origin.shutdown)

        def hold(path):
            started, release = self.server.gates[path]
            self.addCleanup(release.set)
            req = native.Request()
            req.url = f"http://127.0.0.1:{self.server.server_port}{path}"
            job = self.net.submit(req)
            self.assertTrue(started.wait(3))
            return job

        def fetch_image(path):
            url = f"http://127.0.0.1:{origin.server_port}{path}"
            self.assertEqual(
                self.fetch(path, url=url)[:4], (200, "", path.encode(), False)
            )

        # Occupy opposite workers so two different workers must fetch from origin.
        first = hold("/slow")
        fetch_image("/first")
        second = hold("/hold")
        self.server.gates["/slow"][1].set()
        self.assertIs(self.net.next(), first)
        fetch_image("/second")
        self.server.gates["/hold"][1].set()
        self.assertIs(self.net.next(), second)
        self.assertEqual(len(peers), 2)
        self.assertEqual(peers[0], peers[1])

    def test_close_cancels_active_transfers_and_drains_queue(self):
        jobs = []
        for path in ("/slow", "/hold", "/queued"):
            req = native.Request()
            req.url = f"http://127.0.0.1:{self.server.server_port}{path}"
            jobs.append(self.net.submit(req))
        for started, release in self.server.gates.values():
            self.addCleanup(release.set)
            self.assertTrue(started.wait(3))
        self.net.close()
        self.assertCountEqual([self.net.next() for _ in jobs], jobs)
        self.assertIsNone(self.net.next())
        self.assertEqual([job.result()[1] for job in jobs], ["cancelled"] * len(jobs))
        self.assertNotIn("/queued", self.server.visits)

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
        server.gates = {}
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

    def test_mislabeled_and_compressed_media(self):
        payload = b"image bytes" * 100
        # Independent wire fixtures for this payload, from brotli.compress and
        # zstandard.ZstdCompressor().compress. No Python codec needed at runtime.
        br = bytes.fromhex("1b4b04008c54b5bf06119bcb53657435486d155c76fbf102")
        zstd = bytes.fromhex("28b52ffd604c0395000058696d61676520627974657301003ef82f5e")
        valid = {
            "/utf-8": ("uTf-8", payload),
            "/utf8": ("utf8", payload),
            "/gbk": ("GBK", payload),
            "/unknown": ("unknown", payload),
            "/unknown-list": ("unknown, GBK", payload),
            "/gzip": ("gzip", gzip.compress(payload)),
            "/br": ("br", br),
            "/zstd": ("zstd", zstd),
            "/chained": ("br, gzip", gzip.compress(br)),
        }
        invalid = {
            "/broken-gzip": ("gzip", b"invalid gzip"),
            "/broken-br": ("br", b"\xff" * 16),
            "/broken-zstd": ("zstd", b"invalid zstd"),
            "/mixed": ("unknown, gzip", gzip.compress(payload)),
        }
        self.server.encoded = {**valid, **invalid}
        for path in valid:
            with self.subTest(path=path):
                result = self.fetch(path, attempts=3)
                self.assertEqual(result[:5], (200, "", payload, False, 0))
                self.assertEqual(self.server.visits.count(path), 1)
                limited = self.fetch(path, limit=len(payload) - 1)
                self.assertEqual(limited[1], "too_large")
        for path in invalid:
            with self.subTest(path=path):
                self.assertEqual(self.fetch(path)[1], "body_transport")

    def test_content_decoding_matches_legacy_read1(self):
        data = b"media bytes" * 9000
        gz = gzip.compress(data)
        df = zlib.compress(data)
        self.server.encoded = {
            "/gzip-members": (
                "x-gzip",
                gzip.compress(data[:100]) + gzip.compress(data[100:]),
            ),
            "/gzip-trailer": ("gzip", gz + b"extra"),
            "/gzip-truncated": ("gzip", gz[:-2]),
            "/not-a-gzip": ("gzip", df),
            "/deflate-extra": ("deflate", df + b"extra"),
            "/raw-deflate": ("deflate", df[2:-4]),
            "/deflate-first": ("deflate", df + df),
            "/unknown-inner": ("unknown, gzip", gzip.compress(df)),
            "/unknown-outer": ("gzip, unknown", zlib.compress(gz)),
            "/repeated-field": (("gzip", "unknown"), zlib.compress(gz)),
            "/identity-in-list": ("gzip, identity", zlib.compress(gz)),
            "/empty-in-list": ("gzip,", zlib.compress(gz)),
            "/single-whitespace": ("gzip ", gz),
            "/only-unknown": (("utf8", "GBK"), data),
            "/bad-layer": ("unknown, gzip", gz),
        }
        with requests.Session() as legacy:
            legacy.trust_env = False
            for path in self.server.encoded:
                with self.subTest(path=path):
                    url = f"http://127.0.0.1:{self.server.server_port}{path}"
                    with legacy.get(url, stream=True, timeout=5) as response:
                        try:
                            expected = b"".join(
                                iter(
                                    lambda: response.raw.read1(
                                        65536, decode_content=True
                                    ),
                                    b"",
                                )
                            )
                        except DecodeError:
                            expected = None
                    status, error, body = self.fetch(path)[:3]
                    if expected is None:
                        self.assertEqual(error, "body_transport")
                    else:
                        self.assertEqual((status, error, body), (200, "", expected))

    def test_truncated_zstd_and_decoded_size_limit(self):
        # A complete Zstandard frame containing b"image bytes" * 100.
        frame = bytes.fromhex(
            "28b52ffd604c0395000058696d61676520627974657301003ef82f5e"
        )
        self.server.encoded = {
            "/short-zstd": ("zstd", frame[:-1]),
            "/empty-zstd": ("zstd", b""),
            "/tiny-gzip": ("gzip", gzip.compress(b"x")),
        }
        for path in ("/short-zstd", "/empty-zstd"):
            self.assertEqual(self.fetch(path)[1], "body_transport")
        self.assertEqual(self.fetch("/tiny-gzip", limit=1)[:3], (200, "", b"x"))

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
