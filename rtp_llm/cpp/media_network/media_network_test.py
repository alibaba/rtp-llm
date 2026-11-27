import gzip
import socket
import threading
import time
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
        elif self.path == "/policy":
            self.reply(302, b"", "http://denied.invalid/image")
        elif self.path == "/backoff":
            times, started, release = self.server.backoff
            if not times:
                started.set()
                release.wait(5)
            times.append(time.monotonic())
            self.reply(503 if len(times) < 3 else 200, b"image")
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
        self.server.gates = {
            path: (threading.Event(), threading.Event()) for path in ("/slow", "/hold")
        }
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        policy = native.Policy()
        policy.internal = True
        policy.set_networks(["192.0.2.0/24"])
        policy.denied = {"denied.invalid"}
        self.net = native.Network(2, 1, policy)
        self.addCleanup(self.net.close)

    def submit(self, path, **options):
        req = native.Request()
        req.url = "http://127.0.0.1:%d%s" % (self.server.server_port, path)
        for name, value in options.items():
            if name == "body":
                req.set_body(value)
            else:
                setattr(req, name, value)
        return self.net.submit(req)

    def fetch(self, path, **options):
        job = self.submit(path, **options)
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

    def test_cached_image_completes_while_all_network_workers_are_occupied(self):
        self.fetch("/binary", cache_key="warm")
        busy = []
        for path, (started, release) in self.server.gates.items():
            self.addCleanup(release.set)
            busy.append(self.submit(path))
            self.assertTrue(started.wait(3))

        result = self.fetch("/binary", cache_key="warm")
        self.assertEqual(result[:4], (200, "", b"\x00\xffbinary", True))
        self.assertEqual(
            self.fetch("/binary", cache_key="warm", limit=1)[1], "too_large"
        )
        self.assertEqual(
            self.fetch("/binary", cache_key="warm", budget_ms=0)[1], "timeout"
        )
        self.assertEqual(self.server.visits.count("/binary"), 1)
        for _, release in self.server.gates.values():
            release.set()
        self.assertCountEqual([self.net.next() for _ in busy], busy)

    def test_concurrent_cold_image_uses_one_transfer(self):
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        owner = self.submit("/slow", cache_key="image")
        self.assertTrue(started.wait(3))
        followers = [self.submit("/slow", cache_key="image") for _ in range(4)]
        # A different URL can use the other worker while the shared image is in flight.
        self.assertEqual(self.fetch("/binary")[0], 200)
        release.set()
        jobs = [owner, *followers]
        self.assertCountEqual([self.net.next() for _ in jobs], jobs)
        for job in jobs:
            self.assertEqual(job.result()[:4], (200, "", b"slow", False))
        self.assertFalse(owner.result()[8])
        self.assertTrue(all(job.result()[8] for job in followers))
        self.assertEqual(self.server.visits.count("/slow"), 1)
        self.assertTrue(self.fetch("/slow", cache_key="image")[3])

    def test_zero_capacity_bypasses_cache_and_inflight_reuse(self):
        self.net = native.Network(2, 0, native.Policy())
        self.addCleanup(self.net.close)
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        first = self.submit("/slow", cache_key="image")
        self.assertTrue(started.wait(3))
        second = self.submit("/slow", cache_key="image")
        release.set()
        self.assertCountEqual([self.net.next(), self.net.next()], [first, second])
        third = self.fetch("/slow", cache_key="image")
        for result in (first.result(), second.result(), third):
            self.assertEqual(result[:4], (200, "", b"slow", False))
            self.assertFalse(result[8])
        self.assertEqual(self.server.visits.count("/slow"), 3)

    def test_cancelling_first_caller_preserves_other_image_waiters(self):
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        first = self.submit("/slow", cache_key="image")
        self.assertTrue(started.wait(3))
        waiter = self.submit("/slow", cache_key="image")
        first.cancel()
        release.set()
        self.assertCountEqual([self.net.next(), self.net.next()], [first, waiter])
        self.assertEqual(first.result()[1], "cancelled")
        self.assertEqual(waiter.result()[:3], (200, "", b"slow"))
        self.assertEqual(self.server.visits.count("/slow"), 1)

    def test_expired_waiter_does_not_cancel_active_download(self):
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        first = self.submit("/slow", cache_key="image")
        self.assertTrue(started.wait(3))
        waiter = self.submit("/slow", cache_key="image", budget_ms=100)
        self.assertIs(self.net.next(), waiter)
        self.assertEqual(waiter.result()[1], "timeout")
        release.set()
        self.assertIs(self.net.next(), first)
        self.assertEqual(first.result()[:3], (200, "", b"slow"))

    def test_concurrent_limits_are_enforced_per_request(self):
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        small = self.submit("/slow", cache_key="image", limit=1)
        self.assertTrue(started.wait(3))
        large = self.submit("/slow", cache_key="image", limit=8)
        release.set()
        self.assertCountEqual([self.net.next(), self.net.next()], [small, large])
        self.assertEqual(small.result()[1], "too_large")
        self.assertEqual(large.result()[:3], (200, "", b"slow"))

    def test_compatible_waiters_still_merge_after_incompatible_request(self):
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        first = self.submit("/slow", cache_key="image", limit=4)
        self.assertTrue(started.wait(3))
        other = self.submit("/slow", cache_key="image", limit=8)
        waiter = self.submit("/slow", cache_key="image", limit=4)
        release.set()
        jobs = [first, other, waiter]
        self.assertCountEqual([self.net.next() for _ in jobs], jobs)
        for job in jobs:
            self.assertEqual(job.result()[:3], (200, "", b"slow"))
        self.assertLessEqual(self.server.visits.count("/slow"), 2)

    def test_short_owner_deadline_does_not_duplicate_surviving_downloads(self):
        started, release = self.server.gates["/slow"]
        self.addCleanup(release.set)
        owner = self.submit("/slow", cache_key="image", budget_ms=500)
        self.assertTrue(started.wait(3))
        waiters = [
            self.submit("/slow", cache_key="image", budget_ms=5000) for _ in range(4)
        ]
        self.assertIs(self.net.next(), owner)
        self.assertEqual(owner.result()[1], "timeout")
        release.set()
        self.assertCountEqual([self.net.next() for _ in waiters], waiters)
        for waiter in waiters:
            self.assertEqual(waiter.result()[:3], (200, "", b"slow"))
        # The first transfer can expire. All survivors can share the replacement.
        self.assertLessEqual(self.server.visits.count("/slow"), 2)

    def test_failed_image_can_be_fetched_again(self):
        self.assertEqual(self.fetch("/retry", cache_key="image", attempts=1)[0], 503)
        self.assertEqual(self.fetch("/retry", cache_key="image")[:2], (200, ""))
        self.assertTrue(self.fetch("/retry", cache_key="image")[3])
        self.assertEqual(self.server.visits.count("/retry"), 2)

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

    def start_retry_with_one_worker(self, **options):
        self.net = native.Network(1, 1, native.Policy())
        self.addCleanup(self.net.close)
        times, started, release = self.server.backoff = (
            [],
            threading.Event(),
            threading.Event(),
        )
        self.addCleanup(release.set)
        job = self.submit("/backoff", cache_key="image", attempts=3, **options)
        self.assertTrue(started.wait(3))
        return job, times, release

    def test_backoff_frees_only_worker_and_preserves_shared_retries(self):
        owner, times, release = self.start_retry_with_one_worker()
        follower = self.submit("/backoff", cache_key="image", attempts=3)
        fresh = self.submit("/binary")
        release.set()
        self.assertIs(self.net.next(), fresh)
        self.assertEqual(fresh.result()[:3], (200, "", b"\x00\xffbinary"))
        self.assertCountEqual([self.net.next(), self.net.next()], [owner, follower])
        self.assertEqual(owner.result()[:5], (200, "", b"image", False, 2))
        self.assertEqual(follower.result()[:3], (200, "", b"image"))
        self.assertTrue(follower.result()[8])
        # Deliberate HTTP backoff remains source time, separate from FIFO waiting.
        for job in (owner, follower):
            self.assertGreaterEqual(job.result()[6], 550)
            self.assertGreaterEqual(job.result()[7], 0)
        self.assertEqual(
            self.server.visits, ["/backoff", "/binary", "/backoff", "/backoff"]
        )
        self.assertGreaterEqual(times[1] - times[0], 0.19)
        self.assertGreaterEqual(times[2] - times[1], 0.39)

    def test_deadline_shorter_than_backoff_skips_next_attempt(self):
        result = self.fetch("/retry", attempts=3, budget_ms=100)
        self.assertEqual(result[1], "timeout")
        self.assertEqual(self.server.visits, ["/retry"])

    def test_retry_deadline_completes_while_only_worker_is_busy(self):
        # The retry is eligible after 200 ms, before this caller's deadline.
        # A second request keeps the sole worker occupied past both instants.
        retry, _, release_first = self.start_retry_with_one_worker(budget_ms=1000)
        busy = self.submit("/hold")
        started, release_busy = self.server.gates["/hold"]
        self.addCleanup(release_busy.set)
        release_first.set()
        self.assertTrue(started.wait(3))
        self.assertIs(self.net.next(), retry)
        self.assertEqual(retry.result()[1], "timeout")
        self.assertEqual(self.server.visits, ["/backoff", "/hold"])
        release_busy.set()
        self.assertIs(self.net.next(), busy)
        self.assertEqual(busy.result()[:3], (200, "", b"slow"))

    def test_close_drains_backoff_and_its_waiters(self):
        owner, _, release_retry = self.start_retry_with_one_worker()
        follower = self.submit("/backoff", cache_key="image", attempts=3)
        busy = self.submit("/hold")
        started, release_busy = self.server.gates["/hold"]
        self.addCleanup(release_busy.set)
        release_retry.set()
        self.assertTrue(started.wait(3))
        self.net.close()
        jobs = [owner, follower, busy]
        self.assertCountEqual([self.net.next() for _ in jobs], jobs)
        self.assertIsNone(self.net.next())
        self.assertEqual([job.result()[1] for job in jobs], ["cancelled"] * len(jobs))
        self.assertEqual(self.server.visits, ["/backoff", "/hold"])

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


class RangeDownloadTest(unittest.TestCase):
    body = b"0123456789"

    def setUp(self):
        visits = self.visits = []
        data = self.body
        self.barrier = threading.Barrier(2)
        barrier = self.barrier
        occupied = self.occupied = threading.Event()
        release = self.release = threading.Event()
        self.addCleanup(release.set)

        class Origin(Handler):
            def do_HEAD(self):
                visits.append(("HEAD", self.path, None))
                self.send_response(405 if self.path == "/get-only" else 200)
                if self.path != "/no-length":
                    self.send_header("Content-Length", str(len(data)))
                if self.path != "/no-etag":
                    self.send_header("ETag", '"v1"')
                if self.path == "/no-ranges":
                    self.send_header("Accept-Ranges", "none")
                self.end_headers()

            def do_GET(self):
                interval = self.headers.get("Range")
                visits.append(("GET", self.path, interval))
                if not interval or self.path == "/ignore":
                    self.reply(200, data)
                    return
                start, end = map(int, interval.removeprefix("bytes=").split("-"))
                if self.path == "/hold-ranges" and start < 8:
                    barrier.wait(timeout=3)
                    occupied.set()
                    release.wait(5)
                if self.path.endswith("/parallel") and start < 8:
                    barrier.wait(timeout=3)
                if self.headers.get("If-Match") != '"v1"':
                    self.reply(412, b"")
                    return
                part = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(part)))
                reported_end = end + (self.path == "/bad-range")
                self.send_header(
                    "Content-Range", f"bytes {start}-{reported_end}/{len(data)}"
                )
                self.send_header("ETag", '"v2"' if self.path == "/changed" else '"v1"')
                self.end_headers()
                self.wfile.write(part)

        self.origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
        self.base = f"http://127.0.0.1:{self.origin.server_port}"
        self.addCleanup(self.origin.server_close)
        threading.Thread(target=self.origin.serve_forever, daemon=True).start()
        self.addCleanup(self.origin.shutdown)

    def network(self, chunk_size):
        policy = native.Policy()
        policy.internal = True
        policy.range_chunk_bytes = chunk_size
        net = native.Network(2, 1, policy)
        self.addCleanup(net.close)
        return net

    def fetch(self, net, path, proxy=False, **options):
        req = native.Request()
        req.url = self.base + path
        if proxy:
            req.proxy = self.base
            req.url = "http://127.0.0.1:1" + path
        for name, value in options.items():
            setattr(req, name, value)
        job = net.submit(req)
        self.assertIs(net.next(), job)
        return job.result()

    def test_small_equal_and_disabled(self):
        for size in (0, len(self.body), len(self.body) + 1):
            with self.subTest(size=size):
                self.visits.clear()
                result = self.fetch(self.network(size), "/small")
                self.assertEqual(result[:3], (200, "", self.body))
                expected = [("HEAD", "/small", None)] if size else []
                self.assertEqual(self.visits, expected + [("GET", "/small", None)])

    def test_parallel_ranges_and_cache(self):
        for proxy in (False, True):
            with self.subTest(proxy=proxy):
                self.visits.clear()
                path = "/parallel"
                target = "http://127.0.0.1:1" + path if proxy else path
                net = self.network(4)
                self.assertEqual(
                    self.fetch(net, path, proxy, cache_key="image")[:4],
                    (200, "", self.body, False),
                )
                self.assertTrue(self.fetch(net, path, proxy, cache_key="image")[3])
                self.assertEqual(self.visits[0], ("HEAD", target, None))
                self.assertCountEqual(
                    self.visits[1:],
                    [("GET", target, f"bytes={r}") for r in ("0-3", "4-7", "8-9")],
                )

    def test_unsupported_metadata_and_invalid_ranges_fall_back(self):
        net = self.network(4)
        for path in (
            "/get-only",
            "/no-length",
            "/no-etag",
            "/no-ranges",
            "/ignore",
            "/bad-range",
            "/changed",
        ):
            with self.subTest(path=path):
                self.visits.clear()
                self.assertEqual(self.fetch(net, path)[:3], (200, "", self.body))
                self.assertEqual(self.visits[0], ("HEAD", path, None))
                self.assertEqual(self.visits.count(("GET", path, None)), 1)
                if path in ("/get-only", "/no-length", "/no-etag", "/no-ranges"):
                    self.assertEqual(len(self.visits), 2)
                else:
                    self.assertTrue(any(interval for _, _, interval in self.visits))

    def test_queued_request_keeps_its_deadline(self):
        net = self.network(4)
        req = native.Request()
        req.url = self.base + "/hold-ranges"
        busy = net.submit(req)
        try:
            self.assertTrue(self.occupied.wait(3))
            self.assertEqual(self.fetch(net, "/queued", budget_ms=100)[1], "timeout")
            self.assertFalse(any(path == "/queued" for _, path, _ in self.visits))
        finally:
            self.release.set()
        self.assertIs(net.next(), busy)
        self.assertEqual(busy.result()[:3], (200, "", self.body))

    def test_size_limit_before_body(self):
        result = self.fetch(self.network(4), "/limit", limit=len(self.body) - 1)
        self.assertEqual(result[1], "too_large")
        self.assertEqual(self.visits, [("HEAD", "/limit", None)])


if __name__ == "__main__":
    unittest.main()
