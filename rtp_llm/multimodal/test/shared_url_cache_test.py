import multiprocessing
import os
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from unittest.mock import patch

from rtp_llm.multimodal.url_cache import _REQUEST, URLDataCache


def _fetch(handle, url, barrier, results, use_v41):
    from rtp_llm.config.py_config_modules import VitConfig
    from rtp_llm.multimodal import multimodal_util as util
    from rtp_llm.multimodal.mm_preprocess import _worker_initializer
    from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
        load_image_source,
    )

    config = VitConfig()
    config.url_cache_item_num = 2
    _worker_initializer(config, {}, None, url_cache_handle=handle)
    loader = load_image_source if use_v41 else util.get_bytes_io_from_url
    with patch.object(util, "REQUEST_GET", util._default_request_get):
        if barrier is not None:
            barrier.wait(60)
        results.put((os.getpid(), loader(url).read()))


def _abandon(handle, url, started):
    cache = URLDataCache(0)
    cache.attach(handle, 2)

    def download():
        started.set()
        os._exit(42)

    cache.get_or_load(url, "", download)


class SharedURLCacheTest(unittest.TestCase):
    def setUp(self):
        self.owner = URLDataCache(2)
        self.handle = self.owner.share(2)
        self.addCleanup(self.owner.close)
        self.ctx = multiprocessing.get_context("spawn")

    def test_two_cold_workers_fetch_independently_then_a_third_hits(self):
        requests = []
        mutex = threading.Lock()
        release_first = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with mutex:
                    requests.append(self.path)
                    first = len(requests) == 1
                if first and not release_first.wait(12):
                    self.send_error(504)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"slow" if first else b"fast")

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/image"
            results = self.ctx.Queue()
            workers = []

            def start(barrier, use_v41):
                process = self.ctx.Process(
                    target=_fetch, args=(self.handle, url, barrier, results, use_v41)
                )
                process.start()
                workers.append(process)

            try:
                barrier = self.ctx.Barrier(2)
                start(barrier, False)
                start(barrier, True)
                first_pid, data = results.get(timeout=60)
                self.assertEqual(data, b"fast")
                self.assertEqual(requests, ["/image", "/image"])
                release_first.set()
                slow_pid, data = results.get(timeout=5)
                self.assertEqual(data, b"slow")
                start(None, True)
                third_pid, data = results.get(timeout=60)
                self.assertEqual(data, b"fast")
                self.assertEqual(len({first_pid, slow_pid, third_pid}), 3)
                self.assertEqual(len(requests), 2)
            finally:
                release_first.set()
                for worker in workers:
                    worker.join(3)
                    if worker.is_alive():
                        worker.kill()
                        worker.join()
                results.close()
                server.shutdown()
                thread.join()

    def test_failed_owner_does_not_block_replacement(self):
        started = self.ctx.Event()
        worker = self.ctx.Process(target=_abandon, args=(self.handle, "image", started))
        worker.start()
        try:
            self.assertTrue(started.wait(15))
            worker.join(5)
            self.assertEqual(worker.exitcode, 42)
            stream, hit = self.owner.get_or_load("image", "", lambda: BytesIO(b"new"))
            self.assertEqual((stream.read(), hit), (b"new", False))
        finally:
            worker.kill()
            worker.join()

    def test_partial_worker_write_is_invisible(self):
        key = b"x" * 32
        with socket.socket(socket.AF_UNIX) as sock:
            sock.connect(self.handle)
            sock.sendall(_REQUEST.pack(key, 1000) + b"partial")
            self.assertIsNone(self.owner.check_cache(key))
        client = URLDataCache(2)
        client.attach(self.handle, 2)
        client.insert_cache(key, BytesIO(b"whole"))
        self.assertEqual(client.check_cache(key).read(), b"whole")

    def test_unavailable_owner_bypasses_cache(self):
        client = URLDataCache(2)
        client.attach(self.handle, 2)

        def fetch():
            data, hit = client.get_or_load("x", "", lambda: BytesIO(b"source"))
            self.assertEqual(data.read(), b"source")
            return hit

        self.assertFalse(fetch())
        self.assertTrue(fetch())
        with tempfile.TemporaryDirectory() as directory:
            address = os.path.join(directory, "unavailable")
            client.attach(address, 2)
            self.assertFalse(fetch())
            with socket.socket(socket.AF_UNIX) as stalled:
                stalled.bind(address)
                stalled.listen(4)
                begin = time.monotonic()
                with patch.object(
                    client, "_exchange", wraps=client._exchange
                ) as exchange:
                    self.assertFalse(fetch())
                    self.assertEqual(exchange.call_count, 1)
                self.assertLess(time.monotonic() - begin, 1)

    def test_cache_does_not_start_connection_threads_in_vit_process(self):
        client = URLDataCache(2)
        client.attach(self.handle, 2)
        with patch.object(
            threading.Thread,
            "start",
            side_effect=AssertionError(
                "shared cache started a connection thread in ViT"
            ),
        ):
            for _ in range(32):
                client.insert_cache(b"x" * 32, BytesIO(b"cached"))
                self.assertEqual(client.check_cache(b"x" * 32).read(), b"cached")

    def test_different_keys_load_in_parallel_and_share_lru_budget(self):
        barrier = threading.Barrier(2)

        def first_load():
            barrier.wait(3)
            return BytesIO(b"image")

        with ThreadPoolExecutor(2) as pool:
            jobs = [
                pool.submit(self.owner.get_or_load, key, "", first_load)
                for key in ("a", "b")
            ]
            self.assertEqual([job.result()[1] for job in jobs], [False, False])

        def fetch(key, headers=""):
            return self.owner.get_or_load(key, headers, lambda: BytesIO(b"image"))[1]

        self.assertTrue(fetch("a"))
        self.assertFalse(fetch("a", "other credentials"))
        self.assertTrue(fetch("a"))
        self.assertFalse(fetch("b"))
        self.owner.resize_cache(0)
        self.assertFalse(fetch("b"))
        self.assertFalse(fetch("b"))


if __name__ == "__main__":
    unittest.main()
