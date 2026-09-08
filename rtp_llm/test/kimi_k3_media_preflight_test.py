"""CPU tests for K3 image loading, concurrency and lifecycle."""

import asyncio
import base64
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import patch

import torch
from PIL import Image

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal import multimodal_util
from rtp_llm.multimodal.multimodal_mixins.kimi_k3 import kimi_k3_image_processor
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    preflight_kimi_k3_images,
    preflight_kimi_k3_images_async,
    shutdown_kimi_k3_media_executor,
)
from rtp_llm.server.server_args.server_args import EnvArgumentParser
from rtp_llm.server.server_args.vit_group_args import init_vit_group_args


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (12, 8), "red").save(buffer, "PNG")
    return buffer.getvalue()


class KimiK3MediaPreflightConfigTest(unittest.TestCase):
    def setUp(self):
        shutdown_kimi_k3_media_executor()
        self.addCleanup(shutdown_kimi_k3_media_executor)
        self.config = VitConfig()
        self.raw = image_bytes()
        self.url = "data:image/png;base64," + base64.b64encode(self.raw).decode()
        self.cache = multimodal_util.MMDataCache(cache_size=10)
        self.cache_patch = patch.object(multimodal_util, "url_data_cache_", self.cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def _run(self, urls, async_mode):
        if async_mode:
            return asyncio.run(preflight_kimi_k3_images_async(urls, self.config))
        return preflight_kimi_k3_images(urls, self.config)

    def test_cli_values_bind_to_vit_config(self):
        parser = EnvArgumentParser()
        parser.set_root_config(self.config)
        init_vit_group_args(parser, self.config)
        parser.parse_args(
            [
                "--mm_image_max_file_size_kb",
                "32768",
                "--mm_preprocess_max_workers",
                "7",
            ]
        )
        self.assertEqual(self.config.mm_image_max_file_size_kb, 32768)
        self.assertEqual(self.config.mm_preprocess_max_workers, 7)
        self.assertIn("mm_preprocess_max_workers: 7", self.config.to_string())

    def test_empty_request_does_not_create_threads(self):
        with patch.object(kimi_k3_image_processor, "ThreadPoolExecutor") as executor:
            self.assertEqual(preflight_kimi_k3_images([], self.config), ([], []))
            self.assertEqual(
                asyncio.run(preflight_kimi_k3_images_async([], self.config)), ([], [])
            )
        executor.assert_not_called()

    def test_repeated_cached_images_preserve_bytes_and_order(self):
        for async_mode in (False, True):
            with self.subTest(async_mode=async_mode):
                tensors, sizes = self._run([self.url, self.url], async_mode)
                self.assertEqual(sizes, [(12, 8), (12, 8)])
                self.assertEqual([t.numpy().tobytes() for t in tensors], [self.raw] * 2)

    def test_no_aggregate_image_byte_limit(self):
        # Reuse one tensor to cover 130 MiB of logical input without duplicating storage.
        tensor = torch.empty(65 * 1024 * 1024, dtype=torch.uint8)
        with patch.object(
            kimi_k3_image_processor,
            "_preflight_kimi_k3_image",
            return_value=(tensor, (12, 8)),
        ):
            for async_mode in (False, True):
                with self.subTest(async_mode=async_mode):
                    tensors, sizes = self._run(["first", "second"], async_mode)
                    self.assertEqual(sum(t.numel() for t in tensors), 130 * 1024 * 1024)
                    self.assertEqual(sizes, [(12, 8), (12, 8)])

    def test_cached_single_file_limit_uses_current_configuration(self):
        # Cache a valid PNG padded past one KiB; Pillow accepts trailing data.
        self.cache.insert_cache("cached", BytesIO(self.raw + b"x" * 1024))
        self.config.mm_image_max_file_size_kb = 2
        self.assertEqual(
            preflight_kimi_k3_images(["cached"], self.config)[1], [(12, 8)]
        )
        self.config.mm_image_max_file_size_kb = 1
        with self.assertRaisesRegex(FtRuntimeException, "file size"):
            preflight_kimi_k3_images(["cached"], self.config)

    def test_concurrency_and_order_in_both_paths(self):
        for async_mode in (False, True):
            with self.subTest(async_mode=async_mode):
                self.config.mm_preprocess_max_workers = 2
                lock = threading.Lock()
                two_started = threading.Event()
                release = threading.Event()
                active = 0
                peak = 0
                started = []

                def load(url, config):
                    nonlocal active, peak
                    with lock:
                        started.append(url)
                        active += 1
                        peak = max(peak, active)
                        if len(started) == 2:
                            two_started.set()
                    try:
                        if not release.wait(3):
                            raise TimeoutError("test did not release workers")
                        return torch.tensor([int(url)], dtype=torch.uint8), (12, 8)
                    finally:
                        with lock:
                            active -= 1

                with patch.object(
                    kimi_k3_image_processor,
                    "_preflight_kimi_k3_image",
                    side_effect=load,
                ), ThreadPoolExecutor(1) as caller:
                    future = caller.submit(
                        self._run, [str(i) for i in range(5)], async_mode
                    )
                    try:
                        self.assertTrue(two_started.wait(3))
                        self.assertEqual(len(started), 2)
                    finally:
                        release.set()
                    tensors, _ = future.result(timeout=3)
                self.assertLessEqual(peak, 2)
                self.assertEqual([t.item() for t in tensors], list(range(5)))

    def test_one_worker_processes_multiple_batches(self):
        self.config.mm_preprocess_max_workers = 1
        thread_ids = set()

        def load(url, config):
            thread_ids.add(threading.get_ident())
            return torch.tensor([0], dtype=torch.uint8), (12, 8)

        with patch.object(
            kimi_k3_image_processor, "_preflight_kimi_k3_image", side_effect=load
        ):
            tensors, _ = preflight_kimi_k3_images(["a", "b", "c"], self.config)
        self.assertEqual(len(tensors), 3)
        self.assertEqual(len(thread_ids), 1)
        self.assertNotIn(threading.get_ident(), thread_ids)

    def test_shared_executor_and_shutdown(self):
        executor = kimi_k3_image_processor._get_kimi_k3_media_executor(2)
        self.assertIs(executor, kimi_k3_image_processor._get_kimi_k3_media_executor(2))
        self.assertEqual(executor.submit(lambda: 7).result(), 7)
        shutdown_kimi_k3_media_executor()
        with self.assertRaises(RuntimeError):
            executor.submit(lambda: 8)
        self.assertIsNot(
            executor, kimi_k3_image_processor._get_kimi_k3_media_executor(3)
        )

    def test_async_cancellation_stops_later_batches(self):
        started = threading.Event()
        release = threading.Event()
        calls = []
        self.config.mm_preprocess_max_workers = 1

        def load(url, config):
            calls.append(url)
            started.set()
            if not release.wait(3):
                raise TimeoutError("test did not release worker")
            return torch.tensor([0], dtype=torch.uint8), (12, 8)

        async def run():
            task = asyncio.create_task(
                preflight_kimi_k3_images_async(["a", "b"], self.config)
            )
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()

        with patch.object(
            kimi_k3_image_processor, "_preflight_kimi_k3_image", side_effect=load
        ):
            asyncio.run(run())
            shutdown_kimi_k3_media_executor()
        self.assertEqual(calls, ["a"])


if __name__ == "__main__":
    unittest.main()
