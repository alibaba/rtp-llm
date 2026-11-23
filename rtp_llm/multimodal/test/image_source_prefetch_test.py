import asyncio
import io
import os
import threading
import time
import unittest
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.metrics import AccMetrics, kmonitor
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    V41ImageProcessorConfig,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.source_download_executor import SourceDownloadExecutor
from rtp_llm.multimodal.url_cache import URLDataCache
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput


class CpuImages(MultiModalEmbeddingInterface):
    supports_inspection_sources = True
    preprocess_input = staticmethod(DeepSeekV41VisionEmbedding.preprocess_input)
    fetch_preprocess_source = staticmethod(
        DeepSeekV41VisionEmbedding.fetch_preprocess_source
    )

    def get_preprocess_params(self):
        return dict(processor_config=V41ImageProcessorConfig(vision_min_pixels=1764))


def image_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (42, 42), color).save(stream, format="PNG")
    return stream.getvalue()


def media(url):
    return MultimodalInput(
        url,
        1,
        torch.empty(0),
        MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
    )


class ImageSourcePrefetchTest(unittest.TestCase):

    def engine(self, local, *, preprocess_workers=1, download_workers=2):
        cache = URLDataCache(8)
        self.addCleanup(cache.close)
        for module in ("mm_process_engine", "mm_preprocess", "multimodal_util"):
            override = patch("rtp_llm.multimodal." + module + ".url_data_cache_", cache)
            override.start()
            self.addCleanup(override.stop)
        config = VitConfig()
        config.use_local_preprocess = local
        config.mm_preprocess_max_workers = preprocess_workers
        config.vit_source_download_concurrency = download_workers
        config.url_cache_item_num = 8
        config.mm_timeout_ms = 60000
        config.mm_cache_gpu_max_bytes = config.mm_cache_cpu_max_bytes = 0
        config.mm_hash_key_cache_max_bytes = 0
        model = SimpleNamespace(
            mm_model_config=SimpleNamespace(mm_position_ids_style=0),
            mm_related_params=SimpleNamespace(preprocess_batch_size=1),
        )
        engine = MMProcessEngine(
            CpuImages(), model, config, ProfilingDebugLoggingConfig(), device="cpu"
        )
        self.addCleanup(engine.stop)
        self.assertIs(
            engine._image_prefetch.executor, engine._inspection.source_download_executor
        )
        return engine

    def test_download_concurrency_defaults_to_preprocess_workers(self):
        for override, expected in ((0, 19), (2, 2)):
            with self.subTest(override=override), patch(
                "rtp_llm.multimodal.mm_inspection.SourceDownloadExecutor",
                wraps=SourceDownloadExecutor,
            ) as create_pool:
                self.engine(True, preprocess_workers=19, download_workers=override)
                create_pool.assert_called_once_with(expected)

    def exercise_downloads(self, local):
        engine = self.engine(local)
        slow = "https://example.invalid/slow"
        fast = "https://example.invalid/fast"
        sources = {slow: image_bytes("red"), fast: image_bytes("blue")}
        started, release = threading.Event(), threading.Event()
        calls = []

        def get(url, headers, timeout):
            calls.append((url, threading.current_thread().name))
            if url == slow:
                started.set()
                if not release.wait(40):
                    raise AssertionError("Fast image was blocked by slow download")
            return nullcontext(
                SimpleNamespace(
                    status_code=200, iter_content=lambda **kw: iter([sources[url]])
                )
            )

        items = []

        def fetch(urls):
            batch = engine._create_work_items(
                [media(url) for url in urls],
                capture_inspection_sources=True,
                deadline=time.monotonic() + 60,
            )
            items.extend(batch)
            return engine.preprocess_executor.iter_ready(batch)

        def check_result(item, url):
            source = item.prepared_sources[0]
            engine.preprocess_executor.get_result(item)
            self.assertEqual(item.inspection_sources, [sources[url]])
            expected = [255, 0, 0] if url == slow else [0, 0, 255]
            self.assertEqual(item.preprocess_result.rgb[0, 0].tolist(), expected)
            item.release_preprocess_sources()
            if local:
                self.assertTrue(source.closed)
            else:
                self.assertFalse(os.path.exists(source.path))

        with patch("rtp_llm.multimodal.multimodal_util.REQUEST_GET", get):
            try:
                ready = fetch([slow, fast])
                self.assertTrue(started.wait(5))
                first = next(ready)
                self.assertEqual(first.mm_inputs[0].url, fast)
                check_result(first, fast)
                # A second request for the same URL shares the same download pool
                # and can preprocess from the URL cache while the slow GET waits.
                another = fetch([fast])
                check_result(next(another), fast)
                release.set()
                check_result(next(ready), slow)
                self.assertCountEqual([url for url, thread in calls], [slow, fast])
                self.assertTrue(
                    all(thread.startswith("mm-source-download") for _, thread in calls)
                )
            finally:
                release.set()
                for item in items:
                    item.release_preprocess_sources()

    def test_reinspection_loads_the_cached_original_in_the_engine_pool(self):
        engine = self.engine(True)
        url = "https://example.invalid/cached"
        payload = image_bytes("blue")
        seen = []

        class Provider:
            async def reinspect_cached(self, request, inputs, *, load_source):
                return await load_source(
                    inputs[0].url, time.monotonic() + 10, threading.Event()
                )

        def get(url, headers, timeout):
            seen.append(threading.current_thread().name)
            return nullcontext(
                SimpleNamespace(
                    status_code=200, iter_content=lambda **kw: iter([payload])
                )
            )

        with patch.object(engine._inspection, "_greennet_provider", Provider()), patch(
            "rtp_llm.multimodal.multimodal_util.REQUEST_GET", get
        ), patch.object(kmonitor, "report") as report:
            for _ in range(2):
                original = asyncio.run(
                    engine._inspection._reinspect_cached(
                        None, [media(url)], time.monotonic() + 10
                    )
                )
                self.assertEqual(original, payload)
        self.assertEqual(
            [
                c.args[2]
                for c in report.call_args_list
                if c.args[0] == AccMetrics.VIT_URL_CACHE_LOOKUP_QPS_METRIC
            ],
            [{"result": "miss"}, {"result": "hit"}],
        )
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].startswith("mm-source-download"))

    def test_deadline_cancels_a_download_without_rebuilding_cpu_pool(self):
        engine = self.engine(False)
        entered, release = threading.Event(), threading.Event()

        def get(*args, **kwargs):
            entered.set()
            release.wait(10)
            return nullcontext(
                SimpleNamespace(status_code=200, iter_content=lambda **kw: iter([b"x"]))
            )

        with TemporaryDirectory() as directory, patch(
            "rtp_llm.multimodal.multimodal_util.REQUEST_GET", get
        ):
            engine._image_prefetch.directory = directory
            items = engine._create_work_items(
                [media("https://example.invalid/blocked")],
                deadline=time.monotonic() + 2,
            )
            try:
                self.assertTrue(entered.wait(1))
                with patch.object(
                    engine.preprocess_executor, "_on_wait_timeout"
                ) as rebuild:
                    with self.assertRaises(FtRuntimeException) as caught:
                        next(engine.preprocess_executor.iter_ready(items))
                    self.assertEqual(
                        caught.exception.exception_type,
                        ExceptionType.MM_DOWNLOAD_FAILED,
                    )
                    rebuild.assert_not_called()
            finally:
                items[0].release_preprocess_sources()
                release.set()
                engine._inspection.source_download_executor.shutdown()
            self.assertEqual(os.listdir(directory), [])

    def test_one_cpu_process_can_decode_while_another_image_downloads(self):
        self.exercise_downloads(False)

    def test_local_preprocessing_reuses_cached_source_in_memory(self):
        self.exercise_downloads(True)


if __name__ == "__main__":
    unittest.main()
