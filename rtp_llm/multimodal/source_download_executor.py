import asyncio
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor

from rtp_llm.metrics import GaugeMetrics
from rtp_llm.multimodal.vit_metrics import report_vit_preprocess_metric


class SourceDownloadExecutor:
    """Isolate source downloads from the loop's preprocessing/upload workers."""

    def __init__(self, max_workers: int = 0):
        if max_workers < 0:
            raise ValueError("vit_source_download_concurrency must be nonnegative")
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="mm-source-download"
            )
            if max_workers
            else None
        )
        self._closed = False

    def submit(self, func, /, *args, **kwargs):
        if self._closed or self._executor is None:
            raise RuntimeError("A dedicated source download pool is required")
        submitted_ns = time.monotonic_ns()

        def run():
            wait_ms = (time.monotonic_ns() - submitted_ns) / 1_000_000
            report_vit_preprocess_metric(
                GaugeMetrics.VIT_DOWNLOAD_QUEUE_WAIT_MS_METRIC, wait_ms
            )
            return func(*args, **kwargs)

        return self._executor.submit(contextvars.copy_context().run, run)

    async def run(self, func, /, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Source download executor is closed")
        if self._executor is None:
            return await asyncio.to_thread(func, *args, **kwargs)
        return await asyncio.wrap_future(self.submit(func, *args, **kwargs))

    def shutdown(self):
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
