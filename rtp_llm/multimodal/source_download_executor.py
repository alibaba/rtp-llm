import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor


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

    async def run(self, func, /, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Source download executor is closed")
        if self._executor is None:
            return await asyncio.to_thread(func, *args, **kwargs)
        # Preserve request-scoped context just as asyncio.to_thread does.
        context = contextvars.copy_context()
        call = functools.partial(context.run, func, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(self._executor, call)

    def shutdown(self):
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
