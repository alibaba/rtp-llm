import asyncio
import contextvars
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class BoundedEncodeExecutor:
    """Run prompt tokenization away from the frontend event loop.

    The executor is created lazily so a Pipeline constructed before the
    frontend process forks does not leave live threads in the parent. A global
    executor is deliberately not used: tests and embedded callers may create
    several Pipeline instances with independent lifetimes.
    """

    def __init__(self, max_workers: int = 4):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        self._lock = threading.Lock()

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._closed:
                raise RuntimeError("encode executor is closed")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="frontend-prompt-encode",
                )
            return self._executor

    async def run(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("encode executor is closed")
        # Preserve the request context exactly as asyncio.to_thread does.
        context = contextvars.copy_context()
        call = functools.partial(context.run, func, *args, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._get_executor(), call
        )

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)