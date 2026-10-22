"""Shared media producers with independently cancellable waiters."""

import concurrent.futures
import threading
import time


class DownloadSingleFlight:
    def __init__(self, executor):
        self._executor = executor
        self._lock = threading.Lock()
        self._flights = {}

    def submit(self, key, producer):
        """Return (future, owns_producer); cancellation belongs to the waiter."""
        with self._lock:
            if key in self._flights:
                return self._flights[key], False
            future = concurrent.futures.Future()
            self._flights[key] = future
            try:
                self._executor.submit(self._produce, key, future, producer)
            except BaseException:
                self._flights.pop(key, None)
                raise
            return future, True

    def download(
        self, key, producer, check_cache, insert_cache, *, deadline=None,
        cancellation_event=None,
    ):
        """Return shared bytes, retaining Native retry when no cache entry exists."""
        check_download_waiter(deadline, cancellation_event)
        content = check_cache(key)
        if content is not None:
            return content

        def produce_and_cache():
            content = check_cache(key)
            if content is None:
                content = producer()
                insert_cache(key, content)
            return content

        future, owns_producer = self.submit(key, produce_and_cache)
        wait_for_download(
            future, deadline=deadline, cancellation_event=cancellation_event
        )
        if owns_producer:
            content = future.result()
        else:
            content = check_cache(key)
            if content is None:
                # Original Native followers retry when caching is disabled or
                # their shared producer failed. A retry has its own waiter.
                fallback = self._executor.submit(producer)
                wait_for_download(
                    fallback, deadline=deadline, cancellation_event=cancellation_event
                )
                content = fallback.result()
        check_download_waiter(deadline, cancellation_event)
        return content

    def _produce(self, key, future, producer):
        try:
            future.set_result(producer())
        except BaseException as error:
            future.set_exception(error)
        finally:
            with self._lock:
                if self._flights.get(key) is future:
                    self._flights.pop(key)


def check_download_waiter(deadline=None, cancellation_event=None):
    if cancellation_event is not None and cancellation_event.is_set():
        raise concurrent.futures.CancelledError("Media download cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Media download deadline expired")


def wait_for_download(future, *, deadline=None, cancellation_event=None):
    """Wait for completion without cancelling or consuming the producer result.

    The caller consumes result/exception after this function. This keeps a
    producer TimeoutError distinct from expiration of an individual waiter.
    """
    while True:
        check_download_waiter(deadline, cancellation_event)
        if future.done():
            return
        timeout = 0.05 if cancellation_event is not None else None
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            timeout = remaining if timeout is None else min(timeout, remaining)
        concurrent.futures.wait([future], timeout=timeout)
