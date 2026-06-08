import queue
import threading
from typing import Any, Iterator, Optional


_SENTINEL = object()


class StreamChannel:
    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def emit(self, chunk: Any) -> None:
        if self._closed:
            raise RuntimeError("Cannot emit to a closed StreamChannel")
        self._queue.put(chunk)

    def recv(self, timeout: Optional[float] = None) -> Optional[Any]:
        try:
            item = self._queue.get(timeout=timeout)
            if item is _SENTINEL:
                return None
            return item
        except queue.Empty:
            return None

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(_SENTINEL)

    def __iter__(self) -> Iterator[Any]:
        while True:
            item = self.recv(timeout=1.0)
            if item is None and self._closed:
                break
            if item is not None:
                yield item
