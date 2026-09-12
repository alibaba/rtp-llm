import threading
import time
from collections import OrderedDict


class MMTokenIdCache:
    """CPU token IDs with an access window and a separate image-count limit."""

    def __init__(self, max_items=10000, time_window_ms=30 * 60 * 1000):
        self.max_items = max_items
        self.time_window_s = time_window_ms / 1000.0
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def _evict_expired(self, now):
        while self._entries:
            timestamp, _ = next(iter(self._entries.values()))
            if now - timestamp < self.time_window_s:
                break
            self._entries.popitem(last=False)

    def get(self, key):
        with self._lock:
            now = time.monotonic()
            self._evict_expired(now)
            entry = self._entries.pop(key, None)
            if entry is None:
                return None
            self._entries[key] = (now, entry[1])
            return entry[1]

    def put(self, key, token_ids):
        if self.max_items <= 0 or self.time_window_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._evict_expired(now)
            self._entries.pop(key, None)
            self._entries[key] = (now, token_ids)
            while len(self._entries) > self.max_items:
                self._entries.popitem(last=False)
