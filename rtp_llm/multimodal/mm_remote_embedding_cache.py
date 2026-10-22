"""Bounded shared KVCM tier for whole multimodal embedding results.

Only detached CPU eviction snapshots are uploaded. Local eviction never
removes a shared object: KVMeta V1 has no read lease for online reclamation.
"""

import concurrent.futures
import logging
import threading
import time
from collections import Counter

import torch

from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)


class MMRemoteEmbeddingCache:
    def __init__(
        self,
        client,
        *,
        max_object_bytes,
        max_inflight_bytes,
        max_pending=8,
        read_timeout_ms=200,
        cuda_device=None
    ):
        if min(max_object_bytes, max_inflight_bytes, max_pending, read_timeout_ms) <= 0:
            raise ValueError("remote embedding cache limits must be positive")
        self.client = client
        self.max_object_bytes = max_object_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.max_pending = max_pending
        self.read_timeout_ms = read_timeout_ms
        self.cuda_device = cuda_device
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mm-remote-cache"
        )
        self._lock = threading.Lock()
        self._closed = False
        self._bytes = self._tasks = 0
        self._stats = Counter()
        self._pending_writes = set()

    def _count(self, name):
        with self._lock:
            self._stats[name] += 1
            count = self._stats[name]
        if name.endswith("error") and count & (count - 1) == 0:
            logging.warning(
                "ViT remote cache %s count=%d; falling back to local computation/cache",
                name,
                count,
            )

    def _reserve(self, nbytes):
        with self._lock:
            if (
                self._closed
                or self._tasks >= self.max_pending
                or self._bytes + nbytes > self.max_inflight_bytes
            ):
                self._stats["admission_skip"] += 1
                return False
            self._tasks += 1
            self._bytes += nbytes
            return True

    def _release(self, nbytes):
        with self._lock:
            self._tasks -= 1
            self._bytes -= nbytes

    def submit_eviction(self, key, entry):
        """Called after CPU storage detachment; never copies or does network I/O."""
        if entry.pool_owners or entry.result is None or entry.error is not None:
            return
        try:
            plan = plan_object(
                entry.result,
                devices=entry.original_devices,
                max_bytes=self.max_object_bytes,
            )
            if any(t.device.type != "cpu" for t in plan.tensors):
                return
        except (ValueError, TypeError, RecursionError):
            self._count("encode_skip")
            return
        # Snapshot + pack buffer + worst-case contiguous temporary. Account
        # conservatively before retaining the source past this callback.
        cost = 3 * plan.nbytes
        with self._lock:
            if key in self._pending_writes:
                return
            self._pending_writes.add(key)
        if not self._reserve(cost):
            with self._lock:
                self._pending_writes.discard(key)
            return

        def write():
            try:
                self.client.save_one(key, pack_object(plan))
                self._count("write_success")
            except Exception:
                # The same key may be owned by another writer. Never remove
                # or automatically repeat an ambiguous mutation.
                self._count("write_error")

        def done(_):
            with self._lock:
                self._pending_writes.discard(key)
            self._release(cost)

        try:
            future = self._executor.submit(write)
        except RuntimeError:
            done(None)
            return
        future.add_done_callback(done)

    def load(self, key, *, timeout_ms=None):
        timeout_ms = min(self.read_timeout_ms, timeout_ms or self.read_timeout_ms)
        deadline = time.monotonic() + timeout_ms / 1000
        # Admission also bounds metadata probes. Grow the reservation only
        # after Get has supplied a validated exact size, before allocation.
        if not self._reserve(0):
            return None
        cost = 0
        future = None
        try:
            size = self.client.object_size(key, timeout_ms=timeout_ms)
            if size is None:
                self._count("miss")
                return None
            if not 0 < size <= self.max_object_bytes:
                raise ValueError("remote embedding object exceeds byte limit")
            with self._lock:
                if self._bytes + 3 * size > self.max_inflight_bytes:
                    self._stats["admission_skip"] += 1
                    return None
                cost = 3 * size
                self._bytes += cost
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._count("read_timeout")
                return None

            def read():
                buffer = torch.empty(size, dtype=torch.uint8)
                self.client.load_one(key, buffer)
                return buffer

            future = self._executor.submit(read)
            buffer = future.result(timeout=remaining)
            value = unpack_object(
                buffer, max_bytes=self.max_object_bytes, cuda_device=self.cuda_device
            )
            if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
                raise ValueError("remote object is not a multimodal result")
            self._count("hit")
            return value
        except concurrent.futures.TimeoutError:
            self._count("read_timeout")
            if future is not None:
                future.cancel()
            return None
        except Exception:
            self._count("read_error")
            return None
        finally:
            if future is None or future.done():
                self._release(cost)
            else:
                # Keep accounting and destination alive until the actual SDK
                # call returns, even after the requesting thread has timed out.
                future.add_done_callback(lambda _: self._release(cost))

    def stats(self):
        with self._lock:
            return dict(
                self._stats, inflight_bytes=self._bytes, inflight_tasks=self._tasks
            )

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.client.close()
