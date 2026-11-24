"""Bounded exact-shape packed graphs, built only in a GPU-owner idle window.

Misses run eager and retain metadata, never request tensors or output closures.
An existing scheduler calls build_one() AFTER publishing request completion.
Each graph has independent storage; no padding, image reordering or pool sharing.
This is not permission to capture concurrently with uncoordinated CUDA producers.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


@dataclass
class _Entry:
    graph: object
    inputs: list
    outputs: list
    done: object
    charged_bytes: int
    forward_owner: object = None


class PackedVisionGraph:
    def __init__(
        self,
        max_batch_images=None,
        max_patches=None,
        *,
        max_entries=4,
        max_bytes=8 * 1024**3,
    ):
        # Positional arguments are admission ceilings, NOT exact/full shape
        # requirements. Mixed grids and tails have their own cache keys.
        if max_batch_images is not None and max_batch_images < 1:
            raise ValueError("graph image budget must be positive")
        if max_patches is not None and max_patches < 1:
            raise ValueError("graph patch budget must be positive")
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("graph cache budgets must be nonnegative")
        self.max_batch_images = max_batch_images
        self.max_patches = max_patches
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._pending = OrderedDict()
        self._rejected = OrderedDict()
        # Exact shapes can be numerous because request tails differ. Keep a
        # bounded reuse history so one-off tails cannot displace hot full
        # batches, without retaining request tensors or arbitrary metadata.
        self._observations = OrderedDict()
        self._observation_limit = max(16, max_entries * 8)
        self._generation = 0
        self._failed = False
        self.captures = self.replays = self.eager = 0
        self.evictions = self.deferred = self.budget_skips = 0
        self.capture_seconds = 0.0

    def stats(self):
        with self._lock:
            return {
                "captures": self.captures,
                "replays": self.replays,
                "eager": self.eager,
                "entries": len(self._entries),
                "pending": len(self._pending),
                "evictions": self.evictions,
                "deferred": self.deferred,
                "budget_skips": self.budget_skips,
                "charged_bytes": sum(e.charged_bytes for e in self._entries.values()),
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
                "capture_seconds": self.capture_seconds,
                "failed": self._failed,
                "cached_grids": [key[2] for key in self._entries],
                "observed_shapes": len(self._observations),
                "cached_observations": [
                    self._observations.get(key, 0) for key in self._entries
                ],
            }

    def effective_batch_images(self, grid):
        count = grid[0] * grid[1]
        if count <= 0:
            return 0
        limit = self.max_batch_images
        if self.max_patches is not None:
            limit = (
                min(limit, self.max_patches // count)
                if limit
                else self.max_patches // count
            )
        return limit

    def _eligible(self, patches, grids):
        return (
            bool(patches)
            and len(patches) == len(grids)
            and self.max_entries > 0
            and self.max_bytes > 0
            and (self.max_batch_images is None or len(patches) <= self.max_batch_images)
            and (
                self.max_patches is None
                or sum(h * w for h, w in grids) <= self.max_patches
            )
            and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled("cuda")
            and all(
                p.is_cuda and p.is_contiguous() and p.device == patches[0].device
                for p in patches
            )
        )

    def _signature(self, patches, grids, context):
        return (
            self._generation,
            context,
            tuple(tuple(grid) for grid in grids),
            tuple(
                (tuple(p.shape), tuple(p.stride()), p.dtype, p.device) for p in patches
            ),
        )

    def _trim_observations(self):
        protected = self._entries.keys() | self._pending.keys()
        while len(self._observations) > self._observation_limit:
            removable = next(
                (key for key in self._observations if key not in protected), None
            )
            if removable is None:
                break
            self._observations.pop(removable)

    def _observe(self, key):
        count = self._observations.pop(key, 0) + 1
        self._observations[key] = count
        self._trim_observations()

    def _priority(self, key):
        grids = key[2]
        return (
            self._observations.get(key, 0),
            len(grids),
            sum(height * width for height, width in grids),
        )

    def _queue_pending(self, key, estimate):
        if len(self._pending) < self.max_entries:
            self._pending[key] = estimate
            return
        order = {candidate: index for index, candidate in enumerate(self._pending)}
        coldest = min(
            self._pending,
            key=lambda candidate: (self._priority(candidate), order[candidate]),
        )
        if self._priority(key) > self._priority(coldest):
            self._pending.pop(coldest)
            self._pending[key] = estimate
        self.deferred += 1
        self._trim_observations()

    def _select_victims(self, key, charged_bytes):
        """Return cold entries safe to replace, or None when admission loses."""
        entry_overflow = max(0, len(self._entries) + 1 - self.max_entries)
        byte_overflow = max(
            0,
            sum(entry.charged_bytes for entry in self._entries.values())
            + charged_bytes
            - self.max_bytes,
        )
        if not entry_overflow and not byte_overflow:
            return []
        lru_order = {candidate: index for index, candidate in enumerate(self._entries)}
        candidates = sorted(
            self._entries,
            key=lambda candidate: (self._priority(candidate), lru_order[candidate]),
        )
        victims = []
        reclaimed = 0
        for candidate in candidates:
            # Equal-value shapes do not churn. A later observation makes the
            # requesting shape strictly hotter and eligible on its next miss.
            if self._priority(key) <= self._priority(candidate):
                return None
            victims.append(candidate)
            reclaimed += self._entries[candidate].charged_bytes
            if len(victims) >= entry_overflow and reclaimed >= byte_overflow:
                return victims
        return None

    def run(self, patches, grids, forward, consume, *, context=(), estimated_bytes=0):
        with cuda_graph_gate.operation(), self._lock:
            if self._failed:
                raise RuntimeError("packed graph cache stopped after capture failure")
            eligible = self._eligible(patches, grids)
            key = self._signature(patches, grids, context) if eligible else None
            if key is not None:
                self._observe(key)
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                stream = torch.cuda.current_stream(patches[0].device)
                stream.wait_event(entry.done)
                for dst, src in zip(entry.inputs, patches):
                    dst.copy_(src)
                entry.graph.replay()
                try:
                    # consume materializes independently owned request results.
                    result = consume(entry.outputs)
                finally:
                    entry.done.record(stream)
                self.replays += 1
                return result
            self.eager += 1
            result = consume(forward(patches, grids))
            if eligible and key not in self._pending and key not in self._rejected:
                estimate = max(
                    estimated_bytes, sum(p.numel() * p.element_size() for p in patches)
                )
                if estimate > self.max_bytes:
                    self._reject(key)
                else:
                    self._queue_pending(key, estimate)
            return result

    def _reject(self, key):
        self.budget_skips += 1
        self._rejected[key] = None
        self._rejected.move_to_end(key)
        while len(self._rejected) > max(1, self.max_entries):
            self._rejected.popitem(last=False)

    def _evict(self, key):
        entry = self._entries.pop(key)
        entry.done.synchronize()
        self.evictions += 1
        # Dropping both graph and tensors releases its private allocator pool.

    def clear(self):
        with cuda_graph_gate.operation(), self._lock:
            for entry in self._entries.values():
                entry.done.synchronize()
            self._entries.clear()
            self._pending.clear()
            self._rejected.clear()
            self._observations.clear()
            self._generation += 1

    @torch.inference_mode()
    def build_one(self, forward, *, context=(), prepare_forward=None):
        """Try one job on a standalone, currently idle GPU lane.

        try_capture does not enqueue a priority writer; busy foreground work
        wins. Once capture starts, later arrivals wait until it completes.
        """
        with self._lock:
            if not self._pending or self._failed:
                return False
        with cuda_graph_gate.try_capture() as acquired:
            if not acquired:
                with self._lock:
                    self.deferred += 1
                return False
            with self._lock:
                for stale in list(self._pending):
                    if stale[1] != context:
                        self._pending.pop(stale)
                self._trim_observations()
                if not self._pending or self._failed:
                    return False
                key = max(self._pending, key=self._priority)
                estimate = self._pending[key]
                victims = self._select_victims(key, estimate)
                if victims is None:
                    self._pending.pop(key)
                    self.deferred += 1
                    self._trim_observations()
                    return False
                device = key[3][0][3]
                free, total = torch.cuda.mem_get_info(device)
                # graph.__enter__ releases unused ordinary allocator segments.
                # Count only wholly inactive DEFAULT-pool segments, never the
                # unused space reserved inside another live graph's pool.
                reclaimable = sum(
                    segment["total_size"]
                    for segment in torch.cuda.memory_snapshot()
                    if segment["device"] == device.index
                    and segment.get("active_size", -1) == 0
                    and tuple(segment.get("segment_pool_id", ())) == (0, 0)
                )
                reserve = max(256 * 1024**2, total // 20)
                if free + reclaimable < estimate + reserve:
                    # Do not destroy an existing graph before a replacement is
                    # publishable. Drop this job so a smaller pending shape can
                    # run; a later eager miss can requeue it with more evidence.
                    self._pending.pop(key)
                    self.deferred += 1
                    self._trim_observations()
                    return False
                self._pending.pop(key)
                started = time.monotonic()
                try:
                    entry = self._capture(key, estimate, forward, prepare_forward)
                except Exception:
                    # Capture/allocator failures are not safe eager retries.
                    self._failed = True
                    self._pending.clear()
                    raise
                finally:
                    self.capture_seconds += time.monotonic() - started
                if entry.charged_bytes > self.max_bytes:
                    entry.done.synchronize()
                    self._reject(key)
                    return False
                victims = self._select_victims(key, entry.charged_bytes)
                if victims is None:
                    entry.done.synchronize()
                    self.deferred += 1
                    self._trim_observations()
                    return False
                for victim in victims:
                    self._evict(victim)
                self._entries[key] = entry
                self._observations.move_to_end(key)
                self._trim_observations()
                self.captures += 1
                return True

    def _capture(self, key, estimate, forward, prepare_forward):
        grids, specs = key[2], key[3]
        device = specs[0][3]
        with torch.cuda.device(device):
            # The gate only drains host scopes. Finish already submitted GPU
            # uploads/cache/forward work before entering global CUDA capture.
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                if prepare_forward is not None:
                    forward = prepare_forward(grids)
                inputs = [
                    torch.empty_strided(shape, stride, dtype=dtype, device=dev).zero_()
                    for shape, stride, dtype, dev in specs
                ]
                for _ in range(2):
                    warm_outputs = forward(inputs, grids)
                del warm_outputs
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = forward(inputs, grids)
            done = torch.cuda.Event()
            done.record(stream)
            done.synchronize()
            pool = graph.pool()
            pool_bytes = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment.get("segment_pool_id", ())) == tuple(pool)
                and segment["device"] == device.index
            )
            inputs_bytes = sum(p.numel() * p.element_size() for p in inputs)
            owned_bytes = getattr(forward, "graph_owned_bytes", 0)
            # A before/after reserve DELTA is not an upper bound: graph enter
            # can release a larger ordinary allocator cache. If pool IDs are
            # unavailable, charge the whole process allocator reserve instead.
            # Unsupported zero accounting never publishes an unbounded entry.
            accounted = (
                pool_bytes + inputs_bytes + owned_bytes
                if pool_bytes
                else torch.cuda.memory_reserved(device)
            )
            charged = max(estimate, accounted or (self.max_bytes + 1))
            return _Entry(graph, inputs, outputs, done, charged, forward)
