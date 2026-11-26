"""A bounded HF weight transaction, used only while the native engine is paused."""

from __future__ import annotations

import gc
import json
import logging
import queue
import threading
import time
from concurrent.futures import Future

import torch


class RlWeightUpdate:
    """Copy a complete stream of mutable HF weights in-place while paused.

    The engine owns admission, quiescence, cache invalidation, and version
    publication. An error here is fatal to that engine incarnation: earlier
    targets may already have been overwritten. A successful push acknowledges
    that the receiver stream has finished reading/copying that source tensor.
    """

    def __init__(
        self,
        loader,
        weights,
        device,
        source_shapes,
        max_buffer_bytes,
        *,
        borrow_singleton_sources=True,
        source_dtypes=None,
        frozen_qsa_source_names=(),
    ):
        self.loader = loader
        self.weights = weights
        self.device = torch.device(device)
        self.source_shapes = dict(source_shapes)
        self.source_dtypes = None if source_dtypes is None else dict(source_dtypes)
        if self.source_dtypes is not None and (
            set(self.source_dtypes) != set(self.source_shapes)
            or any(
                value not in (torch.bfloat16, torch.float32)
                for value in self.source_dtypes.values()
            )
        ):
            raise ValueError("HF source dtype contract must cover every source exactly")
        self.max_buffer_bytes = max_buffer_bytes
        self.borrow_singleton_sources = bool(borrow_singleton_sources)
        self.frozen_qsa_source_names = frozenset(frozen_qsa_source_names)

        all_targets = self._live_targets()
        frozen_table_keys, frozen_meta_keys = loader.get_rl_static_target_keys(weights)
        self.frozen_qsa_target_keys = loader.get_rl_frozen_qsa_target_keys(
            self.frozen_qsa_source_names
        )
        self.static_table_keys = set(frozen_table_keys)
        self.static_meta_keys = set(frozen_meta_keys)
        self.static_keys = (
            self.static_table_keys | self.static_meta_keys | self.frozen_qsa_target_keys
        )
        self.targets = {
            key: value for key, value in all_targets.items() if key not in self.static_keys
        }

        # Validate target eligibility before the first write.
        self._merged_hc_checks = []
        for key, value in self.targets.items():
            if getattr(value, "_rtp_requires_restart_for_update", False):
                if not getattr(value, "_rtp_allow_paused_inplace_update", False):
                    raise PermissionError(
                        f"target {key!r} requires a service restart and cannot be updated in-place"
                    )
                parent = getattr(value, "_rtp_merged_parent", None)
                tail_start = getattr(value, "_rtp_merged_tail_start", None)
                if parent is None or tail_start is None:
                    raise RuntimeError(
                        f"paused-inplace target {key!r} is missing merged parent metadata"
                    )
                if (
                    value.untyped_storage().data_ptr()
                    != parent.untyped_storage().data_ptr()
                ):
                    raise RuntimeError(
                        f"paused-inplace target {key!r} is detached from its merged parent storage"
                    )
                self._merged_hc_checks.append(
                    (
                        key,
                        parent,
                        int(tail_start),
                        value.untyped_storage().data_ptr(),
                        value.storage_offset(),
                        tuple(value.shape),
                        tuple(value.stride()),
                    )
                )

        self.addresses = {key: value.data_ptr() for key, value in all_targets.items()}
        self._static_meta_snapshots = {
            key: all_targets[key].detach().clone() for key in self.static_meta_keys
        }
        self._frozen_qsa_snapshots = {
            key: all_targets[key].detach().clone()
            for key in self.frozen_qsa_target_keys
        }
        if self.device.type == "cuda" and (
            self._static_meta_snapshots or self._frozen_qsa_snapshots
        ):
            torch.cuda.current_stream(self.device).synchronize()

        self.received = set()
        self.applied = set()
        self._queue = queue.Queue(maxsize=1)
        self._done = Future()
        self._current = None
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="rl-weight-update", daemon=True
        )
        self._thread.start()

    def _live_targets(self):
        targets = {
            (None, key): value for key, value in self.weights.global_weights.items()
        }
        targets.update(
            {
                (layer, key): value
                for layer, group in enumerate(self.weights.weights)
                for key, value in group.items()
            }
        )
        return targets

    def _allocator_memory(self):
        stats = torch.cuda.memory_stats(self.device)
        device_free, device_total = torch.cuda.mem_get_info(self.device)
        return {
            "torch_allocated_bytes": stats["allocated_bytes.all.current"],
            "torch_reserved_bytes": stats["reserved_bytes.all.current"],
            "torch_active_bytes": stats["active_bytes.all.current"],
            "torch_inactive_split_bytes": stats["inactive_split_bytes.all.current"],
            "torch_peak_allocated_bytes": stats["allocated_bytes.all.peak"],
            "torch_peak_reserved_bytes": stats["reserved_bytes.all.peak"],
            "device_free_bytes": device_free,
            "device_total_bytes": device_total,
        }

    def _inputs(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            name, tensor, ack = item
            self._current = ack
            if name not in self.source_shapes:
                raise ValueError(f"unknown HF parameter: {name}")
            if name in self.received:
                raise ValueError(f"duplicate HF parameter: {name}")
            if tuple(tensor.shape) != tuple(self.source_shapes[name]):
                raise ValueError(
                    f"HF shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(self.source_shapes[name])}"
                )
            if (
                tensor.dtype not in (torch.bfloat16, torch.float32)
                or tensor.device != self.device
            ):
                raise ValueError(
                    f"HF parameter must be BF16 or FP32 on {self.device}: {name}"
                )
            if self.source_dtypes is not None and tensor.dtype != self.source_dtypes[name]:
                raise ValueError(
                    f"HF dtype mismatch for {name}: {tensor.dtype} != {self.source_dtypes[name]}"
                )
            yield name, tensor
            # Single post-read/post-copy stream fence before producer ACK so IPC
            # storage can safely be released or overwritten by the producer.
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            del tensor, item
            self.received.add(name)
            ack.set_result(None)
            self._current = None

    def _run(self):
        try:
            with torch.inference_mode():
                is_cuda = self.device.type == "cuda"
                if is_cuda:
                    torch.cuda.set_device(self.device)
                    before_refit = self._allocator_memory()
                for layer, name, value in self.loader.prepare_weights_from_tensors(
                    self._inputs(),
                    str(self.device),
                    max_buffer_bytes=self.max_buffer_bytes,
                    borrow_singleton_sources=self.borrow_singleton_sources,
                    frozen_qsa_source_names=self.frozen_qsa_source_names,
                ):
                    key = (layer, name)
                    if key in self.applied or key not in self.targets:
                        raise ValueError(
                            f"unexpected or repeated internal weight: {key}"
                        )
                    if layer is None:
                        self.weights.update_global_weight(
                            name, value, allow_paused_inplace=True
                        )
                    else:
                        self.weights.update_layer_weight(
                            layer, name, value, allow_paused_inplace=True
                        )
                    self.applied.add(key)
                    del value

                if self.received != set(self.source_shapes) or self.applied != set(
                    self.targets
                ):
                    missing_src = sorted(set(self.source_shapes) - self.received)
                    missing_dst = sorted(set(self.targets) - self.applied)
                    raise ValueError(
                        f"incomplete HF source or internal target coverage: "
                        f"missing_sources={len(missing_src)} ({missing_src[:5]}), "
                        f"missing_targets={len(missing_dst)} ({missing_dst[:5]})"
                    )

                live_targets = self._live_targets()
                if set(live_targets) != set(self.addresses) or any(
                    value.data_ptr() != self.addresses[key]
                    for key, value in live_targets.items()
                ):
                    raise RuntimeError("weight update replaced live tensor storage")

                # Verify BF16 HC merged_down_inject view geometry & zero-padded trailing rows.
                checked_parents = set()
                for (
                    key,
                    parent,
                    tail_start,
                    storage_ptr,
                    storage_offset,
                    shape,
                    stride,
                ) in self._merged_hc_checks:
                    live = live_targets[key]
                    if (
                        live.untyped_storage().data_ptr() != storage_ptr
                        or live.storage_offset() != storage_offset
                        or tuple(live.shape) != shape
                        or tuple(live.stride()) != stride
                    ):
                        raise RuntimeError(
                            f"merged HC view {key!r} lost storage/layout identity"
                        )
                    pid = id(parent)
                    if pid not in checked_parents:
                        checked_parents.add(pid)
                        if tail_start < parent.shape[0] and bool(
                            (parent[tail_start:] != 0).any().item()
                        ):
                            raise RuntimeError(
                                f"merged HC trailing padding rows became non-zero near {key!r}"
                            )

                # Verify immutable static INT64 hash metadata did not change.
                if any(
                    not torch.equal(live_targets[key], before)
                    for key, before in self._static_meta_snapshots.items()
                ):
                    raise RuntimeError("weight update modified immutable PLE hash metadata")
                if any(
                    not torch.equal(live_targets[key], before)
                    for key, before in self._frozen_qsa_snapshots.items()
                ):
                    raise RuntimeError("weight update modified a producer-declared frozen QSA indexer")

                result = {
                    "hf_sources": len(self.received),
                    "internal_tensors": len(self.applied),
                    "static_table_tensors": len(self.static_table_keys),
                    "static_meta_tensors": len(self.static_meta_keys),
                    "static_qsa_tensors": len(self.frozen_qsa_target_keys),
                }
                if is_cuda:
                    torch.cuda.current_stream(self.device).synchronize()
                    self._static_meta_snapshots.clear()
                    self._frozen_qsa_snapshots.clear()
                    before_trim = self._allocator_memory()
                    started = time.monotonic_ns()
                    gc.collect()
                    torch.cuda.empty_cache()
                    trim_wall_ns = time.monotonic_ns() - started
                    allocator_memory = {
                        "device": str(self.device),
                        "before_refit": before_refit,
                        "before_trim": before_trim,
                        "after_trim": self._allocator_memory(),
                        "trim_wall_ns": trim_wall_ns,
                    }
                    result["allocator_memory"] = allocator_memory
                    logging.info(
                        "RL refit allocator memory %s", json.dumps(allocator_memory)
                    )
                self._done.set_result(result)
        except BaseException as exc:
            self._done.set_exception(exc)
            if self._current is not None and not self._current.done():
                self._current.set_exception(exc)

    def push(self, name, tensor, timeout=120):
        with self._lock:
            if self._done.done():
                self._done.result()
                raise RuntimeError("weight transaction is already complete")
            if self._closed:
                raise RuntimeError("weight transaction is closed")
            ack = Future()
            self._queue.put((name, tensor, ack), timeout=timeout)
            return ack.result(timeout=timeout)

    def finish(self, timeout=120):
        with self._lock:
            if self._closed:
                raise RuntimeError("weight transaction is already closed")
            self._closed = True
            if not self._done.done():
                self._queue.put(None, timeout=timeout)
            return self._done.result(timeout=timeout)
