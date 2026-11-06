"""A bounded HF weight transaction, used only while the native engine is paused."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future

import torch


class RlWeightUpdate:
    """Copy a complete stream through the normal loader before allowing commit.

    The engine owns admission, TP quiescence and version publication. An error
    here is fatal to that engine incarnation: earlier groups may be overwritten.
    A successful push acknowledges that its source storage can be reused.
    """

    def __init__(self, loader, weights, device, source_shapes, max_buffer_bytes, *, borrow_singleton_sources=False, fixed_expert_map=False, source_dtypes=None):
        self.loader = loader
        self.weights = weights
        self.device = torch.device(device)
        self.source_shapes = source_shapes
        self.source_dtypes = None if source_dtypes is None else dict(source_dtypes)
        if self.source_dtypes is not None and (
            set(self.source_dtypes) != set(source_shapes)
            or any(value not in (torch.bfloat16, torch.float32) for value in self.source_dtypes.values())
        ):
            raise ValueError("HF source dtype contract must cover every source exactly")
        self.max_buffer_bytes = max_buffer_bytes
        self.borrow_singleton_sources = bool(borrow_singleton_sources)
        all_targets = self._live_targets()
        # These two maps are generated from the fixed physical placement; they
        # have no corresponding HF source. Never silently skip another target.
        static_names = {"moe_eplb.logic_expert_cnt", "moe_eplb.log2phy"}
        self.static_targets = {
            key: value for key, value in all_targets.items()
            if fixed_expert_map and key[0] is not None and key[1] in static_names
        }
        if fixed_expert_map:
            expected_static = {(layer, name) for layer in range(len(weights.weights)) for name in static_names}
            if set(self.static_targets) != expected_static:
                raise ValueError("Fixed expert refit requires both generated maps on every layer")
            for key, value in self.static_targets.items():
                expected_shape = (512,) if key[1].endswith("logic_expert_cnt") else (512, 65)
                if value.dtype != torch.int32 or tuple(value.shape) != expected_shape:
                    raise ValueError("Fixed expert refit requires the Qwen3.8 INT32 map shapes")
        self.targets = {key: value for key, value in all_targets.items() if key not in self.static_targets}
        self.addresses = {key: value.data_ptr() for key, value in all_targets.items()}
        self._static_snapshots = {key: value.detach().clone() for key, value in self.static_targets.items()}
        if self.device.type == "cuda" and self._static_snapshots:
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
        targets = {(None, key): value for key, value in self.weights.global_weights.items()}
        targets.update({(layer, key): value for layer, group in enumerate(self.weights.weights) for key, value in group.items()})
        return targets

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
                raise ValueError(f"HF shape mismatch for {name}: {tuple(tensor.shape)}")
            if tensor.dtype not in (torch.bfloat16, torch.float32) or tensor.device != self.device:
                raise ValueError(f"HF parameter must be BF16 or FP32 on {self.device}: {name}")
            if self.source_dtypes is not None and tensor.dtype != self.source_dtypes[name]:
                raise ValueError(f"HF dtype mismatch for {name}: {tensor.dtype} != {self.source_dtypes[name]}")
            yield name, tensor
            # The loader owns pending group tensors. All reads and in-place
            # writes on our stream are complete before the producer gets ACK.
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            del tensor, item
            self.received.add(name)
            ack.set_result(None)
            self._current = None

    def _run(self):
        try:
            with torch.inference_mode():
                if self.device.type == "cuda":
                    torch.cuda.set_device(self.device)
                for layer, name, value in self.loader.prepare_weights_from_tensors(
                    self._inputs(),
                    str(self.device),
                    max_buffer_bytes=self.max_buffer_bytes,
                    borrow_singleton_sources=self.borrow_singleton_sources,
                ):
                    key = (layer, name)
                    if key in self.applied or key not in self.targets:
                        raise ValueError(
                            f"unexpected or repeated internal weight: {key}"
                        )
                    if layer is None:
                        self.weights.update_global_weight(name, value)
                    else:
                        self.weights.update_layer_weight(layer, name, value)
                    self.applied.add(key)
                    # Converted packed slices may alias the producer's IPC
                    # source. Do not retain them while requesting the next
                    # input, whose ACK permits that storage to be released.
                    del value
                if self.received != set(self.source_shapes) or self.applied != set(
                    self.targets
                ):
                    raise ValueError("incomplete HF source or internal target coverage")
                live_targets = self._live_targets()
                if set(live_targets) != set(self.addresses) or any(
                    value.data_ptr() != self.addresses[key] for key, value in live_targets.items()
                ):
                    raise RuntimeError("weight update replaced live tensor storage")
                if any(not torch.equal(live_targets[key], before) for key, before in self._static_snapshots.items()):
                    raise RuntimeError("weight update changed the fixed expert placement map")
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                self._done.set_result(
                    {
                        "hf_sources": len(self.received),
                        "internal_tensors": len(self.applied),
                    }
                )
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
