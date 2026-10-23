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

    def __init__(self, loader, weights, device, source_shapes, max_buffer_bytes):
        self.loader = loader
        self.weights = weights
        self.device = torch.device(device)
        self.source_shapes = source_shapes
        self.max_buffer_bytes = max_buffer_bytes
        self.targets = {(None, k): v for k, v in weights.global_weights.items()}
        self.targets.update(
            {
                (i, k): v
                for i, group in enumerate(weights.weights)
                for k, v in group.items()
            }
        )
        self.addresses = {k: v.data_ptr() for k, v in self.targets.items()}
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
            if tensor.dtype != torch.bfloat16 or tensor.device != self.device:
                raise ValueError(f"HF parameter must be BF16 on {self.device}: {name}")
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
                if self.received != set(self.source_shapes) or self.applied != set(
                    self.targets
                ):
                    raise ValueError("incomplete HF source or internal target coverage")
                if any(
                    v.data_ptr() != self.addresses[k] for k, v in self.targets.items()
                ):
                    raise RuntimeError("weight update replaced live tensor storage")
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
