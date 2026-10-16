"""CUDA stream ordering and storage ownership for eager CP prefill work.

Callbacks must enqueue all their work on the current stream. In particular,
an asynchronous NCCL Work must be waited on that stream before returning (or
raising), so the completion event also covers NCCL's internal stream. This
helper never synchronizes a CUDA event or stream with the CPU.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

# Tensor-bearing descriptors used by the prefill producer and its consumers.
# Tuple subclasses can additionally own a workspace outside their tuple items.
_TENSOR_FIELDS = (
    "workspace",
    "output",
    "device_offsets",
    "refresh_layout",
    "quant",
    "payload",
    "scale",
    "flat",
)


def _record_stream(value, stream):
    pending, seen, recorded = [value], set(), set()
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(item, torch.Tensor):
            if not item.is_cuda:
                continue
            # Inference-mode views do not retain _base, but still share their
            # storage. Deduplicate that allocation before calling the dispatcher
            # so a ragged batch does not record every request view separately.
            while isinstance(item._base, torch.Tensor):
                item = item._base
            allocation = (item.device, item.untyped_storage().data_ptr())
            if allocation not in recorded:
                item.record_stream(stream)
                recorded.add(allocation)
            continue
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        attributes = getattr(item, "__dict__", None)
        if attributes is not None:
            pending.extend(
                attributes[name] for name in _TENSOR_FIELDS if name in attributes
            )


def _communication_stream(device):
    # Keep importing the helper independent of attention's backend imports.
    from .attention import _get_cp_comm_stream

    return _get_cp_comm_stream(device)


@dataclass
class Pending:
    value: Any
    event: Any
    device: Any = None
    _consumers: set = field(default_factory=set, init=False, repr=False)

    def wait(self):
        """Make this value usable on the calling stream without blocking CPU."""
        if self.event is not None:
            stream = torch.cuda.current_stream(self.device)
            if stream not in self._consumers:
                stream.wait_event(self.event)
                _record_stream(self.value, stream)
                self._consumers.add(stream)
        return self.value


def launch(device, fn: Callable[[], Any], inputs=()) -> Pending:
    """Enqueue ``fn`` after current work, retaining its explicit input storage.

    Include every existing tensor read or written by ``fn`` in ``inputs``.
    Tensors allocated by the callback belong to the communication stream;
    ``wait`` records returned tensors on each consumer stream. Inputs are
    recorded before calling ``fn`` so exceptions cannot prematurely recycle
    them. The callback remains responsible for joining any already-enqueued
    NCCL Work to this stream on its own exceptional paths.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return Pending(fn(), None)
    current = torch.cuda.current_stream(device)
    stream = _communication_stream(device)
    stream.wait_stream(current)
    _record_stream(inputs, stream)
    try:
        with torch.cuda.stream(stream):
            value = fn()
            completed = torch.cuda.Event()
            completed.record(stream)
    except BaseException:
        # No Pending can reach forward's cleanup. Order in-place pool/table
        # reuse after the work already submitted, without a CPU wait.
        current.wait_stream(stream)
        raise
    return Pending(value, completed, device)
