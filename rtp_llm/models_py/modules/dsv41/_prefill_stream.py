"""Reuse the capture decision for an eager prefill on the CUDA default stream."""

from contextlib import contextmanager
from threading import local

import torch

# torch.cuda.current_stream uses the same binding, then constructs a Stream.
# -1 selects the calling thread's actual current device. The tuple identifies
# both the device and its stream, even when multiple GPUs use a null handle.
_current_stream = getattr(torch._C, "_cuda_getCurrentStream", None)
_state = local()


@contextmanager
def default_eager_prefill(enabled: bool):
    previous = getattr(_state, "stream", None)
    stream = _current_stream(-1) if enabled and _current_stream is not None else None
    # A nondefault stream can start capturing again inside this forward.
    _state.stream = stream if stream is not None and stream[0] == 0 else None
    try:
        yield
    finally:
        _state.stream = previous


def prefill_stream_is_capturing() -> bool:
    stream = getattr(_state, "stream", None)
    if stream is not None and _current_stream(-1) == stream:
        return False
    return torch.cuda.is_current_stream_capturing()
