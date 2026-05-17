from __future__ import annotations

import torch


def current_cuda_stream_id():
    """Return the raw CUDA stream handle accepted by RTP-LLM pybind ops.

    Use this helper only at the Python -> custom CUDA/pybind boundary when the
    kernel API still takes a raw stream handle, e.g. ``rtp_llm_ops.rmsnorm(...,
    stream_id)``.  Regular Torch ops should keep using PyTorch's stream
    management directly.

    In eager execution ``torch.cuda.current_stream()`` is normally a
    ``torch.cuda.Stream`` exposing the historical ``.cuda_stream`` attribute.
    Under ``torch.compile`` / Dynamo graph-break execution, PyTorch can hand
    back the newer generic ``torch.Stream`` wrapper instead; that object may
    only expose the CUDA stream protocol via ``__cuda_stream__``.  Support both
    forms so GraphFX-compiled DSV4 paths can still call existing pybind kernels
    without changing their stream argument contract.
    """
    stream = torch.cuda.current_stream()
    stream_id = getattr(stream, "cuda_stream", None)
    if stream_id is not None:
        return stream_id
    cuda_stream = getattr(stream, "__cuda_stream__", None)
    if cuda_stream is None:
        raise AttributeError(f"{type(stream).__name__} does not expose a CUDA stream id")
    protocol_value = cuda_stream()
    if isinstance(protocol_value, tuple):
        return protocol_value[-1]
    return protocol_value
