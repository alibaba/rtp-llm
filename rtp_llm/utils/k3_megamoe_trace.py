"""Read actual routed FC2 outputs from the audited MegaMoE combine buffer.

The final combine region has a full token capacity for each top-k slot. Unlike
the L1/L2 activation rings, it is not overwritten within a kernel invocation.
This private layout is accepted only for the dependency headers audited below.
"""

import hashlib
import importlib
import os
from pathlib import Path

import torch

_AUDITED_HEADERS = {
    (
        "f77e3d70460314db39c47a35d5b95289b92e3eb769351859b488ae3a06e95cfe",
        "291a4147a897b119f7a2b9276270452e9b3d427a65e7dc7add260487d81958bd",
    ),  # RTP deep_gemm 2.6.1+a0bd921.cu132
    (
        "90fdb8f27898b1ce1bead6bc4ae3e54ce5edc9d4f6ee47e3169dbbf2d859ff53",
        "fd1935beff098addd2e54512a7ddc37936c04deb9c7b423128fce46be92ba8f7",
    ),  # vLLM DeepGEMM 8b1392b978f5a03c828dd1711090d7fb50958b8a
    (
        "f77e3d70460314db39c47a35d5b95289b92e3eb769351859b488ae3a06e95cfe",
        "66a7dfca57ed43d49670ab2594c642ce3065d409d672e4dcfc812612080131fd",
    ),  # RTP with the ABI 2 observation-only patch
    (
        "90fdb8f27898b1ce1bead6bc4ae3e54ce5edc9d4f6ee47e3169dbbf2d859ff53",
        "169600e438a24d09665f278beb64c68ee390482c787a9030bf22b3769c99cec1",
    ),  # vLLM with the ABI 2 observation-only patch
}


def prepare_expert_output_view(buffer, deep_gemm):
    """Validate once at buffer creation; retain a view for eager/graph recording.

    K3 runs routed experts here and its shared branch separately. A fused shared
    combine slot, an unknown layout, or debug zeroing must fail explicitly.
    """
    if int(os.environ.get("DG_COMM_KERNEL_DEBUG", "0")):
        raise RuntimeError("DG_COMM_KERNEL_DEBUG erases the FC2 trace buffer")
    package = Path(deep_gemm.__file__).resolve().parent
    headers = tuple(
        hashlib.sha256((package / "include/deep_gemm" / name).read_bytes()).hexdigest()
        for name in ("layout/mega_moe.cuh", "impls/sm100_fp8_fp4_mega_moe.cuh")
    )
    if headers not in _AUDITED_HEADERS:
        raise RuntimeError(f"Unaudited MegaMoE FC2 trace layout: {headers}")

    raw = buffer.buffer
    scales = buffer.l2_acts_sf
    capacity, topk, hidden = (
        buffer.num_max_tokens_per_rank,
        buffer.num_topk,
        buffer.hidden,
    )
    if (
        raw.dtype != torch.int8
        or raw.ndim != 1
        or not raw.is_contiguous()
        or scales.dtype != torch.int32
        or any(value <= 0 for value in (capacity, topk, hidden))
        or hidden % 8
    ):
        raise RuntimeError("Unsupported MegaMoE FC2 trace buffer")
    size = topk * capacity * hidden * 2
    offset = raw.numel() - size
    # Both pinned native slicers expose the preceding SF region with a dense
    # transposed layout. Prove the computed tail starts exactly at its end.
    if (
        offset < 0
        or scales.device != raw.device
        or not scales.transpose(0, 1).is_contiguous()
        or scales.data_ptr() + scales.numel() * scales.element_size()
        != raw.data_ptr() + offset
    ):
        raise RuntimeError("MegaMoE FC2 tail does not follow the routed SF region")
    return raw.narrow(0, offset, size).view(torch.bfloat16).view(topk, capacity, hidden)


def expert_output_tensors(view, expert_ids):
    """Expose token/top-k/hidden order and mask slots the kernel did not write."""
    if expert_ids.ndim != 2 or expert_ids.shape[1] != view.shape[0]:
        raise RuntimeError("MegaMoE FC2 trace routing shape mismatch")
    if expert_ids.shape[0] > view.shape[1]:
        raise RuntimeError("MegaMoE FC2 trace exceeds buffer capacity")
    values = view[:, : expert_ids.shape[0]].transpose(0, 1)
    valid = expert_ids >= 0
    return {
        "values": torch.where(valid.unsqueeze(-1), values, 0),
        "expert_ids": expert_ids,
        "valid": valid,
    }


def prepare_native_trace_factory(deep_gemm):
    """Resolve the matching Python/native ABI before any captured invocation."""
    try:
        native = importlib.import_module(deep_gemm.__name__ + "._C")
        buffers = importlib.import_module(deep_gemm.__name__ + ".mega.k3_trace")
        matching = buffers.ABI_VERSION == 2 and native.k3_trace_abi() == 2
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Install the ABI 2 K3 native MegaMoE trace dependency"
        ) from exc
    if not matching:
        raise RuntimeError("K3 native MegaMoE trace requires matching ABI 2")
    return buffers.K3TraceBuffers


def native_trace_tensors(trace):
    """Yield one source rank at a time so eager recording can spill fragments."""
    for rank in range(trace.tensors["expert_ids"].shape[0]):
        valid = trace.tensors["expert_ids"][rank] >= 0
        yield f"source_rank.{rank}.valid", valid
        for name, tensor in trace.tensors.items():
            value = tensor[rank]
            if name != "expert_ids":
                mask = valid.reshape(valid.shape + (1,) * (value.ndim - valid.ndim))
                value = torch.where(mask, value, 0)
            yield f"source_rank.{rank}.{name}", value
