"""Opt-in, byte-preserving BF16 SiLU/multiply candidate for vision MLPs.

The reference is ``F.silu(gate) * up``: ATen first rounds SiLU to BF16,
then promotes those BF16 values and ``up`` for the multiply. Existing fused
activation kernels that keep SiLU in FP32 do not implement that boundary.

An operator table is computed by the installed CUDA ATen SiLU for all 65536 BF16 bit
patterns. It is immutable operator data, never an image/input/output cache.
This avoids substituting a different exp/div approximation for ATen's.

Tables are private to (device, CUDA stream), initialized on that same stream,
and retained for the module lifetime (128 KiB per encountered device/stream).
There is deliberately no eviction: captured graphs may retain table addresses.
Warm this helper on each execution stream before measuring or graph capture.
Cold graph capture returns None; supported execution errors propagate.

The production vision MLP selects this helper with V41_VISION_SILU_MUL_FUSED
(default off) and retains the original path when it returns None. This module
does not enable itself. Numerical and performance acceptance require the
independent Bazel GPU gates.
"""

import threading
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton = None


_VISION_INTERMEDIATE = 2816
_BLOCK_ELEMENTS = 1024
_TABLES: Dict[Tuple[int, int], torch.Tensor] = {}
_TABLE_LOCK = threading.Lock()


def _supported(gate: torch.Tensor, up: torch.Tensor) -> bool:
    # Cheap metadata rejection precedes every CUDA query. In particular this
    # never initializes CUDA for CPU, meta, training, or unsupported inputs.
    return (
        isinstance(gate, torch.Tensor)
        and isinstance(up, torch.Tensor)
        and not torch.is_grad_enabled()
        and gate.layout == torch.strided
        and up.layout == torch.strided
        and gate.device.type == "cuda"
        and up.device == gate.device
        and gate.dtype == torch.bfloat16
        and up.dtype == torch.bfloat16
        and gate.ndim == 2
        and gate.shape == up.shape
        and gate.shape[0] > 0
        and gate.shape[1] == _VISION_INTERMEDIATE
        and gate.stride(1) == 1
        and up.stride(1) == 1
        and gate.stride(0) >= _VISION_INTERMEDIATE
        and up.stride(0) >= _VISION_INTERMEDIATE
        and torch.cuda.get_device_capability(gate.device)[0] == 10
    )


def _bf16_silu_table(device: torch.device) -> Optional[torch.Tensor]:
    stream = torch.cuda.current_stream(device)
    key = (device.index, stream.cuda_stream)
    table = _TABLES.get(key)
    if table is not None:
        return table
    # Do not publish graph-owned allocations or introduce a hidden table-build
    # workload during capture. Warmup on this stream establishes the table.
    if torch.cuda.is_current_stream_capturing():
        return None
    with _TABLE_LOCK:
        table = _TABLES.get(key)
        if table is None:
            values = torch.arange(65536, device=device, dtype=torch.int32)
            values = values.to(torch.int16).view(torch.bfloat16)
            table = F.silu(values)
            _TABLES[key] = table
    return table


if triton is not None:

    @triton.jit
    def _vision_silu_mul_lut_kernel(
        GATE,
        UP,
        TABLE,
        OUT,
        N: tl.constexpr,
        WIDTH: tl.constexpr,
        GATE_STRIDE: tl.constexpr,
        UP_STRIDE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # The production B128 shape has >2**31 output elements; all indexing,
        # including strided source offsets, must be 64-bit before arithmetic.
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        rows, columns = offsets // WIDTH, offsets % WIDTH
        gate = tl.load(GATE + rows * GATE_STRIDE + columns, offsets < N, other=0)
        up = tl.load(UP + rows * UP_STRIDE + columns, offsets < N, other=0)
        # Bitcast BEFORE any numeric promotion, preserving NaN payload/sign
        # and signed-zero lookup keys. The table is already rounded to BF16.
        index = gate.to(tl.uint16, bitcast=True).to(tl.int32)
        activated = tl.load(TABLE + index).to(tl.float32)
        product = activated * up.to(tl.float32)
        tl.store(OUT + offsets, product, offsets < N)


def apply_vision_silu_mul(
    gate: torch.Tensor, up: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return a fresh contiguous BF16 output, or None for unsupported input.

    Supported: inference/no-grad, CUDA SM10x, [rows, 2816] BF16 split inputs
    with unit column stride and nonoverlapping rows. This includes the actual
    ``w1(x).chunk(2, -1)`` layout (row stride 5632), storage offsets, and two
    independent tensors with different row strides. No inputs are modified.

    A non-None result ALWAYS comes from ``_vision_silu_mul_lut_kernel``. There
    is no concealed eager fallback or output cache. Integration gates may
    wrap this function to count non-None calls and corroborate kernel traces.
    No global counters are used in the production hot path.
    """
    if not _supported(gate, up):
        return None
    if triton is None:
        raise RuntimeError("The supported vision SiLU/multiply path requires Triton")
    with torch.cuda.device(gate.device):
        table = _bf16_silu_table(gate.device)
        if table is None:
            return None
        output = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
        _vision_silu_mul_lut_kernel[(triton.cdiv(gate.numel(), _BLOCK_ELEMENTS),)](
            gate,
            up,
            table,
            output,
            gate.numel(),
            _VISION_INTERMEDIATE,
            gate.stride(0),
            up.stride(0),
            _BLOCK_ELEMENTS,
            enable_fp_fusion=False,
        )
    return output
