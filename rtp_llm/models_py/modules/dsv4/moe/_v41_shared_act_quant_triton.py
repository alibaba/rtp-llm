"""V4.1 shared SwiGLU and native group32 packed-scale quantization."""

import math
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _shared_act_quant(
    GateUp,
    Q,
    S,
    ROWS: tl.constexpr,
    INTER: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    LIMIT: tl.constexpr,
    V2: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    col = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    live = (row < ROWS) & (col < INTER)
    gate = tl.load(GateUp + row * INPUT_STRIDE + col, live, 0).to(tl.float32)
    up = tl.load(GateUp + row * INPUT_STRIDE + INTER + col, live, 0).to(tl.float32)
    if LIMIT > 0:
        up = tl.where(up > LIMIT, LIMIT, up)
        up = tl.where(up < -LIMIT, -LIMIT, up)
        gate = tl.where(gate > LIMIT, LIMIT, gate)
    hidden = (gate * tl.sigmoid(gate)) * up
    # W13SharedExpert rounds the full product to BF16 before w2 quantization.
    hidden = hidden.to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(hidden), 1), 1.1754943508222875e-38)
    if V2:
        bits = (maximum * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
        encoded = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        inverse = ((254 - encoded) << 23).to(tl.float32, bitcast=True)
        # CUDA v2 uses SM120 __fmul2_rn, whose f32x2 multiply flushes subnormals.
        normalized = tl.inline_asm_elementwise(
            "mul.rn.ftz.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[hidden, inverse[:, None]],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        # The legacy CUDA kernel applies an additional 1e-10 scale floor.
        scale = libdevice.exp2(
            libdevice.ceil(
                libdevice.log2(tl.maximum(tl.div_rn(maximum, 448.0), 1.0e-10))
            )
        )
        encoded = (libdevice.log2(scale).to(tl.int32) + 127).to(tl.uint32)
        normalized = tl.div_rn(hidden, scale[:, None])
    payload = tl.clamp(normalized, -448.0, 448.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    tl.store(Q + row * INTER + col, payload, live)
    scale_offset = (groups // 4 * SCALE_STRIDE + row) * 4 + groups % 4
    tl.store(
        S + scale_offset, encoded.to(tl.uint8), (row < ROWS) & (groups < INTER // 32)
    )
    # Padding is private storage. Initialize it without another launch.
    tl.store(S + scale_offset, 0, (row >= ROWS) & (groups < INTER // 32))


def is_supported(gate_up, clamp_limit=0.0):
    return (
        gate_up.is_cuda
        and gate_up.ndim == 2
        and gate_up.dtype == torch.bfloat16
        and gate_up.shape[1] > 0
        and gate_up.shape[1] % 256 == 0
        and gate_up.stride(1) == 1
        and gate_up.stride(0) >= gate_up.shape[1]
        and math.isfinite(clamp_limit)
        and torch.cuda.get_device_capability(gate_up.device) == (12, 0)
        and not torch.cuda.is_current_stream_capturing()
        and not (torch.is_grad_enabled() and gate_up.requires_grad)
    )


def shared_act_quant(gate_up, *, clamp_limit=0.0):
    """Return E4M3 values and column-major TMA packed UE8M0, or None."""
    if not is_supported(gate_up, clamp_limit):
        return None
    rows, width = gate_up.shape
    inter = width // 2
    mode = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    if mode not in ("auto", "legacy", "v2"):
        raise ValueError("DSV4_FP8_QUANT_KERNEL must be auto, legacy, or v2")
    use_v2 = mode == "v2" or (mode == "auto" and rows * inter >= 4 * 1024 * 1024)
    aligned_rows = max(1, triton.cdiv(rows, 4) * 4)
    values = torch.empty(
        (rows, inter), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    storage = torch.empty(
        (inter // 128, aligned_rows), dtype=torch.int32, device=gate_up.device
    )
    scales = storage.T[:rows]
    if rows:
        _shared_act_quant[(aligned_rows, triton.cdiv(inter // 32, 8))](
            gate_up,
            values,
            storage.view(torch.uint8),
            rows,
            inter,
            gate_up.stride(0),
            aligned_rows,
            float(clamp_limit),
            use_v2,
            8,
            num_warps=4,
        )
    return values, scales
