"""Inference kernels preserving GLM vision's intermediate BF16 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_f32(a, b):
    # Prevent narrowing the FP32 multiply/cast into a native half multiply.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rope_f32(x, cos, rotated, sin):
    # Keep the reference's two rounded FP32 products and rounded FP32 sum.
    # Packed f32x2 instruction selection changed BF16 midpoint results on sm100.
    return tl.inline_asm_elementwise(
        "{ .reg .f32 p1, p2; mul.rn.f32 p1, $1, $2; mul.rn.f32 p2, $3, $4; add.rn.f32 $0, p1, p2; }",
        "=f,f,f,f,f",
        [x, cos, rotated, sin],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rms_kernel(X, W, INV_RMS, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    scale = tl.load(INV_RMS + row)
    # The reference casts the normalized value before multiplying the weight.
    normalized = (_mul_f32(x, scale)).to(Y.dtype.element_ty).to(tl.float32)
    w = tl.load(W + col, col < N, 0).to(tl.float32)
    tl.store(Y + row * N + col, _mul_f32(normalized, w), col < N)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Preserve PyTorch's reduction tree and rsqrt: rare BF16 boundary differences amplify
    # through the 24 vision layers. Fuse the remaining elementwise operations.
    inverse_rms = torch.rsqrt(x.float().square().mean(-1) + eps)
    x = x.contiguous()
    out = torch.empty_like(x)
    width = x.shape[-1]
    _rms_kernel[(x.numel() // width,)](
        x,
        weight,
        inverse_rms,
        out,
        width,
        triton.next_power_of_2(width),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _qk_norm_rope_kernel(
    QKV,
    QW,
    KW,
    QINV_RMS,
    KINV_RMS,
    COS,
    SIN,
    Q,
    K,
    ROWS,  # Dynamic packed token count: do not compile once per batch shape.
    HEADS: tl.constexpr,
    D: tl.constexpr,
    ROTARY: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    component = tl.program_id(1)
    col = tl.arange(0, BLOCK_D)
    token = row // HEADS
    head = row % HEADS
    offsets = token[:, None] * (3 * HEADS * D) + component * HEADS * D
    offsets += head[:, None] * D + col[None, :]
    mask = (row[:, None] < ROWS) & (col[None, :] < D)
    x = tl.load(QKV + offsets, mask, 0).to(tl.float32)
    inverse_rms = QINV_RMS if component == 0 else KINV_RMS
    scale = tl.load(inverse_rms + row, row < ROWS, 0)
    wptr = QW if component == 0 else KW
    w = tl.load(wptr + col, col < D, 0).to(tl.float32)
    normalized = (_mul_f32(x, scale[:, None])).to(QKV.dtype.element_ty).to(tl.float32)
    normalized = (
        (_mul_f32(normalized, w[None, :])).to(QKV.dtype.element_ty).to(tl.float32)
    )
    partner = tl.where(col < ROTARY, (col + ROTARY // 2) % ROTARY, col)
    rotated = tl.gather(
        normalized, tl.broadcast_to(partner[None, :], (BLOCK_ROWS, BLOCK_D)), 1
    )
    rotated = tl.where(col[None, :] < ROTARY // 2, -rotated, rotated)
    pos = token[:, None] * ROTARY + col[None, :]
    cmask = (row[:, None] < ROWS) & (col[None, :] < ROTARY)
    cos = tl.load(COS + pos, cmask, 0)
    sin = tl.load(SIN + pos, cmask, 0)
    out = tl.where(
        col[None, :] < ROTARY, _rope_f32(normalized, cos, rotated, sin), normalized
    )
    target = Q if component == 0 else K
    tl.store(target + row[:, None] * D + col[None, :], out, mask)


def qk_norm_rope(qkv, q_weight, k_weight, cos, sin, eps=1e-5):
    tokens, _, heads, dim = qkv.shape
    q_inverse_rms = torch.rsqrt(qkv[:, 0].float().square().mean(-1) + eps)
    k_inverse_rms = torch.rsqrt(qkv[:, 1].float().square().mean(-1) + eps)
    q = torch.empty((tokens, heads, dim), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    _qk_norm_rope_kernel[(triton.cdiv(tokens * heads, 8), 2)](
        qkv,
        q_weight,
        k_weight,
        q_inverse_rms,
        k_inverse_rms,
        cos,
        sin,
        q,
        k,
        tokens * heads,
        heads,
        dim,
        cos.shape[-1],
        8,
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k
