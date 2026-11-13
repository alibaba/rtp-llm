"""SM120 HC consumer with the pinned native BF16 split-K boundaries.

The four injection weights remain a separate BF16 compute view. Each native
split is rounded to BF16 before the final FP32 reduction; replacing it with a
single FP32 GEMV changes the gate. Keep the native path outside this contract.
"""

import torch
import triton
import triton.language as tl

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


@triton.jit
def _inject_bf16_parts(X, W, Parts, M: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    split = tl.program_id(1)
    col = split * BK + tl.arange(0, BK)
    branch = tl.arange(0, 16)
    x = tl.load(X + row[:, None] * 10240 + col[None, :], row[:, None] < M, other=0)
    w = tl.load(
        W + branch[None, :] * 10240 + col[:, None], branch[None, :] < 4, other=0
    )
    partial = tl.dot(x, w)
    tl.store(
        Parts + split * M * 4 + row[:, None] * 4 + branch[None, :],
        partial,
        (row[:, None] < M) & (branch[None, :] < 4),
    )


@triton.jit
def _inject_reduce_down_epilogue(
    Parts, Down, Mix, Gate, M: tl.constexpr, SPLITS: tl.constexpr
):
    row = tl.program_id(0)
    branch = tl.arange(0, 4)
    lane = tl.arange(0, 32)
    value = tl.full((32, 4), 0, tl.float32)
    for group in tl.static_range(triton.cdiv(SPLITS, 32)):
        split = lane + group * 32
        value += tl.load(
            Parts + split[:, None] * M * 4 + row * 4 + branch[None, :],
            split[:, None] < SPLITS,
            other=0,
        ).to(tl.float32)
    projected = tl.sum(value, 0).to(tl.bfloat16).to(tl.float32)
    scaled = (projected * 0.25).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(scaled).to(tl.bfloat16).to(tl.float32) * 2.0
    tl.store(Gate + row * 4 + branch, gate)
    column = tl.arange(0, 512)
    down = tl.load(Down + row * 320 + column, column < 320, other=0).to(tl.float32)
    z = (down * 0.25).to(tl.bfloat16).to(tl.float32)
    tl.store(Mix + row * 320 + column, z * tl.sigmoid(z), column < 320)


def try_splitk_down_inject(normed, weight, down):
    """Return the complete down/gate epilogue, or leave both consumers native."""
    if not (
        normed.is_cuda
        and torch.version.hip is None
        and str(torch.__version__).startswith("2.11.0")
        and torch.version.git_version == _TORCH_GIT
        and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        and not torch.are_deterministic_algorithms_enabled()
        and torch.cuda.get_device_capability(normed.device) == (12, 0)
        and normed.dtype == weight.dtype == down.dtype == torch.bfloat16
        and normed.ndim == 2
        and (4 <= normed.shape[0] <= 32 or 4093 <= normed.shape[0] <= 4096)
        and normed.shape[1] == 10240
        and weight.shape == (4, 10240)
        and down.shape == (normed.shape[0], 320)
        and normed.device == weight.device == down.device
        and normed.is_contiguous()
        and weight.is_contiguous()
        and down.is_contiguous()
        and not any(t.requires_grad for t in (normed, weight, down))
    ):
        return None
    rows = normed.shape[0]
    if rows >= 4093:
        mix = torch.empty_like(down)
        gate = torch.empty((rows, 4), device=normed.device, dtype=torch.bfloat16)
        _inject_serial3_down_epilogue[(triton.cdiv(rows, 32),)](
            normed,
            weight,
            down,
            mix,
            gate,
            M=rows,
            BM=32,
            num_warps=4,
            num_stages=4,
            enable_fp_fusion=False,
        )
        return mix, gate
    splits = 40 if rows <= 16 else 80
    parts = torch.empty((splits, rows, 4), device=normed.device, dtype=torch.bfloat16)
    mix = torch.empty_like(down)
    gate = torch.empty((rows, 4), device=normed.device, dtype=torch.bfloat16)
    _inject_bf16_parts[(triton.cdiv(rows, 16), splits)](
        normed,
        weight,
        parts,
        M=rows,
        BK=10240 // splits,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    _inject_reduce_down_epilogue[(rows,)](
        parts,
        down,
        mix,
        gate,
        M=rows,
        SPLITS=splits,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return mix, gate


@triton.jit
def _inject_serial3_down_epilogue(
    X,
    W,
    Down,
    Mix,
    Gate,
    M: tl.constexpr,
    BM: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    branch = tl.arange(0, 16)
    column = tl.arange(0, 128)
    previous = tl.full((BM, 16), 0, tl.bfloat16)
    # Match native 32x32x128 WMMA serial split-K=3. Each segment has
    # ceil(10240 / (3*128))*128 columns and rounds its accumulated C to BF16.
    for split in tl.static_range(3):
        accumulator = tl.full((BM, 16), 0, tl.float32)
        for local in range(0, 3456, 128):
            k = split * 3456 + local + column
            x = tl.load(
                X + row[:, None] * 10240 + k[None, :],
                (row[:, None] < M) & (k[None, :] < 10240),
                other=0,
            )
            weight = tl.load(
                W + branch[None, :] * 10240 + k[:, None],
                (branch[None, :] < 4) & (k[:, None] < 10240),
                other=0,
            )
            accumulator = tl.dot(x, weight, accumulator)
        previous = (accumulator + previous.to(tl.float32)).to(tl.bfloat16)
    scaled = (previous.to(tl.float32) * 0.25).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(scaled).to(tl.bfloat16).to(tl.float32) * 2.0
    tl.store(
        Gate + row[:, None] * 4 + branch[None, :],
        gate,
        (row[:, None] < M) & (branch[None, :] < 4),
    )
    column = tl.arange(0, 512)
    down = tl.load(
        Down + row[:, None] * 320 + column[None, :],
        (row[:, None] < M) & (column[None, :] < 320),
        other=0,
    ).to(tl.float32)
    z = (down * 0.25).to(tl.bfloat16).to(tl.float32)
    tl.store(
        Mix + row[:, None] * 320 + column[None, :],
        z * tl.sigmoid(z),
        (row[:, None] < M) & (column[None, :] < 320),
    )
