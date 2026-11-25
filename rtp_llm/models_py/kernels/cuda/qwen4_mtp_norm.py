"""Graph-safe Qwen4 MTP input RMSNorms with FP32 zero-centered gains."""

import math

import torch
import triton
import triton.language as tl

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


@triton.jit
def _mtp_embedding_rms_norm(
    X,
    G,
    Y,
    V,
    L: tl.constexpr,
    EPS: tl.constexpr,
    WRITE_VARIANCE: tl.constexpr = False,
):
    row = tl.program_id(0)
    lane = tl.arange(0, L)
    a = tl.full((L,), 0, tl.float32)
    b = tl.full((L,), 0, tl.float32)
    c = tl.full((L,), 0, tl.float32)
    d = tl.full((L,), 0, tl.float32)
    for block in tl.static_range(triton.cdiv(2560, L * 4)):
        col = block * L * 4 + lane * 4
        x0 = tl.load(X + row * 2560 + col, col < 2560, other=0.0).to(tl.float32)
        x1 = tl.load(X + row * 2560 + col + 1, col + 1 < 2560, other=0.0).to(tl.float32)
        x2 = tl.load(X + row * 2560 + col + 2, col + 2 < 2560, other=0.0).to(tl.float32)
        x3 = tl.load(X + row * 2560 + col + 3, col + 3 < 2560, other=0.0).to(tl.float32)
        a += x0 * x0
        b += x1 * x1
        c += x2 * x2
        d += x3 * x3
    sums = ((a + b) + c) + d
    if L > 32:
        sums = tl.sum(sums.reshape(L // 32, 32), 0)
    variance = tl.sum(sums, 0) * (1.0 / 2560.0)
    if WRITE_VARIANCE:
        tl.store(V + row, variance)
    inverse = tl.rsqrt(variance + EPS)
    columns = tl.arange(0, 4096)
    values = tl.load(X + row * 2560 + columns, columns < 2560, other=0.0).to(tl.float32)
    gain = 1.0 + tl.load(G + columns, columns < 2560, other=0.0).to(tl.float32)
    tl.store(Y + row * 2560 + columns, (values * inverse) * gain, columns < 2560)


def try_mtp_embedding_rms_norm(x, gamma, eps):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and not torch.are_deterministic_algorithms_enabled()
        and x.ndim == 2
        and x.shape[1] == 2560
        and 1 <= x.shape[0] <= 32768
        and x.dtype == torch.bfloat16
        and gamma.shape == (2560,)
        and gamma.dtype in (torch.bfloat16, torch.float32)
        and gamma.device == x.device
        and x.is_contiguous()
        and gamma.is_contiguous()
        and not (x.requires_grad or gamma.requires_grad)
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and eps >= 0
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    ):
        return None
    rows = x.shape[0]
    # CUDA uses four input elements per lane and up to 512 threads. Output
    # rows select its x/y block dimensions; preserve that reduction order.
    lanes = max(32, 512 // (1 << (rows.bit_length() - 1)))
    output = torch.empty_like(x)
    _mtp_embedding_rms_norm[(rows,)](
        x,
        gamma,
        output,
        output,
        L=lanes,
        EPS=eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _mtp_wide_rms_norm_row(
    X,
    G,
    OUT,
    ROW,
    TILE_ID,
    WIDTH: tl.constexpr,
    LANES: tl.constexpr,
    EPS: tl.constexpr,
    TILE: tl.constexpr,
):
    # FP32 contiguous mean vectorizes four adjacent values per thread and
    # splits the reduction across all 512 CTA threads for these widths.
    cols = tl.arange(0, 2048)
    partial = tl.full((2048,), 0.0, tl.float32)
    for tile in tl.static_range(WIDTH // 2048):
        values = tl.load(X + ROW * WIDTH + tile * 2048 + cols).to(tl.float32)
        partial += values * values
    even, odd = tl.split(partial.reshape(512, 2, 2))
    a, c = tl.split(even)
    b, d = tl.split(odd)
    sums = ((a + b) + c) + d
    if LANES > 32:
        sums = tl.sum(sums.reshape(512 // LANES, LANES // 32, 32), 1)
    else:
        sums = sums.reshape(512 // LANES, 32)
    variance = tl.sum(tl.sum(sums, 1), 0) * (1.0 / WIDTH)
    columns = TILE_ID * TILE + tl.arange(0, TILE)
    values = tl.load(X + ROW * WIDTH + columns).to(tl.float32)
    gain = 1.0 + tl.load(G + columns).to(tl.float32)
    inverse = tl.rsqrt(variance + EPS)
    tl.store(OUT + ROW * WIDTH + columns, (values * inverse) * gain)


@triton.jit
def _mtp_wide_input_rms_norm(
    EMBEDDING,
    HIDDEN,
    EMBEDDING_GAMMA,
    HIDDEN_GAMMA,
    EMBEDDING_OUT,
    HIDDEN_OUT,
    LANES: tl.constexpr,
    EPS: tl.constexpr,
    EMBEDDING_TILE: tl.constexpr,
    HIDDEN_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if tile < 8192 // EMBEDDING_TILE:
        _mtp_wide_rms_norm_row(
            EMBEDDING,
            EMBEDDING_GAMMA,
            EMBEDDING_OUT,
            row,
            tile,
            8192,
            LANES,
            EPS,
            EMBEDDING_TILE,
        )
    else:
        _mtp_wide_rms_norm_row(
            HIDDEN,
            HIDDEN_GAMMA,
            HIDDEN_OUT,
            row,
            tile - 8192 // EMBEDDING_TILE,
            32768,
            LANES,
            EPS,
            HIDDEN_TILE,
        )


def try_mtp_wide_input_rms_norm(embedding, hidden, embedding_gamma, hidden_gamma, eps):
    if not (
        embedding.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and not torch.are_deterministic_algorithms_enabled()
        and isinstance(eps, (float, int))
        and math.isfinite(eps)
        and eps >= 0
        and embedding.ndim == hidden.ndim == 2
        and embedding.shape[0] == hidden.shape[0]
        and 1 <= embedding.shape[0] <= 32768
        and embedding.shape[1] == 8192
        and hidden.shape[1] == 32768
        and embedding_gamma.shape == (8192,)
        and hidden_gamma.shape == (32768,)
        and embedding.dtype == hidden.dtype == torch.bfloat16
        and all(
            tensor.device == embedding.device
            and tensor.is_contiguous()
            and not tensor.requires_grad
            for tensor in (embedding, hidden, embedding_gamma, hidden_gamma)
        )
        and all(
            gamma.dtype in (torch.bfloat16, torch.float32)
            for gamma in (embedding_gamma, hidden_gamma)
        )
        and torch.cuda.get_device_capability(embedding.device) == (10, 3)
    ):
        return None
    rows = embedding.shape[0]
    lanes = max(32, 512 // (1 << (rows.bit_length() - 1)))
    embedding_tile, hidden_tile = (2048, 4096) if rows <= 32 else (8192, 32768)
    embedding_out = torch.empty_like(embedding)
    hidden_out = torch.empty_like(hidden)
    _mtp_wide_input_rms_norm[(rows, 8192 // embedding_tile + 32768 // hidden_tile)](
        embedding,
        hidden,
        embedding_gamma,
        hidden_gamma,
        embedding_out,
        hidden_out,
        LANES=lanes,
        EPS=eps,
        EMBEDDING_TILE=embedding_tile,
        HIDDEN_TILE=hidden_tile,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return embedding_out, hidden_out
