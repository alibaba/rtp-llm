from __future__ import annotations
import torch
import triton
import triton.language as tl
@triton.jit
def _mxfp8_peer_sum_kernel(
    payload_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    payload_cols: tl.constexpr,
    scale_cols: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < hidden_size
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for peer in tl.static_range(world_size):
        peer_row = peer * n_rows + row
        base = peer_row.to(tl.int64) * payload_cols
        q_u8 = tl.load(payload_ptr + base + cols, mask=mask, other=0)
        q = q_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_idx = cols // 32
        encoded_scale = tl.load(
            payload_ptr + base + hidden_size + scale_idx,
            mask=mask,
            other=127,
        )
        scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        acc += q * scale
    tl.store(output_ptr + row.to(tl.int64) * hidden_size + cols, acc, mask=mask)
def mxfp8_dequant_peer_sum(
    returned_payload: torch.Tensor,
    n_rows: int,
    hidden_size: int,
    world_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if returned_payload.dtype != torch.uint8 or not returned_payload.is_contiguous():
        raise ValueError("returned_payload must be contiguous uint8")
    scale_cols = hidden_size // 32
    payload_cols = hidden_size + scale_cols
    if returned_payload.shape != (world_size * n_rows, payload_cols):
        raise ValueError(
            f"unexpected payload shape {tuple(returned_payload.shape)}, "
            f"expected {(world_size * n_rows, payload_cols)}"
        )
    output = torch.empty(
        (n_rows, hidden_size), dtype=out_dtype, device=returned_payload.device
    )
    block_d = 256
    _mxfp8_peer_sum_kernel[(n_rows, triton.cdiv(hidden_size, block_d))](
        returned_payload,
        output,
        n_rows=n_rows,
        hidden_size=hidden_size,
        payload_cols=payload_cols,
        scale_cols=scale_cols,
        world_size=world_size,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return output


# Decode CUDA-graph peer reduce: All-to-All returns [world, n_rows, D]
# bf16 (peer-major). Load to fp32, sum over world → [n_rows, D] fp32.
# n_rows is local tokens/rank; decode captures 1..1024, or 1..1024*topk.
# No autotune — one JIT per (WORLD, BLOCK_D), n_rows/hidden runtime.
_PEER_SUM_GRID_CAP = 4096
_PEER_SUM_WORLDS = (2, 4, 8)


@triton.jit(do_not_specialize=["n_rows", "hidden", "n_prog"])
def _fp32_peer_sum_kernel(
    src_ptr,
    out_ptr,
    n_rows,
    hidden,
    n_prog,
    WORLD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    n_d = tl.cdiv(hidden, BLOCK_D)
    total = n_rows * n_d
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    for idx in range(pid, total, n_prog):
        row = idx // n_d
        block = idx % n_d
        offs_d = block * BLOCK_D + cols
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if HAS_MASK:
            mask = offs_d < hidden
            for peer in tl.static_range(WORLD):
                peer_row = peer * n_rows + row
                src_offs = peer_row.to(tl.int64) * hidden + offs_d
                acc += tl.load(src_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + row.to(tl.int64) * hidden + offs_d, acc, mask=mask)
        else:
            for peer in tl.static_range(WORLD):
                peer_row = peer * n_rows + row
                src_offs = peer_row.to(tl.int64) * hidden + offs_d
                acc += tl.load(src_ptr + src_offs).to(tl.float32)
            tl.store(out_ptr + row.to(tl.int64) * hidden + offs_d, acc)


@triton.jit(do_not_specialize=["n_rows", "hidden", "n_prog", "world"])
def _fp32_peer_sum_kernel_dyn(
    src_ptr,
    out_ptr,
    n_rows,
    hidden,
    n_prog,
    world,
    BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    n_d = tl.cdiv(hidden, BLOCK_D)
    total = n_rows * n_d
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    for idx in range(pid, total, n_prog):
        row = idx // n_d
        block = idx % n_d
        offs_d = block * BLOCK_D + cols
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        if HAS_MASK:
            mask = offs_d < hidden
            for peer in range(world):
                peer_row = peer * n_rows + row
                src_offs = peer_row.to(tl.int64) * hidden + offs_d
                acc += tl.load(src_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + row.to(tl.int64) * hidden + offs_d, acc, mask=mask)
        else:
            for peer in range(world):
                peer_row = peer * n_rows + row
                src_offs = peer_row.to(tl.int64) * hidden + offs_d
                acc += tl.load(src_ptr + src_offs).to(tl.float32)
            tl.store(out_ptr + row.to(tl.int64) * hidden + offs_d, acc)


def _fp32_peer_sum_launch_cfg(n_rows: int, hidden: int) -> tuple[int, int, int, int]:
    """Decode-graph launch: more CTAs at small n, grid-stride at 1024*topk."""
    if n_rows <= 32:
        block_d, num_warps, num_stages = 64, 2, 2
    elif n_rows <= 256:
        block_d, num_warps, num_stages = 128, 4, 2
    else:
        block_d, num_warps, num_stages = 256, 4, 3
    n_d = triton.cdiv(hidden, block_d)
    total = max(n_rows * n_d, 1)
    n_prog = total if n_rows <= 256 else min(total, _PEER_SUM_GRID_CAP)
    return block_d, num_warps, num_stages, n_prog


def fp32_peer_sum(
    src: torch.Tensor,
    world: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Sum peer-major ``[world * n_rows, D]`` bf16/fp32 into ``[n_rows, D]`` fp32.

    Loads are promoted to fp32; the All-to-All wire can be bf16. CUDA-graph
    safe: writes ``out``, no host sync, no autotune. Tuned for decode
    ``n_rows`` in ``1..1024*topk`` (topk 6/8).
    """
    if src.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"fp32_peer_sum src must be fp32/bf16, got {src.dtype}")
    if out.dtype != torch.float32:
        raise ValueError("fp32_peer_sum expects float32 out")
    if not src.is_contiguous() or not out.is_contiguous():
        raise ValueError("fp32_peer_sum requires contiguous src/out")
    n_rows, hidden = int(out.size(0)), int(out.size(1))
    if src.shape != (world * n_rows, hidden):
        raise ValueError(
            f"src shape {tuple(src.shape)} != {(world * n_rows, hidden)}"
        )
    if n_rows == 0:
        return out
    block_d, num_warps, num_stages, n_prog = _fp32_peer_sum_launch_cfg(
        n_rows, hidden
    )
    has_mask = (hidden % block_d) != 0
    if world in _PEER_SUM_WORLDS:
        _fp32_peer_sum_kernel[(n_prog,)](
            src,
            out,
            n_rows,
            hidden,
            n_prog,
            WORLD=world,
            BLOCK_D=block_d,
            HAS_MASK=has_mask,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _fp32_peer_sum_kernel_dyn[(n_prog,)](
            src,
            out,
            n_rows,
            hidden,
            n_prog,
            world,
            BLOCK_D=block_d,
            HAS_MASK=has_mask,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
