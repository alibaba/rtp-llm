from __future__ import annotations

import torch
import triton
import triton.language as tl

# Eager decode peer reduce: All-to-All returns [world, n_rows, D]
# bf16 (peer-major). Load to fp32, sum over world → [n_rows, D] fp32.
# n_rows is local tokens/rank; decode handles 1..1024, or 1..1024*topk.
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
    """Decode launch: more CTAs at small n, grid-stride at 1024*topk."""
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
        raise ValueError(f"src shape {tuple(src.shape)} != {(world * n_rows, hidden)}")
    if n_rows == 0:
        return out
    block_d, num_warps, num_stages, n_prog = _fp32_peer_sum_launch_cfg(n_rows, hidden)
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
