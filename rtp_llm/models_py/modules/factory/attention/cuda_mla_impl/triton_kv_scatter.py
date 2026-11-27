"""Scatter KV rows through a request-owned reverse map, zero-filling unused rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _populate_rev_kernel(ids_ptr, rev_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    n_vals = offs.to(tl.int32)
    ids_vals = tl.load(ids_ptr + offs, mask=mask)
    tl.store(rev_ptr + ids_vals, n_vals, mask=mask)


@triton.jit
def _fused_scatter_kernel(
    out_ptr,
    src_ptr,
    rev_ptr,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pid_blk = tl.program_id(1)
    offs = pid_blk * BLOCK_HD + tl.arange(0, BLOCK_HD)
    mask = offs < HD
    # Keep row in int64 before multiplying by HD: with HD=16384 and
    # total_q approaching 131k (e.g. long-prefill / large CP shards), the
    # int32 product overflows silently and writes to garbage addresses.
    row_off = row * HD
    src_idx = tl.load(rev_ptr + row)
    if src_idx >= 0:
        d = tl.load(src_ptr + src_idx.to(tl.int64) * HD + offs, mask=mask)
        tl.store(out_ptr + row_off + offs, d, mask=mask)
    else:
        z = tl.zeros([BLOCK_HD], dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_off + offs, z, mask=mask)


def prepare_kv_scatter_indices(ids: torch.Tensor, total_q: int) -> torch.Tensor:
    """Prepare a reverse map for unique ids in [0, total_q), with -1 for holes.

    The caller owns this tensor and must rebuild it whenever ids change. Graph
    callers copy a newly prepared map into their capture-stable plan buffer.
    Preparation and consumption follow normal PyTorch stream ordering rules.
    """
    assert ids.is_cuda and ids.dim() == 1
    assert ids.dtype in (torch.int32, torch.int64)
    reverse_indices = torch.full((total_q,), -1, dtype=torch.int32, device=ids.device)
    if ids.numel():
        _populate_rev_kernel[(triton.cdiv(ids.numel(), 1024),)](
            ids.contiguous(),
            reverse_indices,
            ids.numel(),
            BLOCK=1024,
            num_warps=4,
        )
    return reverse_indices


def triton_kv_scatter(
    src: torch.Tensor,  # [N, H, D]
    ids: torch.Tensor,  # [N], indices into [0, total_q)
    total_q: int,
    out: torch.Tensor | None = None,
    *,
    reverse_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Equivalent to:
    out = torch.zeros(total_q, *src.shape[1:], dtype=src.dtype, device=src.device)
    out[ids] = src
    return out

    A supplied reverse_indices must describe these ids and remain unchanged
    until all consuming scatters finish. When omitted, capture both clearing
    and filling the reverse map so replay can supply different ids.
    """
    assert src.is_cuda and ids.is_cuda
    assert src.device == ids.device
    assert src.dim() == 3
    N, H, D = src.shape
    assert ids.shape[0] == N
    HD = H * D
    src_c = src.contiguous()

    expected_shape = (total_q, H, D)
    if out is None:
        out = torch.empty(expected_shape, dtype=src.dtype, device=src.device)
    elif (
        tuple(out.shape) != expected_shape
        or out.dtype != src.dtype
        or out.device != src.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "triton_kv_scatter out must be a contiguous tensor matching src: "
            f"expected shape/dtype/device={expected_shape}/{src.dtype}/{src.device}, "
            f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
            f"contiguous={out.is_contiguous()}"
        )
    if total_q == 0 or HD == 0:
        return out
    if reverse_indices is None:
        reverse_indices = prepare_kv_scatter_indices(ids, total_q)
    else:
        assert reverse_indices.shape == (total_q,)
        assert reverse_indices.dtype == torch.int32
        assert reverse_indices.device == src.device
        assert reverse_indices.is_contiguous()

    BLOCK_HD = triton.next_power_of_2(HD)
    if BLOCK_HD > 16384:
        BLOCK_HD = 4096
    num_warps = 8 if BLOCK_HD >= 8192 else 4
    n_hd = triton.cdiv(HD, BLOCK_HD)

    _fused_scatter_kernel[(total_q, n_hd)](
        out.view(total_q, HD),
        src_c.view(N, HD),
        reverse_indices,
        HD=HD,
        BLOCK_HD=BLOCK_HD,
        num_warps=num_warps,
    )
    return out
