"""Triton scatter for KV-cache out[ids] = src.

Replaces:
    out = torch.zeros(total_q, H, D, dtype=src.dtype, device=src.device)
    out[ids] = src

Benchmark on B300, GLM-5 prefill shape (total_q=12288, N=6144, H=64, D=256, bf16):
    torch index_put_  : 554 us   (1.0x)
    triton_kv_scatter : 101 us   (5.5x, -453 us)
At total_q=32768/N=16384: torch 1458us → triton 252us (5.8x, -1206 us).

Algorithm:
- One fused kernel writes the entire `out` in a single pass: for each dst row,
  either copy src[n] (target) or write zeros (non-target).
- A prepared request-owned `rev[total_q]` int32 buffer can be reused across
  layers. Unprepared calls retain the generation-tagged lookup buffer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _populate_rev_kernel(
    ids_ptr, rev_ptr, gen, N, TAGGED: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    n_vals = offs.to(tl.int32)
    ids_vals = tl.load(ids_ptr + offs, mask=mask)
    if TAGGED:
        n_vals = (gen << 24) | (n_vals & 0xFFFFFF)
    tl.store(rev_ptr + ids_vals, n_vals, mask=mask)


@triton.jit
def _fused_scatter_kernel(
    out_ptr,
    src_ptr,
    rev_ptr,
    gen,
    TAGGED: tl.constexpr,
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
    if TAGGED:
        valid = ((src_idx >> 24) & 0xFF) == gen
        src_idx = src_idx & 0xFFFFFF
    else:
        valid = src_idx >= 0
    if valid:
        d = tl.load(src_ptr + src_idx.to(tl.int64) * HD + offs, mask=mask)
        tl.store(out_ptr + row_off + offs, d, mask=mask)
    else:
        z = tl.zeros([BLOCK_HD], dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_off + offs, z, mask=mask)


_state: dict = {}


def _next_gen(total_q: int, device: torch.device):
    key = (total_q, device.index)
    st = _state.get(key)
    if st is None:
        rev = torch.zeros((total_q,), dtype=torch.int32, device=device)
        st = {"gen": 0, "rev": rev}
        _state[key] = st
    st["gen"] += 1
    if st["gen"] >= 256:
        st["rev"].zero_()
        st["gen"] = 1
    return st["gen"], st["rev"]


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
            0,
            ids.numel(),
            TAGGED=False,
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
    until all consuming scatters finish. Without one, use the original shared
    generation-tagged workspace and its same-stream, fixed-graph-ids contract.
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
    tagged = reverse_indices is None
    if reverse_indices is None:
        gen, reverse_indices = _next_gen(total_q, src.device)
        ids_i64 = ids if ids.dtype == torch.int64 else ids.to(torch.int64)
        if N:
            _populate_rev_kernel[(triton.cdiv(N, 1024),)](
                ids_i64,
                reverse_indices,
                gen,
                N,
                TAGGED=True,
                BLOCK=1024,
                num_warps=4,
            )
    else:
        gen = 0
        assert reverse_indices.shape == (total_q,)
        assert reverse_indices.dtype == torch.int32
        assert reverse_indices.device == src.device
        assert reverse_indices.is_contiguous()

    if total_q == 0 or HD == 0:
        return out

    BLOCK_HD = triton.next_power_of_2(HD)
    if BLOCK_HD > 16384:
        BLOCK_HD = 4096
    num_warps = 8 if BLOCK_HD >= 8192 else 4
    n_hd = triton.cdiv(HD, BLOCK_HD)

    _fused_scatter_kernel[(total_q, n_hd)](
        out.view(total_q, HD),
        src_c.view(N, HD),
        reverse_indices,
        gen,
        TAGGED=tagged,
        HD=HD,
        BLOCK_HD=BLOCK_HD,
        num_warps=num_warps,
    )
    return out
