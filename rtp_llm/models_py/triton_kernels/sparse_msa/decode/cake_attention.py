"""Cake sparse decode over RTP's unchanged packed NVFP4 cache ABI.

The caller owns one :class:`CakeScaleWorkspace` per CUDA stream and physical
pool, shared by all layers and graph buckets. Only selected physical page/head
pairs are converted on each invocation; their values are never dequantized.
Bucket workspaces own stable metadata/output addresses. Captured graphs sharing
these buffers must be replayed sequentially; a replay on another stream requires
the caller's usual explicit ordering, just like the existing decode workspaces.
Python preparation/launch checks the workspace's eager/capture stream owner.
Warm Cake's compiled programs before capture;
binding a new bucket/layer runner on the capture stream does not compile them.
"""

from dataclasses import dataclass, field
import math

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import NVFP4CacheLayout


@triton.jit
def _prepare_metadata(Q, TOPK, TABLE, LENS, MASK, Q_OUT, TOPK_OUT, TABLE_OUT,
                      LENS_OUT, QS0, QS1, TS0, TS1, KS0, KS1, KS2, LS, MS,
                      ROWS: tl.constexpr, COLS: tl.constexpr,
                      PHYSICAL: tl.constexpr, HAS_MASK: tl.constexpr,
                      COPY_Q: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    active = True
    if HAS_MASK:
        active = tl.load(MASK + row * MS)
    length = tl.load(LENS + row * LS).to(tl.int64)
    length = tl.where(active, tl.minimum(tl.maximum(length, 0), COLS * 128), 0)
    if head == 0:
        tl.store(LENS_OUT + row, length.to(tl.int32))
        col = tl.arange(0, BLOCK)
        physical = tl.load(TABLE + row * TS0 + col * TS1,
                           mask=col < COLS, other=-1).to(tl.int64)
        valid = active & (col < (length + 127) // 128)
        valid = valid & (physical >= 0) & (physical < PHYSICAL)
        tl.store(TABLE_OUT + row * COLS + col,
                 tl.where(valid, physical, -1).to(tl.int32), mask=col < COLS)
    if COPY_Q:
        group = tl.arange(0, 16)[:, None]
        dim = tl.arange(0, 128)[None, :]
        q = tl.load(Q + row * QS0 + (head * 16 + group) * QS1 + dim)
        tl.store(Q_OUT + (row * 64 + head * 16 + group) * 128 + dim,
                 tl.where(active & (length > 0), q, 0))
    slot = tl.arange(0, 16)
    logical = tl.load(TOPK + head * KS0 + row * KS1 + slot * KS2).to(tl.int64)
    valid = active & (logical >= 0) & (logical < COLS)
    valid = valid & (logical < (length + 127) // 128)
    physical = tl.load(TABLE + row * TS0 + tl.where(valid, logical, 0) * TS1,
                       mask=valid, other=-1).to(tl.int64)
    valid = valid & (physical >= 0) & (physical < PHYSICAL)
    ordered = tl.sort(tl.where(valid, logical, 2147483647).to(tl.int32),
                      descending=False)
    tl.store(TOPK_OUT + (head * ROWS + row) * 16 + slot,
             tl.where(ordered == 2147483647, -1, ordered))


@triton.jit
def _selected_valid_prefix(TOPK, TABLE, LENS, SEEN,
                           ROWS: tl.constexpr, COLS: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.arange(0, 16)
    logical = tl.load(TOPK + (head * ROWS + row) * 16 + slot)
    valid = logical >= 0
    page = tl.load(TABLE + row.to(tl.int64) * COLS + tl.where(valid, logical, 0),
                   mask=valid, other=0).to(tl.int64)
    length = tl.load(LENS + row)
    prefix = tl.minimum(128, tl.maximum(0, length - logical * 128))
    # A physical page may be shared by rows with different visible lengths.
    # Preserve every token valid for ANY reader; zero only the global invalid tail.
    # Elect an immutable owner together with the longest visible prefix.
    # A claim/CAS inside the copy kernel can change scalar ownership while
    # other warps in the same CTA are still loading it, producing partial pages.
    owner = row.to(tl.int64) * 16 + slot + 1
    election = (prefix.to(tl.int64) << 32) | owner
    tl.atomic_max(SEEN + page * 4 + head, election, mask=valid, sem="relaxed")


@triton.jit
def _refresh_selected_scales(TOPK, TABLE, SEEN, K_SRC, V_SRC, K_DST, V_DST,
                             K_STRIDE, V_STRIDE, ROWS: tl.constexpr,
                             COLS: tl.constexpr, MMA: tl.constexpr):
    selection = tl.program_id(0)
    row = selection // 16
    slot = selection % 16
    head = tl.program_id(1)
    logical = tl.load(TOPK + (head * ROWS + row) * 16 + slot)
    if logical >= 0:
        page = tl.load(TABLE + row.to(tl.int64) * COLS + logical).to(tl.int64)
        # The previous kernel completed the election. No CTA mutates it here,
        # so every warp observes the same owner and copies the entire page.
        election = tl.load(SEEN + page * 4 + head)
        prefix = election >> 32
        owner = election & 0xFFFFFFFF
        if (prefix > 0) & (owner == selection.to(tl.int64) + 1):
            token = tl.arange(0, 128)[:, None]
            group = tl.arange(0, 8)[None, :]
            if MMA:
                offset = ((group // 4) * 512 + (token % 32) * 16
                          + (token // 32) * 4 + group % 4)
            else:
                offset = token * 8 + group
            k = tl.load(K_SRC + page * K_STRIDE + head * 1024 + offset)
            v = tl.load(V_SRC + page * V_STRIDE + head * 1024 + offset)
            # Masked probabilities multiplied by an uninitialized NaN V scale
            # would still yield NaN. These are invalid tokens, not valid NaNs.
            k = tl.where(token < prefix, k, 0)
            v = tl.where(token < prefix, v, 0)
            tl.store(K_DST + page * 4096 + head * 1024 + token * 8 + group, k)
            swz_token = (token // 4) * 4 + group // 2
            swz_group = (group % 2) * 4 + token % 4
            tl.store(V_DST + page * 4096 + head * 1024
                     + swz_token * 8 + swz_group, v)


@triton.jit
def _mask_output(OUT, LSE, MASK, LENS, MS, HAS_MASK: tl.constexpr):
    row = tl.program_id(0)
    active = tl.load(LENS + row) > 0
    if HAS_MASK:
        active = active & tl.load(MASK + row * MS)
    if not active:
        offset = tl.arange(0, 8192)
        tl.store(OUT + row.to(tl.int64) * 8192 + offset, 0)
        head = tl.arange(0, 64)
        tl.store(LSE + row.to(tl.int64) * 64 + head, -float("inf"))


def _stream(device):
    return torch.cuda.current_stream(device).cuda_stream


@dataclass
class CakeScaleWorkspace:
    """8192 scale bytes plus 32 election bytes per page, shared by buckets.

    Allocation is independent of layer, batch, logical history, and graph bucket.
    Never concurrently use a workspace from different streams or graph replays.
    """

    k: torch.Tensor
    v: torch.Tensor
    seen: torch.Tensor
    owner_stream: int

    @classmethod
    def create(cls, device, physical_pages):
        if physical_pages <= 0 or physical_pages >= 2**31:
            raise ValueError("Cake requires a positive int32-addressable page count")
        shape = (physical_pages, 4, 128, 8)
        return cls(torch.empty(shape, dtype=torch.uint8, device=device),
                   torch.empty(shape, dtype=torch.uint8, device=device),
                   torch.empty((physical_pages, 4), dtype=torch.int64, device=device),
                   _stream(device))

    @property
    def nbytes(self):
        return self.k.numel() + self.v.numel() + self.seen.numel() * 8


def _validate(q, layout, table, topk, lens, mask=None):
    if q.dtype != torch.bfloat16 or q.ndim != 3 or tuple(q.shape[1:]) != (64, 128):
        raise ValueError("Cake requires original BF16 Q [rows,64,128]")
    if not q.is_cuda or q.stride(2) != 1 or q.shape[0] <= 0:
        raise ValueError("Cake Q must be CUDA with contiguous head dimensions")
    if (layout.num_heads, layout.page_size, layout.head_dim) != (4, 128, 128):
        raise ValueError("Cake requires Hkv4/page128/D128")
    rows = q.shape[0]
    if table.ndim != 2 or table.shape[0] != rows or not 0 < table.shape[1] <= 8192:
        raise ValueError("Cake requires table [rows,1..8192]")
    if table.dtype not in (torch.int32, torch.int64):
        raise ValueError("Cake table must be int32/int64")
    if topk.shape != (4, rows, 16) or topk.dtype != torch.int32:
        raise ValueError("Cake requires int32 TopK [4,rows,16]")
    if lens.shape != (rows,) or lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Cake requires int32/int64 lengths [rows]")
    if mask is not None and (mask.shape != (rows,) or mask.dtype != torch.bool):
        raise ValueError("Cake requires bool valid_token_mask [rows]")
    tensors = [table, topk, lens, layout.packed_main, layout.main_scales]
    if mask is not None:
        tensors.append(mask)
    if any(t.device != q.device for t in tensors):
        raise ValueError("Cake operands must share Q's CUDA device")


@dataclass
class CakeAttentionWorkspace:
    """Stable per-bucket buffers using caller-owned cross-layer scale scratch."""

    scales: CakeScaleWorkspace
    q: torch.Tensor
    topk: torch.Tensor
    table: torch.Tensor
    lens: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    runners: dict = field(default_factory=dict)

    @classmethod
    def create(cls, q, layout, block_table, topk_indices, seq_lens, *, scale_workspace):
        _validate(q, layout, block_table, topk_indices, seq_lens)
        if (scale_workspace.k.device != q.device or
                scale_workspace.k.shape[0] != layout.num_blocks):
            raise ValueError("Cake scale workspace does not match physical pool")
        if scale_workspace.owner_stream != _stream(q.device):
            raise ValueError("Cake scale workspace belongs to another CUDA stream")
        rows = q.shape[0]
        return cls(scale_workspace,
                   torch.empty(q.shape, dtype=torch.bfloat16, device=q.device),
                   torch.empty(topk_indices.shape, dtype=torch.int32, device=q.device),
                   torch.empty(block_table.shape, dtype=torch.int32, device=q.device),
                   torch.empty((rows,), dtype=torch.int32, device=q.device),
                   torch.empty(q.shape, dtype=torch.bfloat16, device=q.device),
                   torch.empty((rows, 64), dtype=torch.float32, device=q.device))


@torch.no_grad()
def cake_paged_sparse_decode(q, layout: NVFP4CacheLayout, block_table,
                             topk_indices, seq_lens, *, workspace,
                             mma_scale_layout=True, valid_token_mask=None,
                             sm_scale=None):
    """Consume sorted logical TopK and original BF16 Q; return stable BF16 output.

    Invalid logical/physical IDs and padded rows are removed before Cake runs.
    Valid scale bytes (including NaNs) are preserved, rather than silently
    sanitizing cache corruption. Globally invalid tail tokens get zero scales:
    a masked probability times a NaN V operand otherwise propagates NaN.
    The writer owns valid-token scale initialization.
    ``workspace`` and its scales must outlive every captured graph using them.
    """
    _validate(q, layout, block_table, topk_indices, seq_lens, valid_token_mask)
    ws = workspace
    if (ws.q.shape != q.shape or ws.table.shape != block_table.shape or
            ws.scales.k.shape[0] != layout.num_blocks or ws.q.device != q.device):
        raise ValueError("Cake bucket workspace geometry does not match")
    if ws.scales.owner_stream != _stream(q.device):
        raise ValueError("Cake workspace belongs to another CUDA stream")
    scale = 128**-0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Cake sm_scale must be finite and positive")
    kp, ks = layout.main_plane(0)
    vp, vs = layout.main_plane(1)
    k = kp.view(layout.num_blocks, 4, 128, 64)
    v = vp.view(layout.num_blocks, 4, 128, 64)
    ks, vs = ks.view(torch.uint8), vs.view(torch.uint8)
    # Each Cake work item owns one query row / KV head. Masked NaN queries do
    # not contaminate other rows: they select no pages and output is zeroed.
    # Graph-mutation tests compare this alias against the strided-Q copy path.
    direct_q = q.is_contiguous()
    cake_q = q if direct_q else ws.q
    # Capture retains its bound tensors; eager Q allocations must not become
    # permanent cache entries. A temporary eager runner preserves zero-copy Q
    # without retaining a new Q tensor on every fallback step.
    capturing = torch.cuda.is_current_stream_capturing()
    key = (cake_q.data_ptr(), k.data_ptr(), v.data_ptr(), tuple(k.stride()),
           tuple(v.stride()), scale)
    runner = ws.runners.get(key) if capturing else None
    if runner is None:
        from flashinfer.msa_ops import prepare_msa_nvfp4_sparse_decode
        runner = prepare_msa_nvfp4_sparse_decode(
            cake_q, k, v, ws.topk, k_scale=ws.scales.k, v_scale=ws.scales.v,
            page_table=ws.table, seqused_k=ws.lens, k_global_scale=1.0,
            v_global_scale=1.0, seqlen_q=1, softmax_scale=scale,
            out=ws.output, lse=ws.lse)
        if capturing:
            ws.runners[key] = runner
    rows, cols = q.shape[0], block_table.shape[1]
    _prepare_metadata[(rows, 4)](
        q, topk_indices, block_table, seq_lens, valid_token_mask,
        ws.q, ws.topk, ws.table, ws.lens, q.stride(0), q.stride(1),
        block_table.stride(0), block_table.stride(1), topk_indices.stride(0),
        topk_indices.stride(1), topk_indices.stride(2), seq_lens.stride(0),
        valid_token_mask.stride(0) if valid_token_mask is not None else 0,
        ROWS=rows, COLS=cols, PHYSICAL=layout.num_blocks,
        HAS_MASK=valid_token_mask is not None, COPY_Q=not direct_q,
        BLOCK=triton.next_power_of_2(cols), num_warps=4)
    ws.scales.seen.zero_()
    _selected_valid_prefix[(rows, 4)](
        ws.topk, ws.table, ws.lens, ws.scales.seen,
        ROWS=rows, COLS=cols, num_warps=4)
    _refresh_selected_scales[(rows * 16, 4)](
        ws.topk, ws.table, ws.scales.seen, ks, vs, ws.scales.k, ws.scales.v,
        ks.stride(0), vs.stride(0), ROWS=rows, COLS=cols,
        MMA=mma_scale_layout, num_warps=4)
    runner()
    _mask_output[(rows,)](ws.output, ws.lse, valid_token_mask, ws.lens,
                         valid_token_mask.stride(0) if valid_token_mask is not None else 0,
                         HAS_MASK=valid_token_mask is not None, num_warps=4)
    return ws.output
