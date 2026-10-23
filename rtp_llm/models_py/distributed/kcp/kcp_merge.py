"""Read BF16 B/M in physical rank slots; carry recurrence in FP32."""
from functools import lru_cache
import torch
import triton
import triton.language as tl


@triton.jit
def _merge_physical(HM, H0, OUT, PHYSICAL, STARTS, COUNTS, OFFSETS, SEQS,
                    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    group = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.arange(0, K)
    cols = tl.program_id(2) * BV + tl.arange(0, BV)
    seq = tl.load(SEQS + group).to(tl.int64)
    start = tl.load(STARTS + group)
    count = tl.load(COUNTS + group)
    offset = tl.load(OFFSETS + group)
    state = tl.load(H0 + ((seq * H + head) * K + rows[:, None]) * V + cols[None, :]).to(tl.float32)
    # No global chronological B/M copy. The plan maps each logical segment to
    # its producer's physical [rank, local_segment] slot in the gather buffer.
    for j in range(count - 1):
        physical = tl.load(PHYSICAL + start + j).to(tl.int64)
        base = HM + (physical * H + head) * K * (V + K)
        transition = tl.load(base + rows[:, None] * (V + K) + V + rows[None, :])
        after = tl.load(base + rows[:, None] * (V + K) + cols[None, :]).to(tl.float32)
        high = state.to(tl.bfloat16)
        low = (state - high.to(tl.float32)).to(tl.bfloat16)
        product = tl.dot(transition, high, out_dtype=tl.float32)
        state = after + tl.dot(transition, low, product, out_dtype=tl.float32)
        tl.store(OUT + (((offset + j).to(tl.int64) * H + head) * K + rows[:, None]) * V
                 + cols[None, :], state)


@lru_cache(maxsize=128)
def _metadata(starts, counts, offsets, seqs, device_index, stream):
    return tuple(torch.tensor(x, dtype=torch.int32, device=torch.device('cuda', device_index))
                 for x in (starts, counts, offsets, seqs))


def launch_merge_physical(hm, physical, seq_starts, seq_counts, init_offsets,
                          split_seq_ids, h0, num_non_first):
    assert hm.dtype == torch.bfloat16 and hm.is_contiguous()
    assert h0.dtype == torch.float32 and h0.is_contiguous()
    _, heads, dim, packed_dim = hm.shape
    value_dim = packed_dim - dim
    assert dim == 128 and value_dim == 128
    out = h0.new_empty(num_non_first, heads, dim, value_dim)
    metadata = _metadata(tuple(seq_starts), tuple(seq_counts), tuple(init_offsets),
                         tuple(split_seq_ids), hm.device.index,
                         int(torch.cuda.current_stream(hm.device).cuda_stream))
    _merge_physical[(len(split_seq_ids), heads, value_dim // 64)](
        hm, h0, out, physical, *metadata, H=heads, K=dim, V=value_dim, BV=64,
        num_warps=4, num_stages=2)
    return out


@triton.jit
def _merge_local(HM, H0, OUT, PHYSICAL, STARTS, COUNTS, SEQS,
                 H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
                 RANK: tl.constexpr, PADDED: tl.constexpr, RESIDUAL: tl.constexpr):
    group, head = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, K)
    cols = tl.program_id(2) * BV + tl.arange(0, BV)
    seq = tl.load(SEQS + group).to(tl.int64)
    start, count = tl.load(STARTS + group), tl.load(COUNTS + group)
    state = tl.load(H0 + ((seq * H + head) * K + rows[:, None]) * V + cols[None, :]).to(tl.float32)
    for j in range(count):
        physical = tl.load(PHYSICAL + start + j).to(tl.int64)
        if physical // PADDED == RANK:
            local = physical % PADDED
            tl.store(OUT + ((local * H + head) * K + rows[:, None]) * V + cols[None, :], state)
        if j < count - 1:
            base = HM + (physical * H + head) * K * (V + K)
            transition = tl.load(base + rows[:, None] * (V + K) + V + rows[None, :])
            after = tl.load(base + rows[:, None] * (V + K) + cols[None, :]).to(tl.float32)
            high = state.to(tl.bfloat16)
            low = (state - high.to(tl.float32)).to(tl.bfloat16)
            product = tl.dot(transition, high, out_dtype=tl.float32)
            product = tl.dot(transition, low, product, out_dtype=tl.float32)
            state = after + product + state if RESIDUAL else after + product


def launch_merge_local(hm, physical, seq_starts, seq_counts, split_seq_ids,
                       h0, rank, padded_segments, local_segments, residual=False):
    """Publish only this rank's segment initial states, already in scan order."""
    assert hm.dtype == torch.bfloat16 and hm.is_contiguous()
    assert h0.dtype == torch.float32 and h0.is_contiguous()
    _, heads, dim, packed_dim = hm.shape
    value_dim = packed_dim - dim
    assert dim == value_dim == 128
    out = h0.new_empty(local_segments, heads, dim, value_dim)
    starts, counts, _, seqs = _metadata(tuple(seq_starts), tuple(seq_counts),
        (0,)*len(split_seq_ids), tuple(split_seq_ids), hm.device.index,
        int(torch.cuda.current_stream(hm.device).cuda_stream))
    _merge_local[(len(split_seq_ids), heads, value_dim//64)](
        hm,h0,out,physical,starts,counts,seqs,H=heads,K=dim,V=value_dim,BV=64,
        RANK=rank,PADDED=padded_segments,RESIDUAL=residual,num_warps=4,num_stages=2)
    return out
