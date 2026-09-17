"""Build all request positions in two launches, including unpadding restore."""

import torch
import triton
import triton.language as tl


@triton.jit
def _local_positions(
    D,
    Prefix,
    Padding,
    Relative,
    Global,
    Req,
    Valid,
    cp_size,
    cp_rank,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(D + req * 5)
    length = tl.load(D + req * 5 + 1)
    real_length = tl.load(D + req * 5 + 3)
    padded_start = tl.load(D + req * 5 + 4)
    half = length // 2
    pos = tl.where(
        t < half, cp_rank * half + t, length * cp_size - (cp_rank + 1) * half + t - half
    )
    mask = t < length
    padded = padded_start + pos
    valid = tl.load(Padding + padded, mask=mask, other=0) == 1
    absolute = tl.load(Prefix + req) + tl.minimum(pos, tl.maximum(real_length - 1, 0))
    tl.store(Relative + start + t, padded, mask)
    tl.store(Global + start + t, absolute, mask)
    tl.store(Req + start + t, req, mask)
    tl.store(Valid + start + t, valid, mask)


@triton.jit
def _full_positions(D, Prefix, Restore, FullPos, FullReq, Unpad, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(D + req * 5 + 2)
    length = tl.load(D + req * 5 + 3)
    padded_start = tl.load(D + req * 5 + 4)
    mask = t < length
    restored = tl.load(Restore + padded_start + t, mask=mask, other=0)
    tl.store(FullPos + start + t, tl.load(Prefix + req) + t, mask)
    tl.store(FullReq + start + t, req, mask)
    tl.store(Unpad + start + t, restored, mask)


def build_fused_cp_context(
    cp_info, cp_size, cp_rank, chunk_length, device, position_offset, kv_cache_sharded
):
    from rtp_llm.models_py.modules.dsv4.cp import CPContext

    lengths = [int(v) for v in cp_info.prefill_actual_input_lengths_cpu.tolist()]
    chunks = [int(v) for v in cp_info.prefill_cp_chunk_lengths.cpu().tolist()]
    if len(lengths) != len(chunks) or sum(chunks) != chunk_length:
        raise ValueError("CP request lengths do not cover the rank-local input")
    if any(c < 0 or c % 2 or n < 0 or n > c * cp_size for n, c in zip(lengths, chunks)):
        raise ValueError("Invalid per-request zigzag CP geometry")
    padding = cp_info.prefill_qkv_padding_mask.to(device=device)
    restore = cp_info.prefill_qkv_restore_indice.to(device=device)
    if padding.numel() != cp_size * chunk_length:
        raise ValueError("CP padding length does not match the input")
    batch = len(lengths)
    if isinstance(position_offset, torch.Tensor):
        prefix = position_offset.to(device=device, dtype=torch.long).reshape(-1)
        if prefix.numel() == 1 and batch > 1:
            prefix = prefix.expand(batch)
        prefix = prefix[:batch].contiguous()
        prefix_host = prefix.cpu().tolist()
    else:
        prefix_host = [int(position_offset)] * batch
        prefix = torch.tensor(prefix_host, dtype=torch.long, device=device)
    if len(prefix_host) != batch:
        raise ValueError("CP prefixes must cover every request")
    desc = []
    local_start = real_start = padded_start = 0
    cu = [0]
    for n, c in zip(lengths, chunks):
        desc.append((local_start, c, real_start, n, padded_start))
        local_start += c
        real_start += n
        padded_start += c * cp_size
        cu.append(real_start)
    descriptors = torch.tensor(desc, device=device, dtype=torch.int64)
    relative = torch.empty(chunk_length, dtype=torch.int64, device=device)
    positions = torch.empty_like(relative)
    req = torch.empty(chunk_length, dtype=torch.int32, device=device)
    valid = torch.empty(chunk_length, dtype=torch.bool, device=device)
    full_pos = torch.empty(real_start, dtype=torch.int64, device=device)
    full_req = torch.empty_like(full_pos)
    unpad = torch.empty_like(full_pos)
    if chunk_length:
        _local_positions[(batch, triton.cdiv(max(chunks), 256))](
            descriptors,
            prefix,
            padding,
            relative,
            positions,
            req,
            valid,
            cp_size,
            cp_rank,
            BLOCK=256,
        )
    if real_start:
        _full_positions[(batch, triton.cdiv(max(lengths), 256))](
            descriptors, prefix, restore, full_pos, full_req, unpad, BLOCK=256
        )
    lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    cu_global = torch.tensor(cu, device=device, dtype=torch.int32)
    ctx = CPContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        chunk_length=chunk_length,
        padded_seq_len=padded_start,
        seq_len_full=real_start,
        relative_positions=relative,
        prefix_length=prefix_host[0] if batch else 0,
        global_positions=positions,
        local_is_real=valid,
        unpad_restore=unpad,
        seq_len_total=max((p + n for p, n in zip(prefix_host, lengths)), default=0),
        cp_info=cp_info,
        req_id_per_token=req,
        prefix_lengths=prefix,
        input_lengths_global=lens,
        cu_seqlens_global=cu_global,
        chunk_lengths_per_req=tuple(chunks),
        kv_cache_sharded=bool(kv_cache_sharded),
    )
    ctx.full_prefill_positions = (
        full_pos,
        full_req,
        prefix.to(torch.int32),
        cu_global.to(torch.int64),
    )
    local_cu = [0]
    for c in chunks:
        local_cu.append(local_cu[-1] + c)
    ctx.local_cu_seqlens = torch.tensor(local_cu, device=device, dtype=torch.int32)
    return ctx
