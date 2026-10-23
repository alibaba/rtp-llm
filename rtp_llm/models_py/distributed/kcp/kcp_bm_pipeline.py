"""Persistent BF16 summary exchange with direct cuLA stores into rank slots.

The prescan supports strided head views and retains FP32 accumulation. Buffer
growth, zero padding and communication remain inside the caller's timed scope.
"""
from contextlib import contextmanager
import os

import torch
import torch.distributed as dist


@contextmanager
def scope(name):
    with torch.autograd.profiler.record_function(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


# One buffer per communicator/device/stream/dtype, shared by all KDA layers.
# Use the caller stream for every producer, gather wait and merge consumer.
# This guarantees a later producer cannot overwrite an unfinished merge.
_scratch = {}


def scratch_buffer(group, shape, dtype, device, key):
    import logging, json, math
    numel = math.prod(shape)
    cache_key = (id(group), device.index, int(torch.cuda.current_stream(device).cuda_stream), key, dtype)
    flat = _scratch.get(cache_key)
    if flat is None or flat.numel() < numel:
        if os.environ.get('GLM53_KCP_COMPACT_SCRATCH', '0') == '1':
            element_size = dtype.itemsize
            # CUDA expandable segments map large allocations in 2 MiB pages.
            # Round to pages, avoiding a 1.06 GiB checkpoint growing to 2 GiB.
            page_elements = (2 * 2**20) // element_size
            capacity = ((numel + page_elements - 1) // page_elements) * page_elements
        else:
            capacity = 1 << max(numel - 1, 0).bit_length()
        flat = torch.empty(capacity, dtype=dtype, device=device)
        _scratch[cache_key] = flat
        logging.info('KCP_SCRATCH %s', json.dumps({
            'key':key, 'dtype':str(dtype), 'required_bytes':numel*flat.element_size(),
            'capacity_bytes':capacity*flat.element_size(), 'stream':cache_key[2]}))
    return flat[:numel].view(shape)


def gather_buffer(group, local_shape, dtype, device, key):
    return scratch_buffer(group, (dist.get_world_size(group), *local_shape), dtype, device, key)


def summary_allgather(k, w, u, gk, cu_seqlens, *, group=None, groups=1,
                      overlap=False, last_rank_skips=False, pad_sequences_to=None,
                      strided=False):
    from .cula_prescan_strided import chunk_delta_rule_pre_scan
    if groups != 1 or overlap or last_rank_skips:
        raise ValueError('BF16 physical-slot path uses one summary/gather with every rank participating')
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    nseq = cu_seqlens.numel() - 1
    padded = nseq if pad_sequences_to is None else pad_sequences_to
    if padded < nseq:
        raise ValueError('Gather capacity must cover every local segment')
    heads, dim, value_dim = k.shape[2], k.shape[3], u.shape[3]
    gathered = gather_buffer(group, (padded, heads, dim, value_dim + dim),
                             torch.bfloat16, k.device, 'bm_summary')
    local = gathered[rank]
    with scope('kcp_bm/prescan_direct_bf16'):
        if padded != nseq:
            local[nseq:].zero_()
        chunk_delta_rule_pre_scan(k=k, w=w, u=u, gk=gk,
            cu_seqlens_split=cu_seqlens, S_split=nseq, chunk_size=64,
            out=local[:nseq], identity_residual=True)
    with scope('kcp_bm/inplace_allgather'):
        # NCCL supports in-place AG when send is this rank's receive region.
        # PyTorch supplies the same storage and exact rank offset to NCCL.
        work = dist.all_gather_into_tensor(gathered.flatten(0, 1), local,
                                          group=group, async_op=True)
        work.wait()
    return gathered
