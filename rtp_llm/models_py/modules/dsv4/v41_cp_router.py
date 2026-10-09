"""Preserve CP1 router batch arithmetic in short maintained V4.1 CP4 prefills."""

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
from rtp_llm.models_py.modules.dsv4.cp import CPContext, cp_all_gather_full


def canonical_v41_cp_routes(gate, x, cp):
    """Run the existing gate on restored real rows, then select this CP lane."""
    if (
        not isinstance(cp, CPContext)
        or cp.cp_size != 4
        or cp.kv_cache_sharded
        or not gate.fp32_logits
        or gate.hash
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.shape != (cp.chunk_length, 5120)
        or not x.is_contiguous()
        or torch.cuda.get_device_capability(x.device) != (12, 0)
        or torch.cuda.is_current_stream_capturing()
        or not torch.distributed.is_initialized()
    ):
        raise NotImplementedError("V4.1 CP routing requires eager SM120 unsharded CP4")
    group = _get_group(Group.TP)
    if (
        group is None
        or torch.distributed.get_world_size(group) != 4
        or torch.distributed.get_rank(group) != cp.cp_rank
        or torch.distributed.get_backend(group) != "nccl"
    ):
        raise ValueError("V4.1 CP router disagrees with its registered group")
    lengths, prefixes = cp.input_lengths_global_host, cp.prefix_lengths_host
    if (
        not isinstance(lengths, tuple)
        or not isinstance(prefixes, tuple)
        or len(lengths) != len(prefixes)
        or not lengths
        or any(n <= 0 or p < 0 or n + p > 512 for n, p in zip(lengths, prefixes))
        or sum(lengths) != cp.seq_len_full
        or cp.padded_seq_len != 4 * cp.chunk_length
    ):
        raise ValueError("V4.1 CP router requires current bounded request lengths")
    for name, shape, dtypes in (
        ("req_id_per_token", (cp.chunk_length,), (torch.int32,)),
        ("global_positions", (cp.chunk_length,), (torch.int64,)),
        ("prefix_lengths", (len(lengths),), (torch.int32, torch.int64)),
        ("cu_seqlens_global", (len(lengths) + 1,), (torch.int32, torch.int64)),
        ("unpad_restore", (cp.seq_len_full,), (torch.int64,)),
    ):
        value = getattr(cp, name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype not in dtypes
            or value.device != x.device
            or not value.is_contiguous()
        ):
            raise ValueError(f"Invalid V4.1 CP router {name}")
    requests = cp.req_id_per_token.long()
    rows = (
        cp.cu_seqlens_global[:-1].index_select(0, requests)
        + cp.global_positions
        - cp.prefix_lengths.index_select(0, requests)
    ).long()
    full = cp_all_gather_full(x, cp, profile_name="dsv41.moe.router.cp1_batch")
    # V4.1 has no token-ID hash routing. Padding is mapped to its clamped real
    # request row; the existing CP restoration discards its eventual output.
    weights, indices = gate._forward_local(full, None)
    return weights.index_select(0, rows), indices.index_select(0, rows)
