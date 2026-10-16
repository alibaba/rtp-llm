"""Metadata shared only within one model forward, never across engine steps."""

import os
from contextvars import ContextVar
from functools import wraps

_FORWARD_METADATA = ContextVar("glm53_forward_metadata", default=None)
_SCORE_WORKSPACE_MB = 8192
_SCORE_WORKSPACE_DEMAND = os.environ.get("GLM53_DEMAND_SCORE_BUDGET", "0") == "1"


def metadata_cache():
    return _FORWARD_METADATA.get()


def score_workspace_fits(device, required_bytes):
    """Prove an unchunked score fits the original budget without a CUDA query."""
    import torch

    cache = metadata_cache()
    if cache is None:
        return False
    device = torch.device(device)
    bound = cache.get(
        ("indexer_score_budget", device),
        cache.get(("indexer_score_budget_lower", device), 0),
    )
    return 0 < required_bytes <= bound


def _reusable_allocator_bytes(device):
    import torch

    # Read one coherent allocator snapshot. The scalar convenience functions
    # each flatten and sort the entire statistics tree before selecting a key.
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    reserved = stats.get("reserved_bytes", {}).get("all", {}).get("current", 0)
    allocated = stats.get("allocated_bytes", {}).get("all", {}).get("current", 0)
    return max(0, reserved - allocated)


def score_workspace_budget(device):
    import torch

    cache = metadata_cache()
    key = ("indexer_score_budget", torch.device(device))
    if cache is not None and key in cache:
        return cache[key]
    reusable = _reusable_allocator_bytes(device)
    cap = _SCORE_WORKSPACE_MB * 1024 * 1024
    if reusable >= 2 * cap:
        # The existing formula already saturates at the cap even with zero
        # driver-free bytes. Allocator counters suffice; avoid a CUDA query.
        budget = cap
    else:
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(cap, (free + reusable) // 2)
    if cache is not None:
        cache[key] = budget
    return budget


def tensor_key(tensor):
    if tensor is None:
        return None
    # The forward owns the inputs; pointers cannot be reused while they are live.
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def scoped_forward_metadata(function):
    @wraps(function)
    def wrapped(self, inputs, *args, **kwargs):
        config = getattr(self, "config", None)
        is_glm53 = getattr(config, "model_type", None) == "glm5_3_flash" or getattr(
            config, "is_glm53_mtp", False
        )
        if (
            not is_glm53
            or not inputs.attention_inputs.is_prefill
            or getattr(inputs.attention_inputs, "is_target_verify", False)
        ):
            return function(self, inputs, *args, **kwargs)
        token = _FORWARD_METADATA.set({})
        try:
            import torch

            # Query before this forward queues its projection/collective
            # work: cudaMemGetInfo can otherwise stall the CPU mid-layer.
            with torch.profiler.record_function("glm53.forward_workspace_budget"):
                device = torch.device("cuda", torch.cuda.current_device())
                if _SCORE_WORKSPACE_DEMAND:
                    # free >= 0, so this is a lower bound on the original
                    # min(cap, (free + reusable) // 2). It expires this forward.
                    reusable = _reusable_allocator_bytes(device)
                    metadata_cache()[("indexer_score_budget_lower", device)] = min(
                        _SCORE_WORKSPACE_MB << 20, reusable // 2
                    )
                else:
                    score_workspace_budget(device)
            return function(self, inputs, *args, **kwargs)
        finally:
            # Also release metadata on exceptions and when the same inputs object
            # is reused by the executor for another forward.
            _FORWARD_METADATA.reset(token)

    return wrapped


def cached_cp_context(function):
    @wraps(function)
    def wrapped(
        cp_info,
        cp_size,
        cp_rank,
        chunk_length,
        device,
        position_offset=0,
        kv_cache_sharded=False,
    ):
        cache = metadata_cache()
        if cache is None:
            return function(
                cp_info,
                cp_size,
                cp_rank,
                chunk_length,
                device,
                position_offset,
                kv_cache_sharded,
            )
        key = (
            "cp_context",
            tensor_key(cp_info.prefill_qkv_padding_mask),
            tensor_key(cp_info.prefill_qkv_restore_indice),
            tensor_key(getattr(cp_info, "prefill_actual_input_lengths_cpu", None)),
            tensor_key(getattr(cp_info, "prefill_cp_chunk_lengths", None)),
            cp_size,
            cp_rank,
            chunk_length,
            str(device),
            (
                tensor_key(position_offset)
                if hasattr(position_offset, "data_ptr")
                else int(position_offset)
            ),
            bool(kv_cache_sharded),
        )
        if key not in cache:
            cache[key] = function(
                cp_info,
                cp_size,
                cp_rank,
                chunk_length,
                device,
                position_offset,
                kv_cache_sharded,
            )
        return cache[key]

    return wrapped


def cached_indexer_prefill(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        cache = metadata_cache()
        ctx = getattr(self, "_cp_ctx", None)
        if cache is None or ctx is None or not self.compressor.kpool_mode or args:
            return function(self, *args, **kwargs)
        # Slot maps may only be reused for identical physical block tables and
        # pool geometry. Layer-specific weight/cache payloads are never cached.
        fields = (
            "_state_eb",
            "_kv_eb",
            "_state_tokens_per_block",
            "_kv_tokens_per_block",
            "_kv_owner_tokens_per_block",
        )
        pool = getattr(self, "_kv_pool_view", None)
        key = (
            "indexer_prefill",
            id(ctx),
            self.compress_ratio,
            tuple((name, getattr(self, name, None)) for name in fields),
            tensor_key(getattr(self, "_state_block_table", None)),
            tuple(pool.shape) if pool is not None else None,
            tuple(
                (name, tensor_key(value) if hasattr(value, "data_ptr") else value)
                for name, value in sorted(kwargs.items())
            ),
        )
        if key not in cache:
            cache[key] = function(self, **kwargs)
        return cache[key]

    return wrapped


def cached_prefill_prefix_summary(prefix, host=None):
    """Read immutable prefix metadata once at the start of this forward."""
    import os

    cache = metadata_cache()
    if cache is None or os.environ.get("GLM53_PREFILL_HOST_GEOMETRY", "1") == "0":
        return None
    key = ("prefill_prefix_host", tensor_key(prefix))
    if key not in cache:
        if host is None:
            host = prefix.detach().cpu()
        elif (
            host.device.type != "cpu"
            or host.shape != prefix.shape
            or host.dtype != prefix.dtype
        ):
            raise ValueError("prefix host mirror must match the device tensor")
        values = tuple(int(v) for v in host.reshape(-1).tolist())
        cache[key] = (host, values)
    values = cache[key][1]
    return any(v > 0 for v in values), bool(values) and all(v > 0 for v in values)


def cached_kda_chunk_indices(cu, chunk_size):
    """Build KDA's logical chunk map from this forward's CPU boundaries."""
    import os

    import torch

    cache = metadata_cache()
    if cache is None or os.environ.get("GLM53_PREFILL_HOST_GEOMETRY", "1") == "0":
        return None
    cu_key = tensor_key(cu)
    boundaries = cache.get(("prefill_cu_host", cu_key))
    if boundaries is None:
        return None
    key = ("kda_chunk_indices", cu_key, chunk_size)
    if key not in cache:
        # Match FLA's cumsum(eq(chunk_id, 0)) sequence numbering, including
        # its omission of empty sequences. No GPU values or dynamic shapes
        # are read here, and the context expires before inputs can be reused.
        rows = []
        sequence = 0
        for start, stop in zip(boundaries, boundaries[1:]):
            count = (stop - start + chunk_size - 1) // chunk_size
            if count:
                rows.extend((sequence, chunk) for chunk in range(count))
                sequence += 1
        host = torch.tensor(rows, dtype=cu.dtype, pin_memory=cu.is_cuda).reshape(-1, 2)
        cache[key] = host.to(device=cu.device, non_blocking=True)
    return cache[key]


def cached_indexer_host_lengths(cu, prefix, input_lengths, ratio, device):
    """Share logical length geometry without caching physical pool mappings."""
    import os
    from itertools import accumulate

    import torch

    cache = metadata_cache()
    if cache is None or os.environ.get("GLM53_PREFILL_HOST_GEOMETRY", "1") == "0":
        return None
    if cu.numel() != prefix.numel() + 1 or input_lengths.numel() != prefix.numel():
        return None
    key = (
        "indexer_host_lengths",
        tensor_key(cu),
        tensor_key(prefix),
        tensor_key(input_lengths),
        ratio,
        str(device),
    )
    if key in cache:
        return cache[key]
    cu_key = ("prefill_cu_host", tensor_key(cu))
    boundaries = cache.get(cu_key)
    if boundaries is None:
        boundaries = tuple(int(v) for v in cu.detach().cpu().tolist())
        cache[cu_key] = boundaries
    cached_prefill_prefix_summary(prefix)
    prefixes = cache[("prefill_prefix_host", tensor_key(prefix))][1]
    totals = tuple(p + b - a for p, a, b in zip(prefixes, boundaries, boundaries[1:]))
    counts = tuple(n // ratio for n in totals)
    if any(n < 0 or n >= 2**31 for n in counts) or sum(counts) >= 2**31:
        return None
    offsets = (0, *accumulate(counts))
    host_counts = torch.tensor(
        counts, dtype=torch.int32, pin_memory=torch.device(device).type == "cuda"
    )
    host_cu = torch.tensor(
        offsets, dtype=torch.int32, pin_memory=torch.device(device).type == "cuda"
    )
    result = (
        host_counts.to(device, non_blocking=True),
        host_cu.to(device, non_blocking=True),
        offsets[-1],
        totals[0],
        host_counts,
        host_cu,
    )
    cache[key] = result
    return result


def indexer_varlen_bounds(req_ids, positions, counts, cu_keys, ratio, device):
    """Reuse immutable logical bounds across layers within one GLM53 prefill."""
    import os

    import torch

    cache = (
        metadata_cache()
        if os.environ.get("GLM53_INDEXER_SHARED_BOUNDS", "0") == "1"
        else None
    )
    sources = (req_ids, positions, counts, cu_keys)
    key = None
    if cache is not None:
        key = (
            "indexer_varlen_bounds",
            tuple(tensor_key(tensor) for tensor in sources),
            ratio,
            str(device),
        )
        if key in cache:
            return cache[key][1]
    positions_d = positions.to(device=device, dtype=torch.int32).contiguous()
    req_id_long = req_ids.to(device=device, dtype=torch.int64)
    count_per_token = counts.to(torch.int64).index_select(0, req_id_long)
    offsets = cu_keys.to(torch.int64).index_select(0, req_id_long)
    ks = offsets.to(torch.int32).contiguous()
    ke = (
        (
            offsets
            + ((positions_d.to(torch.int64) + 1) // ratio).clamp_max(count_per_token)
        )
        .to(torch.int32)
        .contiguous()
    )
    cu_per_token = offsets.to(torch.int32).contiguous()
    result = positions_d, ks, ke, cu_per_token
    if cache is not None:
        # Keep source tensors alive too: a later layer must not reuse a freed
        # allocation's address for different metadata during this forward.
        cache[key] = sources, result
    return result
