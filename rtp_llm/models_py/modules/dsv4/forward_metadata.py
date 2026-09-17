"""Metadata shared only within one model forward, never across engine steps."""

from contextvars import ContextVar
from functools import wraps

_FORWARD_METADATA = ContextVar("glm53_forward_metadata", default=None)
_SCORE_WORKSPACE_MB = 8192


def metadata_cache():
    return _FORWARD_METADATA.get()


def score_workspace_budget(device):
    import torch

    cache = metadata_cache()
    key = ("indexer_score_budget", torch.device(device))
    if cache is not None and key in cache:
        return cache[key]
    free, _ = torch.cuda.mem_get_info(device)
    reusable = max(
        0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    budget = min(_SCORE_WORKSPACE_MB * 1024 * 1024, (free + reusable) // 2)
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
                score_workspace_budget(
                    torch.device("cuda", torch.cuda.current_device())
                )
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
