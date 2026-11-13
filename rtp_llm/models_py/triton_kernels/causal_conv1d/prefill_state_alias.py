"""Host-side alias planning for Prefill only; no device synchronization."""


def prefill_conv_alias_sources(prefixes, cu_seqlens, block_table, page_size):
    """Return safe gather indices and a per-request continuation mask.

    Unselected outputs are negative/zero. Each selected output must own a
    distinct physical block within this forward; a repeated output ID is a
    writer race, which an input snapshot cannot repair.
    """
    if (
        page_size <= 0
        or len(cu_seqlens) != len(prefixes) + 1
        or len(block_table) != len(prefixes)
    ):
        raise ValueError("inconsistent Prefill conv alias metadata")
    if cu_seqlens[0] != 0:
        raise ValueError("Prefill cu_seqlens must start at zero")
    sources, aliases = [], []
    for row, prefix in enumerate(prefixes):
        length = cu_seqlens[row + 1] - cu_seqlens[row]
        if prefix < 0 or length < 0:
            raise ValueError("negative Prefill prefix or query length")
        table = block_table[row]
        if length == 0:
            sources.append(0)
            aliases.append(False)
            continue
        start = prefix // page_size
        end = (prefix + length - 1) // page_size + 1
        if end > len(table):
            raise ValueError("Prefill conv output is outside the execution row")
        outputs = [block for block in table[start:end] if block > 0]
        if len(outputs) != len(set(outputs)):
            raise ValueError(
                "multiple Prefill conv outputs alias one physical state block"
            )
        source = table[(prefix - 1) // page_size] if prefix else 0
        if prefix and source <= 0:
            raise ValueError("Prefill conv is missing its initial state")
        alias = bool(prefix and source in outputs)
        sources.append(source if alias else 0)
        aliases.append(alias)
    return sources, aliases


def snapshot_prefill_conv_input(
    conv_states, attn_inputs, page_size, metadata, *, world_rank=None
):
    """Gather only when this Prefill's old state can be overwritten in place.

    Cache the immutable index/mask tensors across layers, never their contents:
    every layer owns different conv history. index_select completes on the
    current CUDA stream before the subsequent convolution can overwrite it.
    """
    import torch

    key = ("prefill_conv_alias", attn_inputs, page_size)
    cache = metadata.prefill_cuda_inputs
    if key not in cache:
        mirrors = (
            attn_inputs.prefix_lengths,
            attn_inputs.cu_seqlens,
            attn_inputs.kv_cache_kernel_block_id,
        )
        if any(value.device.type != "cpu" for value in mirrors):
            raise ValueError(
                "Prefill conv alias planning requires host metadata mirrors"
            )
        sources, aliases = prefill_conv_alias_sources(
            attn_inputs.prefix_lengths.tolist(),
            attn_inputs.cu_seqlens.tolist(),
            attn_inputs.kv_cache_kernel_block_id.tolist(),
            page_size,
        )
        if any(aliases):
            source_host = torch.tensor(sources, dtype=torch.int64, pin_memory=True)
            mask_host = torch.tensor(aliases, dtype=torch.bool, pin_memory=True)
            cache[key] = (
                source_host.to(conv_states.device, non_blocking=True),
                mask_host.to(conv_states.device, non_blocking=True),
                source_host,
                mask_host,
            )
        else:
            cache[key] = None
    indices = cache[key]
    if indices is None:
        return None, None
    snapshot = conv_states.index_select(0, indices[0])
    return snapshot, indices[1]
