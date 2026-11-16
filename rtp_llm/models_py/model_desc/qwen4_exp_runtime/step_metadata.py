"""Invocation-local reuse of layer-independent QSA metadata.

The model creates a new dictionary for each forward. During CUDA graph capture,
the first layer records the producer kernels and later layers consume the same
tensors. Every graph replay therefore refreshes values from the current inputs.
No dictionary is retained across eager forwards or different MTP phases.
"""

import torch


def tensor_signature(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.data_ptr(),
    )


def cached_step_value(context, key, factory):
    cache = getattr(context, "metadata_cache", None)
    if cache is None:
        return factory()
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def cached_query_geometry(context, prefixes, query_len, ratio):
    """Share one query window across layers, for decode and verification."""

    def materialize():
        if prefixes.is_cuda:
            from rtp_llm.models_py.kernels.cuda.qsa.target_metadata_triton import (
                try_target_metadata,
            )

            geometry = try_target_metadata(prefixes, query_len, ratio)
            if geometry is not None:
                return geometry
        offsets = torch.arange(
            1, query_len + 1, dtype=torch.int32, device=prefixes.device
        )
        visible = prefixes[:, None] + offsets[None, :]
        # Preserve the target's layer-local handling of inactive positions.
        return visible, visible // ratio, None, None

    return cached_step_value(
        context,
        ("query_geometry", tensor_signature(prefixes), query_len, ratio),
        materialize,
    )


def cached_text_rope(
    context, positions, rope_config, indexer, *, token_count, dtype, device
):
    """Share identical text-position cos/sin within this model forward."""
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
        build_qsa_rope_from_logical_positions,
    )

    get_frequency = getattr(indexer, "rope_inv_freq", None)
    frequency = get_frequency(rope_config) if callable(get_frequency) else None
    frequency_key = None
    if frequency is not None:
        recipe = getattr(indexer, "_rope_inv_freq_key", None)
        if recipe is None or (not torch.is_inference(frequency) and frequency._version):
            frequency_key = tensor_signature(frequency)
        else:
            frequency_key = (
                recipe,
                frequency.dtype,
                frequency.device,
                tuple(frequency.shape),
            )
    settings = (str(rope_config.style),) + tuple(
        getattr(rope_config, name, None)
        for name in (
            "base",
            "dim",
            "scale",
            "index_factor",
            "indexer_is_neox_style",
            "mrope_interleaved",
            "mrope_dim1",
            "mrope_dim2",
            "mrope_dim3",
        )
    )
    return cached_step_value(
        context,
        (
            "text_rope",
            tensor_signature(positions),
            settings,
            frequency_key,
            token_count,
            dtype,
            device,
        ),
        lambda: build_qsa_rope_from_logical_positions(
            positions,
            rope_config,
            token_count=token_count,
            dtype=dtype,
            device=device,
            inv_freq=frequency,
        ),
    )
