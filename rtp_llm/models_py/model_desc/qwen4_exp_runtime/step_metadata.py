"""Invocation-local reuse of layer-independent QSA metadata.

The model creates a new dictionary for each forward. Captured query geometry
reads live device prefixes on replay. Incremental prefill geometry uses host
lengths and prefixes and its callers restrict it to eager execution.
"""

from dataclasses import dataclass
from typing import NamedTuple

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


@dataclass(frozen=True)
class PrefillRequestGeometry:
    prefixes: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    logical_positions: torch.Tensor
    visible_lengths: torch.Tensor
    compressed_lengths: torch.Tensor
    final_compressed_lengths: torch.Tensor


class _PrefillGeometryEntry(NamedTuple):
    source_lengths: torch.Tensor
    source_prefixes: torch.Tensor
    geometry: PrefillRequestGeometry


def cached_prefill_geometry(context, lengths, prefixes, ratio, device):
    """Share eager prefill geometry from validated request lengths and prefixes."""
    inputs = context.main_inputs
    source_lengths, source_prefixes = inputs.input_lengths, inputs.prefix_lengths

    def materialize():
        device_prefixes = source_prefixes.to(
            device=device, dtype=torch.int64, non_blocking=True
        ).contiguous()
        device_lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        cu[1:] = device_lengths.cumsum(0)
        visible_ends = device_prefixes + device_lengths
        cu_kv = torch.zeros_like(cu)
        cu_kv[1:] = visible_ends.to(torch.int32).cumsum(0)
        logical = torch.cat(
            [
                torch.arange(prefix, prefix + length, dtype=torch.int64, device=device)
                for prefix, length in zip(prefixes, lengths)
            ]
        )
        offsets = torch.arange(max(lengths), dtype=torch.int64, device=device)
        valid = offsets[None, :] < device_lengths[:, None]
        visible = torch.where(valid, device_prefixes[:, None] + offsets[None, :] + 1, 0)
        geometry = PrefillRequestGeometry(
            device_prefixes,
            cu,
            cu_kv,
            logical,
            visible,
            (visible // ratio).to(torch.int32),
            (visible_ends // ratio).to(torch.int32),
        )
        # Retain native tensor owners along with derived values for this
        # invocation, even if a layer-local context is released first.
        return _PrefillGeometryEntry(source_lengths, source_prefixes, geometry)

    return cached_step_value(
        context,
        (
            "prefill_geometry",
            tensor_signature(source_lengths),
            tensor_signature(source_prefixes),
            tuple(lengths),
            tuple(prefixes),
            ratio,
            device,
        ),
        materialize,
    ).geometry


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
