# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical whole-page layout for BTC's single-host worker."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from typing import TYPE_CHECKING

import torch
from vllm.v1.kv_cache_interface import MambaSpec, iter_layer_specs

from vllm_torchtpu.models.model_hooks import qwen4_options

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _physical_regions(config: KVCacheConfig) -> list[tuple[list[str], int]]:
    # Keep the physical row and stable alias ordering of the original
    # kvs_connector.kv_cache_layout geometry.
    tensors = config.kv_cache_tensors
    sizes = {tensor.size for tensor in tensors}
    if len(sizes) != 1 or config.num_blocks <= 0:
        raise ValueError("BTC KV placements need one paged backing allocation")
    backing_size = sizes.pop()
    num_blocks = config.num_blocks
    block_outermost = all(tensor.layer_stride > 0 for tensor in tensors) and (
        (config.kv_cache_layout or "").startswith("B")
        or (
            all(num_blocks * t.block_stride == backing_size for t in tensors)
            and any(t.layer_stride < t.block_stride for t in tensors)
        )
    )
    regions: dict[tuple[int, int], list[str]] = {}
    for tensor in tensors:
        if not tensor.layers or tensor.block_stride <= 0:
            raise ValueError("BTC KV placement requires layers and a positive stride")
        if block_outermost and num_blocks * tensor.block_stride != backing_size:
            raise ValueError("BTC block-outermost stride disagrees with backing size")
        page_bytes = tensor.layer_stride if block_outermost else tensor.block_stride
        for index, name in enumerate(tensor.layers):
            offset = tensor.offset + index * tensor.layer_stride
            if block_outermost:
                offset *= num_blocks
            regions.setdefault((offset, num_blocks * page_bytes), []).append(name)

    result = []
    end = 0
    for (offset, size), layers in sorted(regions.items()):
        if offset != end:
            raise ValueError("BTC requires contiguous, non-overlapping native pools")
        result.append((layers, size))
        end += size
    if end != backing_size:
        raise ValueError("BTC KV regions do not cover their backing allocation")
    return result


def _companion_ple_bytes(config: VllmConfig, layers: set[str]) -> int:
    # Separate companion PLE carriers are outside the scheduler's placements.
    if not any(name.endswith(".linear_attn") for name in layers) or any(
        name.endswith(".ple") for name in layers
    ):
        return 0
    text = config.model_config.hf_text_config
    ple_ids = getattr(text, "ple_layer_ids", None)
    if not ple_ids:
        return 0
    valid_ple_ids = {
        int(layer) for layer in ple_ids if 1 <= int(layer) <= text.num_hidden_layers
    }
    if not valid_ple_ids:
        return 0
    if (
        min(
            text.ple_conv_kernel_size - 1,
            text.ngram_size,
            text.hidden_size,
            text.hc_count,
            text.indexer_head_dim,
        )
        <= 0
    ):
        return 0
    state_bytes = (
        (text.ple_conv_kernel_size - 1)
        * text.ngram_size
        * text.hidden_size
        * text.hc_count
        * torch.bfloat16.itemsize
    )
    dtype = qwen4_options(config).get("indexer_cache_dtype", "bfloat16")
    itemsize = 1 if dtype in ("fp8", "fp8_e4m3") else 2
    width = ceil((text.indexer_head_dim + (itemsize == 1)) / 128) * 128
    carrier_row_bytes = 4 * width
    rows = ceil(state_bytes / carrier_row_bytes)
    return len(valid_ple_ids) * rows * carrier_row_bytes


def rank_page_bytes(config: VllmConfig, kv_config: KVCacheConfig) -> int:
    """Calculate the packed physical bytes per block and local rank."""
    regions = _physical_regions(kv_config)
    layers = {name for names, _ in regions for name in names}
    has_mamba = any(
        isinstance(spec, MambaSpec)
        for group in kv_config.kv_cache_groups
        for spec in iter_layer_specs(group.kv_cache_spec)
    )
    size = sum(size for _, size in regions) // kv_config.num_blocks
    return size + (_companion_ple_bytes(config, layers) if has_mamba else 0)


def _storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    try:
        pointer = int(storage.data_ptr())
    except RuntimeError:
        # Some lazy tensor storages expose only an allocation size.
        pointer = 0
    return pointer or id(tensor), storage.nbytes()


def _cache_tensor(cache: object, name: str) -> torch.Tensor:
    if isinstance(cache, (tuple, list)) and len(cache) == 1:
        cache = cache[0]
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"BTC KV cache {name!r} must be backed by one tensor")
    if cache.ndim < 2 or cache.shape[0] == 0:
        raise ValueError(f"BTC KV cache {name!r} has no block axis")
    if cache.storage_offset() != 0 or not cache.is_contiguous():
        raise ValueError(f"BTC KV cache {name!r} must be contiguous at offset zero")
    return cache


def resolve_rank_page(
    config: KVCacheConfig,
    kv_caches: Mapping[str, object],
) -> list[tuple[torch.Tensor, int]]:
    """Resolve the complete physical carriers and their bytes per manager block."""
    regions = _physical_regions(config)
    configured = {name for names, _ in regions for name in names}
    carriers: dict[tuple[int, int], torch.Tensor] = {}
    num_blocks = config.num_blocks
    for names, region_size in regions:
        region: dict[tuple[int, int], torch.Tensor] = {}
        for name in names:
            tensor = _cache_tensor(kv_caches[name], name)
            region.setdefault(_storage_key(tensor), tensor)
            index_name = f"{name}.compressed_k"
            if index_name in kv_caches:
                index = _cache_tensor(kv_caches[index_name], index_name)
                region.setdefault(_storage_key(index), index)
        if sum(key[1] for key in region) != region_size:
            raise ValueError(
                f"BTC KV region {names} does not match its configured size"
            )
        for key, tensor in region.items():
            carriers.setdefault(key, tensor)

    companions = sorted(
        name for name in kv_caches if name.endswith(".ple") and name not in configured
    )
    if (
        companions
        and not any(name.endswith(".linear_attn") for name in configured)
        and not any(
            isinstance(spec, MambaSpec)
            for group in config.kv_cache_groups
            if getattr(group.kv_cache_spec, "prefix_cacheable", True)
            for spec in iter_layer_specs(group.kv_cache_spec)
        )
    ):
        raise ValueError("BTC companion PLE requires a Mamba state owner")
    for name in companions:
        tensor = _cache_tensor(kv_caches[name], name)
        key = _storage_key(tensor)
        if tensor.shape[0] != num_blocks or key in carriers:
            raise ValueError(f"BTC companion PLE {name!r} needs its own full pool")
        carriers[key] = tensor

    result = []
    for (_, storage_bytes), tensor in carriers.items():
        page_bytes, remainder = divmod(storage_bytes, num_blocks)
        if not page_bytes or remainder:
            raise ValueError("BTC KV carrier does not contain whole manager blocks")
        result.append((tensor, page_bytes))
    # Match the existing persisted order, including insertion order for ties.
    result.sort(key=lambda item: (item[1], str(item[0].dtype), tuple(item[0].shape)))
    return result
