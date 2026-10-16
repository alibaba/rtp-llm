from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Sequence

import torch

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import PyAttentionInputs
else:
    PyAttentionInputs = Any


class PreparedBlockMaps(NamedTuple):
    device_groups: Sequence[torch.Tensor]
    kernel_host_groups: Sequence[torch.Tensor]
    host_groups: Sequence[torch.Tensor]
    layer_groups: Optional[tuple[int, ...]]


def prepare_block_maps(
    attention_inputs: PyAttentionInputs,
) -> Optional[PreparedBlockMaps]:
    """Snapshot pybind vector wrappers once for one model forward."""
    device_groups = attention_inputs.kv_cache_kernel_block_id_device_by_group
    if device_groups is None:
        return None
    layer_to_group = attention_inputs.kv_cache_layer_to_group
    if layer_to_group is None:
        layer_groups = None
    elif isinstance(layer_to_group, torch.Tensor):
        # The engine publishes this as pinned CPU int32 metadata. Converting it
        # once avoids one Tensor.item() and three pybind vector materializations
        # per layer without retaining request-owned data across forwards.
        layer_groups = tuple(int(group) for group in layer_to_group.tolist())
    else:
        layer_groups = tuple(int(group) for group in layer_to_group)
    return PreparedBlockMaps(
        device_groups,
        attention_inputs.kv_cache_kernel_block_id_host_by_group,
        attention_inputs.kv_cache_block_id_host_by_group,
        layer_groups,
    )


def select_block_map_for_layer(
    attention_inputs: PyAttentionInputs,
    layer_idx: int,
    prepared: Optional[PreparedBlockMaps] = None,
) -> Optional[int]:
    # Each pybind vector getter materializes a Python list. The prepared path
    # snapshots the tables for one forward; the default preserves existing
    # callers that select a single layer.
    if prepared is None:
        device_groups = attention_inputs.kv_cache_kernel_block_id_device_by_group
        kernel_host_groups = (
            attention_inputs.kv_cache_kernel_block_id_host_by_group
        )
        host_groups = attention_inputs.kv_cache_block_id_host_by_group
        layer_to_group = attention_inputs.kv_cache_layer_to_group
        gid = 0 if layer_to_group is None else int(layer_to_group[layer_idx].item())
    else:
        device_groups = prepared.device_groups
        kernel_host_groups = prepared.kernel_host_groups
        host_groups = prepared.host_groups
        gid = 0 if prepared.layer_groups is None else prepared.layer_groups[layer_idx]
    if device_groups is None:
        return

    if device_groups:
        attention_inputs.kv_cache_kernel_block_id_device = device_groups[gid]
    if kernel_host_groups:
        attention_inputs.kv_cache_kernel_block_id_host = kernel_host_groups[gid]
    if host_groups:
        attention_inputs.kv_cache_block_id_host = host_groups[gid]
    return gid
