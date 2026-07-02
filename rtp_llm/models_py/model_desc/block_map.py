from typing import Any

from rtp_llm.ops.compute_ops import PyAttentionInputs


def select_block_map_for_layer(
    attention_inputs: PyAttentionInputs, kv_cache: Any, local_layer_idx: int
) -> int:
    by_group = attention_inputs.kv_cache_kernel_block_id_device_by_group
    if by_group is None or len(by_group) == 0:
        return 0

    gid = 0
    if kv_cache is not None and kv_cache.layer_to_group_ids:
        if local_layer_idx < 0 or local_layer_idx >= len(kv_cache.layer_to_group_ids):
            raise RuntimeError(
                f"local layer {local_layer_idx} is out of layer_to_group_ids range "
                f"{len(kv_cache.layer_to_group_ids)}"
            )
        group_ids = kv_cache.layer_to_group_ids[local_layer_idx]
        if len(group_ids) != 1:
            raise RuntimeError(
                f"local layer {local_layer_idx} owns groups {group_ids}; "
                "select_block_map_for_layer requires a single default group"
            )
        gid = int(group_ids[0])

    if gid < 0 or gid >= len(by_group):
        raise RuntimeError(
            f"local layer {local_layer_idx} maps to invalid group {gid}; "
            f"available groups={len(by_group)}"
        )
    attention_inputs.kv_cache_kernel_block_id_device = by_group[gid]
    return gid
