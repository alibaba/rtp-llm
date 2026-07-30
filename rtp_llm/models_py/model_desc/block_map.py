import os

from rtp_llm.ops.compute_ops import PyAttentionInputs


def select_block_map_for_layer(
    attention_inputs: PyAttentionInputs,
    layer_idx: int,
    group_id: int | None = None,
) -> int:
    if attention_inputs.kv_cache_kernel_block_id_device_by_group is None:
        return

    gid = 0 if group_id is None else int(group_id)
    if group_id is None:
        host_map = (
            getattr(attention_inputs, "kv_cache_layer_to_group_host", None)
            if os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
            else None
        )
        if host_map is not None and host_map.numel():
            gid = int(host_map[layer_idx])
        elif attention_inputs.kv_cache_layer_to_group is not None:
            # Compatibility fallback for older bindings. K3 supplies the
            # pinned host mirror (or an explicit static group id), so its hot
            # path does not issue this CUDA scalar read.
            gid = int(attention_inputs.kv_cache_layer_to_group[layer_idx].item())

    if attention_inputs.kv_cache_kernel_block_id_device_by_group is not None and len(
        attention_inputs.kv_cache_kernel_block_id_device_by_group
    ):
        attention_inputs.kv_cache_kernel_block_id_device = (
            attention_inputs.kv_cache_kernel_block_id_device_by_group[gid]
        )
    if attention_inputs.kv_cache_kernel_block_id_host_by_group is not None and len(
        attention_inputs.kv_cache_kernel_block_id_host_by_group
    ):
        attention_inputs.kv_cache_kernel_block_id_host = (
            attention_inputs.kv_cache_kernel_block_id_host_by_group[gid]
        )
    if attention_inputs.kv_cache_block_id_host_by_group is not None and len(
        attention_inputs.kv_cache_block_id_host_by_group
    ):
        attention_inputs.kv_cache_block_id_host = (
            attention_inputs.kv_cache_block_id_host_by_group[gid]
        )
    return gid
