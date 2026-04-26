from rtp_llm.ops.compute_ops import PyAttentionInputs


def select_block_map_for_layer(
    attention_inputs: PyAttentionInputs, layer_idx: int
) -> int:
    if attention_inputs.kv_cache_kernel_block_id_device_by_group is None:
        return

    gid = 0
    if attention_inputs.kv_cache_layer_to_group is not None:
        gid = int(attention_inputs.kv_cache_layer_to_group[layer_idx].item())

    if attention_inputs.kv_cache_kernel_block_id_device_by_group is not None and len(
        attention_inputs.kv_cache_kernel_block_id_device_by_group
    ):
        attention_inputs.kv_cache_kernel_block_id_device = (
            attention_inputs.kv_cache_kernel_block_id_device_by_group[gid]
        )
    # Host block-id metadata aliases group 0 only; hybrid callers needing
    # per-layer host data must derive it explicitly from device state.
    return gid
