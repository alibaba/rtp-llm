from rtp_llm.ops.compute_ops import PyAttentionInputs


def select_block_map_for_layer(
    attention_inputs: PyAttentionInputs, layer_idx: int
) -> None:
    if attention_inputs.kv_cache_block_id_device_by_group is None:
        return

    gid = 0
    if attention_inputs.kv_cache_layer_to_group is not None:
        gid = int(attention_inputs.kv_cache_layer_to_group[layer_idx].item())

    if attention_inputs.kv_cache_block_id_device_by_group is not None and len(
        attention_inputs.kv_cache_block_id_device_by_group
    ):
        attention_inputs.kv_cache_block_id_device = (
            attention_inputs.kv_cache_block_id_device_by_group[gid]
        )

    if attention_inputs.kv_cache_block_id_host_by_group is not None and len(
        attention_inputs.kv_cache_block_id_host_by_group
    ):
        attention_inputs.kv_cache_block_id_host = (
            attention_inputs.kv_cache_block_id_host_by_group[gid]
        )
