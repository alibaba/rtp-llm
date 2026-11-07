"""Topology admission for block128 FP8-source, contiguous online NVFP4 experts."""


def use_nvfp4_tensor_parallel(config) -> bool:
    pc = config.parallelism_config
    quant = getattr(config.model_config, "quant_config", None)
    return (
        quant is not None
        and quant.get_method() == "modelopt_fp4"
        and not quant.is_quanted()
        and quant.bits == 4
        and quant.group_size() == 16
        and quant.source_fp8_block_size == 128
        and quant.require_contiguous_experts
        and quant.activation_amax == 32.0
        and not quant.mixed_attention
        and config.tp_size == 2
        and config.ep_size == 1
        and pc.world_size == 2
        and pc.get_ffn_tp_size() == 2
        and pc.ffn_sp_size == 1
        and not pc.ffn_disaggregate_config.enable_ffn_disaggregate
    )
