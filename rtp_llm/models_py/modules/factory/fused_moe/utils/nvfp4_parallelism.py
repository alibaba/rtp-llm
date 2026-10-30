"""Topology admission for Qwen4 NVFP4 tensor parallelism."""

from rtp_llm.config.quant_config import QWEN4_NVFP4_RECIPE


def use_nvfp4_tensor_parallel(config) -> bool:
    pc = config.parallelism_config
    quant = getattr(config.model_config, "quant_config", None)
    online = (
        quant is not None
        and quant.get_method() == "modelopt_fp4"
        and not quant.is_quanted()
        and quant.bits == 4
        and quant.group_size() == 16
        and quant.source_fp8_block_size == 128
        and quant.require_contiguous_experts
        and quant.activation_amax == 32.0
        and not quant.mixed_attention
    )
    mixed_nvfp4 = (
        getattr(config.model_config, "_qwen4_nvfp4_mixed_source", False)
        and getattr(quant, "model_recipe", None) == QWEN4_NVFP4_RECIPE
    )
    return (
        (online or mixed_nvfp4)
        and config.tp_size == 2
        and config.ep_size == 1
        and pc.world_size == 2
        and pc.dp_size == 1
        and pc.pp_size == 1
        and not pc.prefill_cp_config.is_enabled()
        and not pc.prefill_cp_config.is_prefill_enabled()
        and pc.get_ffn_tp_size() == 2
    )
