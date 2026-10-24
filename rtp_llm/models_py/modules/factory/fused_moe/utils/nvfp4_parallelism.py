"""Topology admission for the complete Qwen4 FP8/NVFP4 recipe."""


def use_nvfp4_tensor_parallel(config) -> bool:
    pc = config.parallelism_config
    return (
        bool(getattr(config.model_config, "_qwen4_full_fp8_nvfp4", False))
        and config.tp_size == 2
        and config.ep_size == 1
        and pc.world_size == 2
        and pc.get_ffn_tp_size() == 2
        and pc.ffn_sp_size == 1
        and not pc.ffn_disaggregate_config.enable_ffn_disaggregate
    )
