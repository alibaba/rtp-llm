"""Admission for the experimental Qwen NVFP4 tensor-parallel path."""

import os


def use_nvfp4_tensor_parallel(config) -> bool:
    if os.environ.get("RTP_LLM_QWEN4_NVFP4_TP_EXPERIMENTAL", "0") != "1":
        return False
    pc = config.parallelism_config
    return (
        bool(getattr(config.model_config, "_qwen4_moe_only_fp4", False))
        and config.tp_size == 2
        and config.ep_size == 1
        and pc.world_size == 2
        and pc.get_ffn_tp_size() == 2
        and pc.ffn_sp_size == 1
        and not pc.ffn_disaggregate_config.enable_ffn_disaggregate
    )
