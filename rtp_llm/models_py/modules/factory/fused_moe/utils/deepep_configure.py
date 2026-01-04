from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)


def calc_low_latency_max_token_per_rank(
    max_generate_batch_size: int,
    tp_size: int,
    quant_config: FusedMoEQuantConfig,
) -> int:
    ll_num_max_token_per_rank = (max_generate_batch_size + tp_size - 1) // tp_size
    # deepgemm masked with max_m < 64 get incorrect result, related: https://github.com/deepseek-ai/DeepGEMM/issues/268
    if not quant_config.is_quantized or quant_config.is_block_quantized:
        matched_tokens = [64, 128]
    elif quant_config.is_per_act_token:
        matched_tokens = [
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
        ]
    else:
        raise ValueError("Unsupported quantization config")
    if ll_num_max_token_per_rank > 128:
        ll_num_max_token_per_rank = ((ll_num_max_token_per_rank + 127) // 128) * 128
        return ll_num_max_token_per_rank
    for t in matched_tokens:
        if ll_num_max_token_per_rank <= t:
            ll_num_max_token_per_rank = t
            return ll_num_max_token_per_rank
    return 128
