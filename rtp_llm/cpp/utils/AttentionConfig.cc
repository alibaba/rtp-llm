#include "rtp_llm/cpp/utils/AttentionConfig.h"

namespace rtp_llm {

void registerFMHAType(py::module m) {
    py::enum_<FMHAType>(m, "FMHAType")
        .value("NONE", FMHAType::NONE)
        .value("PAGED_TRT_V2", FMHAType::PAGED_TRT_V2)
        .value("TRT_V2", FMHAType::TRT_V2)
        .value("PAGED_OPEN_SOURCE", FMHAType::PAGED_OPEN_SOURCE)
        .value("OPEN_SOURCE", FMHAType::OPEN_SOURCE)
        .value("TRT_V1", FMHAType::TRT_V1)
        .value("FLASH_INFER", FMHAType::FLASH_INFER)
        .value("XQA", FMHAType::XQA)
        .value("AITER_PREFILL", FMHAType::AITER_PREFILL)
        .value("AITER_DECODE", FMHAType::AITER_DECODE);
}

std::string AttentionConfigs::DebugAttentionConfigStr() const {
    std::ostringstream oss;
    oss << "AttentionConfigs Debug Info:" << std::endl;
    oss << "  head_num: " << head_num << std::endl;
    oss << "  kv_head_num: " << kv_head_num << std::endl;
    oss << "  size_per_head: " << size_per_head << std::endl;
    oss << "  hidden_size: " << hidden_size << std::endl;
    oss << "  tokens_per_block: " << tokens_per_block << std::endl;
    oss << "  mask_type: " << mask_type << std::endl;
    oss << "  q_scaling: " << q_scaling << std::endl;
    oss << "  fuse_qkv_add_bias: " << fuse_qkv_add_bias << std::endl;
    oss << "  use_logn_attn: " << use_logn_attn << std::endl;
    oss << "  use_mla: " << use_mla << std::endl;
    oss << "  q_lora_rank: " << q_lora_rank << std::endl;
    oss << "  kv_lora_rank: " << kv_lora_rank << std::endl;
    oss << "  nope_head_dim: " << nope_head_dim << std::endl;
    oss << "  rope_head_dim: " << rope_head_dim << std::endl;
    oss << "  v_head_dim: " << v_head_dim << std::endl;
    oss << "  softmax_extra_scale: " << softmax_extra_scale << std::endl;
    oss << "  kv_cache_dtype: " << static_cast<int>(kv_cache_dtype) << std::endl;
    oss << "  skip_append_kv_cache: " << skip_append_kv_cache << std::endl;
    oss << rope_config.DebugRopeConfigStr();
    return oss.str();
}

}  // namespace rtp_llm
