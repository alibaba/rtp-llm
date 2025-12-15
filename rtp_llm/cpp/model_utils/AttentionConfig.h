#pragma once

#include "rtp_llm/cpp/model_utils/RopeConfig.h"

namespace rtp_llm {

enum class FMHAType {
    FLASH_INFER,
    NONE,
    OPEN_SOURCE,
    PAGED_OPEN_SOURCE,
    PAGED_TRT_V2,
    TRT_V1,
    TRT_V2,
    XQA,
    AITER_PREFILL,
    AITER_DECODE,
    PY_FLASH_INFER_PREFILL,
    PY_FLASH_INFER_DECODE
};

enum AttentionMaskType {
    // ones matrix, for bert model.
    noMask,
    causalMask,
};

enum class KvCacheDataType : int8_t {
    BASE = 0,
    INT8 = 1,
    FP8  = 2
};

KvCacheDataType inline loadKvCacheDataTypeFromString(const std::string& str) {
    if (str == "base" || str == "fp16") {
        return KvCacheDataType::BASE;
    } else if (str == "int8") {
        return KvCacheDataType::INT8;
    } else if (str == "fp8") {
        return KvCacheDataType::FP8;
    } else {
        return KvCacheDataType::BASE;
    }
}

struct AttentionConfigs {
    size_t head_num;
    size_t kv_head_num;
    size_t size_per_head;
    size_t hidden_size;

    // rotary embending config
    RopeConfig rope_config;

    // kv cache block
    size_t tokens_per_block;

    AttentionMaskType mask_type         = noMask;
    float             q_scaling         = 1.0f;
    bool              fuse_qkv_add_bias = true;
    bool              use_logn_attn     = false;

    // mla config
    bool   use_mla = false;
    size_t q_lora_rank;
    size_t kv_lora_rank;
    size_t nope_head_dim;
    size_t rope_head_dim;
    size_t v_head_dim;

    // softmax config
    float           softmax_extra_scale  = 1.0f;
    KvCacheDataType kv_cache_dtype       = KvCacheDataType::BASE;
    bool            skip_append_kv_cache = false;

public:
    std::string DebugAttentionConfigStr() const;
};

}  // namespace rtp_llm
