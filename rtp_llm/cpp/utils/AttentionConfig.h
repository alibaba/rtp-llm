#pragma once

#include "rtp_llm/cpp/utils/RopeConfig.h"
#include "rtp_llm/cpp/utils/EnumUtils.h"
#include <torch/extension.h>

namespace rtp_llm {

enum class FMHAType {
    NONE,
    PAGED_TRT_V2,
    TRT_V2,
    PAGED_OPEN_SOURCE,
    OPEN_SOURCE,
    TRT_V1,
    FLASH_INFER,
    XQA
};

enum AttentionMaskType {
    // ones matrix, for bert model.
    noMask,
    causalMask,
};

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
};

void registerFMHAType(py::module m);

}  // namespace rtp_llm
