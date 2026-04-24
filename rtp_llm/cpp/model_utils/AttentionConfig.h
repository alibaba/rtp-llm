#pragma once

#include "rtp_llm/cpp/model_utils/RopeConfig.h"
#include <c10/core/ScalarType.h>

namespace rtp_llm {

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

    // rotary embending config
    RopeConfig rope_config;

    // kv cache block
    size_t tokens_per_block        = 8;
    size_t kernel_tokens_per_block = 0;

    float q_scaling         = 1.0f;
    bool  fuse_qkv_add_bias = true;
    bool  use_logn_attn     = false;
    bool  is_causal         = true;

    // mla config
    bool   use_mla = false;
    size_t q_lora_rank;
    size_t kv_lora_rank;
    size_t nope_head_dim;
    size_t rope_head_dim;
    size_t v_head_dim;

    // softmax config
    float           softmax_extra_scale = 1.0f;
    KvCacheDataType kv_cache_dtype      = KvCacheDataType::BASE;
    bool            need_rope_kv_cache  = true;

    // sparse attention config
    bool is_sparse        = false;
    int  indexer_head_dim = 0;
    int  indexer_head_num = 0;
    int  indexer_topk     = 0;

    // DeepSeek-V4 hybrid attention (CSA + HCA + SWA-bypass)
    // 0 = not V4 / non-compressed; 4 = CSA (compress every 4 tokens); 128 = HCA (every 128).
    // Per-layer compress ratio drives layer-type dispatch (CSA/HCA/SWA-only).
    std::vector<int> layer_compress_ratios = {};
    bool             use_v4_hybrid_attn    = false;
    int              o_groups              = 0;
    int              o_lora_rank           = 0;
    int              sliding_window        = 0;        // 0 = disabled
    bool             has_attention_sink    = false;    // per-head learnable sink logit
    int              hc_mult               = 0;        // mHC residual expansion factor n_hc (0 = no mHC)
    int              hc_sinkhorn_iters     = 20;       // Sinkhorn-Knopp iterations for B projection
    float            hc_eps                = 1e-6f;
    int              compress_rope_theta   = 0;        // separate RoPE base for compressed KV branch (0 = use rope_config.base)

    // data type for attention computation
    c10::ScalarType dtype = c10::ScalarType::Half;

    // maximum sequence length for RoPE cache generation
    size_t max_seq_len = 32768;

    // speculative decoding: tokens generated per cycle (0 = no speculative decoding)
    int64_t gen_num_per_cycle = 0;

public:
    std::string DebugAttentionConfigStr() const;
};

}  // namespace rtp_llm
