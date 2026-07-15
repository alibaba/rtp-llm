#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpecType.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

struct KVCacheSpec;
using KVCacheSpecPtr = std::shared_ptr<KVCacheSpec>;

struct KVCacheSpecDesc {
    std::string     tag;
    KVCacheSpecType cache_type = KVCacheSpecType::MultiHeadAttention;
    DataType        dtype      = DataType::TYPE_INVALID;
};

struct SpecBuildContext {
    DataType                     dtype                   = DataType::TYPE_INVALID;
    uint32_t                     seq_size_per_block      = 0;
    const AttentionConfigs*      attn_config             = nullptr;
    const LinearAttentionConfig* linear_attention_config = nullptr;
    const ParallelismConfig*     parallelism_config      = nullptr;
    uint32_t                     gen_num_per_cycle       = 0;
};

class SpecBuilder {
public:
    static KVCacheSpecPtr   build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx);
    static CacheGroupType   groupType(const KVCacheSpecDesc& desc);
    static CacheGroupPolicy groupPolicy(const KVCacheSpecDesc& desc);
};

}  // namespace rtp_llm
