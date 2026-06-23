#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/KVCacheSpecDescTypes.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct SpecBuildContext {
    DataType dtype              = DataType::TYPE_INVALID;
    uint32_t seq_size_per_block = 0;
};

class SpecBuilder {
public:
    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(!desc.tag.empty(), "KVCacheSpecDesc tag must not be empty");
        auto spec = buildTyped(desc);
        spec->tag                = desc.tag;
        spec->local_head_num_kv  = valueOr(desc.local_head_num_kv, 1);
        spec->seq_size_per_block = valueOr(desc.seq_size_per_block, valueOr(ctx.seq_size_per_block, 1));
        spec->dtype              = dataTypeOr(desc.dtype, dataTypeOr(ctx.dtype, desc.store_dtype));
        return spec;
    }

    static CacheGroupType groupType(const KVCacheSpecDesc& desc) {
        switch (desc.cache_type) {
            case CacheType::LINEAR:
                return CacheGroupType::LINEAR;
            case CacheType::FIXED_STATE:
                return CacheGroupType::SWA;
            case CacheType::MHA:
            case CacheType::MLA:
            case CacheType::COMPRESSED_KV:
                return CacheGroupType::FULL;
        }
        return CacheGroupType::FULL;
    }

private:
    static uint32_t valueOr(uint32_t value, uint32_t fallback) {
        return value == 0 ? fallback : value;
    }

    static DataType dataTypeOr(DataType value, DataType fallback) {
        return value == DataType::TYPE_INVALID ? fallback : value;
    }

    static KVCacheSpecPtr buildTyped(const KVCacheSpecDesc& desc) {
        switch (desc.cache_type) {
            case CacheType::MHA: {
                auto spec           = std::make_shared<MHAKVCacheSpec>();
                spec->size_per_head = desc.size_per_head;
                return spec;
            }
            case CacheType::MLA: {
                auto spec          = std::make_shared<MLAKVCacheSpec>();
                spec->kv_lora_rank = desc.kv_lora_rank;
                spec->rope_head_dim = desc.rope_head_dim;
                return spec;
            }
            case CacheType::LINEAR: {
                auto spec                = std::make_shared<LinearKVCacheSpec>();
                spec->local_num_k_heads = desc.local_num_k_heads;
                spec->local_num_v_heads = desc.local_num_v_heads;
                spec->head_k_dim        = desc.head_k_dim;
                spec->head_v_dim        = desc.head_v_dim;
                spec->conv_kernel_dim   = desc.conv_kernel_dim;
                spec->ssm_state_dtype   = dataTypeOr(desc.ssm_state_dtype, DataType::TYPE_BF16);
                spec->conv_state_dtype  = dataTypeOr(desc.conv_state_dtype, DataType::TYPE_BF16);
                return spec;
            }
            case CacheType::COMPRESSED_KV: {
                auto spec                         = std::make_shared<CompressedKVCacheSpec>();
                spec->entry_elems                 = desc.entry_elems;
                spec->entries_per_block           = desc.entries_per_block;
                spec->compression_ratio           = valueOr(desc.compression_ratio, 1);
                spec->store_dtype                 = desc.store_dtype;
                spec->block_size_bytes_alignment  = desc.block_size_bytes_alignment;
                return spec;
            }
            case CacheType::FIXED_STATE: {
                auto spec                             = std::make_shared<FixedStateCacheSpec>();
                spec->state_dim                       = desc.entry_elems;
                spec->entries_per_block               = desc.entries_per_block;
                spec->store_dtype                     = desc.store_dtype;
                spec->block_size_bytes_override       = desc.block_size_bytes_override;
                spec->block_size_bytes_alignment      = desc.block_size_bytes_alignment;
                spec->block_size_alignment_min_entries = desc.block_size_alignment_min_entries;
                spec->is_state_cache                  = desc.is_state_cache;
                spec->skip_prefix_reuse               = desc.skip_prefix_reuse;
                return spec;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(false, "unknown CacheType=%d", static_cast<int>(desc.cache_type));
        return nullptr;
    }
};

}  // namespace rtp_llm
