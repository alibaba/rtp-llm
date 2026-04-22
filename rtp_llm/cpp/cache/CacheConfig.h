#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct KVCacheAllocatorConfig {
    size_t   model_id  = 0;
    uint32_t layer_num = 0;

    rtp_llm::DataType dtype     = rtp_llm::DataType::TYPE_FP16;
    bool              use_mla   = false;
    bool              is_sparse = false;

    // KVCacheSpec & group structure
    std::vector<KVCacheSpecPtr>   cache_specs;
    std::vector<CacheGroupType>   group_types;
    std::vector<CacheGroupType>   layer_attn_types;   // per-layer attention type (FULL/LINEAR)
    std::vector<std::vector<int>> layer_ids;          // [group_id][local_layer_idx]
    std::vector<int>              layer_to_group_id;  // layer_id -> group_id
    std::vector<int>              layer_to_block_stride_bytes;

    // Block configuration
    uint32_t block_num          = 0;
    size_t   seq_size_per_block = 1;

    // Per-block sizes (all layers combined)
    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size_bytes    = 0;

    // Per-block strides (one layer)
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    // Attention-specific
    int linear_step      = 1;
    int group_layer_num  = 1;
    int linear_group_num = 0;
    int full_group_num   = 0;

    int groupNums() const {
        return std::max<int>(1, static_cast<int>(cache_specs.size()));
    }

    std::string debugString(size_t indent = 0) const {
        const std::string  pad = std::string(indent, ' ');
        std::ostringstream os;
        os << pad << "KVCacheAllocatorConfig{model_id=" << model_id << ", layer_num=" << layer_num
           << ", block_num=" << block_num << ", block_size_bytes=" << block_size_bytes
           << ", kv_block_stride_bytes=" << kv_block_stride_bytes << ", groups=" << groupNums() << "}";
        return os.str();
    }
};

struct CacheConfig {
    // Cross-model shared / aggregate parameters only.
    // Per-model parameters live in allocator_configs[model_id].

    // Block configuration (shared across all models)
    size_t seq_size_per_block        = 1;
    size_t kernel_seq_size_per_block = 0;

    // Total layer count across all models (main + all MTP modules)
    uint32_t layer_all_num = 0;

    // Cross-model combined layer mappings (global layer_id space: main + MTP concatenated).
    // Used by connectors (P/D separation) that transfer KV data across all models.
    std::vector<int> layer_to_group_id;
    std::vector<int> layer_to_block_stride_bytes;

    // Per-model allocator configurations.
    // Non-MTP: allocator_configs.size() == 1 (main model only).
    // MTP:     allocator_configs.size() == 1 + num_mtp_modules.
    std::vector<KVCacheAllocatorConfig> allocator_configs;

    CacheConfig() {}

    size_t kernelBlocksPerKvBlock() const {
        if (kernel_seq_size_per_block == 0 || seq_size_per_block == 0) {
            return 1;
        }
        return seq_size_per_block / kernel_seq_size_per_block;
    }

    size_t modelNum() const {
        return std::max<size_t>(1, allocator_configs.size());
    }

    // Returns group count for the main model (delegates to allocator_configs[0]).
    int groupNums() const {
        if (allocator_configs.empty()) {
            return 1;
        }
        return allocator_configs[0].groupNums();
    }

    const KVCacheAllocatorConfig& getAllocatorConfig(size_t model_id = 0) const {
        RTP_LLM_CHECK_WITH_INFO(model_id < allocator_configs.size(),
                                "model_id %zu out of range (allocator_configs.size()=%zu)",
                                model_id,
                                allocator_configs.size());
        return allocator_configs[model_id];
    }

    std::string debugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

        // Cross-model shared block configuration
        os << indent1 << "# Block Configuration (shared):\n";
        OUTPUT_FIELD(seq_size_per_block);
        OUTPUT_FIELD(layer_all_num);
        os << "\n";

        // Allocator configs section
        os << indent1 << "# Allocator Configs:\n";
        OUTPUT_FIELD_EXPR("allocator_configs.size()", allocator_configs.size());
        for (size_t i = 0; i < allocator_configs.size(); ++i) {
            os << indent1 << "allocator_configs[" << i << "]: " << allocator_configs[i].debugString() << "\n";
        }
        os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

        os << indent_str << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
