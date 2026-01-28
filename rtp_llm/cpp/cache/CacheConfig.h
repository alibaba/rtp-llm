#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct CacheConfig {
    // Cache specification and layer mapping
    std::vector<KVCacheSpecPtr>   cache_specs;
    std::vector<std::vector<int>> global_layer_ids;  // including mtp module layers
    std::vector<std::vector<int>> layer_ids;
    std::vector<std::vector<int>> linear_groups;  // for hybrid attention
    std::vector<std::vector<int>> full_groups;    // for hybrid attention
    std::vector<CacheGroupType>   group_types;    // for hybrid attention
    std::vector<int>              layer_to_group_id;
    std::vector<int>              layer_to_block_stride_bytes;

    // Model configuration
    rtp_llm::DataType dtype;
    uint32_t          layer_num;      // the number of main model layers
    uint32_t          layer_all_num;  // the number of all layers including mtp modules
    bool              use_mla = false;

    // Block configuration
    uint32_t block_num;
    size_t   seq_size_per_block = 1;

    // Block sizing information
    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size_bytes    = 0;  // (kv + scales together)

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    // Attention-specific configuration
    int linear_step      = 1;  // For Linear attention: keep one cache block every `linear_step` blocks
    int group_layer_num  = 1;  // Number of layers per group for hybrid attention
    int linear_group_num = 0;  // Number of linear attention groups
    int full_group_num   = 0;  // Number of full attention groups

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    int groupNums() const {
        return std::max<int>(1, static_cast<int>(cache_specs.size()));
    }

    std::string debugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";
        const std::string indent2    = indent_str + "    ";

        std::ostringstream os;
        os << indent_str << "CacheConfig{\n";

// Macro to simplify repeated field output and eliminate duplicate field names
#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"

// Helper macro for complex expressions
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

        // Model configuration section
        os << indent1 << "# Model Configuration:\n";
        OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
        OUTPUT_FIELD(layer_num);
        OUTPUT_FIELD(layer_all_num);
        OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
        os << "\n";

        // Block configuration section
        os << indent1 << "# Block Configuration:\n";
        OUTPUT_FIELD(block_num);
        OUTPUT_FIELD(seq_size_per_block);
        os << "\n";

        // Block sizing information section
        os << indent1 << "# Block Sizing Information:\n";
        OUTPUT_FIELD(kv_block_size_bytes);
        OUTPUT_FIELD(kv_scale_size_bytes);
        OUTPUT_FIELD(block_size_bytes);
        OUTPUT_FIELD(kv_block_stride_bytes);
        OUTPUT_FIELD(kv_scale_stride_bytes);
        os << "\n";

        // Attention-specific configuration section
        os << indent1 << "# Attention Configuration:\n";
        OUTPUT_FIELD(linear_step);
        OUTPUT_FIELD(group_layer_num);
        OUTPUT_FIELD(linear_group_num);
        OUTPUT_FIELD(full_group_num);
        os << "\n";

        // Cache specification section
        os << indent1 << "# Cache Specifications:\n";
        OUTPUT_FIELD_EXPR("cache_specs.size()", cache_specs.size());
        for (size_t i = 0; i < cache_specs.size(); ++i) {
            const auto& spec = cache_specs[i];
            if (!spec) {
                os << indent1 << "cache_specs[" << i << "]=null\n";
                continue;
            }

            os << indent1 << "cache_specs[" << i << "] {\n";
            os << spec->debugString(indent + 2);
            os << indent1 << "}\n";
        }
        os << "\n";

        // Layer mapping section
        os << indent1 << "# Layer Mapping:\n";
        OUTPUT_FIELD_EXPR("global_layer_ids.size()", global_layer_ids.size());
        os << indent1 << "global_layer_ids=" << rtp_llm::vectorsToString(global_layer_ids) << "\n";
        OUTPUT_FIELD_EXPR("layer_ids.size()", layer_ids.size());
        os << indent1 << "layer_ids=" << rtp_llm::vectorsToString(layer_ids) << "\n";
        OUTPUT_FIELD_EXPR("group_types.size()", group_types.size());
        os << indent1 << "group_types=[";
        for (size_t i = 0; i < group_types.size(); ++i) {
            os << static_cast<int>(group_types[i]);
            if (i + 1 < group_types.size()) {
                os << ",";
            }
        }
        os << "]\n";
        os << "\n";

        // mtp configurations section
        os << indent1 << "# MTP Configurations:\n";
        OUTPUT_FIELD_EXPR("mtp_sub_configs.size()", mtp_sub_configs.size());
        for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
            const auto& sub = mtp_sub_configs[i];
            if (!sub) {
                os << indent1 << "mtp_sub_configs[" << i << "]=null\n";
                continue;
            }
            os << indent1 << "mtp_sub_configs[" << i << "]:\n";
            os << sub->debugString(indent + 4);
        }
        os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

        os << indent_str << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
