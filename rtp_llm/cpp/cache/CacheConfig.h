#pragma once

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct CacheConfig {
    // Cache specification and layer mapping
    std::vector<KVCacheSpecPtr>    cache_specs;
    std::vector<std::vector<int>>  global_layer_ids;  // including mtp module layers
    std::vector<std::vector<int>>  layer_ids;
    std::vector<std::vector<int>>  linear_groups;  // for hybrid attention
    std::vector<std::vector<int>>  full_groups;    // for hybrid attention
    std::vector<CacheGroupType>    group_types;    // for hybrid attention
    std::vector<CacheGroupType>    layer_group_types;
    std::vector<KVCacheRegionName> group_region_names;        // group id -> cache identity
    std::vector<std::vector<int>>  layer_to_group_ids;        // layer id -> all group ids needed by the layer
    std::vector<std::vector<int>>  layer_region_to_group_id;  // layer id -> region name id -> group id
    std::vector<int>               layer_to_group_id;
    std::vector<int>               layer_to_block_stride_bytes;
    std::vector<size_t>            group_seq_size_per_block;
    std::vector<size_t>            group_kv_block_stride_bytes;
    std::vector<size_t>            group_kv_scale_stride_bytes;
    std::vector<size_t>            group_block_size_bytes;
    std::vector<uint32_t>          group_block_nums;
    uint32_t                       dsv4_fixed_pool_blocks                  = 0;
    bool                           use_independent_block_pools              = false;
    bool                           use_typed_cache_regions                  = false;
    bool                           use_opaque_kv_cache_store                = false;
    bool                           disable_decode_first_malloc_device_reuse = false;

    // Model configuration
    rtp_llm::DataType dtype;
    uint32_t          layer_num;      // the number of main model layers
    uint32_t          layer_all_num;  // the number of all layers including mtp modules
    bool              use_mla   = false;
    bool              is_sparse = false;

    // Block configuration
    uint32_t block_num;
    size_t   seq_size_per_block        = 1;
    size_t   kernel_seq_size_per_block = 1;

    // Returns how many kernel blocks fit inside one physical (kv-manager) block.
    size_t kernelBlocksPerKvBlock() const {
        if (kernel_seq_size_per_block == 0) {
            return 1;
        }
        return std::max<size_t>(1, seq_size_per_block / kernel_seq_size_per_block);
    }

    // Block sizing information
    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size_bytes  = 0;
    size_t kv_scale_size_bytes  = 0;
    size_t block_size_bytes     = 0;  // (kv + scales together)
    size_t swa_block_size_bytes = 0;  // SWA groups (joint allocation with full groups)

    // Per-block bytes of all DSV4 STATE pools (INDEXER_STATE, CSA_STATE,
    // HCA_STATE). Excluded from the HBM fixed-pool reservation only when
    // state_pool_uses_pinned_cpu is true.
    size_t state_block_size_bytes = 0;

    // True when STATE pools should be allocated on pinned CPU memory and
    // excluded from HBM fixed-pool reservation.
    bool state_pool_uses_pinned_cpu = false;

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    // Bytes pre-reserved for fixed-allocation pools (e.g. DSV4 state / SWA pools).
    // CacheConfigCreator deducts this from kv_cache_mem_size before computing the
    // paged block_num, so paged pools don't overcommit HBM. 0 means no reservation.
    size_t fixed_pool_reserve_bytes = 0;

    // Attention-specific configuration
    int linear_step = 1;  // For Linear attention: keep one cache block every `linear_step` blocks
    int linear_fixed_cap =
        0;  // >0 = ring buffer of this many blocks per LINEAR group (per request); 0 = legacy unbounded
    int group_layer_num  = 1;  // Number of layers per group for hybrid attention
    int linear_group_num = 0;  // Number of linear attention groups
    int swa_group_num    = 0;  // Number of sliding-window attention groups
    int full_group_num   = 0;  // Number of full attention groups

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    int groupNums() const {
        return std::max<int>(1, static_cast<int>(cache_specs.size()));
    }

    void finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) {
        (void)runtime_config;
        if (!use_independent_block_pools || group_block_nums.empty()) {
            fixed_pool_reserve_bytes = 0;
            return;
        }

        const int step    = std::max(1, linear_step);
        size_t    reserve = 0;
        for (size_t gid = 0; gid < group_block_nums.size(); ++gid) {
            const bool is_swa = gid < group_types.size() && group_types[gid] == CacheGroupType::SWA;
            const auto region = gid < group_region_names.size() ? group_region_names[gid] : KVCacheRegionName::DEFAULT;
            const bool is_state                       = isStateRegion(region);
            const bool is_dsv4_fixed_region           = is_state || region == KVCacheRegionName::SWA_KV;
            const bool use_explicit_dsv4_fixed_blocks = is_dsv4_fixed_region && dsv4_fixed_pool_blocks > 0;
            uint32_t   rule_blocks;
            if (use_explicit_dsv4_fixed_blocks) {
                rule_blocks = dsv4_fixed_pool_blocks;
            } else if (is_swa && step > 1 && global_block_num > 0) {
                rule_blocks = std::max(1u, global_block_num / static_cast<uint32_t>(step));
            } else {
                rule_blocks = global_block_num;
            }
            group_block_nums[gid] = rule_blocks;

            // Explicit DSV4 fixed pools are allocated outside the paged FULL
            // pool budget. The linear-step fallback is accounted by the
            // effective block-size formula instead, so no reserve is needed.
            const bool exclude_from_reserve = is_state && state_pool_uses_pinned_cpu;
            if (use_explicit_dsv4_fixed_blocks && gid < group_block_size_bytes.size() && !exclude_from_reserve) {
                reserve += static_cast<size_t>(rule_blocks) * group_block_size_bytes[gid];
            }
        }
        fixed_pool_reserve_bytes = reserve;
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
        OUTPUT_FIELD(kernel_seq_size_per_block);
        os << "\n";

        // Block sizing information section
        os << indent1 << "# Block Sizing Information:\n";
        OUTPUT_FIELD(kv_block_size_bytes);
        OUTPUT_FIELD(kv_scale_size_bytes);
        OUTPUT_FIELD(block_size_bytes);
        OUTPUT_FIELD(swa_block_size_bytes);
        OUTPUT_FIELD(kv_block_stride_bytes);
        OUTPUT_FIELD(kv_scale_stride_bytes);
        os << "\n";

        // Attention-specific configuration section
        os << indent1 << "# Attention Configuration:\n";
        OUTPUT_FIELD(linear_step);
        OUTPUT_FIELD(linear_fixed_cap);
        OUTPUT_FIELD(group_layer_num);
        OUTPUT_FIELD(linear_group_num);
        OUTPUT_FIELD(swa_group_num);
        OUTPUT_FIELD(full_group_num);
        OUTPUT_FIELD(dsv4_fixed_pool_blocks);
        OUTPUT_FIELD(use_independent_block_pools);
        OUTPUT_FIELD(use_typed_cache_regions);
        OUTPUT_FIELD(use_opaque_kv_cache_store);
        OUTPUT_FIELD(disable_decode_first_malloc_device_reuse);
        os << indent1 << "group_block_nums=" << rtp_llm::vectorToString(group_block_nums) << "\n";
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
        OUTPUT_FIELD_EXPR("group_region_names.size()", group_region_names.size());
        os << indent1 << "group_region_names=[";
        for (size_t i = 0; i < group_region_names.size(); ++i) {
            os << static_cast<int>(group_region_names[i]);
            if (i + 1 < group_region_names.size()) {
                os << ",";
            }
        }
        os << "]\n";
        OUTPUT_FIELD_EXPR("layer_to_group_ids.size()", layer_to_group_ids.size());
        os << indent1 << "layer_to_group_ids=" << rtp_llm::vectorsToString(layer_to_group_ids) << "\n";
        OUTPUT_FIELD_EXPR("layer_group_types.size()", layer_group_types.size());
        os << indent1 << "layer_group_types=[";
        for (size_t i = 0; i < layer_group_types.size(); ++i) {
            os << static_cast<int>(layer_group_types[i]);
            if (i + 1 < layer_group_types.size()) {
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
