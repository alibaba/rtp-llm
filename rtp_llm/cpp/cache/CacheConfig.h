#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

struct CacheConfig {
private:
    std::shared_ptr<const CacheTopology> cache_topology;

public:
    std::vector<int> layer_to_block_stride_bytes;
    bool             group_block_layout_initialized           = false;
    bool             use_independent_block_pools              = false;
    bool             use_typed_cache_regions                  = false;
    bool             use_opaque_kv_cache_store                = false;
    bool             disable_decode_first_malloc_device_reuse = false;

    rtp_llm::DataType dtype         = rtp_llm::DataType::TYPE_INVALID;
    uint32_t          layer_num     = 0;  // the number of main model layers
    uint32_t          layer_all_num = 0;  // the number of all layers including mtp modules
    bool              use_mla       = false;
    bool              is_sparse     = false;

    // Block configuration
    uint32_t block_num                 = 0;
    size_t   seq_size_per_block        = 1;
    size_t   kernel_seq_size_per_block = 0;

    // How many kernel blocks fit inside one physical block of a cache group.
    // Derived and validated, so it stays a method; every plain per-group field is
    // read directly from the tagged group record via group(tag).
    size_t kernelBlocksPerKvBlock(std::string_view tag) const {
        const auto& group_config = group(tag);
        if (group_config.kernel_seq_size_per_block == 0) {
            return 1;
        }
        RTP_LLM_CHECK_WITH_INFO(
            group_config.seq_size_per_block % group_config.kernel_seq_size_per_block == 0,
            "group seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu), tag=%s",
            group_config.seq_size_per_block,
            group_config.kernel_seq_size_per_block,
            group_config.tag.c_str());
        return std::max<size_t>(1, group_config.seq_size_per_block / group_config.kernel_seq_size_per_block);
    }

    // Legacy scalar view: how many kernel blocks fit inside one global physical block.
    size_t kernelBlocksPerKvBlock() const {
        if (kernel_seq_size_per_block == 0) {
            return 1;
        }
        RTP_LLM_CHECK_WITH_INFO(seq_size_per_block % kernel_seq_size_per_block == 0,
                                "seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                                seq_size_per_block,
                                kernel_seq_size_per_block);
        return std::max<size_t>(1, seq_size_per_block / kernel_seq_size_per_block);
    }

    // Block sizing information
    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size_bytes    = 0;  // (kv + scales together)

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    // Attention-specific configuration
    int    linear_step     = 1;  // For Linear attention: keep one cache block every `linear_step` blocks
    int    group_layer_num = 1;  // Number of layers per group for hybrid attention
    size_t explicitly_sized_pool_reserve_bytes = 0;

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    static uint32_t
    mtpGlobalLayerId(uint32_t main_layer_num, int module_index, uint32_t module_layer_num, int local_layer_id) {
        constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
        if (module_index < 0 || module_layer_num == 0 || local_layer_id < 0
            || static_cast<uint32_t>(local_layer_id) >= module_layer_num) {
            return invalid;
        }
        const uint64_t global_layer_id = static_cast<uint64_t>(main_layer_num)
                                         + static_cast<uint64_t>(module_index) * module_layer_num
                                         + static_cast<uint32_t>(local_layer_id);
        return global_layer_id < invalid ? static_cast<uint32_t>(global_layer_id) : invalid;
    }

    int groupNums() const {
        return cache_topology == nullptr ? 0 : static_cast<int>(cache_topology->groups().size());
    }

    const CacheTopology& topology() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheConfig topology is not initialized");
        return *cache_topology;
    }

    const std::shared_ptr<const CacheTopology>& topologyPtr() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheConfig topology is not initialized");
        return cache_topology;
    }

    const GroupBase& group(std::string_view tag) const {
        return topology().group(tag);
    }

    const std::vector<std::string>& groupsForLayer(int layer_id) const {
        return topology().layer(layer_id).group_tags;
    }

    const GroupBase& groupForLayer(int layer_id, std::string_view tag) const {
        return topology().groupForLayer(layer_id, tag);
    }

    const GroupBase& soleGroupForLayer(int layer_id) const {
        return topology().soleGroupForLayer(layer_id);
    }

    // Total bytes one logical block of a cache group occupies across all of the
    // group's layers. Derived, so it stays a method.
    size_t blockSizeBytes(std::string_view tag) const {
        const auto& group_config = group(tag);
        return group_config.layer_ids.size()
               * (group_config.kv_block_stride_bytes + group_config.kv_scale_stride_bytes);
    }

    uint32_t localKvHeadNum(std::string_view tag) const {
        const auto& group_config = group(tag);
        RTP_LLM_CHECK_WITH_INFO(group_config.local_kv_head_num > 0,
                                "CacheConfig::localKvHeadNum invalid local_kv_head_num=%u tag=%s",
                                group_config.local_kv_head_num,
                                group_config.tag.c_str());
        return group_config.local_kv_head_num;
    }

    void setGroupPolicies(const std::vector<CacheGroupPolicy>& policies);

    void setGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                             const std::vector<size_t>&   kv_block_stride_bytes,
                             const std::vector<size_t>&   kv_scale_stride_bytes);

    std::shared_ptr<CacheConfig>
    mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);

    uint32_t explicitIndependentBlocks(std::string_view tag) const {
        return group(tag).policy.explicit_block_num;
    }

    bool usesExplicitIndependentBlocks(std::string_view tag) const {
        return explicitIndependentBlocks(tag) > 0;
    }

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);

    void        setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers);
    void        fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                 const std::vector<std::vector<int>>& layers_by_group,
                                 const std::vector<CacheGroupType>&   types,
                                 const std::vector<std::string>&      tags     = {},
                                 const std::vector<CacheGroupPolicy>& policies = {});
    void        finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config);
    std::string debugString(size_t indent = 0) const;
};

}  // namespace rtp_llm
