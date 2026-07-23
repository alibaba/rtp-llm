#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
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

    size_t kernelBlocksPerKvBlock(std::string_view tag) const {
        const auto& group        = topology().group(tag);
        const auto  group_seq    = group.seq_size_per_block;
        const auto  group_kernel = group.kernel_seq_size_per_block;
        if (group_kernel == 0) {
            return 1;
        }
        RTP_LLM_CHECK_WITH_INFO(
            group_seq % group_kernel == 0,
            "group seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu), tag=%s",
            group_seq,
            group_kernel,
            group.tag.c_str());
        return std::max<size_t>(1, group_seq / group_kernel);
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

    const GroupBase& group(const std::string& tag) const {
        return topology().group(tag);
    }

    CacheTopology::GroupRefs groupsForLayer(int layer_id) const {
        return topology().groupsForLayer(layer_id);
    }

    const GroupBase& groupForLayer(int layer_id, const std::string& tag) const {
        return topology().groupForLayer(layer_id, tag);
    }

    const GroupBase& soleGroupForLayer(int layer_id) const {
        return topology().soleGroupForLayer(layer_id);
    }

    std::vector<CacheGroupType> groupTypesSnapshot() const {
        const auto& snapshot = topology().groupTypesSnapshot();
        return {snapshot.begin(), snapshot.end()};
    }

    std::vector<KVCacheSpecType> groupSpecTypesSnapshot() const {
        const auto& snapshot = topology().groupSpecTypesSnapshot();
        return {snapshot.begin(), snapshot.end()};
    }

    std::vector<std::string> groupTagsSnapshot() const {
        const auto& snapshot = topology().groupTagsSnapshot();
        return {snapshot.begin(), snapshot.end()};
    }

    std::vector<CacheGroupPolicy> groupPoliciesSnapshot() const {
        std::vector<CacheGroupPolicy> policies;
        policies.reserve(topology().groups().size());
        for (const auto& group : topology().groups()) {
            policies.push_back(group.policy);
        }
        return policies;
    }

    std::vector<size_t> groupSeqBlockSizesSnapshot() const {
        std::vector<size_t> values;
        values.reserve(topology().groups().size());
        for (const auto& group : topology().groups()) {
            values.push_back(group.seq_size_per_block);
        }
        return values;
    }

    std::vector<size_t> groupKernelSeqBlockSizesSnapshot() const {
        std::vector<size_t> values;
        values.reserve(topology().groups().size());
        for (const auto& group : topology().groups()) {
            values.push_back(group.kernel_seq_size_per_block);
        }
        return values;
    }

    std::vector<size_t> groupKernelBlocksPerKvBlockSnapshot() const {
        std::vector<size_t> values;
        values.reserve(topology().groups().size());
        for (const auto& group : topology().groups()) {
            values.push_back(kernelBlocksPerKvBlock(group.tag));
        }
        return values;
    }

    std::vector<uint32_t> groupBlockNumsSnapshot() const {
        if (!group_block_layout_initialized) {
            return {};
        }
        std::vector<uint32_t> block_nums;
        block_nums.reserve(topology().groups().size());
        for (const auto& group : topology().groups()) {
            block_nums.push_back(group.block_num);
        }
        return block_nums;
    }

    std::vector<size_t> groupBlockSizeBytesSnapshot() const {
        std::vector<size_t> result;
        result.reserve(static_cast<size_t>(groupNums()));
        for (const auto& group : topology().groups()) {
            result.push_back(blockSizeBytes(group.tag));
        }
        return result;
    }

    std::vector<size_t> groupKvBlockStrideBytesSnapshot() const {
        if (!group_block_layout_initialized) {
            return {};
        }
        std::vector<size_t> strides;
        strides.reserve(topology().groups().size());
        for (const auto& group : topology().groups()) {
            strides.push_back(group.kv_block_stride_bytes);
        }
        return strides;
    }

    std::vector<size_t> groupKvScaleStrideBytesSnapshot() const {
        if (!group_block_layout_initialized) {
            return {};
        }
        std::vector<size_t> strides;
        strides.reserve(topology().groups().size());
        for (const auto& group : topology().groups()) {
            strides.push_back(group.kv_scale_stride_bytes);
        }
        return strides;
    }

    size_t blockSizeBytes(std::string_view tag) const {
        const auto& group = topology().group(tag);
        return group.layer_ids.size() * (group.kv_block_stride_bytes + group.kv_scale_stride_bytes);
    }

    uint32_t localKvHeadNum(std::string_view tag) const {
        const auto& group = topology().group(tag);
        RTP_LLM_CHECK_WITH_INFO(group.local_kv_head_num > 0,
                                "CacheConfig::localKvHeadNum invalid local_kv_head_num=%u tag=%s",
                                group.local_kv_head_num,
                                group.tag.c_str());
        return group.local_kv_head_num;
    }

    void setGroupPolicies(const std::vector<CacheGroupPolicy>& policies);

    void setGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                             const std::vector<size_t>&   kv_block_stride_bytes,
                             const std::vector<size_t>&   kv_scale_stride_bytes);

    std::shared_ptr<CacheConfig>
    mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);

    uint32_t explicitIndependentBlocks(std::string_view tag) const {
        return topology().group(tag).policy.explicit_block_num;
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
